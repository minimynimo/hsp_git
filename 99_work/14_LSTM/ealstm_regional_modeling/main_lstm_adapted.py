"""
Adapted (regional model fine-tuned per basin) evaluation for the true EA-LSTM regional
model trained by `main.py` in this directory.

Why this file exists instead of reusing `x_lstm_regional_modeling-master/eval_adapted.py`:
that script's `Model.forward` only ever calls `self.lstm(x_d)` when not concat_static/
no_static -- wait, more precisely, its `FineTuneDataset` always builds `x` by tiling static
attributes into every timestep of the dynamic sequence (the "concat_static" convention) and
its fine-tuning/eval loops call `basin_model(x)` with a single tensor, never passing a
separate `x_s`. A checkpoint from THIS project's `main.py` (no_static=False,
concat_static=False, i.e. true EA-LSTM) has `input_size_dyn == 3` (dynamic only) and its
`EALSTM` layer requires `x_s` of shape [batch, n_static] passed SEPARATELY
(`papercode/ealstm.py: EALSTM.forward(x_d, x_s)`). Feeding it `eval_adapted.py`'s
concat-static-tiled `x` of width `3 + n_static` would be a hard shape mismatch against the
loaded weights (`weight_ih` sized for `input_size_dyn=3`), not a silent numerical difference
-- so that script cannot be reused as-is for a true-EA-LSTM checkpoint.

This file instead fine-tunes per basin using this directory's own `HydroDataset` (main.py)
and evaluates using this directory's own `EvalDataset` / `evaluate_basin` (eval.py), so the
x_d/x_s calling convention and per-basin scalers exactly match how the base regional model
was trained. Usage:

    python adapted.py --run_dir runs/run_XXXX_US_seedNN --n_basins 20 --ft_epochs 5

Output goes to /data0/funato/hsp2/0_out/LSTM/{loc}/{dataset}/adapted/ (`dataset` being the run's
--dataset, 'camels' or 'caravan' -- see main_lstm_regional.py), distinct from both the
old x_lstm-based .../adapted/ and this directory's own regional/ output.

RUN USING (--run_dir is the output directory printed by main_lstm_regional.py's training run;
loc/dataset/hyperparameters are all read back from that run's own cfg.json, not passed here):
nohup python main_lstm_adapted.py --run_dir runs/run_<timestamp>_US_camels_seed<N> > adapted_output.log 2>&1 &

Quick smoke test (few basins):
python main_lstm_adapted.py --run_dir runs/run_<timestamp>_US_camels_seed<N> --n_basins 5
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_lstm_regional import Model, HydroDataset, DEVICE
from papercode.lstm_run_registry import load_lstm_state_dict
from papercode.datautils import load_attributes
from papercode.nseloss import NSELoss
from papercode.utils import get_basin_list
from eval import EvalDataset, BMK, benchmark_list, _load_scalers, _param_tag


def evaluate_basin(model, basin, dates, cfg, scalers, attributes):
    """Local variant of eval.py's `evaluate_basin`, adjusted for `main.Model.forward`'s
    return signature (`out, h_n, c_n` tuple) instead of eval.py's own `Model` class (plain
    tensor) -- otherwise identical (same EvalDataset, same per-basin rescaling)."""
    model.eval()
    if basin not in scalers['basin_scalers']:
        return None, None, None

    dataset = EvalDataset(basin, dates, cfg, scalers, attributes)
    if len(dataset) == 0:
        return None, None, None

    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    bscaler = scalers['basin_scalers'][basin]

    true_ealstm = (not cfg.get('no_static', True)) and (not cfg.get('concat_static', False))

    preds_list, obs_list = [], []
    with torch.no_grad():
        for x, x_s, y_raw in loader:
            x = x.to(DEVICE)
            if true_ealstm:
                preds_scaled = model(x, x_s.to(DEVICE))[0].cpu().numpy().flatten()
            else:
                preds_scaled = model(x)[0].cpu().numpy().flatten()

            preds_rescaled = preds_scaled * bscaler['q_std'] + bscaler['q_mean']
            preds_rescaled[preds_rescaled < 0] = 0
            preds_list.extend(preds_rescaled)
            obs_list.extend(y_raw.numpy())

    return np.array(preds_list), np.array(obs_list), dataset.dates


def fine_tune_basin(base_model, basin, cfg, scalers, attributes, ft_epochs, ft_lr):
    """Deep-copy the regional model and fine-tune it for a few epochs on ONE basin's own
    training-period data, using that basin's own per-basin scaler (already computed at
    regional-training time and stored in scalers.json) -- mirrors main.py's train_epoch, just
    scoped to a single basin's DataLoader."""
    single_basin_scalers = {'basin_scalers': {basin: scalers['basin_scalers'][basin]}}
    if 'static_means' in scalers:
        single_basin_scalers['static_means'] = scalers['static_means']
        single_basin_scalers['static_stds'] = scalers['static_stds']

    ds = HydroDataset([basin], (cfg['train_start'], cfg['train_end']), cfg,
                       single_basin_scalers, attributes)
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=min(cfg['batch_size'], len(ds)), shuffle=True)

    model = copy.deepcopy(base_model).to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
    loss_func = nn.MSELoss() if cfg.get('use_mse', False) else NSELoss()

    true_ealstm = (not cfg['no_static']) and (not cfg['concat_static'])

    for _ in range(ft_epochs):
        for x, x_s, y, q_params in loader:
            optimizer.zero_grad()
            x, y, q_params = x.to(DEVICE), y.to(DEVICE), q_params.to(DEVICE)
            if true_ealstm:
                predictions = model(x, x_s.to(DEVICE))[0]
            else:
                predictions = model(x)[0]

            if cfg.get('use_mse', False):
                loss = loss_func(predictions, y)
            else:
                q_means_batch = q_params[:, 0].unsqueeze(1)
                q_stds_batch = q_params[:, 1].unsqueeze(1)
                pred_rescaled = predictions * q_stds_batch + q_means_batch
                y_rescaled = y * q_stds_batch + q_means_batch
                loss = loss_func(pred_rescaled, y_rescaled, q_stds_batch.flatten())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Adapted (per-basin fine-tuned) EA-LSTM evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                         help='Regional model run dir produced by main.py train')
    parser.add_argument('--model_epoch', type=int, default=None,
                         help='Which model_epoch{N}.pt to start fine-tuning from. Default: last available.')
    parser.add_argument('--n_basins', type=int, default=None)
    parser.add_argument('--ft_epochs', type=int, default=5, help='Fine-tuning epochs per basin')
    parser.add_argument('--ft_lr', type=float, default=1e-4, help='Fine-tuning learning rate')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print("Adapted EA-LSTM Evaluation (regional model fine-tuned per basin)")
    print(f"Run directory: {run_dir}")

    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)
    cfg['train_start'] = pd.to_datetime(cfg['train_start'])
    cfg['train_end'] = pd.to_datetime(cfg['train_end'])
    val_start = pd.to_datetime(cfg['val_start'])
    val_end = pd.to_datetime(cfg['val_end'])

    scalers = _load_scalers(run_dir)

    model_epoch = args.model_epoch if args.model_epoch is not None else cfg['epochs']
    model_file = run_dir / f"model_epoch{model_epoch}.pt"
    if not model_file.exists():
        model_file = run_dir / "lstm_model_final.pt"
        if not model_file.exists():
            candidates = sorted(run_dir.glob('model_epoch*.pt'))
            model_file = candidates[-1] if candidates else None
    if model_file is None or not model_file.exists():
        raise ValueError(f"No .pt model file found in {run_dir}")
    print(f"Loading base regional model from: {model_file}")

    is_no_static = cfg.get('no_static', False)
    is_concat_static = cfg.get('concat_static', False)
    input_size_stat = cfg.get('input_size_stat', 0)
    input_size_dyn = cfg.get('input_size_dyn', 3)

    base_model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat,
                        hidden_size=cfg['hidden_size'], dropout=cfg.get('dropout', 0.0),
                        concat_static=is_concat_static, no_static=is_no_static,
                        initial_forget_bias=cfg.get('initial_forget_gate_bias', 5)).to(DEVICE)
    base_model.load_state_dict(load_lstm_state_dict(model_file, map_location=DEVICE))

    loc = cfg['loc']
    dataset = cfg.get('dataset', 'camels')
    basins = get_basin_list(loc=loc, dataset=dataset)
    if args.n_basins:
        basins = basins[:args.n_basins]
    basins = [b for b in basins if b in scalers['basin_scalers']]

    attributes = None
    if not is_no_static:
        db_path = str(run_dir / 'attributes.db')
        attributes = load_attributes(db_path, basins)

    output_base = Path(f"/data0/funato/hsp2/0_out/LSTM/{loc}/{dataset}/adapted")
    tag = _param_tag(cfg, model_epoch) + f"_ft{args.ft_epochs}_ftlr{args.ft_lr}"

    period_dates = {'cal': (cfg['train_start'], cfg['train_end']), 'eva': (val_start, val_end)}
    pred_paths = {}
    results_paths = {}
    for name in ['cal', 'eva']:
        pred_paths[name] = output_base / "predict" / f"LSTM_predict_{tag}_{name}.csv"
        pred_paths[name].parent.mkdir(parents=True, exist_ok=True)
        results_paths[name] = output_base / "results" / f"LSTM_results_{tag}_{name}.csv"
        results_paths[name].parent.mkdir(parents=True, exist_ok=True)

    # A restart after a kill previously redid every basin's fine-tuning from scratch (no
    # per-basin skip) and only wrote metrics CSVs once, after the ENTIRE basin loop finished --
    # worse than the individual-model scripts, where at least a partial results CSV survived a
    # kill. A basin only counts as completed if it has a row in BOTH the cal and eva results
    # CSVs (a basin present in only one implies a kill mid-basin).
    results = {'cal': [], 'eva': []}
    completed_basins = set()
    if results_paths['cal'].exists() and results_paths['eva'].exists():
        # dtype=str is required: gauge_id values like "01022500" have a leading zero, and
        # pd.read_csv's default type inference silently parses an all-digit column as int64,
        # dropping it (-> 1022500). That mismatches the loop's string gauge_id below, so the
        # "already completed" skip check never matches and every basin gets reprocessed and
        # duplicated on every resume -- see DEBUG_README.md §22.2 for the same bug found (and
        # fixed) in all four *_indiv_resume.py modules.
        df_cal = pd.read_csv(results_paths['cal'], dtype={'gauge_id': str})
        df_eva = pd.read_csv(results_paths['eva'], dtype={'gauge_id': str})
        cal_ids = set(df_cal['gauge_id'].astype(str))
        eva_ids = set(df_eva['gauge_id'].astype(str))
        completed_basins = cal_ids & eva_ids
        results['cal'] = df_cal[df_cal['gauge_id'].astype(str).isin(completed_basins)].to_dict('records')
        results['eva'] = df_eva[df_eva['gauge_id'].astype(str).isin(completed_basins)].to_dict('records')
    if completed_basins:
        print(f"Resuming: {len(completed_basins)} basins already completed, skipping them.")

    # Predict-CSV header is written on the first row ever appended to a NEW file -- keyed off
    # file existence (not a per-run flag) so a resumed run never re-truncates a predict CSV
    # that already has earlier basins' rows in it.
    first_basin = {name: not pred_paths[name].exists() for name in ['cal', 'eva']}

    for basin in tqdm(basins, desc="Fine-tuning + evaluating basins"):
        if str(basin) in completed_basins:
            continue

        ft_model = fine_tune_basin(base_model, basin, cfg, scalers, attributes,
                                    args.ft_epochs, args.ft_lr)
        if ft_model is None:
            continue

        for name, dates in period_dates.items():
            preds, obs, eval_dates = evaluate_basin(ft_model, basin, dates, cfg, scalers, attributes)
            if preds is None:
                continue

            basin_row_dict = {d.strftime('%Y-%m-%d'): preds[i] for i, d in enumerate(eval_dates)}
            basin_row = pd.DataFrame([basin_row_dict], index=[basin])
            basin_row.index.name = 'gauge_id'
            basin_row.to_csv(pred_paths[name], mode='w' if first_basin[name] else 'a',
                              header=first_basin[name])
            first_basin[name] = False

            metrics_row = {'gauge_id': basin}
            for benchmark in benchmark_list:
                metrics_row[benchmark] = BMK(obs, preds, benchmark)
            results[name].append(metrics_row)

        # Save results incrementally so a kill mid-run only loses the basin in flight, not
        # everything processed so far.
        for name in ['cal', 'eva']:
            if results[name]:
                pd.DataFrame(results[name]).to_csv(results_paths[name], index=False)

    for name in ['cal', 'eva']:
        if not pred_paths[name].exists():
            print(f"No data to evaluate for {name}.")
            continue
        temp_df = pd.read_csv(pred_paths[name], index_col='gauge_id', dtype={'gauge_id': str})
        temp_df.sort_index().to_csv(pred_paths[name])
        print(f"Predictions saved to {pred_paths[name]}")

        if not results[name]:
            continue
        metrics_df = pd.DataFrame(results[name])
        metrics_df.to_csv(results_paths[name], index=False)
        print(f"Metrics saved to {results_paths[name]}")
        print(f"\n--- {name.upper()} Period Summary ---\n{metrics_df.describe()}\n")


if __name__ == "__main__":
    main()
