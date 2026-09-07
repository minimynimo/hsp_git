"""
Evaluation script for the EA-LSTM regional model trained by main.py in this directory.

ADAPTED for this project (hsp2 / 99_work/14_LSTM/ealstm_regional_modeling), modeled on
`x_lstm_regional_modeling-master/eval_regional.py`'s structure: load cfg.json + scalers.json,
rebuild the model, evaluate per-basin on the cal (train) and eval (val) periods, and compute
the same BMK-style metrics (NSE, KGE, E1, VE, d, RMSE, MAE) used elsewhere in this repo
(see run_LSTM.py's `BMK`).

Reproducibility note: the old `eval_regional.py` tagged its output CSVs only with
`hs{hidden}_do{dropout}_e{epoch}`, which does NOT encode no_static/concat_static/seq_length --
two differently-configured runs (e.g. no_static=True vs true EA-LSTM) could silently overwrite
each other's results. `_param_tag` below additionally encodes no_static, concat_static and
seq_length so that cannot happen. Output also goes to a location dedicated to this script
(`0_out/LSTM/{loc}/regional_ealstm/`), distinct from
`x_lstm_regional_modeling-master/eval_regional.py`'s `0_out/LSTM/{loc}/regional/`, so the two
implementations' results can never collide either.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.datautils import load_attributes
from papercode.utils import get_basin_list

sys.path.insert(0, "/data0/funato/hsp2/99_work/00_analysis/data")
from ml_predict_long_format import save_regional_predictions_long_format

from main_lstm_regional import DYN_COLS, Q_COL, load_data_for_basin  # reuse training script's data conventions
from papercode.lstm_run_registry import load_lstm_state_dict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

# ####################### #
# Synthetic precipitation scenarios (regional-model only; see main()'s --synthetic flag)
# ####################### #
#
# Full 14-scenario list, ported from the legacy Keras `x_main_LSTM_regional.py` (its
# `syn_precip_cols`, module-level, lines ~46-49). The legacy PyTorch
# `x_lstm_regional_modeling-master/eval_regional.py` only had 13 of these (missing
# `syn_p_pet`) -- this list restores the full set.
#
# Each varssim CSV (`2_MERV/varssim_{loc}/varssim_{gauge_id}.csv`) is expected to already contain a
# pre-computed synthetic-Precip-substitute column named exactly like the scenario (e.g.
# `syn_no_precip`, `syn_peak1_rate10`), plus, for the 'peak' scenarios only, companion columns
# `Temp_at_peak`/`PET_at_peak`/`Obs_flow_at_peak` that replace the normal `Temp`/`PET`/
# `Obs flow` columns for that scenario's evaluation window.
#
# NOTE (data-availability caveat, same pattern as the GB-static-attribute / JP-no-data caveats
# documented in ANALYSIS_README.md): as of this pass, `syn_p_pet` does NOT actually exist as a
# column in either location's varssim CSVs, even though
# `2_MERV/x_create_csv_from_varssim.py` lists it among the columns it intends to copy. This is
# a real upstream data gap, not a bug introduced here -- `run_synthetic_evaluation` below
# detects a missing synthetic column per-basin and skips that scenario for that basin with a
# one-time warning, rather than crashing or silently fabricating data.
SYN_PRECIP_COLS = [
    'syn_no_precip', 'syn_p_pet',
    'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
    'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
    'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50',
]


# ####################### #
# Model Definition        #
# ####################### #


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with a fully connected layer (mirrors main.py)."""

    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int,
                 initial_forget_bias: int = 5, dropout: float = 0.0,
                 concat_static: bool = False, no_static: bool = False):
        super(Model, self).__init__()
        self.concat_static = concat_static
        self.no_static = no_static
        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn, hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat,
                               hidden_size=hidden_size, initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None):
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        return self.fc(last_h)


# ####################### #
# Metrics (same formulas as run_LSTM.py's BMK / x_lstm eval_regional.py's BMK)
# ####################### #


def BMK(obs_data, sim_data, benchmark):
    obs_data = np.asarray(obs_data).flatten()
    sim_data = np.asarray(sim_data).flatten()
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data, sim_data = obs_data[mask], sim_data[mask]

    if len(obs_data) == 0:
        return np.nan
    if np.sum(obs_data) == 0 and benchmark not in ["RMSE", "MAE"]:
        return np.nan

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.square(obs_data - obs_ave))
        if denom == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / denom)

    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave, sim_ave = np.mean(obs_data), np.mean(sim_data)
        obs_std, sim_std = np.std(obs_data), np.std(sim_data)
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave)
        return 1 - np.sqrt((r - 1) ** 2 + ((sim_std / obs_std) - 1) ** 2 + ((sim_ave / obs_ave) - 1) ** 2)

    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)

    elif benchmark == "VE":
        if np.sum(obs_data) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)

    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return 1.0 if numer == 0 else np.nan
        return 1 - numer / denom

    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))

    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))


# ####################### #
# Data Handling           #
# ####################### #


class EvalDataset(Dataset):
    """Builds evaluation sequences for one basin, with a `seq_length - 1` day warm-up window
    before `dates[0]` (so the first predicted day still has a full input sequence), using the
    same per-basin scalers computed at training time.

    Like main.py's HydroDataset, always yields (x_d, x_s, y): x_s is the real static vector
    only in the true-EA-LSTM case (not no_static, not concat_static); otherwise it is an
    unused zero-length placeholder, since with concat_static the static attributes are already
    tiled into x_d, and with no_static there are none.
    """

    def __init__(self, basin: str, dates, cfg, scalers, attributes, syn_col: str = None):
        """`syn_col`, if given, is one of `SYN_PRECIP_COLS` (a synthetic-precipitation-scenario
        column name already present in the basin's varssim CSV). When set, sequences are built
        from that column in place of the real `Precip`, and -- for 'peak' scenarios only -- from
        the `Temp_at_peak`/`PET_at_peak`/`Obs_flow_at_peak` companion columns in place of the
        normal `Temp`/`PET`/`Obs flow` columns (non-peak scenarios keep the normal Temp/PET/Obs
        flow columns, only Precip is swapped). The SAME per-basin `dynamic_means`/`dynamic_stds`
        scalers computed from real training-period data are reused -- scalers are never
        recomputed from synthetic data, since they represent basin climatology, not the
        counterfactual precip signal being tested. If the required synthetic/companion column is
        missing from this basin's data, `self.x` is left empty (caller skips this basin+scenario
        combo, see `run_synthetic_evaluation`'s missing-column warning)."""
        super(EvalDataset, self).__init__()
        self.seq_length = cfg['seq_length']
        self.x, self.x_s, self.y, self.dates = [], [], [], []
        self.missing_columns = []

        if basin not in scalers['basin_scalers']:
            return

        no_static = cfg.get("no_static", True)
        concat_static = cfg.get("concat_static", False)
        true_ealstm = (not no_static) and (not concat_static)

        df = load_data_for_basin(cfg['loc'], basin)
        warmup_start = dates[0] - pd.DateOffset(days=self.seq_length - 1)
        df_period = df.loc[warmup_start:dates[1]].copy()
        if len(df_period) < self.seq_length:
            return

        bscaler = scalers['basin_scalers'][basin]

        if syn_col is not None:
            is_peak = 'peak' in syn_col
            temp_col = 'Temp_at_peak' if is_peak else 'Temp'
            pet_col = 'PET_at_peak' if is_peak else 'PET'
            q_col_use = 'Obs_flow_at_peak' if is_peak else Q_COL
            # dyn_cols must stay in DYN_COLS' ['Precip', 'PET', 'Temp'] order since the scaler
            # arrays below are indexed positionally to that order.
            dyn_cols_use = [syn_col, pet_col, temp_col]
            self.missing_columns = [c for c in dyn_cols_use + [q_col_use] if c not in df_period.columns]
            if self.missing_columns:
                return
            dyn_scaled = (df_period[dyn_cols_use].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
            q_raw = df_period[q_col_use].values
        else:
            dyn_scaled = (df_period[DYN_COLS].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
            q_raw = df_period[Q_COL].values

        static_scaled = None
        if not no_static:
            static_data = attributes.loc[basin].values
            static_scaled = ((static_data - scalers['static_means'])
                              / scalers['static_stds']).astype(np.float32)

        is_nan = pd.DataFrame(dyn_scaled).isnull().any(axis=1).values
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < self.seq_length:
                continue

            block_dyn_scaled = dyn_scaled[start_loc:end_loc]
            block_q_raw = q_raw[start_loc:end_loc]

            for i in range(block_len - self.seq_length + 1):
                target_idx = i + self.seq_length - 1
                current_date = df_period.index[start_loc + target_idx]
                if current_date < dates[0]:
                    continue  # part of warm-up only, not itself an evaluation day

                dyn_seq = block_dyn_scaled[i:target_idx + 1]
                final_features = dyn_seq
                if not no_static and concat_static:
                    static_repeated = np.tile(static_scaled, (self.seq_length, 1))
                    final_features = np.concatenate([dyn_seq, static_repeated], axis=1)

                self.x.append(final_features.astype(np.float32))
                self.x_s.append(static_scaled if true_ealstm else np.zeros(1, dtype=np.float32))
                self.y.append(block_q_raw[target_idx])
                self.dates.append(current_date)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x_s[idx], self.y[idx]


def evaluate_basin(model, basin, dates, cfg, scalers, attributes, syn_col: str = None):
    """Return (predictions, observations, dates) for one basin, or (None, None, None).

    `syn_col` optionally selects a synthetic-precipitation-scenario column (see
    `EvalDataset`'s docstring / `SYN_PRECIP_COLS`) -- omit for the normal real-data evaluation
    path, which is unaffected."""
    model.eval()
    if basin not in scalers['basin_scalers']:
        return None, None, None

    dataset = EvalDataset(basin, dates, cfg, scalers, attributes, syn_col=syn_col)
    if len(dataset) == 0:
        if syn_col is not None and dataset.missing_columns:
            print(f"  [synthetic] basin {basin}, scenario '{syn_col}': missing column(s) "
                  f"{dataset.missing_columns} -- skipped.")
        return None, None, None

    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    bscaler = scalers['basin_scalers'][basin]

    true_ealstm = (not cfg.get('no_static', True)) and (not cfg.get('concat_static', False))

    preds_list, obs_list = [], []
    with torch.no_grad():
        for x, x_s, y_raw in loader:
            x = x.to(DEVICE)
            if true_ealstm:
                preds_scaled = model(x, x_s.to(DEVICE)).cpu().numpy().flatten()
            else:
                preds_scaled = model(x).cpu().numpy().flatten()

            preds_rescaled = preds_scaled * bscaler['q_std'] + bscaler['q_mean']
            preds_rescaled[preds_rescaled < 0] = 0
            preds_list.extend(preds_rescaled)
            obs_list.extend(y_raw.numpy())

    return np.array(preds_list), np.array(obs_list), dataset.dates


def run_synthetic_evaluation(model, basins, cfg, scalers, attributes, output_base: Path,
                              syn_scenarios=None):
    """Run all (or a subset of) SYN_PRECIP_COLS synthetic-precipitation scenarios against the
    given model/basins, writing one CSV per scenario to `output_base / 'predict' / 'synthetic'`
    -- basins as rows, dates as columns, matching the legacy Keras
    `x_main_LSTM_regional.py`'s final save-loop transposition format (lines ~402-429) so any
    downstream consumer built against that shape keeps working.

    Peak scenarios ('syn_peak*') are evaluated over a short window from `cfg['train_start']`
    (this project's `start_date_cal`) to '2001-02-10' -- a synthetic rainfall-pulse test, not a
    full-period reforecast. Non-peak scenarios ('syn_no_precip', 'syn_p_pet') use the full
    `[val_start, val_end]` evaluation period, same as the real-data eval path.

    Uses the SAME per-basin scalers / true-EA-LSTM x_d,x_s calling convention as
    `evaluate_basin` -- no separate model-calling logic duplicated here.
    """
    scenarios = syn_scenarios if syn_scenarios is not None else SYN_PRECIP_COLS
    train_start = pd.to_datetime(cfg['train_start'])
    val_start = pd.to_datetime(cfg['val_start'])
    val_end = pd.to_datetime(cfg['val_end'])
    peak_end = pd.to_datetime('2001-02-10')

    syn_dir = output_base / "predict" / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)

    scenario_predictions: Dict[str, Dict[str, pd.Series]] = {}

    for syn_col in scenarios:
        dates = (train_start, peak_end) if 'peak' in syn_col else (val_start, val_end)

        basin_rows = {}
        n_skipped_missing = 0
        for basin in tqdm(basins, desc=f"[synthetic] {syn_col}"):
            preds, obs, eval_dates = evaluate_basin(model, basin, dates, cfg, scalers,
                                                      attributes, syn_col=syn_col)
            if preds is None:
                n_skipped_missing += 1
                continue
            basin_rows[basin] = pd.Series(
                preds, index=[d.strftime('%Y-%m-%d') for d in eval_dates])

        if not basin_rows:
            print(f"[synthetic] '{syn_col}': no basins produced predictions "
                  f"({n_skipped_missing} skipped, e.g. missing column) -- no CSV written.")
            continue

        # basins as rows, dates as columns (outer-joins basins with slightly different date
        # coverage, matching the legacy script's pd.DataFrame(...).join(how='outer') behavior).
        combined = pd.DataFrame(basin_rows).T
        combined.index.name = 'gauge_id'
        combined = combined.sort_index()
        combined = combined[sorted(combined.columns)]

        out_path = syn_dir / f"LSTM_predict_{syn_col.replace('syn_', '')}.csv"
        combined.to_csv(out_path)
        print(f"[synthetic] '{syn_col}': saved {combined.shape[0]} basins x "
              f"{combined.shape[1]} dates to {out_path}"
              + (f" ({n_skipped_missing} basins skipped, missing column)" if n_skipped_missing else ""))

        scenario_predictions[syn_col] = {}
        for basin, series in basin_rows.items():
            s = series.copy()
            s.index = pd.to_datetime(s.index)
            scenario_predictions[syn_col][basin] = s

    return scenario_predictions


def _load_scalers(run_dir: Path) -> Dict:
    with open(run_dir / 'scalers.json', 'r') as fp:
        scalers = json.load(fp)
    if 'basin_scalers' in scalers:
        for b_scaler in scalers['basin_scalers'].values():
            for key, val in b_scaler.items():
                if isinstance(val, list):
                    b_scaler[key] = np.array(val)
    if 'static_means' in scalers:
        scalers['static_means'] = np.array(scalers['static_means'])
    if 'static_stds' in scalers:
        scalers['static_stds'] = np.array(scalers['static_stds'])
    return scalers


def _param_tag(cfg: Dict, model_epoch) -> str:
    """Config tag for output filenames. Encodes everything that changes the model/data
    pipeline (no_static, concat_static, hidden_size, dropout, epochs, seq_length) so two
    differently-configured runs can never silently overwrite each other's results -- see
    module docstring."""
    return (f"ns{cfg.get('no_static', False)}_cs{cfg.get('concat_static', False)}"
           f"_hs{cfg['hidden_size']}_do{cfg.get('dropout', 0.0)}"
           f"_e{model_epoch}_sl{cfg['seq_length']}")


def main():
    parser = argparse.ArgumentParser(description='EA-LSTM Regional Model Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--n_basins', type=int, default=None)
    parser.add_argument('--model_epoch', type=int, default=None,
                        help='Which model_epoch{N}.pt to evaluate. Default: last available.')
    # Was action='store_true' (default off) -- every other model family's regional eval script
    # (RF/XGB/TCN) always runs synthetic scenarios unconditionally, so LSTM silently skipping
    # them by default was an inconsistency, not an intentional per-family difference. Flipped
    # to on-by-default; pass --no_synthetic to reproduce the old real-data-only behavior.
    parser.add_argument('--synthetic', dest='synthetic', action='store_true', default=True,
                        help='REGIONAL ONLY. After the normal real-data cal/eval evaluation, '
                             'also run all 14 SYN_PRECIP_COLS synthetic-precipitation scenarios '
                             '(see module docstring) and write one prediction CSV per scenario '
                             'to <run output>/predict/synthetic/. On by default.')
    parser.add_argument('--no_synthetic', dest='synthetic', action='store_false',
                        help='Skip synthetic-precipitation-scenario evaluation (real-data-only).')
    parser.add_argument('--syn_scenarios', type=str, default=None,
                        help='Comma-separated subset of synthetic scenario names to run with '
                             '--synthetic (e.g. "syn_no_precip,syn_peak1_rate10") -- for fast '
                             'smoke-testing without all 14 x all basins. Default: all of '
                             'SYN_PRECIP_COLS. Ignored if --synthetic is not passed.')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print("EA-LSTM Regional Model Evaluation")
    print(f"Run directory: {run_dir}")

    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)
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
    print(f"Loading model from: {model_file}")

    is_no_static = cfg.get('no_static', False)
    is_concat_static = cfg.get('concat_static', False)

    input_size_stat = cfg.get('input_size_stat', 0)
    input_size_dyn = cfg.get('input_size_dyn', 3)

    model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat,
                  hidden_size=cfg['hidden_size'], dropout=cfg.get('dropout', 0.0),
                  concat_static=is_concat_static, no_static=is_no_static,
                  initial_forget_bias=cfg.get('initial_forget_gate_bias', 5)).to(DEVICE)
    model.load_state_dict(load_lstm_state_dict(model_file, map_location=DEVICE))

    loc = cfg['loc']
    basins = get_basin_list(loc=loc)
    if args.n_basins:
        basins = basins[:args.n_basins]

    attributes = None
    if not is_no_static:
        db_path = str(run_dir / 'attributes.db')
        attributes = load_attributes(db_path, basins)

    train_start = pd.to_datetime(cfg['train_start'])
    train_end = pd.to_datetime(cfg['train_end'])
    val_start = pd.to_datetime(cfg['val_start'])
    val_end = pd.to_datetime(cfg['val_end'])

    output_base = Path(f"/data0/funato/hsp2/0_out/LSTM/{loc}/regional")
    tag = _param_tag(cfg, model_epoch)

    og_series_by_basin: Dict[str, pd.Series] = {}

    for name, dates in {'cal': (train_start, train_end), 'eva': (val_start, val_end)}.items():
        print(f"\nEvaluating {name.upper()} period...")

        pred_path = output_base / "predict" / f"LSTM_predict_{tag}_{name}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        all_metrics = []
        first_basin = True

        for basin in tqdm(basins, desc=f"Processing basins for {name}"):
            preds, obs, eval_dates = evaluate_basin(model, basin, dates, cfg, scalers, attributes)
            if preds is None:
                continue

            basin_row_dict = {d.strftime('%Y-%m-%d'): preds[i] for i, d in enumerate(eval_dates)}
            basin_row = pd.DataFrame([basin_row_dict], index=[basin])
            basin_row.index.name = 'gauge_id'

            basin_row.to_csv(pred_path, mode='w' if first_basin else 'a', header=first_basin)
            first_basin = False

            s = pd.Series(preds, index=pd.to_datetime(eval_dates))
            og_series_by_basin[basin] = pd.concat([og_series_by_basin.get(basin, pd.Series(dtype=float)), s]).sort_index()

            metrics_row = {'gauge_id': basin}
            for benchmark in benchmark_list:
                metrics_row[benchmark] = BMK(obs, preds, benchmark)
            all_metrics.append(metrics_row)

        if all_metrics:
            temp_df = pd.read_csv(pred_path, index_col='gauge_id', dtype={'gauge_id': str})
            temp_df.sort_index().to_csv(pred_path)
            print(f"Predictions saved to {pred_path}")

            metrics_df = pd.DataFrame(all_metrics)
            results_path = output_base / "results" / f"LSTM_results_{tag}_{name}.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(results_path, index=False)
            print(f"Metrics saved to {results_path}")
            print(f"\n--- {name.upper()} Period Summary ---\n{metrics_df.describe()}\n")
        else:
            print(f"No data to evaluate for {name}.")

    scenario_predictions: Dict[str, Dict[str, pd.Series]] = {}
    if args.synthetic:
        print("\nRunning synthetic-precipitation-scenario evaluation...")
        syn_scenarios = (args.syn_scenarios.split(',') if args.syn_scenarios else None)
        scenario_predictions = run_synthetic_evaluation(
            model, basins, cfg, scalers, attributes, output_base, syn_scenarios=syn_scenarios)

    # Write the long-format per-basin prediction files directly (og + all synthetic scenarios),
    # matching reshape_ML_predict.py's old output exactly so that reshape step is no longer
    # needed. The wide cal/eva/synthetic matrices above are still written too, for backward
    # compatibility with anything else that reads them.
    scenario_predictions["og"] = og_series_by_basin
    save_regional_predictions_long_format("LSTM", loc, scenario_predictions, basins)


if __name__ == "__main__":
    main()
