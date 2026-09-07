"""
Hyperparameter-search variant of `main_lstm_regional.py` (the canonical true EA-LSTM regional
training script in this directory). See `main_lstm_regional.py`'s module docstring for full
provenance (Kratzert et al. 2019 / `papercode/*`, per-basin scaling, data conventions, etc.) --
none of that is repeated or duplicated here.

This is a THIN wrapper, not a rewrite: it reuses `main_lstm_regional.py`'s `Model`,
`HydroDataset`, `compute_basin_scalers`, `train_epoch`, `str2bool`, `DYN_COLS`, `_dump_cfg`,
`_convert_for_json`, `DEVICE` and `eval.py`'s `Model` (renamed `EvalModel` here),
`evaluate_basin`, `BMK`, `benchmark_list` via import -- matching the reuse convention already
established for `main_lstm_adapted.py` / `main_lstm_shap_regional.py` in this directory. No
training or evaluation logic is duplicated.

Differences from `main_lstm_regional.py` (the production train script -- unchanged by this
file, its own CLI/behavior/defaults are untouched):
  * `--no_static`/`--concat_static` are NOT exposed as CLI flags here -- hardcoded to
    False/False (true EA-LSTM) so this HP search can only ever explore the canonical
    configuration, never the `no_static=True` misconfiguration that this whole pipeline was
    rebuilt to move away from (see ANALYSIS_README.md ## LSTM).
  * `--seq_length` IS exposed as a CLI arg here. `main_lstm_regional.py` hardcodes it at 270
    inside `GLOBAL_SETTINGS` and deliberately does not expose it (its own defaults must not
    change); this HP variant instead builds its own cfg dict with an overridable seq_length,
    passed as a parameter through `HydroDataset`/`compute_basin_scalers`/`evaluate_basin`,
    without touching `main_lstm_regional.py`'s hardcoded global.
  * `--epochs` defaults to 10 (vs. production's 30) and `--n_basins` defaults to 40 (vs. all
    614 US / 424 GB / 87 JP basins) -- see the argparse help text below for the reasoning.
    This is for HP *search* (relative ranking of configs), not a final publishable fit; run
    `main_lstm_regional.py` directly (all basins, 30 epochs) for the winning combo afterward.
  * After training, automatically evaluates the resulting checkpoint against the cal
    (train-period) and eval (val-period) splits -- reusing `eval.py`'s own evaluation
    machinery -- and writes `hp_metrics_cal.csv` / `hp_metrics_eva.csv` (per-basin
    NSE/KGE/E1/VE/d/RMSE/MAE) into the SAME run directory, so a downstream script can later
    glob `runs/hp/{loc}/*/*/hp_metrics_*.csv` and compare configs without a separate manual
    eval step per combo.

Run directory layout (mirrors the legacy `x_lstm_regional_modeling-master/main_hp.py`'s
`runs/hp/{loc}/h{h}_d{d}/...` convention, extended with lr/seq_length since this HP search
covers more dimensions than that legacy script did):
    ealstm_regional_modeling/runs/hp/{loc}/h{hidden_size}_d{dropout}_lr{learning_rate}_sl{seq_length}/
        run_<timestamp>_seed{seed}/
            cfg.json, scalers.json, attributes.db, model_epoch{N}.pt, lstm_model_final.pt,
            hp_metrics_cal.csv, hp_metrics_eva.csv

CPU-only machine: --n_basins/--epochs defaults are deliberately small for fast HP-search
iteration; scale both up (or run main_lstm_regional.py directly) for a final confirmation run.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_lstm_regional import (
    Model, HydroDataset, compute_basin_scalers, train_epoch, str2bool,
    DYN_COLS, DEVICE, _dump_cfg, _convert_for_json,
)
from papercode.datautils import add_basin_attributes, load_attributes
from papercode.utils import get_basin_list
from papercode.lstm_run_registry import find_completed_run
from papercode.nseloss import NSELoss

from eval import Model as EvalModel, evaluate_basin, BMK, benchmark_list

# HP-search-only cal/val split: calibration 1990-1998, validation 1999-2000. This
# deliberately does NOT match main_lstm_regional.py's production GLOBAL_SETTINGS
# (train 1990-2000 / val 2001-2006) -- HP search must never touch the real 2001-2006
# held-out test period, so results from this script are NOT directly comparable to
# production regional-train runs or to the individual-model baseline.
TRAIN_START = pd.to_datetime('1990-01-01', format='%Y-%m-%d')
TRAIN_END = pd.to_datetime('1998-12-31', format='%Y-%m-%d')
VAL_START = pd.to_datetime('1999-01-01', format='%Y-%m-%d')
VAL_END = pd.to_datetime('2000-12-31', format='%Y-%m-%d')


def get_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='EA-LSTM Regional Model Hyperparameter Search '
                     '(true EA-LSTM only -- no_static=False, concat_static=False, fixed)')
    parser.add_argument('--loc', type=str, default='US', choices=['US', 'GB', 'JP'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument(
        '--n_basins', type=int, default=40,
        help='Subsample of basins for faster HP-search iteration (default 40 -- roughly a '
             'tenth of the 424 GB / 614 US basins; large enough to average out single-basin '
             'noise while keeping one CPU epoch over ~40 basins tractable for a search that '
             'has to repeat over many HP combos). Use main_lstm_regional.py directly (all '
             'basins) for the final confirmation run of the winning combo.')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument(
        '--seq_length', type=int, default=270,
        help='Overridable here (unlike main_lstm_regional.py, which hardcodes this in '
             'GLOBAL_SETTINGS) so the HP grid can sweep it without touching that script.')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Default 10, well below production\'s 30 -- for HP *search* (relative ranking of '
             'configs), not a final publishable fit. The paper\'s own LR schedule applies its '
             'first drop at epoch 11, so most of the early-training signal needed to rank '
             'configs against each other is already visible by epoch 10.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--initial_forget_gate_bias', type=int, default=5)
    parser.add_argument('--use_mse', type=str2bool, default=False)

    cfg = vars(parser.parse_args())
    # Fixed to true EA-LSTM -- see module docstring. Not CLI-overridable by design.
    cfg['no_static'] = False
    cfg['concat_static'] = False
    cfg['train_start'] = TRAIN_START
    cfg['train_end'] = TRAIN_END
    cfg['val_start'] = VAL_START
    cfg['val_end'] = VAL_END
    return cfg


def _hp_str(cfg: Dict) -> str:
    return f"h{cfg['hidden_size']}_d{cfg['dropout']}_lr{cfg['learning_rate']}_sl{cfg['seq_length']}"


def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now()
    hp_str = _hp_str(cfg)
    run_name = f'{hp_str}/run_{now.strftime("%m%d_%H%M%S")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs" / "hp" / cfg['loc'] / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    return cfg


def train(cfg: Dict):
    """Train one HP combo. Mirrors main_lstm_regional.py's train(), but always true-EA-LSTM,
    writes under runs/hp/{loc}/h{..}_d{..}_lr{..}_sl{..}/, and returns everything
    evaluate_and_save() needs instead of exiting."""
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = get_basin_list(loc=cfg['loc'])
    if cfg.get('n_basins'):
        basins = basins[:cfg['n_basins']]

    # run_lstm_hp_tuning_regional.sh launches one process per hyperparameter
    # combination; a restart after a kill would otherwise retrain every
    # combination from scratch since _setup_run's leaf directory is
    # timestamped. Skip only if a run with this exact HP combination already
    # exists under runs/hp/{loc}/{hp_str}/ AND has both the final model and the
    # hp_metrics CSVs this function itself writes below -- so a run killed after
    # saving the model but before evaluate_and_save() finishes is correctly not
    # treated as complete. 'seed' is deliberately NOT a match key -- see
    # main_lstm_regional.py's train() for why (get_args() defaults --seed to a
    # random draw when omitted, matching the upstream reference implementation's
    # traceability-only use of the seed, not an experiment identity). This
    # particular driver does pin --seed via $REGIONAL_SEED, but excluding it
    # keeps this check correct even for a manual invocation that doesn't.
    match_keys = ['loc', 'n_basins', 'hidden_size', 'dropout', 'learning_rate', 'seq_length',
                  'epochs', 'batch_size', 'initial_forget_gate_bias', 'use_mse']
    search_root = Path(__file__).parent / "runs" / "hp" / cfg['loc'] / _hp_str(cfg)
    required_files = ['lstm_model_final.pt', 'hp_metrics_cal.csv', 'hp_metrics_eva.csv']
    existing_run = find_completed_run(search_root, cfg, match_keys, required_files)
    if existing_run is not None:
        print(f"[HP] Skipping: a completed {cfg['epochs']}-epoch run with these exact hyperparameters already exists at {existing_run}")
        return None, None, None, None

    cfg = _setup_run(cfg)

    print("\n[HP] Calculating per-basin scalers from clean training-period data...")
    basin_scalers = compute_basin_scalers(cfg['loc'], basins, cfg['train_start'], cfg['train_end'])
    scalers = {'basin_scalers': basin_scalers}
    print(f"[HP] Usable basins (had a clean training block >= seq_length): "
          f"{len(basin_scalers)}/{len(basins)}")

    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(loc=cfg['loc'], db_path=db_path)
    attributes = load_attributes(db_path=db_path, basins=list(basin_scalers.keys()))
    scalers['static_means'] = attributes.mean().values
    scalers['static_stds'] = attributes.std().values

    input_size_stat = attributes.shape[1]
    input_size_dyn = len(DYN_COLS)
    cfg['input_size_stat'] = input_size_stat
    cfg['input_size_dyn'] = input_size_dyn

    _dump_cfg(cfg)
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(_convert_for_json(scalers), fp, indent=4)
    print(f"[HP] Scalers saved to {cfg['run_dir'] / 'scalers.json'}")

    dataset = HydroDataset(list(basin_scalers.keys()), (cfg['train_start'], cfg['train_end']), cfg,
                            scalers, attributes)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=cfg["num_workers"])

    model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"], initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"], concat_static=False, no_static=False).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = nn.MSELoss() if cfg["use_mse"] else NSELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        train_epoch(model, optimizer, loss_func, loader, cfg, epoch)

    model_file = cfg['run_dir'] / f"model_epoch{cfg['epochs']}.pt"
    torch.save(model.state_dict(), str(model_file))
    torch.save(model.state_dict(), str(cfg['run_dir'] / "lstm_model_final.pt"))
    print(f"[HP] Model saved to {model_file}")

    return cfg, model_file, scalers, attributes


def evaluate_and_save(cfg: Dict, model_file: Path, scalers: Dict, attributes):
    """Evaluate the just-trained checkpoint on cal+eval periods and write hp_metrics_*.csv into
    the run directory, reusing eval.py's evaluate_basin/BMK (no duplicated eval logic).

    eval.py's `Model` class is rebuilt here (rather than reusing the training `Model` from
    main_lstm_regional.py) because its forward() returns a single prediction tensor, matching
    what evaluate_basin() expects, whereas the training Model's forward() returns a
    (prediction, h_n, c_n) tuple. Both share the same lstm/dropout/fc submodules, so the
    state_dict from training loads into eval.py's Model unmodified.
    """
    eval_model = EvalModel(input_size_dyn=cfg['input_size_dyn'], input_size_stat=cfg['input_size_stat'],
                            hidden_size=cfg['hidden_size'], dropout=cfg['dropout'],
                            concat_static=False, no_static=False,
                            initial_forget_bias=cfg['initial_forget_gate_bias']).to(DEVICE)
    eval_model.load_state_dict(torch.load(model_file, map_location=DEVICE))

    basins = list(scalers['basin_scalers'].keys())
    eval_cfg = dict(cfg, no_static=False, concat_static=False)

    for name, dates in {'cal': (cfg['train_start'], cfg['train_end']),
                         'eva': (cfg['val_start'], cfg['val_end'])}.items():
        all_metrics = []
        for basin in tqdm(basins, desc=f"[HP] Evaluating {name}"):
            preds, obs, _ = evaluate_basin(eval_model, basin, dates, eval_cfg, scalers, attributes)
            if preds is None:
                continue
            row = {'gauge_id': basin}
            for benchmark in benchmark_list:
                row[benchmark] = BMK(obs, preds, benchmark)
            all_metrics.append(row)

        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = cfg['run_dir'] / f"hp_metrics_{name}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[HP] {name.upper()} metrics saved to {metrics_path}")
        if len(metrics_df):
            print(f"--- {name.upper()} summary ---\n{metrics_df.describe()}\n")


def main():
    cfg = get_args()
    cfg, model_file, scalers, attributes = train(cfg)
    if cfg is None:
        return  # train() already printed why (a completed run with these hyperparameters exists)
    evaluate_and_save(cfg, model_file, scalers, attributes)
    print(f"\n[HP] Done. Run directory: {cfg['run_dir']}")


if __name__ == "__main__":
    main()
