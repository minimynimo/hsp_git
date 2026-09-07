"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

--------------------------------------------------------------------------------------------
ADAPTED for this project (hsp2 / 99_work/14_LSTM/ealstm_regional_modeling):

This is a from-scratch rebuild starting from the pristine reference implementation at
/data0/funato/hsp2/reference/ealstm_regional_modeling (do not edit that copy). The
`papercode/ealstm.py`, `papercode/lstm.py` and `papercode/nseloss.py` modules are used
UNCHANGED from the reference -- they are generic architecture/loss code with no data-path
assumptions. Only the data loading (this file + `papercode/datautils.py` +
`papercode/utils.py`) is adapted to this project's data:

  * Dynamic forcing + discharge come from per-basin CSVs at
        /data0/funato/hsp2/2_MERV/varssim_{loc}/varssim_{gauge_id}.csv
    with exactly 3 dynamic inputs (Precip, PET, Temp) and target column 'Obs flow', matching
    the convention already used by `run_LSTM.py` (individual model) and
    `x_lstm_regional_modeling-master/main_lstm.py` (previous regional model).
  * Static attributes come from the combined CSV per location (see `papercode/datautils.py`),
    restricted to VALID_ATTR, with per-location caveats documented there.
  * Discharge and dynamic-input normalization is done PER BASIN (mean/std computed from that
    basin's own clean training-period data), not with one global scaler. This mirrors the
    (already-fixed) per-basin scaling logic in `x_lstm_regional_modeling-master/main_lstm.py`
    -- pooling many basins with wildly different discharge magnitudes under one global scaler
    would badly distort the NSE loss. Static-attribute normalization uses a single global
    mean/std across basins, which is fine since catchment attributes aren't a moving target
    the way discharge magnitude is.
  * Defaults restore the paper's actual hyperparameters, including no_static=False,
    concat_static=False (true EA-LSTM) -- this is the configuration that was empirically
    verified to beat the individually-trained LSTM on US data (NSE 0.399 vs 0.328), unlike
    the `x_lstm_regional_modeling-master/main_lstm.py` script, which historically was always
    run with no_static=True and therefore never actually exercised the EA-LSTM mechanism.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from papercode.datautils import add_basin_attributes, load_attributes
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.nseloss import NSELoss
from papercode.utils import get_basin_list
from papercode.lstm_run_registry import find_completed_run, find_resumable_run

# ####################### #
# Global Definitions      #
# ####################### #

# Dates match the individual-model cal/eval split (see run_LSTM.py, main_LSTM_indiv.py) and
# the previous regional script (x_lstm_regional_modeling-master/main_lstm.py) so results are
# directly comparable.
GLOBAL_SETTINGS = {
    'seq_length': 365,
    'train_start': pd.to_datetime('1990-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d'),
}

DYN_COLS = ['Precip', 'PET', 'Temp']  # exactly 3 dynamic inputs, per this project's data
Q_COL = 'Obs flow'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ####################### #
# Model Definition        #
# ####################### #


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with a fully connected layer."""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


# ####################### #
# Utility Functions       #
# ####################### #


def load_data_for_basin(loc: str, gauge_id) -> pd.DataFrame:
    """Load one basin's forcing+discharge CSV. See module docstring for the data convention."""
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{loc}"
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _clean_blocks(is_nan: np.ndarray):
    """Return list of (start, stop) slices of contiguous non-NaN rows."""
    clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
    if not isinstance(clumps, list):
        clumps = [clumps] if clumps else []
    return clumps


# ####################### #
# Data Handling           #
# ####################### #


class HydroDataset(Dataset):
    """Per-basin-scaled PyTorch Dataset, covering both plain LSTM and EA-LSTM configurations.

    Discharge and dynamic inputs are scaled with THIS basin's own train-period mean/std
    (`scalers['basin_scalers'][basin]`), not a single pooled/global scaler -- see module
    docstring. Static attributes (when used) are scaled with a single global mean/std since
    they are not a moving target across basins the way discharge magnitude is.
    """

    def __init__(self, basins: List[str], dates: Tuple[pd.Timestamp, pd.Timestamp], cfg: Dict,
                 scalers: Dict, attributes: pd.DataFrame):
        super(HydroDataset, self).__init__()
        self.basins = basins
        self.dates = dates
        self.cfg = cfg
        self.scalers = scalers
        self.attributes = attributes
        self.seq_length = cfg['seq_length']
        self.no_static = cfg['no_static']
        self.concat_static = cfg['concat_static']

        self.x, self.x_s, self.y, self.q_params = self._create_dataset()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x_s[idx], self.y[idx], self.q_params[idx]

    def _create_dataset(self):
        x_list, xs_list, y_list, q_params_list = [], [], [], []
        # True EA-LSTM (not no_static, not concat_static) needs the static vector passed to
        # the model SEPARATELY from the dynamic sequence (EALSTM.forward(x_d, x_s), x_s of
        # shape [batch, n_static], not tiled across time). concat_static instead tiles static
        # attributes into every timestep of x_d and feeds a plain LSTM. no_static uses only
        # dynamic inputs. `x_s` below is a real per-sample static vector in the true-EA-LSTM
        # case, and an empty placeholder (unused) otherwise, so __getitem__ can stay uniform.
        true_ealstm = (not self.no_static) and (not self.concat_static)

        for basin in tqdm(self.basins, desc="Preparing dataset"):
            if basin not in self.scalers['basin_scalers']:
                continue

            df = load_data_for_basin(self.cfg['loc'], basin)
            df_period = df.loc[self.dates[0]:self.dates[1]]

            bscaler = self.scalers['basin_scalers'][basin]

            dyn_data_scaled = (df_period[DYN_COLS].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
            q_data_scaled = (df_period[Q_COL].values - bscaler['q_mean']) / bscaler['q_std']

            static_data_scaled = None
            if not self.no_static:
                static_data = self.attributes.loc[basin].values
                static_data_scaled = ((static_data - self.scalers['static_means'])
                                       / self.scalers['static_stds']).astype(np.float32)

            is_nan = df_period[DYN_COLS + [Q_COL]].isnull().any(axis=1).values
            for s in _clean_blocks(is_nan):
                start_loc, end_loc = s.start, s.stop
                block_len = end_loc - start_loc
                if block_len < self.seq_length:
                    continue

                block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
                block_q_scaled = q_data_scaled[start_loc:end_loc]

                for i in range(block_len - self.seq_length + 1):
                    target_idx = i + self.seq_length - 1
                    dyn_seq = block_dyn_scaled[i:target_idx + 1]

                    final_features = dyn_seq
                    if not self.no_static and self.concat_static:
                        static_repeated = np.tile(static_data_scaled, (self.seq_length, 1))
                        final_features = np.concatenate([dyn_seq, static_repeated], axis=1)

                    x_list.append(final_features)
                    xs_list.append(static_data_scaled if true_ealstm else np.zeros(1, dtype=np.float32))
                    y_list.append(block_q_scaled[target_idx])
                    q_params_list.append([bscaler['q_mean'], bscaler['q_std']])

        return (np.array(x_list, dtype=np.float32),
                np.array(xs_list, dtype=np.float32),
                np.array(y_list, dtype=np.float32).reshape(-1, 1),
                np.array(q_params_list, dtype=np.float32))


def compute_basin_scalers(loc: str, basins: List[str], train_start: pd.Timestamp,
                           train_end: pd.Timestamp) -> Dict:
    """Compute per-basin mean/std for discharge and dynamic inputs from clean (no-NaN) rows
    within the training period. See module docstring: this is the per-basin scaling that
    replaces a single pooled/global scaler."""
    basin_scalers = {}
    for basin in tqdm(basins, desc="Computing per-basin scalers"):
        df = load_data_for_basin(loc, basin)
        df_period = df.loc[train_start:train_end]

        is_nan = df_period[DYN_COLS + [Q_COL]].isnull().any(axis=1).values
        clean_blocks = [df_period.iloc[s] for s in _clean_blocks(is_nan)]
        if not clean_blocks:
            continue

        basin_clean_df = pd.concat(clean_blocks)
        if len(basin_clean_df) < 2:
            continue

        basin_scalers[basin] = {
            'q_mean': basin_clean_df[Q_COL].mean(),
            'q_std': basin_clean_df[Q_COL].std(),
            'dynamic_means': basin_clean_df[DYN_COLS].mean().values,
            'dynamic_stds': basin_clean_df[DYN_COLS].std().values,
        }
    return basin_scalers


def _convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# ####################### #
# Main Training Logic     #
# ####################### #


def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train"])
    parser.add_argument('--loc', type=str, default='US', choices=['US', 'GB', 'JP'])
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--num_workers', type=int, default=2)
    # Optional basin-count cap for quick tests/smoke runs (mirrors eval.py's --n_basins).
    # Default None = use all basins for `--loc`, matching original full-scale behavior.
    parser.add_argument('--n_basins', type=int, default=None)
    # Static-attribute / architecture switches. Defaults restore true EA-LSTM (the
    # configuration verified to beat the individually-trained model, see module docstring).
    parser.add_argument('--no_static', type=str2bool, default=False)
    parser.add_argument('--concat_static', type=str2bool, default=False)
    parser.add_argument('--use_mse', type=str2bool, default=False)
    # Model hyperparameters -- paper defaults (Kratzert et al. 2019)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--initial_forget_gate_bias', type=int, default=5)
    # Training hyperparameters -- paper defaults
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    cfg = vars(parser.parse_args())
    if cfg["seed"] is None:
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
    cfg.update(GLOBAL_SETTINGS)
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run under ealstm_regional_modeling/runs/."""
    now = datetime.now()
    run_name = f'run_{now.strftime("%m%d_%H%M%S")}_{cfg["loc"]}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs" / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    return cfg


def _dump_cfg(cfg: Dict):
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, (Path, PosixPath)):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%Y-%m-%d")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int):
    """Train model for a single epoch."""
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    true_ealstm = (not cfg["no_static"]) and (not cfg["concat_static"])

    for x, x_s, y, q_params in pbar:
        optimizer.zero_grad()
        x, y, q_params = x.to(DEVICE), y.to(DEVICE), q_params.to(DEVICE)

        if true_ealstm:
            x_s = x_s.to(DEVICE)
            predictions = model(x, x_s)[0]
        else:
            predictions = model(x)[0]

        if cfg["use_mse"]:
            loss = loss_func(predictions, y)
        else:  # NSE Loss with per-basin scaling
            q_means_batch = q_params[:, 0].unsqueeze(1)
            q_stds_batch = q_params[:, 1].unsqueeze(1)

            # Rescale predictions and observations to physical units for this batch, so the
            # NSE loss (and its per-sample q_std weighting) operates on real discharge values.
            pred_rescaled = predictions * q_stds_batch + q_means_batch
            y_rescaled = y * q_stds_batch + q_means_batch

            loss = loss_func(pred_rescaled, y_rescaled, q_stds_batch.flatten())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.5f}")


def train(cfg):
    """Main training function."""
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = get_basin_list(loc=cfg['loc'])
    if cfg.get('n_basins'):
        basins = basins[:cfg['n_basins']]

    # _setup_run's run_dir is timestamped, so a restart after a kill would
    # otherwise retrain from scratch even if training already ran to
    # cfg["epochs"] under these exact hyperparameters. Skip if a completed run
    # (one with lstm_model_final.pt, only written after the full epoch loop
    # finishes) already exists under runs/ with matching hyperparameters. 'seed'
    # is deliberately NOT a match key: get_args() defaults --seed to a random
    # draw when omitted, exactly like the upstream reference implementation
    # (/data0/funato/hsp2/reference/ealstm_regional_modeling/main.py), which
    # logs that random seed in the run folder name purely for traceability, not
    # as an experiment identity -- requiring it to match would mean two
    # invocations of the identical hyperparameters, neither passing --seed
    # explicitly, could never recognize each other and resumability would never
    # trigger (this was confirmed happening in practice).
    match_keys = ['loc', 'n_basins', 'no_static', 'concat_static', 'use_mse', 'batch_size',
                  'hidden_size', 'dropout', 'initial_forget_gate_bias', 'epochs',
                  'learning_rate']
    search_root = Path(__file__).parent / "runs"
    existing_run = find_completed_run(search_root, cfg, match_keys)
    if existing_run is not None:
        print(f"Skipping: a completed {cfg['epochs']}-epoch run with these exact hyperparameters already exists at {existing_run}")
        return

    # Resume from a partially-trained run with matching hyperparameters instead of always
    # starting a fresh timestamped directory from epoch 0 -- this training loop already
    # checkpoints every epoch, that progress just wasn't being reused before. Only the
    # model/optimizer state is restored here (scaler computation, static attributes, and the
    # dataset below are all still redone -- cheap relative to a training epoch, and
    # deterministic given the same basins/period, so recomputing them is harmless).
    resume_dir, resume_epoch = find_resumable_run(search_root, cfg, match_keys)
    if resume_dir is not None:
        cfg['run_dir'] = resume_dir
        print(f"Resuming existing run at {resume_dir} from epoch {resume_epoch + 1}/{cfg['epochs']}")
    else:
        cfg = _setup_run(cfg)

    print("\nCalculating per-basin scalers from clean training-period data...")
    basin_scalers = compute_basin_scalers(cfg['loc'], basins, cfg['train_start'], cfg['train_end'])
    scalers = {'basin_scalers': basin_scalers}
    print(f"Usable basins (had a clean training block >= seq_length): {len(basin_scalers)}/{len(basins)}")

    attributes = None
    if not cfg["no_static"]:
        db_path = str(cfg['run_dir'] / 'attributes.db')
        add_basin_attributes(loc=cfg['loc'], db_path=db_path)
        attributes = load_attributes(db_path=db_path, basins=list(basin_scalers.keys()))
        scalers['static_means'] = attributes.mean().values
        scalers['static_stds'] = attributes.std().values

    input_size_stat = 0 if cfg["no_static"] else attributes.shape[1]
    input_size_dyn = len(DYN_COLS)
    if not cfg["no_static"] and cfg["concat_static"]:
        input_size_dyn = len(DYN_COLS) + input_size_stat
    cfg['input_size_stat'] = input_size_stat
    cfg['input_size_dyn'] = input_size_dyn

    _dump_cfg(cfg)
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(_convert_for_json(scalers), fp, indent=4)
    print(f"Scalers saved to {cfg['run_dir'] / 'scalers.json'}")

    dataset = HydroDataset(list(basin_scalers.keys()), (cfg['train_start'], cfg['train_end']), cfg,
                           scalers, attributes)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"]).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = nn.MSELoss() if cfg["use_mse"] else NSELoss()

    if resume_epoch > 0:
        ckpt_path = cfg['run_dir'] / f"model_epoch{resume_epoch}.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            # Old checkpoint format: a bare model.state_dict(), no optimizer state saved --
            # optimizer starts fresh (Adam re-warms its moment estimates over a few steps,
            # not a correctness issue for resuming).
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from epoch {resume_epoch}")

    for epoch in range(resume_epoch + 1, cfg["epochs"] + 1):
        train_epoch(model, optimizer, loss_func, loader, cfg, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, str(cfg["run_dir"] / f"model_epoch{epoch}.pt"))

    torch.save(model.state_dict(), str(cfg["run_dir"] / "lstm_model_final.pt"))
    print(f"Final model saved to {cfg['run_dir'] / 'lstm_model_final.pt'}")
    print(f"Run directory: {cfg['run_dir']}")


if __name__ == "__main__":
    config = get_args()
    train(config)
