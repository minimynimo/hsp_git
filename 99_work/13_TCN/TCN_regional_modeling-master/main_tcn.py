"""
TCN Regional Hydrological Modeling

This version is corrected to use per-basin scaling for regional training,
properly handles missing data, and integrates end-of-run diagnostic plotting.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Force matplotlib to use non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from papercode.datautils import add_basin_attributes, load_attributes, VALID_ATTR
from papercode.utils import get_basin_list
from papercode.tcn_run_registry import find_completed_run, find_resumable_run

# ####################### #
# Global Definitions      #
# ####################### #

GLOBAL_SETTINGS = {
    'loc': 'US',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1990-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

gauge_id_list = get_basin_list(GLOBAL_SETTINGS['loc'])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ####################### #
# Utility Functions       #
# ####################### #

def load_data_for_basin(gauge_id):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ####################### #
# TCN Model Definition    #
# ####################### #

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])

# ####################### #
# Data Handling           #
# ####################### #

class HydroDataset(Dataset):
    def __init__(self, basins: List, cfg: Dict, scalers: Dict, attributes: pd.DataFrame):
        self.basins = basins
        self.cfg = cfg
        self.scalers = scalers
        self.attributes = attributes
        self.samples = []
        self.sample_info = []
        self._prepare_data()

    def _prepare_data(self):
        print("Preparing and scaling data for all basins...")
        dyn_cols = ['Precip', 'PET', 'Temp']
        q_col = 'Obs flow'

        for basin in tqdm(self.basins, desc="Processing basins"):
            if basin not in self.scalers['basin_scalers']: continue

            df = load_data_for_basin(basin)
            df_period = df.loc[self.cfg['train_start']:self.cfg['train_end']]

            bscaler = self.scalers['basin_scalers'][basin]

            dyn_data_scaled = (df_period[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
            q_data_scaled = (df_period[q_col].values - bscaler['q_mean']) / bscaler['q_std']
            
            static_data_scaled = None
            if not self.cfg["no_static"]:
                static_data = self.attributes.loc[basin].values
                static_data_scaled = np.where(
                    self.scalers['static_stds'] != 0,
                    (static_data - self.scalers['static_means']) / self.scalers['static_stds'],
                    0.0
                )
            
            is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
            clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
            if not isinstance(clumps, list):
                clumps = [clumps] if clumps else []

            for s in clumps:
                start_loc, end_loc = s.start, s.stop
                block_len = end_loc - start_loc
                if block_len < self.cfg['seq_length']: continue

                block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
                block_q_scaled = q_data_scaled[start_loc:end_loc]

                for i in range(block_len - self.cfg['seq_length'] + 1):
                    target_idx_in_block = i + self.cfg['seq_length'] - 1
                    dyn_seq = block_dyn_scaled[i:target_idx_in_block + 1]
                    
                    final_features = dyn_seq
                    if not self.cfg["no_static"]:
                        static_seq = np.tile(static_data_scaled, (self.cfg['seq_length'], 1))
                        final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                    
                    self.samples.append((final_features.astype(np.float32), block_q_scaled[target_idx_in_block].astype(np.float32)))
                    self.sample_info.append({'basin': basin, 'date': df_period.index[start_loc + target_idx_in_block]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

# ####################### #
# Main Execution          #
# ####################### #

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train"])
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(10000000))
    parser.add_argument('--run_dir', type=str, help="For evaluation mode.")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no_static', type=str2bool, default=False)
    # Fixed at 6 (dilations [1,2,4,8,16,32]) per HYPERPARAMETERS.md's "LAYERS 1 2 4 8 16 32" row,
    # same convention as the individual TCN script -- see DEBUG_README.md §15.1. Was previously
    # 8 here, inconsistent with that convention (run_main_tcn.sh doesn't pass --num_levels, so
    # this default is what production actually got).
    parser.add_argument('--num_levels', type=int, default=6, help='Number of dilation levels (fixed at 6)')
    
    # TCN specific arguments. Defaults are the selected regional combo from the completed
    # 162/162-combo HP grid (154-basin subsample, NSE_median=0.284) -- see
    # SELECTED_HYPERPARAMETERS.md and DEBUG_README.md's TCN follow-up section.
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_channels', type=int, nargs='+', default=None)

    cfg = vars(parser.parse_args())
    cfg.update(GLOBAL_SETTINGS)
    # Ensure TCN channel definition matches eval_regional/eval_adapted logic.
    # If num_filters/num_levels are provided, derive num_channels from them.
    if cfg.get("num_filters") is not None and cfg.get("num_levels") is not None:
        cfg["num_channels"] = [cfg["num_filters"]] * cfg["num_levels"]
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    run_name = f'tcn_run_{now.strftime("%m%d_%H%M")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs_tcn" / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        temp_cfg = {k: v for k, v in cfg.items() if not isinstance(v, (pd.Timestamp, Path))}
        temp_cfg['train_start'] = cfg['train_start'].strftime('%Y-%m-%d')
        temp_cfg['train_end'] = cfg['train_end'].strftime('%Y-%m-%d')
        temp_cfg['run_dir'] = str(cfg['run_dir'])
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
    return cfg

def train(cfg):
    """Train TCN model"""
    print("Starting TCN training...")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])
    
    basins = gauge_id_list

    # _setup_run's run_dir is timestamped, so a restart after a kill would
    # otherwise retrain from scratch even if training already ran to
    # cfg["epochs"] under these exact hyperparameters. Skip if a completed run
    # (one with a checkpoint at the required epoch count) already exists under
    # runs_tcn/. 'seed' is deliberately NOT a match key -- get_args() defaults
    # --seed to a random draw when omitted (same convention as the upstream
    # EA-LSTM reference implementation, which logs that random seed in the run
    # folder name purely for traceability), so requiring it to match would defeat
    # resumability for any invocation that doesn't pin --seed.
    match_keys = ['batch_size', 'epochs', 'learning_rate', 'dropout', 'kernel_size',
                  'num_filters', 'num_levels', 'no_static']
    search_root = Path(__file__).parent / "runs_tcn"
    target_checkpoint = f"tcn_model_epoch{cfg['epochs']}.pt"
    existing_run = find_completed_run(search_root, cfg, match_keys, target_checkpoint)
    if existing_run is not None:
        print(f"Skipping: a completed {cfg['epochs']}-epoch run with these exact hyperparameters already exists at {existing_run}")
        return

    # Resume from a partially-trained run with matching hyperparameters instead of always
    # starting a fresh timestamped directory from epoch 0 -- this training loop already
    # checkpoints every 5 epochs, that progress just wasn't being reused before. Only the
    # model/optimizer state is restored here (data loading, scaler computation, and static
    # attributes below are all still redone -- cheap relative to a training epoch, and
    # deterministic given the same basins/period, so recomputing them is harmless).
    resume_dir, resume_epoch = find_resumable_run(search_root, cfg, match_keys,
                                                   "tcn_model_epoch*.pt",
                                                   lambda p: int(p.stem.replace("tcn_model_epoch", "")))
    if resume_dir is not None:
        cfg['run_dir'] = resume_dir
        print(f"Resuming existing run at {resume_dir} from epoch {resume_epoch + 1}/{cfg['epochs']}")
    else:
        cfg = _setup_run(cfg)
    for key, val in cfg.items(): print(f"{key}: {val}")

    print("\nCalculating per-basin scalers from clean data blocks...")
    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'
    basin_scalers = {}

    for basin in tqdm(basins, desc="Loading basin data for scaling"):
        df = load_data_for_basin(basin)
        df_period = df.loc[cfg['train_start']:cfg['train_end']]

        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clumps = np.ma.clump_unmasked(np.ma.masked_array(df_period.index, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []
        clean_blocks = [df_period.iloc[s] for s in clumps]

        if not clean_blocks:
            continue

        basin_clean_df = pd.concat(clean_blocks)

        basin_scalers[basin] = {
            'q_mean': basin_clean_df[q_col].mean(),
            'q_std': basin_clean_df[q_col].std(),
            'dynamic_means': basin_clean_df[dyn_cols].mean().values,
            'dynamic_stds': basin_clean_df[dyn_cols].std().values
        }

    scalers = {'basin_scalers': basin_scalers}

    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)

    attributes = None
    if not cfg["no_static"]:
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True, keep_features=VALID_ATTR)
        scalers['static_means'] = attributes.mean().values
        scalers['static_stds'] = attributes.std().values
        # Replace zero stds with 1.0 to avoid division by zero
        scalers['static_stds'] = np.where(scalers['static_stds'] == 0, 1.0, scalers['static_stds'])

    def convert_for_json(obj):
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.generic): return obj.item()
        else: return obj
            
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(convert_for_json(scalers), fp, indent=4)
    print(f"\nScalers successfully calculated and saved to {cfg['run_dir'] / 'scalers.json'}")

    dataset = HydroDataset(basins=basins, cfg=cfg, scalers=scalers, attributes=attributes)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    input_size = len(dyn_cols)
    if not cfg["no_static"]:
        input_size += len(attributes.columns)

    model = TCNModel(input_size=input_size, 
                     num_channels=cfg["num_channels"], 
                     kernel_size=cfg["kernel_size"], 
                     dropout=cfg["dropout"]).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = nn.MSELoss()

    if resume_epoch > 0:
        ckpt_path = cfg['run_dir'] / f"tcn_model_epoch{resume_epoch}.pt"
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

    print("\nStarting training loop...")
    all_preds_scaled, all_obs_scaled = [], []
    for epoch in range(resume_epoch + 1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{cfg["epochs"]}')
        
        epoch_preds, epoch_obs = [], []
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_func(predictions, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            if epoch == cfg["epochs"]: # Store results only on last epoch
                epoch_preds.extend(predictions.detach().cpu().numpy())
                epoch_obs.extend(y.detach().cpu().numpy())
        
        if epoch == cfg["epochs"]:
            all_preds_scaled = np.array(epoch_preds).flatten()
            all_obs_scaled = np.array(epoch_obs).flatten()

        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, cfg["run_dir"] / f"tcn_model_epoch{epoch}.pt")

    # Was previously print-only -- final_model_path was computed and logged as "saved" but
    # never actually passed to torch.save(), so tcn_model_final.pt never existed (same bug
    # already documented for main_tcn_hp.py). The epoch%5==0-or-last-epoch checkpoint above
    # already wrote tcn_model_epoch{cfg["epochs"]}.pt, which is what eval_regional.py's
    # --model_epoch default actually relies on; this just makes the "_final.pt" convenience
    # alias (used by eval_adapted.py's fallback when a specific --model_epoch isn't passed)
    # real.
    final_model_path = cfg["run_dir"] / "tcn_model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining finished. Final model saved to {final_model_path}")

    print("\nGenerating debug plots for a sample of basins...")
    plot_dir = cfg['run_dir'] / "debug_plots"
    plot_dir.mkdir(exist_ok=True)
    
    df_samples = pd.DataFrame(dataset.sample_info)
    df_samples['pred_scaled'] = all_preds_scaled
    df_samples['obs_scaled'] = all_obs_scaled
    
    unique_basins = df_samples['basin'].unique().tolist()
    basins_to_plot = random.sample(unique_basins, min(5, len(unique_basins)))

    for basin_id in basins_to_plot:
        print(f"  Plotting basin {basin_id}...")
        
        basin_samples = df_samples[df_samples['basin'] == basin_id]
        bscaler = scalers['basin_scalers'][basin_id]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle(f'TCN Training Debug Analysis for Basin {basin_id}', fontsize=16)

        # Rescaled Hydrograph
        obs_rescaled = (basin_samples['obs_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        pred_rescaled = (basin_samples['pred_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        ax1.plot(basin_samples['date'], obs_rescaled, label='Observed', color='black')
        ax1.plot(basin_samples['date'], pred_rescaled, label='Predicted', color='blue', linestyle='--')
        ax1.set_title('Rescaled Hydrograph (Training Period)')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Scaled Hydrograph
        ax2.plot(basin_samples['date'], basin_samples['obs_scaled'], label='Observed (scaled)', color='black')
        ax2.plot(basin_samples['date'], basin_samples['pred_scaled'], label='Predicted (scaled)', color='blue', linestyle='--')
        ax2.set_title('Scaled Hydrograph (What the model sees)')
        ax2.set_ylabel('Standardized Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_dir / f"basin_{basin_id}_training_debug.png")
        plt.close(fig)

    print("\nUse a separate evaluation script with the saved model and scalers.")

if __name__ == "__main__":
    config = get_args()
    if config["mode"] == "train":
        train(config)

