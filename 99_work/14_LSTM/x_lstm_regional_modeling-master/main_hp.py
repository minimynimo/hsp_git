"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

This version is modified to correctly use per-basin scaling for regional training.
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

from papercode.datautils import (add_basin_attributes, load_attributes)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.nseloss import NSELoss
from papercode.utils import get_basin_list

# ####################### #
# Global Definitions      #
# ####################### #

GLOBAL_SETTINGS = {
    'loc': 'US',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ####################### #
# Model Definition        #
# ####################### #

class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connected layer"""
    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = True,
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

def file_name(input_num, total_len):
    return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ####################### #
# Data Handling           #
# ####################### #

class HydroDataset(Dataset):
    """Custom PyTorch Dataset for hydrological data with per-basin scaling."""
    def __init__(self, basins, dates, cfg, scalers, attributes):
        super(HydroDataset, self).__init__()
        self.basins = basins
        self.dates = dates
        self.cfg = cfg
        self.scalers = scalers
        self.attributes = attributes
        self.seq_length = cfg['seq_length']
        self.no_static = cfg['no_static']
        self.concat_static = cfg['concat_static']
        
        self.x, self.y, self.q_params = self._create_dataset()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.q_params[idx]

    def _create_dataset(self):
        x_list, y_list, q_params_list = [], [], []
        
        dyn_cols = ['Precip', 'PET', 'Temp']
        q_col = 'Obs flow'

        for basin in tqdm(self.basins, desc="Preparing training data"):
            if basin not in self.scalers['basin_scalers']: continue

            df = load_data_for_basin(int(basin))
            df_period = df.loc[self.dates[0]:self.dates[1]]
            
            bscaler = self.scalers['basin_scalers'][basin]
            
            dyn_data_scaled = (df_period[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
            q_data_scaled = (df_period[q_col].values - bscaler['q_mean']) / bscaler['q_std']
            
            static_data_scaled = None
            if not self.no_static:
                # basin is string but attributes has integer index, convert for lookup
                static_data = self.attributes.loc[int(basin)].values
                static_data_scaled = (static_data - self.scalers['static_means']) / self.scalers['static_stds']

            is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
            clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
            if not isinstance(clumps, list):
                clumps = [clumps] if clumps else []

            for s in clumps:
                start_loc, end_loc = s.start, s.stop
                block_len = end_loc - start_loc
                if block_len < self.seq_length: continue

                block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
                block_q_scaled = q_data_scaled[start_loc:end_loc]

                for i in range(block_len - self.seq_length + 1):
                    target_idx_in_block = i + self.seq_length - 1
                    dyn_seq = block_dyn_scaled[i:target_idx_in_block + 1]
                    
                    final_features = dyn_seq
                    if not self.no_static and self.concat_static:
                        static_repeated = np.tile(static_data_scaled, (self.seq_length, 1))
                        final_features = np.concatenate([dyn_seq, static_repeated], axis=1)
                    
                    x_list.append(final_features)
                    y_list.append(block_q_scaled[target_idx_in_block])
                    q_params_list.append([bscaler['q_mean'], bscaler['q_std']])

        return (np.array(x_list, dtype=np.float32), 
                np.array(y_list, dtype=np.float32).reshape(-1, 1),
                np.array(q_params_list, dtype=np.float32))

# ####################### #
# Main Training Logic     #
# ####################### #

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train"])
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no_static', type=str2bool, default=False) # Default to using static data
    parser.add_argument('--concat_static', type=str2bool, default=True)
    
    # Fixed hyperparameters based on Kratzert et al. (2018, 2019)
    parser.add_argument('--batch_size', type=int, default=256, help='Fixed at 256 for regional')
    parser.add_argument('--epochs', type=int, default=30, help='30-50 epochs recommended')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Fixed at 1e-3')
    parser.add_argument('--use_mse', type=str2bool, default=True, help='Use MSE loss instead of NSE')
    
    # Hyperparameters for grid search
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    cfg = vars(parser.parse_args())
    cfg.update(GLOBAL_SETTINGS)
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    hp_str = f"h{cfg['hidden_size']}_d{cfg['dropout']}_lr{cfg['learning_rate']}"
    run_name = f'{hp_str}/run_{now.strftime("%d%m_%H%M")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs" / "hp"/ GLOBAL_SETTINGS['loc'] / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, (Path, PosixPath)): temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp): temp_cfg[key] = val.strftime(format="%Y-%m-%d")
            else: temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
    return cfg

def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module, loader: DataLoader, cfg: Dict, epoch: int):
    """Train model for a single epoch."""
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    for x, y, q_params in pbar:
        optimizer.zero_grad()
        x, y, q_params = x.to(DEVICE), y.to(DEVICE), q_params.to(DEVICE)
        
        predictions = model(x)[0]
        
        if cfg["use_mse"]:
            loss = loss_func(predictions, y)
        else: # NSE Loss with per-basin scaling
            q_means_batch = q_params[:, 0].unsqueeze(1)
            q_stds_batch = q_params[:, 1].unsqueeze(1)
            
            # Rescale predictions and observations for this specific batch
            pred_rescaled = predictions * q_stds_batch + q_means_batch
            y_rescaled = y * q_stds_batch + q_means_batch

            # The third argument to NSELoss is the std of each sample's observation
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
    
    if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
    else: file_tot_num = 424
    basins = get_basin_list(file_tot_num=file_tot_num)
    basins = basins[::4]
    
    cfg = _setup_run(cfg)

    print("\nCalculating per-basin scalers from clean data blocks...")
    dyn_cols, q_col = ['Precip', 'PET', 'Temp'], 'Obs flow'
    basin_scalers = {}

    for basin in tqdm(basins, desc="Loading basin data for scaling"):
        df = load_data_for_basin(int(basin))
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        
        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clean_blocks = []
        clumps = np.ma.clump_unmasked(np.ma.masked_array(df_period.index, is_nan))
        if not isinstance(clumps, list): clumps = [clumps] if clumps else []

        for s in clumps:
            clean_blocks.append(df_period.iloc[s])
        
        if not clean_blocks: continue
            
        basin_clean_df = pd.concat(clean_blocks)
        
        basin_scalers[basin] = {
            'q_mean': basin_clean_df[q_col].mean(),
            'q_std': basin_clean_df[q_col].std(),
            'dynamic_means': basin_clean_df[dyn_cols].mean().values,
            'dynamic_stds': basin_clean_df[dyn_cols].std().values
        }

    scalers = {'basin_scalers': basin_scalers}
    
    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name="ver2_0")
    
    attributes = None
    if not cfg["no_static"]:
        # Convert string basins to integers for attributes DataFrame
        attributes = load_attributes(db_path=db_path, basins=[int(b) for b in basins], drop_lat_lon=True)
        scalers['static_means'] = attributes.mean().values
        static_stds = attributes.std().values.copy()  # .values can be a read-only view
        # Replace zero stds with 1 to avoid division by zero for constant features
        static_stds[static_stds == 0] = 1.0
        # Also handle NaN stds (in case of missing data)
        static_stds[np.isnan(static_stds)] = 1.0
        scalers['static_stds'] = static_stds

    def convert_for_json(obj):
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.generic): return obj.item()
        else: return obj
            
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(convert_for_json(scalers), fp, indent=4)
    print(f"Scalers saved to {cfg['run_dir'] / 'scalers.json'}")
    
    dataset = HydroDataset(basins, [cfg['train_start'], cfg['train_end']], cfg, scalers, attributes)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    # Get actual number of static features from attributes DataFrame
    input_size_stat = 0 if cfg["no_static"] else attributes.shape[1]
    input_size_dyn = 3
    if not cfg["no_static"] and cfg["concat_static"]:
        input_size_dyn = 3 + input_size_stat

    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"]).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = nn.MSELoss() if cfg["use_mse"] else NSELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        train_epoch(model, optimizer, loss_func, loader, cfg, epoch)
        torch.save(model.state_dict(), str(cfg["run_dir"] / f"model_epoch{epoch}.pt"))

    torch.save(model.state_dict(), str(cfg["run_dir"] / "lstm_model_final.pt"))
    print(f"Final LSTM model saved to {cfg['run_dir'] / 'lstm_model_final.pt'}")


if __name__ == "__main__":
    config = get_args()
    train(config)

