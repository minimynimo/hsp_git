"""
TCN Regional Hydrological Modeling

Adapted from LSTM regional modeling code to use Temporal Convolutional Network (TCN) instead of LSTM.
TCN preserves the temporal structure while using dilated convolutions.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_basin_attributes, load_attributes
from papercode.utils import create_h5_files, get_basin_list

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

# Global Settings
GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'seq_length': 30,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}



ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

# Basin configuration
if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
    attribute_size = 196
    output_mean = np.array([3.8526752089320553])
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 396
    attribute_size = 209
    output_mean = np.array([1.838397528976445])

file_num_list = list(map(str, range(1, file_tot_num + 1)))

def file_name(input_num, total_len):
    """Generate zero-padded filename"""
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data_for_basin(file_num):
    """Load data for a specific basin"""
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
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

def save_evaluation_predictions(model, basins, cfg):
    """Generate and save predictions for the evaluation period"""
    print("\n--- Generating Evaluation Period Predictions ---")
    try:
        # Create evaluation dataset
        eval_file = cfg["train_dir"] / 'eval_data.h5'
        create_h5_files(out_file=eval_file,
                        basins=basins,
                        dates=[cfg["val_start"], cfg["val_end"]],
                        with_basin_str=True,
                        seq_length=cfg["seq_length"])
        
        eval_ds = CamelsH5(h5_file=str(eval_file),
                          basins=basins,
                          db_path=cfg["db_path"],
                          concat_static=cfg["concat_static"],
                          cache=False,
                          no_static=cfg["no_static"])
        
        eval_loader = DataLoader(eval_ds, batch_size=1024, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_basins = []
        
        with torch.no_grad():
            for data in tqdm(eval_loader, desc="Generating evaluation predictions"):
                if cfg["no_static"]:
                    x, y, q_stds = data
                else:
                    x, y, q_stds = data
                
                # Move to device
                x = x.to(DEVICE)
                
                # Make predictions
                predictions = model(x).cpu().numpy().flatten()
                targets = y.numpy().flatten()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Get basin information for each sample
        basin_dates = []
        for i in range(len(eval_ds)):
            with h5py.File(str(eval_file), 'r') as f:
                basin = f["sample_2_basin"][i].decode("ascii")
                basin_dates.append(basin)
        
        # Create evaluation results DataFrame
        eval_df = pd.DataFrame({
            'basin': basin_dates,
            'prediction': all_predictions,
            'observation': all_targets
        })
        
        # Save evaluation predictions
        os.makedirs(str(cfg["run_dir"]) + "/predict", exist_ok=True)
        eval_output_path = f"{cfg['run_dir']}/predict/TCN_evaluation_predictions.csv"
        eval_df.to_csv(eval_output_path, index=False)
        print(f"Evaluation predictions saved to {eval_output_path}")
        
        # Also save by basin for easier analysis
        for basin in basins:
            basin_data = eval_df[eval_df['basin'] == basin].copy()
            if not basin_data.empty:
                basin_output_path = f"{cfg['run_dir']}/predict/TCN_evaluation_basin_{basin}.csv"
                basin_data.to_csv(basin_output_path, index=False)
        
        print("Evaluation period predictions completed and saved.")
        
    except Exception as e:
        print(f"Error generating evaluation predictions: {e}")
        print("Continuing without evaluation predictions...")

def predict_synthetic_data(model, basins, mean_dyn, std_dyn, cfg):
    """Generate predictions for synthetic data"""
    syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}
    
    for file_num in basins:
        print(f"Processing synthetic data for basin {file_num}...")
        df_current_basin = load_data_for_basin(int(file_num))
        
        for syn_precip_col in syn_precip_cols:
            if syn_precip_col.startswith('syn_peak'):
                end_date_eva_syn = '2001-02-10'
            else:
                end_date_eva_syn = cfg['val_end'].strftime('%Y-%m-%d')
            
            start_date = cfg['train_start'].strftime('%Y-%m-%d')
            df_current = df_current_basin[start_date:end_date_eva_syn].copy()
            
            # Select appropriate temperature and PET columns
            syn_temp_col = 'Temp_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'Temp'
            syn_pet_col = 'PET_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'PET'
            
            # Create input data array
            syn_input_dyn = np.column_stack([
                df_current[syn_precip_col].values,
                df_current[syn_temp_col].values,
                df_current[syn_pet_col].values
            ])
            
            # Remove NaN values
            valid = ~np.isnan(syn_input_dyn).any(axis=1)
            syn_input_clean = syn_input_dyn[valid]
            syn_dates = df_current.index[valid]
            
            if len(syn_input_clean) == 0:
                continue
            
            # Normalize using training statistics
            syn_input_norm = (syn_input_clean - mean_dyn) / std_dyn
            
            # Create sequences for TCN (preserve temporal structure)
            seq_length = cfg['seq_length']
            X_syn_list = []
            dates_syn_list = []
            
            for i in range(len(syn_input_norm) - seq_length + 1):
                X_syn_list.append(syn_input_norm[i:i+seq_length])
                dates_syn_list.append(syn_dates[i+seq_length-1])
            
            if len(X_syn_list) == 0:
                continue
                
            X_syn = torch.tensor(np.array(X_syn_list), dtype=torch.float32).to(DEVICE)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                syn_predict = model(X_syn).cpu().numpy().flatten()
            syn_predict[syn_predict < 0] = 0
            
            # Store results
            temp_syn_df = pd.DataFrame({
                'Date': dates_syn_list,
                'prediction': syn_predict
            })
            syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df
    
    return syn_predict_dfs_combined

def save_synthetic_predictions(syn_predict_dfs_combined, output_dir):
    """Save synthetic predictions to CSV files"""
    os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)
    
    for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
        if not basin_preds_dict:
            continue
        
        combined_syn_df = pd.DataFrame()
        for file_num, df_preds in basin_preds_dict.items():
            df_preds_renamed = df_preds.set_index('Date').rename(columns={'prediction': f'file_{file_num}'})
            if combined_syn_df.empty:
                combined_syn_df = df_preds_renamed
            else:
                combined_syn_df = combined_syn_df.join(df_preds_renamed, how='outer')
        
        combined_syn_df.sort_index(inplace=True)
        combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/TCN_predict_{syn_precip_col.replace('syn_', '')}.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TemporalBlock(nn.Module):
    """Single temporal block for TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
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

class Chomp1d(nn.Module):
    """Chomp excess padding"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """TCN network"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    """TCN model for hydrological modeling"""
    def __init__(self, input_size_dyn, input_size_stat=0, num_channels=[25, 50, 100], 
                 kernel_size=2, dropout=0.2, concat_static=True, no_static=False):
        super(TCNModel, self).__init__()
        
        self.concat_static = concat_static
        self.no_static = no_static
        
        # Always concatenate static features for simplicity
        if not no_static and concat_static:
            total_input_size = input_size_dyn + input_size_stat
        else:
            total_input_size = input_size_dyn
            
        self.tcn = TemporalConvNet(total_input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
        # TCN expects: (batch_size, features, seq_length)
        x = x.transpose(1, 2)
        
        # TCN forward pass
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], seq_length)
        
        # Take the last time step
        out = tcn_out[:, :, -1]  # (batch_size, num_channels[-1])
        
        # Final prediction
        out = self.linear(out)  # (batch_size, 1)
        
        return out

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(10000000),
                        help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--cache_data', type=bool, default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static', type=str2bool, default=True,
                        help="If True, trains TCN without static features")
    parser.add_argument('--concat_static', type=bool, default=True,
                        help="Always True for TCN - concatenate static features")
    parser.add_argument('--use_mse', type=bool, default=False,
                        help="If True, uses mean squared error as loss function.")
    
    # TCN specific arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout probability")
    parser.add_argument('--nb_filters', type=int, default=128,
                        help="Number of filters in TCN")
    parser.add_argument('--kernel_size', type=int, default=4,
                        help="Kernel size for TCN")
    parser.add_argument('--nb_stacks', type=int, default=3,
                        help="Number of stacks in TCN")
    
    cfg = vars(parser.parse_args())
    
    # Add global settings
    cfg.update(GLOBAL_SETTINGS)
    
    # Override concat_static to always be True for TCN
    cfg["concat_static"] = True
    
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{month}{day}_{hour}{minute}_seed{cfg["seed"]}'

    cfg['run_dir'] = Path(__file__).parent / "runs" / run_name
    if not cfg['run_dir'].exists():
        cfg['run_dir'].mkdir(parents=True)
    
    # Create file paths
    cfg['train_dir'] = cfg['run_dir'] / 'data'
    cfg['train_dir'].mkdir(parents=True, exist_ok=True)
    cfg['db_path'] = str(cfg['run_dir'] / 'attributes.db')
    
    # Store config
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        temp_cfg = cfg.copy()
        temp_cfg['train_start'] = temp_cfg['train_start'].strftime('%d/%m/%Y')
        temp_cfg['train_end'] = temp_cfg['train_end'].strftime('%d/%m/%Y') 
        temp_cfg['val_start'] = temp_cfg['val_start'].strftime('%d/%m/%Y')
        temp_cfg['val_end'] = temp_cfg['val_end'].strftime('%d/%m/%Y')
        temp_cfg['run_dir'] = str(temp_cfg['run_dir'])
        temp_cfg['train_dir'] = str(temp_cfg['train_dir'])
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
        
    return cfg

def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Prepare training data"""
    add_basin_attributes(db_path=cfg["db_path"], loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)
    
    # Create .h5 files for train data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])
    
    return cfg

def train(cfg):
    """Train TCN model"""
    print("Starting TCN training...")
    
    # Set random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    
    # Print configuration
    for key, val in cfg.items():
        print(f"{key}: {val}")
    
    basins = file_num_list
    
    # Create folder structure for this run  
    cfg = _setup_run(cfg)
    
    # Prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    ds = CamelsH5(h5_file=str(cfg["train_file"]),
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=cfg["concat_static"],
                  cache=cfg["cache_data"],
                  no_static=cfg["no_static"])
    
    loader = DataLoader(ds,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=cfg["num_workers"])
    
    # Create model
    input_size_stat = 0 if cfg["no_static"] else 27
    input_size_dyn = 3 if (cfg["no_static"] or not cfg["concat_static"]) else 30
    
    print(f"Creating TCN model with input_size_dyn={input_size_dyn}, input_size_stat={input_size_stat}")
    model = TCNModel(input_size_dyn=input_size_dyn,
                     input_size_stat=input_size_stat,
                     num_channels=[cfg["nb_filters"], cfg["nb_filters"]*2, cfg["nb_filters"]*4],
                     kernel_size=cfg["kernel_size"],
                     dropout=cfg["dropout"],
                     concat_static=cfg["concat_static"],
                     no_static=cfg["no_static"]).to(DEVICE)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        # Use MSE for simplicity in TCN
        loss_func = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for batch_idx, data in enumerate(pbar):
            optimizer.zero_grad()
            
            if cfg["no_static"]:
                x, y, q_stds = data
                x, y = x.to(DEVICE), y.to(DEVICE)
            else:
                x, y, q_stds = data
                x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            predictions = model(x)
            
            # Calculate loss
            loss = loss_func(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{cfg['epochs']}, Average Loss: {avg_loss:.6f}")
        
        # Save model checkpoint
        model_path = cfg["run_dir"] / f"tcn_model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))
    
    # Save final model
    final_model_path = cfg["run_dir"] / "tcn_model_final.pt"
    torch.save(model.state_dict(), str(final_model_path))
    print(f"Final model saved to {final_model_path}")
    
    # Generate synthetic data predictions
    print("\n--- Generating Synthetic Data Predictions ---")
    try:
        # Calculate normalization statistics from the dataset
        # We need to recreate the dataset to get the normalization stats
        temp_ds = CamelsH5(h5_file=str(cfg["train_file"]),
                          basins=basins,
                          db_path=cfg["db_path"],
                          concat_static=cfg["concat_static"],
                          cache=False,
                          no_static=cfg["no_static"])
        
        # Sample some data to calculate statistics
        sample_data = []
        for i in range(min(1000, len(temp_ds))):
            x, _, _ = temp_ds[i]
            sample_data.append(x[:, :3])  # First 3 features are dynamic
        
        if sample_data:
            sample_array = torch.cat(sample_data, dim=0)
            mean_dyn = sample_array.mean(dim=0).numpy()
            std_dyn = sample_array.std(dim=0).numpy()
            std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)
            
            syn_predictions = predict_synthetic_data(model, basins, mean_dyn, std_dyn, cfg)
            save_synthetic_predictions(syn_predictions, str(cfg["run_dir"]))
            print("Synthetic data predictions completed and saved.")
            
            # Generate evaluation period predictions
            save_evaluation_predictions(model, basins, cfg)
        else:
            print("No data available for synthetic predictions.")
        
    except Exception as e:
        print(f"Error generating synthetic predictions: {e}")
        print("Continuing without synthetic predictions...")

def evaluate(cfg):
    """Evaluate trained TCN model"""
    print("TCN evaluation not implemented yet")
    pass

if __name__ == "__main__":
    config = get_args()
    
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "evaluate":
        evaluate(config)