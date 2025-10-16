"""
Adapted TCN Evaluation Script

This script evaluates a trained TCN (Temporal Convolutional Network) model using an adapted approach 
where the model is fine-tuned for 5 additional epochs on each basin individually before making predictions.

Usage:
    python eval_adapted.py --run_dir path/to/trained/model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from papercode.datasets import CamelsH5
from papercode.datautils import add_basin_attributes, load_attributes, rescale_features
from papercode.utils import create_h5_files, get_basin_list

# Global Settings (matching main.py)
GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'seq_length': 30,  # TCN uses shorter sequences
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def normalize_basin_data(data, basin_stats):
    """Normalize data using basin-specific statistics"""
    normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        mean = basin_stats['mean'][i]
        std = basin_stats['std'][i]
        if std > 0:
            normalized[:, i] = (data[:, i] - mean) / std
        else:
            normalized[:, i] = data[:, i] - mean
    return normalized

def denormalize_predictions(predictions, basin_stats, feature_idx=3):
    """Denormalize predictions using basin-specific output statistics"""
    mean = basin_stats['mean'][feature_idx]
    std = basin_stats['std'][feature_idx]
    if std > 0:
        return predictions * std + mean
    else:
        return predictions + mean

def create_sequences_tcn(data, seq_length, static_features=None):
    """Create sequences for TCN"""
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data)):
        # Get sequence (keep temporal structure)
        seq = data[i-seq_length:i, :3]  # Only dynamic features (P, PET, Temp)
        
        # Concatenate static features to each time step if provided
        if static_features is not None:
            static_expanded = np.tile(static_features, (seq_length, 1))
            seq = np.concatenate([seq, static_expanded], axis=1)
        
        sequences.append(seq)
        targets.append(data[i, 3])  # Assuming discharge is at index 3
        
    return np.array(sequences), np.array(targets)

# TCN Model Components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
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

class TemporalConvNet(nn.Module):
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
                 kernel_size=2, dropout=0.2, no_static=False):
        super(TCNModel, self).__init__()
        
        self.no_static = no_static
        
        # Always concatenate static features for simplicity
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

def fine_tune_basin_model(base_model, train_loader, run_cfg, basin):
    """Fine-tune the base model for a specific basin"""
    # Create a copy of the base model for this basin
    basin_model = copy.deepcopy(base_model)
    basin_model.train()
    
    # Set up optimizer for fine-tuning with a smaller learning rate
    fine_tune_lr = run_cfg.get("learning_rate", 0.001) * 0.1  # Use 10% of original learning rate
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=fine_tune_lr)
    
    # Loss function
    loss_func = nn.MSELoss()
    
    print(f"  Fine-tuning model for basin {basin} for 5 epochs...")
    
    # Fine-tune for 5 epochs
    for epoch in range(5):
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            x_d, x_s, y = batch
            x_d, y = x_d.to(DEVICE), y.to(DEVICE)
            
            # Normalize targets using basin-specific rescaling
            y_rescaled = rescale_features(y, variable='output', loc=GLOBAL_SETTINGS['loc'])
            
            optimizer.zero_grad()
            
            # Forward pass - use the concatenated input
            predictions = basin_model(x_d)
            
            # Calculate loss
            loss = loss_func(predictions.squeeze(), y_rescaled.squeeze())
            loss.backward()
            
            # Gradient clipping
            if run_cfg.get("clip_norm", True):
                torch.nn.utils.clip_grad_norm_(basin_model.parameters(), 
                                             run_cfg.get("clip_value", 1))
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        print(f"  Epoch {epoch+1}/5, Loss: {avg_loss:.6f}")
    
    return basin_model

def evaluate_basin_adapted_period(basin_model, basin, start_date, end_date, run_cfg, static_attrs=None):
    """Evaluate the fine-tuned model on a basin for a specific period"""
    # Load data for this basin
    df = load_data_for_basin(int(basin))
    
    # Filter by date range with warmup
    warmup_start = start_date - pd.DateOffset(days=run_cfg["seq_length"] - 1)
    period_data = df[warmup_start:end_date].copy()
    
    if len(period_data) < run_cfg["seq_length"]:
        return None, None, None
    
    # Prepare input data (P, PET, Temp, Obs flow for normalization)
    input_cols = ['Precip', 'PET', 'Temp', 'Obs flow']
    input_data = period_data[input_cols].values
    
    # Calculate basin-specific normalization statistics
    basin_stats = {
        'mean': np.nanmean(input_data, axis=0),
        'std': np.nanstd(input_data, axis=0)
    }
    
    # Normalize the data per basin
    input_data_norm = normalize_basin_data(input_data, basin_stats)
    
    # Create sequences
    static_features = static_attrs[basin] if static_attrs and basin in static_attrs else None
    sequences, targets = create_sequences_tcn(input_data_norm, run_cfg["seq_length"], static_features)
    
    # Get dates for predictions (after warmup period)
    dates = period_data.index[run_cfg["seq_length"]:]
    
    # Only include sequences within the actual period (not warmup)
    actual_period_mask = dates >= start_date
    sequences = sequences[actual_period_mask]
    targets = targets[actual_period_mask]
    dates = dates[actual_period_mask]
    
    if len(sequences) == 0:
        return None, None, None
    
    # Convert to tensors
    sequences = torch.FloatTensor(sequences)
    batch_size = 128  # Set batch size for inference
    predictions = []
    basin_model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size].to(DEVICE)
            batch_pred = basin_model(batch).cpu().numpy().flatten()
            predictions.append(batch_pred)
    predictions = np.concatenate(predictions)
    # Rescale predictions
    predictions_rescaled = rescale_features(predictions, variable='output', loc=GLOBAL_SETTINGS['loc'])
    predictions_rescaled[predictions_rescaled < 0] = 0
    # Get actual targets (denormalized)
    targets_rescaled = denormalize_predictions(targets, basin_stats)
    return predictions_rescaled, targets_rescaled, dates

def get_param_string(run_cfg):
    """Create parameter string for file naming"""
    # Check if the config uses the new format (nb_filters, nb_stacks) or old format (num_channels)
    if 'nb_filters' in run_cfg and 'nb_stacks' in run_cfg:
        nb_filters = run_cfg.get("nb_filters", 128)
        nb_stacks = run_cfg.get("nb_stacks", 3)
        param_channels = f"f{nb_filters}_s{nb_stacks}"
    else:
        num_channels = run_cfg.get("num_channels", [25, 50, 100])
        param_channels = f"ch{'-'.join(map(str, num_channels))}"
    
    kernel_size = run_cfg.get("kernel_size", 2)
    dropout = run_cfg.get("dropout", 0.2)
    learning_rate = run_cfg.get("learning_rate", 0.001)
    seq_length = run_cfg.get("seq_length", 30)
    
    return f"{param_channels}_k{kernel_size}_d{dropout}_lr{learning_rate}_w{seq_length}"

def evaluate_adapted(model, run_cfg, output_dir):
    """Evaluate model using adapted approach (fine-tuning per basin)"""
    print("\n--- Adapted Evaluation (Fine-tuning per basin) ---")
    
    basins = file_num_list
    
    # Define periods
    periods = {
        'cal': (GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]),
        'eva': (GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"])
    }
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"\nEvaluating {period_name.upper()} period ({start_date.date()} to {end_date.date()})")
        
        # Collect all predictions and targets by basin
        basin_predictions = {}
        basin_dates = {}
        all_results = []
        
        for basin in tqdm(basins, desc=f"Processing basins for {period_name}"):
            try:
                # Create training dataset for this basin (always use training period for fine-tuning)
                train_file = output_dir.parent / "temp" / f'train_data_basin_{basin}.h5'
                train_file.parent.mkdir(parents=True, exist_ok=True)
                
                create_h5_files(out_file=train_file,
                                basins=[basin],
                                dates=[GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]],
                                with_basin_str=True,
                                seq_length=run_cfg["seq_length"])
                
                # Create datasets
                db_path = str(run_cfg["run_dir"] / "attributes.db")
                
                train_ds = CamelsH5(h5_file=str(train_file),
                                   basins=[basin],
                                   db_path=db_path,
                                   cache=False,
                                   no_static=run_cfg["no_static"])
                
                # Create data loader for training
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                
                # Fine-tune model for this basin
                basin_model = fine_tune_basin_model(model, train_loader, run_cfg, basin)
                
                # Evaluate on the current period using direct data loading
                predictions, targets, dates = evaluate_basin_adapted_period(
                    basin_model, basin, start_date, end_date, run_cfg)
                
                if predictions is not None:
                    basin_predictions[basin] = predictions
                    basin_dates[basin] = dates
                    
                    # Calculate metrics for this basin
                    valid_idx = targets >= 0
                    if np.sum(valid_idx) > 0:
                        obs_valid = targets[valid_idx]
                        pred_valid = predictions[valid_idx]
                        
                        # Calculate benchmarks
                        metrics = {}
                        for benchmark in benchmark_list:
                            metrics[benchmark] = BMK(obs_valid, pred_valid, benchmark)
                        
                        result_row = {'basin': basin}
                        result_row.update(metrics)
                        all_results.append(result_row)
                        
                        print(f"Basin {basin}: NSE = {metrics['NSE']:.4f}")
                
                # Clean up temporary files
                train_file.unlink(missing_ok=True)
                
            except Exception as e:
                print(f"Error processing basin {basin}: {e}")
                continue
        
        if not basin_predictions:
            print(f"No valid predictions for {period_name} period")
            continue
        
        # Create prediction matrix (basins x dates)
        all_dates = []
        for dates in basin_dates.values():
            all_dates.extend(dates)
        unique_dates = sorted(list(set(all_dates)))
        
        # Create prediction DataFrame
        pred_matrix = []
        basin_order = []
        
        for basin in sorted(basin_predictions.keys(), key=int):
            basin_order.append(basin)
            basin_preds = basin_predictions[basin]
            basin_dates_list = basin_dates[basin]
            
            # Create row for this basin
            row = {}
            for i, date in enumerate(basin_dates_list):
                row[date.strftime('%Y-%m-%d')] = basin_preds[i]
            
            pred_matrix.append(row)
        
        # Convert to DataFrame with basins as index and dates as columns
        pred_df = pd.DataFrame(pred_matrix, index=basin_order)
        pred_df = pred_df.reindex(columns=sorted(pred_df.columns))
        
        # Save prediction file
        param_string = get_param_string(run_cfg)
        pred_output_path = output_dir / "predict" / f"TCN_predict_{param_string}_{period_name}.csv"
        pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_output_path, index_label='basin')
        print(f"Predictions saved to {pred_output_path}")
        
        # Create results DataFrame (basins x benchmarks)
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.set_index('basin')
            
            # Save results file
            results_output_path = output_dir / "results" / f"TCN_results_{param_string}_{period_name}.csv"
            results_output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_output_path, index_label='basin')
            print(f"Results saved to {results_output_path}")
            
            # Print summary
            print(f"{period_name.upper()} Period Summary:")
            #print(f"Mean NSE: {results_df['NSE'].mean():.4f}")
            #print(f"Median NSE: {results_df['NSE'].median():.4f}")
            #print(f"NSE > 0.5: {(results_df['NSE'] > 0.5).sum()}/{len(results_df)} basins")

def BMK(obs_data, sim_data, benchmark):    
    """Calculate benchmark metrics"""
    # Ensure data are flat for calculations if they come in (N, 1) shape
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

    # Handle cases where obs_data might be all zero or very close to zero
    if np.sum(obs_data) == 0 and benchmark not in ["RMSE", "MAE"]:
        return np.nan

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        if np.sum(np.square(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
        
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        if np.sum(np.abs(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
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

def main():
    parser = argparse.ArgumentParser(description='Adapted TCN Evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--model_epoch', type=int, default=50,
                        help='Which epoch model to use for evaluation')
    parser.add_argument('--n_basins', type=int, default=None,
                        help='Number of basins to process (for testing)')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist")
    
    # Load run configuration
    with open(run_dir / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)
    
    run_cfg["run_dir"] = run_dir  # Add run_dir to config
    
    print("Adapted TCN Evaluation")
    print(f"Model directory: {run_dir}")
    print(f"Using model from epoch: {args.model_epoch}")
    
    # Limit number of basins for testing if specified
    global file_num_list
    if args.n_basins is not None:
        file_num_list = file_num_list[:args.n_basins]
        print(f"Processing only first {args.n_basins} basins for testing")
    
    # Create output directory
    output_dir = Path("/data0/funato/hsp/0_out/TCN/GB/adapted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base model
    input_size_stat = 0 if run_cfg["no_static"] else 209  # GB attribute size
    input_size_dyn = 3 if (run_cfg["no_static"]) else 3
    
    # Handle different parameter formats
    if 'nb_filters' in run_cfg and 'nb_stacks' in run_cfg:
        # New format: nb_filters and nb_stacks
        nb_filters = run_cfg.get("nb_filters", 128)
        nb_stacks = run_cfg.get("nb_stacks", 3)
        num_channels = [nb_filters] * nb_stacks
    else:
        # Old format: num_channels list
        num_channels = run_cfg.get("num_channels", [25, 50, 100])
    
    model = TCNModel(input_size_dyn=input_size_dyn,
                     input_size_stat=input_size_stat,
                     num_channels=num_channels,
                     kernel_size=run_cfg.get("kernel_size", 2),
                     dropout=run_cfg.get("dropout", 0.2),
                     no_static=run_cfg["no_static"]).to(DEVICE)
    
    # Load trained model weights
    model_file = run_dir / f"model_epoch{args.model_epoch}.pt"
    if not model_file.exists():
        model_file = run_dir / "tcn_model_final.pt"
        if not model_file.exists():
            raise ValueError(f"Model file not found. Looked for {run_dir}/model_epoch{args.model_epoch}.pt and {run_dir}/tcn_model_final.pt")
    
    print(f"Loading base model from: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    # Evaluate using adapted approach
    evaluate_adapted(model, run_cfg, output_dir)
    
    print("\nAdapted evaluation completed successfully!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()