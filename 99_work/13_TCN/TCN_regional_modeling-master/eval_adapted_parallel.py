"""
Parallel Adapted TCN Evaluation Script

This script evaluates a trained TCN model using an adapted approach with parallel processing.
Each process handles a subset of basins, allowing for faster evaluation across multiple cores.

Usage:
    python eval_adapted_parallel.py --run_dir path/to/trained/model --process_id 0 --total_processes 10
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import copy
from sklearn.metrics import mean_squared_error
import re
import pickle

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

benchmarks = ["NSE", "KGE", "E1", "VE", "RMSE", "MAE"]

syn_precip_cols = ['syn_no_precip', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # Force CPU for debugging


if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
elif GLOBAL_SETTINGS['loc'] == "GB": file_tot_num = 424
elif GLOBAL_SETTINGS['loc'] == "US": file_tot_num = 614
file_num_list = list(map(str, range(1, file_tot_num + 1)))

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
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
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
# Utility Functions       #
# ####################### #

def file_name(input_num, total_len): return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv", index_col='Date', parse_dates=True)
    return df

def load_attributes(db_path, basins, drop_lat_lon=True):
    from papercode.datautils import load_attributes as load_attributes_from_lib
    return load_attributes_from_lib(db_path, basins, drop_lat_lon)

def BMK(obs_data, sim_data, benchmark):    
    """Calculate benchmark metrics"""
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

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

# ####################### #
# Data Handling           #
# ####################### #

class FineTuneDataset(Dataset):
    def __init__(self, basin, dates, cfg, static_attrs):
        self.samples, self.scalers = [], {}
        self._prepare_data(basin, dates, cfg, static_attrs)

    def _prepare_data(self, basin, dates, cfg, static_attrs):
        df = load_data_for_basin(int(basin))
        df_period = df.loc[dates[0]:dates[1]]
        dyn_cols, q_col = ['Precip', 'PET', 'Temp'], 'Obs flow'

        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clean_blocks = [df_period.iloc[s] for s in np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))]
        if not clean_blocks: return
        
        basin_clean_df = pd.concat(clean_blocks)
        self.scalers = {
            'q_mean': basin_clean_df[q_col].mean(), 'q_std': basin_clean_df[q_col].std(),
            'dynamic_means': basin_clean_df[dyn_cols].mean().values, 'dynamic_stds': basin_clean_df[dyn_cols].std().values
        }

        dyn_scaled = (df_period[dyn_cols].values - self.scalers['dynamic_means']) / self.scalers['dynamic_stds']
        q_scaled = (df_period[q_col].values - self.scalers['q_mean']) / self.scalers['q_std']
        
        static_scaled = None
        if static_attrs is not None and not cfg["no_static"]:
            static_data = static_attrs.loc[int(basin)].values
            static_scaled = static_data

        for block_df in clean_blocks:
            if len(block_df) < cfg['seq_length']: continue
            start_loc = df_period.index.get_loc(block_df.index[0])
            end_loc = df_period.index.get_loc(block_df.index[-1])
            block_dyn, block_q = dyn_scaled[start_loc:end_loc+1], q_scaled[start_loc:end_loc+1]
            
            for i in range(len(block_df) - cfg['seq_length'] + 1):
                target_idx = i + cfg['seq_length'] - 1
                dyn_seq = block_dyn[i:target_idx + 1]
                final_features = dyn_seq
                if static_scaled is not None:
                    static_seq = np.tile(static_scaled, (cfg['seq_length'], 1))
                    final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                self.samples.append((final_features.astype(np.float32), block_q[target_idx].astype(np.float32)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor([y])

class SyntheticFineTuneDataset(Dataset):
    """Dataset for synthetic precipitation scenarios"""
    def __init__(self, basin, scenario_col, dates, cfg, static_attrs, base_scalers):
        self.samples = []
        self.scalers = base_scalers
        self._prepare_synthetic_data(basin, scenario_col, dates, cfg, static_attrs)

    def _prepare_synthetic_data(self, basin, scenario_col, dates, cfg, static_attrs):
        df = load_data_for_basin(int(basin))
        warmup_start = dates[0] - pd.DateOffset(days=cfg['seq_length'] - 1)
        df_period = df.loc[warmup_start:dates[1]]
        
        if scenario_col not in df_period.columns:
            return
        
        df_syn = df_period.copy()
        df_syn['Precip'] = df_syn[scenario_col]
        
        if 'peak' in scenario_col:
            if 'Temp_at_peak' in df_syn.columns:
                df_syn['Temp'] = df_syn['Temp_at_peak']
            if 'PET_at_peak' in df_syn.columns:
                df_syn['PET'] = df_syn['PET_at_peak']
        
        dyn_cols = ['Precip', 'PET', 'Temp']
        dyn_scaled = (df_syn[dyn_cols].values - self.scalers['dynamic_means']) / self.scalers['dynamic_stds']
        
        static_scaled = None
        if static_attrs is not None and not cfg["no_static"]:
            static_data = static_attrs.loc[int(basin)].values
            static_scaled = static_data

        is_nan = df_syn[dyn_cols].isnull().any(axis=1)
        clean_blocks = [df_syn.iloc[s] for s in np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))]
        
        for block_df in clean_blocks:
            if len(block_df) < cfg['seq_length']:
                continue
            start_loc = df_syn.index.get_loc(block_df.index[0])
            end_loc = df_syn.index.get_loc(block_df.index[-1])
            block_dyn = dyn_scaled[start_loc:end_loc+1]
            
            for i in range(len(block_df) - cfg['seq_length'] + 1):
                target_idx = i + cfg['seq_length'] - 1
                sample_date = block_df.index[i + cfg['seq_length'] - 1]
                
                if sample_date < dates[0]:
                    continue
                
                dyn_seq = block_dyn[i:target_idx + 1]
                final_features = dyn_seq
                if static_scaled is not None:
                    static_seq = np.tile(static_scaled, (cfg['seq_length'], 1))
                    final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                
                self.samples.append((final_features.astype(np.float32), sample_date))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, date = self.samples[idx]
        return torch.from_numpy(x), date

# ####################### #
# Main Logic              #
# ####################### #

def save_basin_model(basin, model, scalers, save_dir):
    """Save fine-tuned model and scalers for a basin."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"basin_{basin}_model.pt"
    scalers_path = save_dir / f"basin_{basin}_scalers.pkl"
    
    torch.save(model.state_dict(), model_path)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)

def load_basin_model(basin, base_model, save_dir):
    """Load fine-tuned model and scalers for a basin if they exist."""
    save_dir = Path(save_dir)
    model_path = save_dir / f"basin_{basin}_model.pt"
    scalers_path = save_dir / f"basin_{basin}_scalers.pkl"
    
    if model_path.exists() and scalers_path.exists():
        model = copy.deepcopy(base_model)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        return model, scalers
    return None, None

def fine_tune_basin_model(base_model, basin, cfg, static_attrs):
    """Fine-tune the base model once for a basin using calibration period data."""
    ft_dataset = FineTuneDataset(basin, (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), cfg, static_attrs)
    if len(ft_dataset) == 0:
        return None, None

    ft_loader = DataLoader(ft_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)
    
    basin_model = copy.deepcopy(base_model).train()
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    
    for _ in range(5):
        for x, y in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = basin_model(x)
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()
            
    basin_model.eval()
    return basin_model, ft_dataset.scalers

def evaluate_period(basin_model, basin, dates, cfg, static_attrs, scalers):
    """Evaluate an already fine-tuned model on a specific period."""
    eval_dataset = FineTuneDataset(basin, dates, cfg, static_attrs)
    if len(eval_dataset) == 0:
        return None, None, None
        
    eval_loader = DataLoader(eval_dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False)
    
    predictions, observations, eval_dates = [], [], []
    with torch.no_grad():
        for x, y_scaled in eval_loader:
            x = x.to(DEVICE)
            preds_scaled = basin_model(x).cpu().numpy().flatten()
            
            preds_rescaled = (preds_scaled * scalers['q_std']) + scalers['q_mean']
            obs_rescaled = y_scaled.numpy().flatten() * scalers['q_std'] + scalers['q_mean']
            
            predictions.extend(preds_rescaled)
            observations.extend(obs_rescaled)

    eval_dates = [s[2] if hasattr(s, '__len__') and len(s) > 2 else None for s in eval_dataset.samples]
    if None in eval_dates:
        eval_dates = list(range(len(eval_dataset)))
        
    return np.array(predictions), np.array(observations), eval_dates

def custom_collate_fn(batch):
    """Custom collate function to handle Timestamp objects."""
    x_batch = torch.stack([item[0] for item in batch])
    dates_batch = [item[1] for item in batch]
    return x_batch, dates_batch

def evaluate_synthetic(basin_model, basin, scenario_col, dates, cfg, static_attrs, scalers):
    """Evaluate an already fine-tuned model on synthetic scenario."""
    syn_ds = SyntheticFineTuneDataset(basin, scenario_col, dates, cfg, static_attrs, scalers)
    if len(syn_ds) == 0:
        return None, None
    
    syn_loader = DataLoader(syn_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    
    preds_list, dates_list = [], []
    with torch.no_grad():
        for x_batch, date_batch in syn_loader:
            x_batch = x_batch.to(DEVICE)
            y_pred = basin_model(x_batch)
            preds_list.extend(y_pred.cpu().numpy().flatten())
            dates_list.extend(date_batch)
    
    preds_rescaled = np.array(preds_list) * scalers['q_std'] + scalers['q_mean']
    return preds_rescaled, dates_list

def main():
    parser = argparse.ArgumentParser(description='Parallel Adapted TCN Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=int, default=30)
    parser.add_argument('--process_id', type=int, required=True, help='Process ID (0-indexed)')
    parser.add_argument('--total_processes', type=int, required=True, help='Total number of parallel processes')
    parser.add_argument('--stage', type=str, default='all', choices=['finetune', 'evaluate', 'all'],
                        help='Stage to run: finetune only, evaluate only, or all')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)

    print(f"Parallel Adapted TCN Evaluation - Process {args.process_id}/{args.total_processes}")
    
    # Distribute basins across processes
    all_basins = file_num_list
    basins_per_process = len(all_basins) // args.total_processes
    start_idx = args.process_id * basins_per_process
    if args.process_id == args.total_processes - 1:
        # Last process takes remaining basins
        end_idx = len(all_basins)
    else:
        end_idx = start_idx + basins_per_process
    
    basins_to_process = all_basins[start_idx:end_idx]
    print(f"Process {args.process_id} handling basins {start_idx+1} to {end_idx} ({len(basins_to_process)} basins)")
    
    model_file = run_dir / f"tcn_model_epoch{args.model_epoch}.pt"
    if not model_file.exists(): model_file = run_dir / "tcn_model_final.pt"
    print(f"Loading base model from: {model_file}")

    is_no_static = cfg.get('no_static', False)
    input_size = 3
    if not is_no_static:
        input_size += 27 

    base_model = TCNModel(input_size=input_size, 
                          num_channels=cfg["num_channels"], 
                          kernel_size=cfg["kernel_size"], 
                          dropout=cfg["dropout"]).to(DEVICE)
    base_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    static_attrs = None
    if not is_no_static:
        db_path = run_dir / "attributes.db"
        static_attrs = load_attributes(db_path, basins=file_num_list, drop_lat_lon=True)

    output_dir = Path(f"/data0/funato/hsp2/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted")
    adapted_models_dir = Path("/data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/runs_tcn_adapted")
    adapted_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Fine-tuning stage
    if args.stage in ['finetune', 'all']:
        print("\n" + "="*60)
        print(f"Process {args.process_id}: Fine-tuning basins...")
        print("="*60)
        
        for basin in tqdm(basins_to_process, desc=f"P{args.process_id} Fine-tuning"):
            # Check if already fine-tuned
            basin_model, scalers = load_basin_model(basin, base_model, adapted_models_dir)
            
            if basin_model is None:
                # Fine-tune if not found
                basin_model, scalers = fine_tune_basin_model(base_model, basin, cfg, static_attrs)
                if basin_model is not None:
                    save_basin_model(basin, basin_model, scalers, adapted_models_dir)
                    print(f"Process {args.process_id}: Saved fine-tuned model for basin {basin}")
    
    # Evaluation stage
    if args.stage in ['evaluate', 'all']:
        print("\n" + "="*60)
        print(f"Process {args.process_id}: Evaluating basins...")
        print("="*60)
        
        # Load all fine-tuned models for this process's basins
        basin_models = {}
        basin_scalers = {}
        
        for basin in basins_to_process:
            basin_model, scalers = load_basin_model(basin, base_model, adapted_models_dir)
            if basin_model is not None:
                basin_models[basin] = basin_model
                basin_scalers[basin] = scalers
        
        print(f"Process {args.process_id}: Loaded {len(basin_models)} fine-tuned models")
        
        # Evaluate on cal and eva periods
        for name, dates in {'cal': (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), 
                            'eva': (GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end'])}.items():
            print(f"\nProcess {args.process_id}: Evaluating {name.upper()} period...")
            
            pred_path = output_dir / "predict" / f"TCN_predict_{name}_p{args.process_id}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            
            all_results = []
            first_basin = True
            
            for basin in tqdm(basins_to_process, desc=f"P{args.process_id} {name}"):
                if basin not in basin_models:
                    continue
                    
                preds, obs, eval_dates = evaluate_period(basin_models[basin], basin, dates, cfg, static_attrs, basin_scalers[basin])
                if preds is not None:
                    # Create a Series with dates as index and predictions as values
                    basin_series = pd.Series(preds, index=eval_dates, name=basin)
                    # Convert to DataFrame with basin as row and dates as columns
                    basin_row = basin_series.to_frame().T
                    basin_row.index.name = 'basin'
                    
                    if first_basin:
                        basin_row.to_csv(pred_path, mode='w', header=True)
                        first_basin = False
                    else:
                        basin_row.to_csv(pred_path, mode='a', header=False)
                    
                    metrics = {}
                    for benchmark in benchmarks:
                        metrics[benchmark] = BMK(obs, preds, benchmark)
                    result_row = {'basin': basin}
                    result_row.update(metrics)
                    all_results.append(result_row)

            if all_results:
                temp_df = pd.read_csv(pred_path, index_col='basin')
                temp_df.index = temp_df.index.astype(int)
                temp_df = temp_df.sort_index()
                temp_df.to_csv(pred_path)
                print(f"Process {args.process_id}: Predictions saved to {pred_path}")

                results_df = pd.DataFrame(all_results).set_index('basin')
                results_path = output_dir / "results" / f"TCN_results_{name}_p{args.process_id}.csv"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(results_path)
                print(f"Process {args.process_id}: Results saved to {results_path}")
        
        # Evaluate on synthetic scenarios
        print("\n" + "="*60)
        print(f"Process {args.process_id}: Synthetic Scenario Evaluation...")
        print("="*60)
        
        val_start = GLOBAL_SETTINGS["val_start"]
        val_end = GLOBAL_SETTINGS["val_end"]
        
        for scenario_col in syn_precip_cols:
            print(f"\nProcess {args.process_id}: Processing scenario: {scenario_col}")
            syn_end = pd.to_datetime('2001-02-10') if 'peak' in scenario_col else val_end
            dates = (val_start, syn_end)
            
            pred_path = output_dir / "predict" / "synthetic" / f"TCN_predict_{scenario_col}_eva_p{args.process_id}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            
            first_basin = True
            
            for basin in tqdm(basins_to_process, desc=f"P{args.process_id} {scenario_col}"):
                if basin not in basin_models:
                    continue
                
                preds_rescaled, dates_list = evaluate_synthetic(basin_models[basin], basin, scenario_col, dates, cfg, static_attrs, basin_scalers[basin])
                
                if preds_rescaled is not None:
                    basin_preds_df = pd.DataFrame({str(basin): preds_rescaled}, index=pd.DatetimeIndex(dates_list))
                    basin_preds_df = basin_preds_df[~basin_preds_df.index.duplicated(keep='first')]
                    basin_row = basin_preds_df.T
                    basin_row.index.name = 'basin'
                    
                    if first_basin:
                        basin_row.to_csv(pred_path, mode='w', header=True)
                        first_basin = False
                    else:
                        basin_row.to_csv(pred_path, mode='a', header=False)
            
            if pred_path.exists():
                temp_df = pd.read_csv(pred_path, index_col='basin')
                temp_df.index = temp_df.index.astype(int)
                temp_df = temp_df.sort_index()
                temp_df.to_csv(pred_path)
                print(f"Process {args.process_id}: Saved predictions to {pred_path}")
    
    print(f"\nProcess {args.process_id}: TCN adapted evaluation completed!")

if __name__ == "__main__":
    main()
