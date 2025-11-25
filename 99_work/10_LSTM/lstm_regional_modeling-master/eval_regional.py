"""
LSTM Regional Hydrological Model Evaluation

This script evaluates a trained LSTM model using the same per-basin data
preparation pipeline as the training script to ensure consistency.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# This is necessary to load the saved model architecture
from papercode.lstm import LSTM
from papercode.ealstm import EALSTM

# ####################### #
# Global Definitions      #
# ####################### #

syn_precip_cols = ['syn_no_precip', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

GLOBAL_SETTINGS = {'loc': 'GB', 
                   'ver': 2, 
                   'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
                   'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
                   'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
                   'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')}

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # Force CPU for debugging

# ####################### #
# Model Definition        #
# ####################### #

class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connected layer"""
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int,
                 initial_forget_bias: int = 5, dropout: float = 0.0,
                 concat_static: bool = True, no_static: bool = False):
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
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
            last_h = self.dropout(h_n[:, -1, :])
            out = self.fc(last_h)
            return out
        else:
            h_n, c_n = self.lstm(x_d, x_s)
            last_h = self.dropout(h_n[:, -1, :])
            out = self.fc(last_h)
            return out

# ####################### #
# Utility Functions       #
# ####################### #

def file_name(input_num, total_len):
    return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def load_attributes(db_path: str, basins: list, drop_lat_lon: bool = True) -> pd.DataFrame:
    from papercode.datautils import load_attributes as load_attributes_from_lib
    return load_attributes_from_lib(db_path, basins, drop_lat_lon)

def BMK(obs_data, sim_data, benchmark):    
    """Calculate benchmark metrics"""
    # Ensure data are flat for calculations if they come in (N, 1) shape
    obs_data = np.asarray(obs_data).flatten()
    sim_data = np.asarray(sim_data).flatten()
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data, sim_data = obs_data[mask], sim_data[mask]

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

# ####################### #
# Data Handling           #
# ####################### #

class HydroDataset(Dataset):
    """Custom Dataset for evaluation with per-basin scaling."""
    def __init__(self, basin, dates, cfg, scalers, attributes, syn_col=None):
        super(HydroDataset, self).__init__()
        self.seq_length = cfg['seq_length']
        self.x, self.y, self.dates = [], [], []

        if basin not in scalers['basin_scalers']:
            return

        df = load_data_for_basin(int(basin))
        warmup_start = dates[0] - pd.DateOffset(days=self.seq_length - 1)
        df_period = df.loc[warmup_start:dates[1]].copy()
        
        if len(df_period) < self.seq_length:
            return

        bscaler = scalers['basin_scalers'][basin]
        dyn_cols = ['Precip', 'PET', 'Temp']
        q_col = 'Obs flow'

        if syn_col:
            dyn_df = pd.DataFrame(index=df_period.index)
            dyn_df['Precip'] = df_period[syn_col]
            dyn_df['PET'] = df_period['PET_at_peak'] if 'peak' in syn_col else df_period['PET']
            dyn_df['Temp'] = df_period['Temp_at_peak'] if 'peak' in syn_col else df_period['Temp']
            q_raw = np.full(len(dyn_df), np.nan)
        else:
            dyn_df = df_period[dyn_cols]
            q_raw = df_period[q_col].values
            
        dyn_scaled = (dyn_df.values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
        
        static_scaled = None
        if not cfg.get("no_static", True):
            static_data = attributes.loc[int(basin)].values
            static_scaled = (static_data - scalers['static_means']) / scalers['static_stds']

        is_nan = pd.DataFrame(dyn_scaled).isnull().any(axis=1).values
        if not syn_col:
            is_nan = is_nan | pd.isnull(q_raw)

        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list): clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < self.seq_length: continue

            block_dyn_scaled = dyn_scaled[start_loc:end_loc]
            block_q_raw = q_raw[start_loc:end_loc]

            for i in range(block_len - self.seq_length + 1):
                target_idx_in_block = i + self.seq_length - 1
                current_date = df_period.index[start_loc + target_idx_in_block]
                if current_date < dates[0]: continue

                dyn_seq = block_dyn_scaled[i:target_idx_in_block + 1]
                
                final_features = dyn_seq
                if not cfg.get("no_static", True) and cfg.get("concat_static", False):
                    static_repeated = np.tile(static_scaled, (self.seq_length, 1))
                    final_features = np.concatenate([dyn_seq, static_repeated], axis=1)

                self.x.append(final_features.astype(np.float32))
                self.y.append(block_q_raw[target_idx_in_block])
                self.dates.append(current_date)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ####################### #
# Main Evaluation Logic   #
# ####################### #

def evaluate_basin_incremental(model, basin, dates, cfg, scalers, attributes, syn_col=None):
    """Evaluate a single basin and return predictions."""
    model.eval()
    
    # Skip basins without scalers
    if basin not in scalers['basin_scalers']:
        return None, None, None
        
    dataset = HydroDataset(basin, dates, cfg, scalers, attributes, syn_col)
    if len(dataset) == 0:
        return None, None, None
    
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    bscaler = scalers['basin_scalers'][basin]
    
    basin_preds_list = []
    basin_obs_list = []
    
    with torch.no_grad():
        for x, y_raw in loader:
            x = x.to(DEVICE)
            preds_scaled = model(x).cpu().numpy().flatten()
            
            # Rescale using the correct per-basin scaler
            preds_rescaled = (preds_scaled * bscaler['q_std']) + bscaler['q_mean']
            preds_rescaled[preds_rescaled < 0] = 0

            basin_preds_list.extend(preds_rescaled)
            basin_obs_list.extend(y_raw.numpy())

    return np.array(basin_preds_list), np.array(basin_obs_list), dataset.dates

def evaluate(model, basins, dates, cfg, scalers, attributes, period_name, syn_col=None):
    """Main evaluation function for a given period and scenario."""
    model.eval()
    all_preds, all_obs, all_basins, all_dates = [], [], [], []

    for basin in tqdm(basins, desc=f"Processing basins for {period_name}"):
        # Skip basins without scalers
        if basin not in scalers['basin_scalers']:
            continue
            
        dataset = HydroDataset(basin, dates, cfg, scalers, attributes, syn_col)
        if len(dataset) == 0: continue
        
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        
        bscaler = scalers['basin_scalers'][basin]
        
        basin_preds_list = []
        basin_obs_list = []
        
        with torch.no_grad():
            for x, y_raw in loader:
                x = x.to(DEVICE)
                preds_scaled = model(x).cpu().numpy().flatten()
                
                # Rescale using the correct per-basin scaler
                preds_rescaled = (preds_scaled * bscaler['q_std']) + bscaler['q_mean']
                preds_rescaled[preds_rescaled < 0] = 0

                basin_preds_list.extend(preds_rescaled)
                basin_obs_list.extend(y_raw.numpy())

        all_preds.extend(basin_preds_list)
        all_obs.extend(basin_obs_list)
        all_basins.extend([basin] * len(dataset.dates))
        all_dates.extend(dataset.dates)

    if not all_preds:
        print(f"No data to evaluate for {period_name}.")
        return None, None
        
    results_df = pd.DataFrame({
        'date': all_dates, 'basin': all_basins,
        'prediction': all_preds, 'observation': all_obs
    })

    metrics = None
    if not syn_col:
        for benchmark in benchmark_list:
            metrics = results_df.groupby('basin').apply(
                lambda x: pd.Series({
                    benchmark: BMK(x['observation'], x['prediction'], benchmark)
                }), include_groups=False
            ).reset_index() if metrics is None else metrics.merge(
                results_df.groupby('basin').apply(
                    lambda x: pd.Series({
                        benchmark: BMK(x['observation'], x['prediction'], benchmark)
                    }), include_groups=False
                ).reset_index(), on='basin'
            )            

    return results_df, metrics

def main():
    parser = argparse.ArgumentParser(description='Regional LSTM Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--n_basins', type=int, default=None)
    parser.add_argument('--model_epoch', type=int, default=30, 
                        help='Which model epoch to use for evaluation (default: 30)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    print("Regional LSTM Evaluation")
    print(f"Model directory: {run_dir}")
    
    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)
    with open(run_dir / 'scalers.json', 'r') as fp:
        scalers = json.load(fp)
        if 'basin_scalers' in scalers:
            for b_id, b_scaler in scalers['basin_scalers'].items():
                for key, val in b_scaler.items():
                    if isinstance(val, list): b_scaler[key] = np.array(val)
        if 'static_means' in scalers: scalers['static_means'] = np.array(scalers['static_means'])
        if 'static_stds' in scalers: scalers['static_stds'] = np.array(scalers['static_stds'])

    model_file = run_dir / f"model_epoch{args.model_epoch}.pt"
    if not model_file.exists():
        model_file = run_dir / "lstm_model_final.pt"
        if not model_file.exists(): model_file = next(run_dir.glob('*.pt'), None) 
    if not model_file: raise ValueError(f"No .pt model file found in {run_dir}")
    
    print(f"Loading model from: {model_file}")
    
    is_no_static = cfg.get('no_static', False)
    is_concat_static = cfg.get('concat_static', False)
    
    input_size_stat = 0 if is_no_static else 27
    input_size_dyn = 3
    if not is_no_static and is_concat_static:
        input_size_dyn = 3 + input_size_stat
        
    model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat, 
                  hidden_size=cfg['hidden_size'], dropout=cfg.get('dropout', 0.0),
                  concat_static=is_concat_static, no_static=is_no_static,
                  initial_forget_bias=cfg.get('initial_forget_bias', 5)).to(DEVICE)
                  
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
    else: file_tot_num = 424
    # Use zero-padded basin IDs to match scalers.json format
    basins = [str(i).zfill(3) for i in range(1, file_tot_num + 1)]
    if args.n_basins:
        basins = basins[:args.n_basins]
        
    attributes = None
    if not is_no_static:
        db_path = str(run_dir / 'attributes.db')
        attributes = load_attributes(db_path, basins, drop_lat_lon=True)
    
    train_start = pd.to_datetime(cfg['train_start'])
    train_end = pd.to_datetime(cfg['train_end'])
    val_start = pd.to_datetime(GLOBAL_SETTINGS['val_start'])
    val_end = pd.to_datetime(GLOBAL_SETTINGS['val_end'])
    
    output_base = Path(f"/data0/funato/hsp/0_out/LSTM/{GLOBAL_SETTINGS['loc']}/regional")
    
    # Evaluate cal and eva periods with incremental saving
    for name, dates in {'cal': (train_start, train_end), 'eva': (val_start, val_end)}.items():
        print(f"\nEvaluating {name.upper()} period...")
        
        pred_path = output_base / "predict" / f"LSTM_predict_{name}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        first_basin = True
        
        for basin in tqdm(basins, desc=f"Processing basins for {name}"):
            preds, obs, eval_dates = evaluate_basin_incremental(model, basin, dates, cfg, scalers, attributes, None)
            
            if preds is not None:
                # Create row with basin as integer index and dates as columns
                basin_row_dict = {}
                for i, date in enumerate(eval_dates):
                    basin_row_dict[date.strftime('%Y-%m-%d')] = preds[i]
                
                # Use integer basin ID
                basin_int = int(basin)
                basin_row = pd.DataFrame([basin_row_dict], index=[basin_int])
                basin_row.index.name = 'basin'
                
                # Append to CSV incrementally
                if first_basin:
                    basin_row.to_csv(pred_path, mode='w', header=True)
                    first_basin = False
                else:
                    basin_row.to_csv(pred_path, mode='a', header=False)
                
                # Calculate metrics
                metrics_row = {'basin': basin_int}
                for benchmark in benchmark_list:
                    metrics_row[benchmark] = BMK(obs, preds, benchmark)
                all_metrics.append(metrics_row)
        
        if all_metrics:
            # Sort CSV by basin ID numerically
            temp_df = pd.read_csv(pred_path, index_col='basin')
            temp_df.index = temp_df.index.astype(int)
            temp_df.to_csv(pred_path)
            print(f"Predictions saved to {pred_path}")
            
            # Save metrics
            metrics_df = pd.DataFrame(all_metrics)
            results_path = output_base / "results" / f"LSTM_results_{name}.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(results_path, index=False)
            print(f"Metrics saved to {results_path}")
            print(f"\n--- {name.upper()} Period Summary ---\n{metrics_df.describe()}\n")
    


    print("\n" + "="*40 + "\nStarting Synthetic Scenario Evaluation...\n" + "="*40)
    for syn_col in syn_precip_cols:
        syn_dates = (val_start, pd.to_datetime('2001-02-10') if 'peak' in syn_col else val_end)
        
        pred_path = output_base / "predict" / "synthetic" / f"LSTM_predict_{syn_col}_eva.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        
        first_basin = True
        
        for basin in tqdm(basins, desc=f"Processing basins for {syn_col}"):
            preds, obs, eval_dates = evaluate_basin_incremental(model, basin, syn_dates, cfg, scalers, attributes, syn_col)
            
            if preds is not None:
                # Create row with basin as integer index and dates as columns
                basin_row_dict = {}
                for i, date in enumerate(eval_dates):
                    basin_row_dict[date.strftime('%Y-%m-%d')] = preds[i]
                
                # Use integer basin ID
                basin_int = int(basin)
                basin_row = pd.DataFrame([basin_row_dict], index=[basin_int])
                basin_row.index.name = 'basin'
                
                # Append to CSV incrementally
                if first_basin:
                    basin_row.to_csv(pred_path, mode='w', header=True)
                    first_basin = False
                else:
                    basin_row.to_csv(pred_path, mode='a', header=False)
        
        # Sort CSV by basin ID numerically
        if pred_path.exists():
            temp_df = pd.read_csv(pred_path, index_col='basin')
            temp_df.index = temp_df.index.astype(int)
            temp_df.to_csv(pred_path)
            print(f"Synthetic predictions for '{syn_col}' saved to {pred_path}")

if __name__ == "__main__":
    main()

