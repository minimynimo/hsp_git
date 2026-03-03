"""
Adapted LSTM Evaluation Script

This script evaluates a pre-trained LSTM model using an adapted approach where
the model is fine-tuned for a few epochs on each basin's clean, correctly-scaled
training data before making predictions.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import copy
import pickle

from papercode.lstm import LSTM
from papercode.ealstm import EALSTM
from papercode.nseloss import NSELoss

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

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

"""syn_precip_cols = ['syn_no_precip', 'syn_p_pet',
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']"""

syn_precip_cols = ['syn_no_precip', 
                   'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#åDEVICE = torch.device("cpu")  # Force CPU for debugging
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
elif GLOBAL_SETTINGS['loc'] == "US": file_tot_num = 614
elif GLOBAL_SETTINGS['loc'] == "GB": file_tot_num = 424 # changed from 424 due to change of dataset from Caravan to CamelGB
file_num_list = list(map(str, range(1, file_tot_num + 1)))

# ####################### #
# Model Definition        #
# ####################### #

class Model(nn.Module):
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int,
                 initial_forget_bias: int = 5, dropout: float = 0.0,
                 concat_static: bool = True, no_static: bool = False):
        super(Model, self).__init__()
        self.input_size_dyn, self.input_size_stat, self.hidden_size = input_size_dyn, input_size_stat, hidden_size
        self.concat_static, self.no_static = concat_static, no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn, hidden_size=hidden_size, initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat, hidden_size=hidden_size, initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None):
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
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
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def load_attributes(db_path, basins, drop_lat_lon=True):
    from papercode.datautils import load_attributes as load_attributes_from_lib
    return load_attributes_from_lib(db_path, basins, drop_lat_lon)

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
        if not cfg["no_static"]:
            static_data = static_attrs.loc[int(basin)].values
            # For adapted eval, static attrs don't need scaling as they don't vary within the basin
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
                if not cfg["no_static"]:
                    static_seq = np.tile(static_scaled, (cfg['seq_length'], 1))
                    final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                sample_date = block_df.index[i + cfg['seq_length'] - 1]
                # store (features, scaled_target, date) so evaluation can reference sample dates
                self.samples.append((final_features.astype(np.float32), block_q[target_idx].astype(np.float32), sample_date))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y, date = self.samples[idx]
        return torch.from_numpy(x), torch.tensor([y]), date

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
        if not cfg["no_static"]:
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
                if not cfg["no_static"]:
                    static_seq = np.tile(static_scaled, (cfg['seq_length'], 1))
                    final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                
                self.samples.append((final_features.astype(np.float32), sample_date))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, date = self.samples[idx]
        return torch.from_numpy(x), str(date)

# ####################### #
# Main Logic              #
# ####################### #

def custom_collate(batch):
    """Custom collate function to handle Timestamp dates."""
    xs, ys, dates = zip(*batch)
    xs_batch = torch.stack([x for x in xs])
    ys_batch = torch.stack([y for y in ys])
    # Keep dates as a list (don't try to stack/collate them)
    return xs_batch, ys_batch, list(dates)

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

    ft_loader = DataLoader(ft_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate)
    
    basin_model = copy.deepcopy(base_model).to(DEVICE).train()
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    
    is_concat_static = cfg.get('concat_static', False)
    
    for _ in range(5):  # 5 fine-tuning epochs
        for x, y, _date in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            # Model expects concatenated input when concat_static=True
            preds = basin_model(x)  # x already has static features concatenated by dataset
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

    eval_loader = DataLoader(eval_dataset, batch_size=2048, shuffle=False, collate_fn=custom_collate)
    
    predictions, observations, eval_dates = [], [], []
    with torch.no_grad():
        for i, (x, y_scaled, dates_batch) in enumerate(eval_loader):
            x = x.to(DEVICE)
            preds_scaled = basin_model(x).cpu().numpy().flatten()

            preds_rescaled = (preds_scaled * scalers['q_std']) + scalers['q_mean']
            raw_obs = y_scaled.numpy().flatten()
            raw_obs_rescaled = (raw_obs * scalers['q_std']) + scalers['q_mean']

            predictions.extend(preds_rescaled)
            observations.extend(raw_obs_rescaled)
            eval_dates.extend(list(dates_batch))

    if len(eval_dates) != len(predictions):
        eval_dates = [eval_dataset.samples[i][2] for i in range(len(eval_dataset))]

    return np.array(predictions), np.array(observations), eval_dates

def evaluate_synthetic(basin_model, basin, scenario_col, dates, cfg, static_attrs, scalers):
    """Evaluate an already fine-tuned model on synthetic scenario."""
    syn_dataset = SyntheticFineTuneDataset(basin, scenario_col, dates, cfg, static_attrs, scalers)
    if len(syn_dataset) == 0:
        return None, None
    
    syn_loader = DataLoader(syn_dataset, batch_size=2048, shuffle=False)
    
    predictions, eval_dates = [], []
    with torch.no_grad():
        for batch in syn_loader:
            if len(batch) == 2:
                xs_batch, dates_batch = batch
                xs_batch = xs_batch.to(DEVICE)
                preds = basin_model(xs_batch)  # Model returns tensor directly
                preds_rescaled = preds.cpu().numpy() * scalers['q_std'] + scalers['q_mean']
                predictions.extend(preds_rescaled.flatten())
                eval_dates.extend(dates_batch)
    
    return np.array(predictions) if predictions else None, eval_dates

def fine_tune_and_evaluate(base_model, basin, dates, cfg, static_attrs):
    """Fine-tune a model for a specific basin and then evaluate it."""
    # 1. Create dataset and loader for fine-tuning
    ft_dataset = FineTuneDataset(basin, (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), cfg, static_attrs)
    if len(ft_dataset) == 0:
        print(f"Skipping basin {basin}: Not enough clean data for fine-tuning.")
        return None, None, None

    ft_loader = DataLoader(ft_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate)
    
    # 2. Fine-tune the model (ensure it stays on GPU after deep copy)
    basin_model = copy.deepcopy(base_model).to(DEVICE).train()
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss() # Using simple MSE for fine-tuning
    
    print(f"  Fine-tuning basin {basin} for 5 epochs...")
    for _ in range(5):
        for x, y, _date in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = basin_model(x)
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()
            
    # 3. Evaluate the fine-tuned model
    basin_model.eval()
    eval_dataset = FineTuneDataset(basin, dates, cfg, static_attrs)
    if len(eval_dataset) == 0: return None, None, None # Should not happen if fine-tuning worked

    eval_loader = DataLoader(eval_dataset, batch_size=2048, shuffle=False, collate_fn=custom_collate)
    
    predictions, observations, eval_dates = [], [], []
    with torch.no_grad():
        for i, (x, y_scaled, dates_batch) in enumerate(eval_loader):
            x = x.to(DEVICE)
            preds_scaled = basin_model(x).cpu().numpy().flatten()

            # Use the scalers from the fine-tuning dataset to rescale
            scalers = ft_dataset.scalers
            preds_rescaled = (preds_scaled * scalers['q_std']) + scalers['q_mean']

            # y_scaled comes from the loader as a tensor of shape (batch, 1); convert to flat numpy
            raw_obs = y_scaled.numpy().flatten()
            raw_obs_rescaled = (raw_obs * scalers['q_std']) + scalers['q_mean']

            predictions.extend(preds_rescaled)
            observations.extend(raw_obs_rescaled)

            # Append dates for this batch (dates_batch is a list/array of timestamps)
            # Ensure they are native python datetimes or pandas Timestamps
            eval_dates.extend(list(dates_batch))

    # eval_dates have been collected per-batch above; ensure they match predictions length
    if len(eval_dates) != len(predictions):
        # Fallback: reconstruct from dataset samples if something went wrong
        eval_dates = [eval_dataset.samples[i][2] for i in range(len(eval_dataset))]

    return np.array(predictions), np.array(observations), eval_dates

def main():
    parser = argparse.ArgumentParser(description='Adapted LSTM Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=int, default=100)
    parser.add_argument('--n_basins', type=int, default=None)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)

    print("Adapted LSTM Evaluation")
    
    basins_to_process = file_num_list
    if args.n_basins:
        basins_to_process = file_num_list[:args.n_basins]
        
    model_file = run_dir / f"model_epoch{args.model_epoch}.pt"
    if not model_file.exists(): model_file = run_dir / "lstm_model_final.pt"
    print(f"Loading base model from: {model_file}")

    is_no_static = cfg.get('no_static', False)
    is_concat_static = cfg.get('concat_static', False)
    input_size_stat = 0 if is_no_static else 27
    input_size_dyn = 3 + input_size_stat if not is_no_static and is_concat_static else 3
    
    base_model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat, 
                  hidden_size=cfg['hidden_size'], dropout=cfg.get('dropout', 0.0),
                  concat_static=is_concat_static, no_static=is_no_static).to(DEVICE)
    base_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    static_attrs = None
    if not is_no_static:
        db_path = run_dir / "attributes.db"
        static_attrs = load_attributes(db_path, basins=basins_to_process, drop_lat_lon=True)

    output_dir = Path(f"/data0/funato/hsp2/0_out/LSTM/{GLOBAL_SETTINGS['loc']}/adapted")

    param_tag = f"hs{cfg['hidden_size']}_do{cfg.get('dropout', 0.0)}_e{cfg['epochs']}"

    
    # Directory for saving/loading fine-tuned models
    adapted_models_dir = Path(f"/data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/runs_lstm_adapted/{GLOBAL_SETTINGS['loc']}")
    adapted_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Store fine-tuned models and scalers for each basin (fine-tune once, use multiple times)
    basin_models = {}
    basin_scalers = {}
    
    print("\n" + "="*60)
    print("Loading or fine-tuning models for each basin...")
    print("="*60)
    for basin in tqdm(basins_to_process, desc="Processing basins"):
        # Try to load existing fine-tuned model
        basin_model, scalers = load_basin_model(basin, base_model, adapted_models_dir)
        
        if basin_model is None:
            # Fine-tune if not found
            basin_model, scalers = fine_tune_basin_model(base_model, basin, cfg, static_attrs)
            if basin_model is not None:
                # Save the fine-tuned model
                save_basin_model(basin, basin_model, scalers, adapted_models_dir)
        
        if basin_model is not None:
            basin_models[basin] = basin_model
            basin_scalers[basin] = scalers
    
    # Evaluate on cal and eva periods using the fine-tuned models
    for name, dates in {'cal': (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), 
                        'eva': (GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end'])}.items():
        print(f"\nEvaluating {name.upper()} period...")
        
        pred_path = output_dir / "predict_p" / f"LSTM_predict_{param_tag}_{name}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        first_basin = True
        
        for basin in tqdm(basins_to_process, desc=f"Processing basins for {name}"):
            if basin not in basin_models:
                continue
                
            preds, obs, eval_dates = evaluate_period(basin_models[basin], basin, dates, cfg, static_attrs, basin_scalers[basin])
            if preds is not None:
                # Create row with basin as index and dates as columns
                basin_series = pd.Series(preds, index=eval_dates, name=basin)
                basin_row = basin_series.to_frame().T
                basin_row.index.name = 'basin'
                
                # Append to CSV incrementally
                if first_basin:
                    basin_row.to_csv(pred_path, mode='w', header=True)
                    first_basin = False
                else:
                    basin_row.to_csv(pred_path, mode='a', header=False)
                
                metrics = {}
                for benchmark in benchmark_list:
                    metrics[benchmark] = BMK(obs, preds, benchmark)
                result_row = {'basin': basin}
                result_row.update(metrics)
                all_metrics.append(result_row)

        if all_metrics:
            # Sort CSV by basin ID numerically
            temp_df = pd.read_csv(pred_path, index_col='basin')
            temp_df.index = temp_df.index.astype(int)
            temp_df.to_csv(pred_path)
            print(f"Predictions saved to {pred_path}")

            results_df = pd.DataFrame(all_metrics).set_index('basin')
            results_path = output_dir / "results_p" / f"LSTM_results_{param_tag}_{name}.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path)
            print(f"Results saved to {results_path}")
            print(f"\n{name.upper()} Period Summary:\n{results_df.describe()}")
    
    # Evaluate on synthetic scenarios using the fine-tuned models
    print("\n" + "="*60)
    print("Starting Synthetic Scenario Evaluation...")
    print("="*60)
    
    val_start = GLOBAL_SETTINGS["val_start"]
    val_end = GLOBAL_SETTINGS["val_end"]
    
    for scenario_col in syn_precip_cols:
        print(f"\nProcessing scenario: {scenario_col}")
        syn_end = pd.to_datetime('2001-02-10') if 'peak' in scenario_col else val_end
        dates = (val_start, syn_end)
        
        pred_path = output_dir / "predict_p" / "synthetic" / f"LSTM_predict_{param_tag}_{scenario_col}_eva.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        
        first_basin = True
        
        for basin in tqdm(basins_to_process, desc=f"Processing {scenario_col}"):
            if basin not in basin_models:
                continue
            
            predictions, eval_dates = evaluate_synthetic(basin_models[basin], basin, scenario_col, dates, cfg, static_attrs, basin_scalers[basin])
            
            if predictions is not None and len(predictions) > 0 and len(eval_dates) == len(predictions):
                # Create row with basin as index and dates as columns
                clipped_preds = np.clip(predictions, a_min=0, a_max=None)
                basin_series = pd.Series(clipped_preds, index=eval_dates, name=basin)
                basin_row = basin_series.to_frame().T
                basin_row.index.name = 'basin'
                
                # Append to CSV incrementally
                if first_basin:
                    basin_row.to_csv(pred_path, mode='w', header=True)
                    first_basin = False
                else:
                    basin_row.to_csv(pred_path, mode='a', header=False)
        
        # Sort CSV by basin ID numerically if file exists
        if pred_path.exists():
            temp_df = pd.read_csv(pred_path, index_col='basin')
            temp_df.index = temp_df.index.astype(int)
            temp_df.to_csv(pred_path)
            print(f"Synthetic predictions saved to {pred_path}")
    
    print("\nLSTM adapted evaluation completed!")

if __name__ == "__main__":
    main()

