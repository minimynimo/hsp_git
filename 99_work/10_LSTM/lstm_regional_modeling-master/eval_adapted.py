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

from papercode.lstm import LSTM
from papercode.ealstm import EALSTM
from papercode.nseloss import NSELoss

# ####################### #
# Global Definitions      #
# ####################### #

GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
else: file_tot_num = 396
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

def file_name(input_num, total_len): return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv", index_col='Date', parse_dates=True)
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
    parser.add_argument('--model_epoch', type=int, default=30)
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

    output_dir = Path(f"/data0/funato/hsp/0_out/LSTM/{GLOBAL_SETTINGS['loc']}/adapted")
    
    for name, dates in {'cal': (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), 
                        'eva': (GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end'])}.items():
        print(f"\nEvaluating {name.upper()} period...")
        
        all_preds_df, all_metrics = pd.DataFrame(), []
        for basin in tqdm(basins_to_process, desc=f"Processing basins for {name}"):
            preds, obs, eval_dates = fine_tune_and_evaluate(base_model, basin, dates, cfg, static_attrs)
            if preds is not None:
                preds_df = pd.DataFrame({'date': eval_dates, 'prediction': preds}).set_index('date')
                all_preds_df[basin] = preds_df['prediction']
                metrics = {}
                for benchmark in benchmark_list:
                    metrics[benchmark] = BMK(obs, preds, benchmark)
                result_row = {'basin': basin}
                result_row.update(metrics)
                all_metrics.append(result_row)

        if all_metrics:
            pred_path = output_dir / "predict" / f"LSTM_predict_{name}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            all_preds_df.sort_index().to_csv(pred_path)
            print(f"Predictions saved to {pred_path}")

            results_df = pd.DataFrame(all_metrics).set_index('basin')
            results_path = output_dir / "results" / f"LSTM_results_{name}.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path)
            print(f"Results saved to {results_path}")
            print(f"\n{name.upper()} Period Summary:\n{results_df.describe()}")

if __name__ == "__main__":
    main()
