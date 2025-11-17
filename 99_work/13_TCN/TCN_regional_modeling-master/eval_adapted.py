"""
Adapted TCN Evaluation Script

This script evaluates a trained TCN model using an adapted approach where the model
is fine-tuned for 5 additional epochs on each basin individually before making predictions.
This version correctly handles missing data and uses per-basin scaling.

Usage:
    python eval_adapted_tcn.py --run_dir path/to/trained/model
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
from sklearn.metrics import mean_squared_error
import re

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

benchmarks = ["NSE", "KGE", "E1", "VE", "RMSE", "MAE"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
else: file_tot_num = 396
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


def get_seed_label_from_run_cfg(run_cfg):
    """Return seed label like 'seed_300' extracted from run_cfg['run_dir'] or path string.

    If no seed found, returns 'seed_unknown'.
    """
    try:
        rd = run_cfg.get("run_dir") if isinstance(run_cfg, dict) else run_cfg
        if rd is None:
            return 'seed_unknown'
        s = str(rd)
        m = re.search(r"seed[_-]?(\d+)", s)
        if m:
            return f"seed_{m.group(1)}"
    except Exception:
        pass
    return 'seed_unknown'


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
            # For adapted eval, static attrs don't need scaling as they are constant for the basin
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

# ####################### #
# Main Logic              #
# ####################### #

def fine_tune_and_evaluate(base_model, basin, dates, cfg, static_attrs):
    """Fine-tune a model for a specific basin and then evaluate it."""
    ft_dataset = FineTuneDataset(basin, (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), cfg, static_attrs)
    if len(ft_dataset) == 0:
        print(f"Skipping basin {basin}: Not enough clean data for fine-tuning.")
        return None, None, None

    ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True)
    
    basin_model = copy.deepcopy(base_model).train()
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    
    print(f"  Fine-tuning basin {basin} for 5 epochs...")
    for _ in range(5):
        for x, y in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = basin_model(x)
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()
            
    basin_model.eval()
    
    # Create a separate dataset for the evaluation period
    eval_dataset = FineTuneDataset(basin, dates, cfg, static_attrs)
    if len(eval_dataset) == 0: return None, None, None
    eval_loader = DataLoader(eval_dataset, batch_size=1024, shuffle=False)
    
    predictions, observations, eval_dates = [], [], []
    with torch.no_grad():
        for x, y_scaled in eval_loader:
            x = x.to(DEVICE)
            preds_scaled = basin_model(x).cpu().numpy().flatten()
            
            # Use the scalers from the *fine-tuning* dataset to rescale
            scalers = ft_dataset.scalers
            preds_rescaled = (preds_scaled * scalers['q_std']) + scalers['q_mean']
            obs_rescaled = y_scaled.numpy().flatten() * scalers['q_std'] + scalers['q_mean']
            
            predictions.extend(preds_rescaled)
            observations.extend(obs_rescaled)

    eval_dates = [s[2] for s in eval_dataset.samples] if hasattr(eval_dataset.samples[0], '__len__') and len(eval_dataset.samples[0]) > 2 else range(len(eval_dataset))
    # print metric for NSE
    if observations and predictions:
        nse_value = BMK(np.array(observations), np.array(predictions), "NSE")
        print(f"    Basin {basin} NSE: {nse_value:.4f}")

    return np.array(predictions), np.array(observations), eval_dates

def main():
    parser = argparse.ArgumentParser(description='Adapted TCN Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=int, default=30)
    parser.add_argument('--n_basins', type=int, default=None)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)

    print("Adapted TCN Evaluation")
    
    basins_to_process = file_num_list
    if args.n_basins:
        basins_to_process = file_num_list[:args.n_basins]
        
    model_file = run_dir / f"tcn_model_epoch{args.model_epoch}.pt"
    if not model_file.exists(): model_file = run_dir / "tcn_model_final.pt"
    print(f"Loading base model from: {model_file}")

    is_no_static = cfg.get('no_static', False)
    input_size = 3
    if not is_no_static:
        # This assumes the number of static features is known and consistent
        # A better approach would be to save this in cfg.json during training
        # For now, we hardcode a common value for GB attributes
        input_size += 27 

    base_model = TCNModel(input_size=input_size, 
                          num_channels=cfg["num_channels"], 
                          kernel_size=cfg["kernel_size"], 
                          dropout=cfg["dropout"]).to(DEVICE)
    base_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    static_attrs = None
    if not is_no_static:
        db_path = run_dir / "attributes.db"
        static_attrs = load_attributes(db_path, basins=basins_to_process, drop_lat_lon=True)

    output_dir = Path(f"/data0/funato/hsp/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted")
    
    for name, dates in {'cal': (GLOBAL_SETTINGS['train_start'], GLOBAL_SETTINGS['train_end']), 
                        'eva': (GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end'])}.items():
        print(f"\nEvaluating {name.upper()} period...")
        
        all_preds_df, all_results = pd.DataFrame(), []
        for basin in tqdm(basins_to_process, desc=f"Processing basins for {name}"):
            preds, obs, eval_dates = fine_tune_and_evaluate(base_model, basin, dates, cfg, static_attrs)
            if preds is not None:
                preds_df = pd.DataFrame({'date': eval_dates, 'prediction': preds}).set_index('date')
                all_preds_df[basin] = preds_df['prediction']
                # Calculate benchmarks for this basin
                metrics = {}
                for benchmark in benchmarks:
                    metrics[benchmark] = BMK(obs, preds, benchmark)
                result_row = {'basin': basin}
                result_row.update(metrics)
                all_results.append(result_row)

        if all_results:
            seed_label = get_seed_label_from_run_cfg(cfg)
            pred_path = output_dir / "predict" / f"TCN_predict_{seed_label}_{name}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            all_preds_df.sort_index().to_csv(pred_path)
            print(f"Predictions saved to {pred_path}")

            results_df = pd.DataFrame(all_results).set_index('basin')
            results_path = output_dir / "results" / f"TCN_results_{name}.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path)
            print(f"Results saved to {results_path}")
            print(f"\n{name.upper()} Period Summary:\n{results_df.describe()}")

if __name__ == "__main__":
    main()
