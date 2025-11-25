"""
XGBoost Regional Hydrological Modeling

This version correctly implements per-basin scaling for regional training and 
integrates end-of-run diagnostic plotting.
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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt

from papercode.datautils import add_basin_attributes, load_attributes

# Global Settings
GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

# Basin configuration
if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 424

file_num_list = list(map(str, range(1, file_tot_num + 1)))

def file_name(input_num, total_len):
    """Generate zero-padded filename"""
    return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    """Load data for a specific basin"""
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train"])
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(10000000))
    parser.add_argument('--run_dir', type=str, help="For evaluation mode.")
    parser.add_argument('--no_static', type=bool, default=True)
    
    # XGBoost specific arguments
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=7) 
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    
    parser.add_argument(
        '--peak_weight', 
        type=float, 
        default=3.0, 
        help="Weight to apply to errors on high-flow events (e.g., top 5%% of flows)."
    )
    
    cfg = vars(parser.parse_args())
    cfg.update(GLOBAL_SETTINGS)
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    run_name = f'xgb_run_{now.strftime("%m%d_%H%M")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs_xgb" / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        temp_cfg = {k: v for k, v in cfg.items() if not isinstance(v, (pd.Timestamp, Path))}
        temp_cfg['train_start'] = cfg['train_start'].strftime('%Y-%m-%d')
        temp_cfg['train_end'] = cfg['train_end'].strftime('%Y-%m-%d')
        temp_cfg['run_dir'] = str(cfg['run_dir'])
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
    return cfg

def create_weighted_mse_objective(peak_weight: float):
    """
    Creates and returns a custom objective function with a specified weight for high flows.
    """
    def weighted_mse_obj(y_pred: np.ndarray, y_true: xgb.DMatrix) -> (np.ndarray, np.ndarray):
        y_true_labels = y_true.get_label()
        high_flow_threshold = np.percentile(y_true_labels, 95)
        weights = np.where(y_true_labels >= high_flow_threshold, peak_weight, 1.0)
        grad = -2 * (y_true_labels - y_pred) * weights
        hess = 2 * weights
        return grad, hess
    return weighted_mse_obj

def train(cfg):
    """Train XGBoost model and generate debug plots."""
    print("Starting XGBoost training...")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    
    basins = file_num_list
    cfg = _setup_run(cfg)
    for key, val in cfg.items(): print(f"{key}: {val}")

    print("\nCalculating per-basin scalers from clean data blocks...")
    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'
    basin_scalers = {}

    for basin in tqdm(basins, desc="Loading basin data for scaling"):
        df = load_data_for_basin(int(basin))
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        
        # ## MODIFICATION START: Calculate scalers ONLY from clean, contiguous data ##
        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clean_blocks = []
        clumps = np.ma.clump_unmasked(np.ma.masked_array(df_period.index, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        for s in clumps:
            start_date = df_period.index[s.start]
            end_date = df_period.index[s.stop - 1]
            clean_blocks.append(df_period.loc[start_date:end_date])
        
        if not clean_blocks:
            continue # Skip basins with no clean data
            
        basin_clean_df = pd.concat(clean_blocks)
        
        basin_scalers[basin] = {
            'q_mean': basin_clean_df[q_col].mean(),
            'q_std': basin_clean_df[q_col].std(),
            'dynamic_means': basin_clean_df[dyn_cols].mean().values,
            'dynamic_stds': basin_clean_df[dyn_cols].std().values
        }
        # ## MODIFICATION END ##

    # For static attributes, we still use a global scaler
    scalers = {'basin_scalers': basin_scalers}
    
    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)
    
    attributes = None
    if not cfg["no_static"]:
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True)
        # Calculate global mean/std for static attributes only
        scalers['static_means'] = attributes.mean().values
        scalers['static_stds'] = attributes.std().values
    
    # Custom serializer for numpy types
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj
            
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(convert_for_json(scalers), fp, indent=4)
    print(f"\nScalers successfully calculated and saved to {cfg['run_dir'] / 'scalers.json'}")

    print("\nPreparing training data for XGBoost from clean contiguous blocks...")
    X_list, y_list = [], []
    sample_info = []

    for basin in tqdm(basins, desc="Processing basins for training"):
        if basin not in scalers['basin_scalers']: continue # Skip if no scalers were generated

        df = load_data_for_basin(int(basin))
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        bscaler = scalers['basin_scalers'][basin]
        
        dyn_data_scaled = (df_period[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
        q_data_scaled = (df_period[q_col].values - bscaler['q_mean']) / bscaler['q_std']

        static_data_scaled = None
        if not cfg["no_static"]:
            static_data = attributes.loc[int(basin)].values
            static_data_scaled = (static_data - scalers['static_means']) / scalers['static_stds']

        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < cfg['seq_length']:
                continue

            block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
            block_q_scaled = q_data_scaled[start_loc:end_loc]
            
            for i in range(block_len - cfg['seq_length'] + 1):
                target_idx_in_block = i + cfg['seq_length'] - 1
                dyn_seq_flat = block_dyn_scaled[i:target_idx_in_block + 1].flatten()
                
                final_features = dyn_seq_flat
                if not cfg["no_static"]:
                    final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
                
                X_list.append(final_features)
                y_list.append(block_q_scaled[target_idx_in_block])
                sample_info.append({'basin': basin, 'date': df_period.index[start_loc + target_idx_in_block]})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).flatten()
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")

    print("\nConverting data to DMatrix for custom objective training...")
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        'max_depth': cfg["max_depth"],
        'learning_rate': cfg["learning_rate"],
        'subsample': cfg["subsample"],
        'colsample_bytree': cfg["colsample_bytree"],
        'random_state': cfg["seed"],
        'verbosity': 1
    }
    
    if torch.cuda.is_available():
        print("CUDA is available. Training XGBoost on GPU.")
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    else:
        print("CUDA not available. Training XGBoost on CPU.")
        params['tree_method'] = 'hist'
        
    print(f"\nTraining XGBoost model with custom weighted loss (peak_weight={cfg['peak_weight']})...")
    weighted_mse_obj = create_weighted_mse_objective(cfg['peak_weight'])
    
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=cfg["n_estimators"],
        obj=weighted_mse_obj
    )
    
    model_path = cfg["run_dir"] / "xgb_model.json"
    booster.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    y_pred_scaled = booster.predict(dtrain)
    y_pred_rescaled = np.zeros_like(y_pred_scaled)
    y_rescaled = np.zeros_like(y)
    for i, info in enumerate(sample_info):
        basin = info['basin']
        bscaler = scalers['basin_scalers'][basin]
        y_pred_rescaled[i] = (y_pred_scaled[i] * bscaler['q_std']) + bscaler['q_mean']
        y_rescaled[i] = (y[i] * bscaler['q_std']) + bscaler['q_mean']
    train_rmse = np.sqrt(mean_squared_error(y_rescaled, y_pred_rescaled))
    print(f"\nTraining RMSE (rescaled): {train_rmse:.6f}")
    
    print("\nGenerating debug plots for a sample of basins...")
    plot_dir = cfg['run_dir'] / "debug_plots"
    plot_dir.mkdir(exist_ok=True)
    
    df_samples = pd.DataFrame(sample_info)
    df_samples['pred_scaled'] = y_pred_scaled
    df_samples['obs_scaled'] = y
    
    unique_basins_in_training = df_samples['basin'].unique().tolist()
    basins_to_plot = random.sample(unique_basins_in_training, min(5, len(unique_basins_in_training)))

    for basin_id in basins_to_plot:
        print(f"  Plotting basin {basin_id}...")
        basin_samples = df_samples[df_samples['basin'] == basin_id]
        bscaler = scalers['basin_scalers'][basin_id]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
        fig.suptitle(f'Training Debug Analysis for Basin {basin_id}', fontsize=18)

        ax1.set_title('Rescaled Hydrograph (Training Period)', fontsize=14)
        obs_rescaled = (basin_samples['obs_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        pred_rescaled = (basin_samples['pred_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        ax1.plot(basin_samples['date'], obs_rescaled, label='Observed', color='black')
        ax1.plot(basin_samples['date'], pred_rescaled, label='Predicted', color='red', linestyle='--')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.set_title('Scaled Hydrograph (Training Period)', fontsize=14)
        ax2.plot(basin_samples['date'], basin_samples['obs_scaled'], label='Observed (scaled)', color='black')
        ax2.plot(basin_samples['date'], basin_samples['pred_scaled'], label='Predicted (scaled)', color='red', linestyle='--')
        ax2.set_ylabel('Standardized Value')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3.set_title('Original Input Data (Training Period)', fontsize=14)
        df_basin_raw = load_data_for_basin(int(basin_id))
        df_basin_period = df_basin_raw.loc[cfg['train_start']:cfg['train_end']]

        ax3.plot(df_basin_period.index, df_basin_period['PET'], label='PET', color='orange')
        ax3.plot(df_basin_period.index, df_basin_period['Temp'], label='Temp (°C)', color='green')
        ax3.set_ylabel('PET & Temp')
        ax3.legend(loc='upper left')

        ax3_twin = ax3.twinx()
        ax3_twin.bar(df_basin_period.index, df_basin_period['Precip'], label='Precipitation', color='lightblue', width=1.0)
        ax3_twin.set_ylabel('Precipitation (mm)')
        ax3_twin.legend(loc='upper right')
        ax3_twin.invert_yaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_dir / f"basin_{basin_id}_training_debug.png")
        plt.close(fig)
    
    print("\nTraining finished.")

if __name__ == "__main__":
    config = get_args()
    if "mode" in config and config["mode"] == "train":
        train(config)

