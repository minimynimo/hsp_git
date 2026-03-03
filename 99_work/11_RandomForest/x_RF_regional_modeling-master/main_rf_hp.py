"""
Random Forest Regional Hydrological Modeling

This version is corrected to use per-basin scaling for regional training,
properly handles missing data, and integrates end-of-run diagnostic plotting.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from papercode.datautils import add_basin_attributes, load_attributes

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

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 424

file_num_list = list(map(str, range(1, file_tot_num + 1)))

# ####################### #
# Utility Functions       #
# ####################### #

def file_name(input_num, total_len):
    return str(input_num).zfill(total_len)

def load_data_for_basin(file_num):
    varssim_dir = "/data0/funato/hsp2/2_MERV/varssim"
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
# Main Execution          #
# ####################### #

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train"])
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(10000000))
    parser.add_argument('--run_dir', type=str, help="For evaluation mode.")
    parser.add_argument('--no_static', type=str2bool, default=False)
    
    # Random Forest specific arguments
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth (None for unlimited)')
    parser.add_argument('--min_samples_leaf', type=int, default=2, help='Minimum samples at leaf node')
    parser.add_argument('--max_features', type=str, default='sqrt', help='Max features: sqrt, 1/3, or float')
    
    cfg = vars(parser.parse_args())
    
    # Handle max_features conversion
    if cfg['max_features'] == 'sqrt':
        cfg['max_features'] = 'sqrt'
    elif cfg['max_features'] == '1/3':
        cfg['max_features'] = 1.0/3.0
    else:
        try:
            cfg['max_features'] = float(cfg['max_features'])
        except:
            cfg['max_features'] = 'sqrt'
    
    cfg.update(GLOBAL_SETTINGS)
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    max_feat_str = cfg['max_features'] if isinstance(cfg['max_features'], str) else f"{cfg['max_features']:.2f}"
    hp_str = f"n{cfg['n_estimators']}_d{cfg['max_depth'] if cfg['max_depth'] is not None else 'None'}_l{cfg['min_samples_leaf']}_f{max_feat_str}"
    run_name = f'{hp_str}/rf_run_{now.strftime("%m%d_%H%M")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs_rf" / "hp" / run_name
    cfg['run_dir'].mkdir(parents=True, exist_ok=True)
    
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        temp_cfg = {k: v for k, v in cfg.items() if not isinstance(v, (pd.Timestamp, Path))}
        temp_cfg['train_start'] = cfg['train_start'].strftime('%Y-%m-%d')
        temp_cfg['train_end'] = cfg['train_end'].strftime('%Y-%m-%d')
        temp_cfg['run_dir'] = str(cfg['run_dir'])
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
    return cfg

def train(cfg):
    """Train Random Forest model"""
    print("Starting Random Forest training...")
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
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)
    
    attributes = None
    if not cfg["no_static"]:
        # Convert string basins to integers for attributes DataFrame
        attributes = load_attributes(db_path=db_path, basins=[int(b) for b in basins], drop_lat_lon=True)
        scalers['static_means'] = attributes.mean().values
        scalers['static_stds'] = attributes.std().values

    def convert_for_json(obj):
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.generic): return obj.item()
        else: return obj
            
    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(convert_for_json(scalers), fp, indent=4)
    print(f"\nScalers successfully calculated and saved to {cfg['run_dir'] / 'scalers.json'}")

    print("\nPreparing training data for Random Forest from clean contiguous blocks...")
    X_list, y_list, sample_info = [], [], []

    for basin in tqdm(basins, desc="Processing basins for training"):
        if basin not in scalers['basin_scalers']: continue

        df = load_data_for_basin(int(basin))
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        bscaler = scalers['basin_scalers'][basin]
        
        dyn_data_scaled = (df_period[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
        q_data_scaled = (df_period[q_col].values - bscaler['q_mean']) / bscaler['q_std']

        static_data_scaled = None
        if not cfg["no_static"]:
            # basin is string but attributes has integer index, convert for lookup
            static_data = attributes.loc[int(basin)].values
            static_data_scaled = (static_data - scalers['static_means']) / scalers['static_stds']

        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list): clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < cfg['seq_length']: continue

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

    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        criterion='squared_error',
        bootstrap=True,
        min_samples_split=2,
        random_state=cfg["seed"],
        n_jobs=1,  # Use single core as specified
        verbose=1
    )
    
    model.fit(X, y)
    
    model_path = cfg["run_dir"] / "rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    y_pred_scaled = model.predict(X)
    
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
    
    unique_basins = df_samples['basin'].unique().tolist()
    basins_to_plot = random.sample(unique_basins, min(5, len(unique_basins)))

    for basin_id in basins_to_plot:
        print(f"  Plotting basin {basin_id}...")
        
        basin_samples = df_samples[df_samples['basin'] == basin_id]
        bscaler = scalers['basin_scalers'][basin_id]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle(f'Random Forest Training Debug Analysis for Basin {basin_id}', fontsize=16)
        
        # Rescaled Hydrograph
        obs_rescaled = (basin_samples['obs_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        pred_rescaled = (basin_samples['pred_scaled'] * bscaler['q_std']) + bscaler['q_mean']
        ax1.plot(basin_samples['date'], obs_rescaled, label='Observed', color='black')
        ax1.plot(basin_samples['date'], pred_rescaled, label='Predicted', color='green', linestyle='--')
        ax1.set_title('Rescaled Hydrograph (Training Period)')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Scaled Hydrograph
        ax2.plot(basin_samples['date'], basin_samples['obs_scaled'], label='Observed (scaled)', color='black')
        ax2.plot(basin_samples['date'], basin_samples['pred_scaled'], label='Predicted (scaled)', color='green', linestyle='--')
        ax2.set_title('Scaled Hydrograph (What the model sees)')
        ax2.set_ylabel('Standardized Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_dir / f"basin_{basin_id}_training_debug.png")
        plt.close(fig)
        
    print("\nUse a separate evaluation script for full evaluation.")

if __name__ == "__main__":
    config = get_args()
    if config["mode"] == "train":
        train(config)
