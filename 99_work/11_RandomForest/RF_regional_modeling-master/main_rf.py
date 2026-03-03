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

# Force matplotlib to use non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from papercode.datautils import add_basin_attributes, load_attributes

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

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
    file_num_list = list(map(str, range(1, file_tot_num + 1)))
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 424
    file_num_list = list(map(str, range(1, file_tot_num + 1)))
elif GLOBAL_SETTINGS['loc'] == "US":
    # Load small basins list (area < 1000 km²) for US
    small_basins_csv = "/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/small_basins_1000.csv"
    small_basins_df = pd.read_csv(small_basins_csv)
    file_num_list = small_basins_df['basin_id'].astype(str).tolist()

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
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--max_features', type=float, default=0.7)
    parser.add_argument('--max_samples', type=float, default=0.8)
    parser.add_argument('--min_samples_split', type=int, default=5)
    
    cfg = vars(parser.parse_args())
    cfg.update(GLOBAL_SETTINGS)
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    run_name = f'rf_run_{now.strftime("%m%d_%H%M")}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs_rf" / GLOBAL_SETTINGS['loc'] / run_name
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

    print("\nUsing global normalization scalers (matching LSTM approach)...")
    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'
    
    # Global scalers - same as LSTM's SCALER values
    # Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
    global_scalers = {
        'dynamic_means': np.array([3.0267525200819776, 13.110970171836367, 10.81853041139944]),
        'dynamic_stds': np.array([1.085855688036388, 5.04530884159448, 5.45962822094668]),
        'q_mean': 1.6033382557458762,
        'q_std': 1.6091810953328294
    }

    scalers = {'global_scalers': global_scalers}
    
    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)
    
    # Define the 18 valid basin attributes to use
    VALID_ATTR = [
        'p_mean', 'pet_mean', 'aridity', 'seasonality', 'frac_snow', 'high_prec_freq',
        'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'ele_mt_smn', 'slp_dg_sav', 'area',
        'for_pc_sse', 'snd_pc_sav', 'slt_pc_sav', 'cly_pc_sav','soc_th_sav', 'swc_pc_syr']
    
    attributes = None
    if not cfg["no_static"]:
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True, keep_features=VALID_ATTR)
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
        df = load_data_for_basin(int(basin))
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        gscaler = scalers['global_scalers']
        
        dyn_data_scaled = (df_period[dyn_cols].values - gscaler['dynamic_means']) / gscaler['dynamic_stds']
        q_data_scaled = (df_period[q_col].values - gscaler['q_mean']) / gscaler['q_std']

        static_data_scaled = None
        if not cfg["no_static"]:
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
        max_features=cfg["max_features"],
        max_samples=cfg["max_samples"],
        min_samples_split=cfg["min_samples_split"],
        random_state=cfg["seed"],
        n_jobs=-1,
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
        gscaler = scalers['global_scalers']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle(f'Random Forest Training Debug Analysis for Basin {basin_id}', fontsize=16)
        
        # Rescaled Hydrograph
        obs_rescaled = (basin_samples['obs_scaled'] * gscaler['q_std']) + gscaler['q_mean']
        pred_rescaled = (basin_samples['pred_scaled'] * gscaler['q_std']) + gscaler['q_mean']
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
