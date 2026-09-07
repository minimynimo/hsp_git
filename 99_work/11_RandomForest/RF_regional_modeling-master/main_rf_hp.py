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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from papercode.datautils import add_basin_attributes, load_attributes, VALID_ATTR
from papercode.utils import get_basin_list
from papercode.rf_features import load_or_build_dataset
from papercode.rf_run_registry import find_completed_run

DATASET_CACHE_ROOT = Path(__file__).parent / "cache" / "hp"

# ####################### #
# Global Definitions      #
# ####################### #


# HP-search cal/val split -- must NOT touch the real 2001-01-01..2006-12-31 test period.
GLOBAL_SETTINGS = {
    'loc': 'US',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1990-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('1998-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('1999-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d')
}

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

gauge_id_list = get_basin_list(GLOBAL_SETTINGS['loc'])

# ####################### #
# Utility Functions       #
# ####################### #

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
    parser.add_argument('--n_jobs', type=int, default=1, help='Threads used by RandomForestRegressor (tree-level parallelism)')
    
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

def _hp_str(cfg: Dict) -> str:
    max_feat_str = cfg['max_features'] if isinstance(cfg['max_features'], str) else f"{cfg['max_features']:.2f}"
    return f"n{cfg['n_estimators']}_d{cfg['max_depth'] if cfg['max_depth'] is not None else 'None'}_l{cfg['min_samples_leaf']}_f{max_feat_str}"

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    hp_str = _hp_str(cfg)
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
    
    basins = gauge_id_list

    # run_rf_hp_tuning.sh launches one process per (n_estimators, max_depth,
    # min_samples_leaf, max_features) combination; a restart after a kill would
    # otherwise retrain every combination from scratch since _setup_run's leaf
    # directory is timestamped. Skip if a completed run with this exact HP
    # combination already exists under runs_rf/hp/{hp_str}/. 'seed' is deliberately
    # NOT a match key -- get_args() defaults --seed to a random draw when omitted
    # (same convention as the upstream reference implementation, which logs that
    # random seed in the run folder name purely for traceability), so requiring it
    # to match would defeat resumability for any invocation that doesn't pin --seed.
    match_keys = ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'no_static']
    search_root = Path(__file__).parent / "runs_rf" / "hp" / _hp_str(cfg)
    existing_run = find_completed_run(search_root, cfg, match_keys)
    if existing_run is not None:
        print(f"Skipping: a completed run with these exact hyperparameters already exists at {existing_run}")
        return

    cfg = _setup_run(cfg)
    for key, val in cfg.items(): print(f"{key}: {val}")

    db_path = str(cfg['run_dir'] / 'attributes.db')
    add_basin_attributes(db_path=db_path, loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)

    attributes = None
    if not cfg["no_static"]:
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True,
                                      keep_features=VALID_ATTR)

    # X/y/scalers depend only on (loc, ver, seq_length, train period, no_static, basins) --
    # not on any RF hyperparameter -- so they are cached once under DATASET_CACHE_ROOT and
    # reused by every hyperparameter job launched by run_rf_hp_tuning.sh instead of being
    # rebuilt (via a slow per-sample Python loop) in each of the parallel processes.
    X, y, sample_info_df, basin_scalers, static_means, static_stds = load_or_build_dataset(
        cfg, basins, DATASET_CACHE_ROOT, attributes)
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")

    scalers = {'basin_scalers': basin_scalers}
    if static_means is not None:
        scalers['static_means'] = static_means
        scalers['static_stds'] = static_stds

    def convert_for_json(obj):
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.generic): return obj.item()
        else: return obj

    with (cfg['run_dir'] / 'scalers.json').open('w') as fp:
        json.dump(convert_for_json(scalers), fp, indent=4)
    print(f"\nScalers successfully saved to {cfg['run_dir'] / 'scalers.json'}")

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
        n_jobs=cfg["n_jobs"],
        verbose=1
    )
    
    model.fit(X, y)
    
    model_path = cfg["run_dir"] / "rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    y_pred_scaled = model.predict(X)
    
    basin_q_mean = sample_info_df['basin'].map(lambda b: scalers['basin_scalers'][b]['q_mean']).values
    basin_q_std = sample_info_df['basin'].map(lambda b: scalers['basin_scalers'][b]['q_std']).values
    y_pred_rescaled = (y_pred_scaled * basin_q_std) + basin_q_mean
    y_rescaled = (y * basin_q_std) + basin_q_mean

    train_rmse = np.sqrt(mean_squared_error(y_rescaled, y_pred_rescaled))
    print(f"\nTraining RMSE (rescaled): {train_rmse:.6f}")

    print("\nGenerating debug plots for a sample of basins...")
    plot_dir = cfg['run_dir'] / "debug_plots"
    plot_dir.mkdir(exist_ok=True)
    
    df_samples = sample_info_df.copy()
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
