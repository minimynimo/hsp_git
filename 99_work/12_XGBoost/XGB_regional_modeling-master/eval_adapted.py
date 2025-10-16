"""
Adapted XGBoost Evaluation Script

This script evaluates a trained XGBoost model using an adapted approach where a new,
basin-specific model is trained for each basin before making predictions.
This version correctly handles missing data by operating only on contiguous
blocks of clean data.

Usage:
    python eval_adapted.py --run_dir path/to/regional/model/run_dir
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

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

benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
else:
    file_tot_num = 396

file_num_list = list(map(str, range(1, file_tot_num + 1)))

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
# Core Logic              #
# ####################### #

def train_basin_model(basin: str, run_cfg: Dict, static_attrs: pd.DataFrame = None):
    """Train an XGBoost model for a single basin using its clean training data."""
    df = load_data_for_basin(int(basin))
    train_start, train_end = GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]
    df_period = df.loc[train_start:train_end]

    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'

    # ## CRITICAL FIX: Find clean blocks and calculate scalers ONLY from them ##
    is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
    clean_blocks = []
    clumps = np.ma.clump_unmasked(np.ma.masked_array(df_period.index, is_nan))
    if not isinstance(clumps, list):
        clumps = [clumps] if clumps else []
    for s in clumps:
        clean_blocks.append(df_period.iloc[s])
    
    if not clean_blocks:
        return None, None
        
    basin_clean_df = pd.concat(clean_blocks)
    
    basin_stats = {
        'dynamic_means': basin_clean_df[dyn_cols].mean().values,
        'dynamic_stds': basin_clean_df[dyn_cols].std().values,
        'q_mean': basin_clean_df[q_col].mean(),
        'q_std': basin_clean_df[q_col].std()
    }

    # Prepare training data from the clean blocks
    X_list, y_list = [], []
    seq_length = run_cfg['seq_length']
    
    dyn_data_scaled = (df_period[dyn_cols].values - basin_stats['dynamic_means']) / basin_stats['dynamic_stds']
    
    static_data_scaled = None
    if static_attrs is not None and not static_attrs.empty:
        # Note: Static attributes are not scaled here as they are constant for the basin
        static_data_scaled = static_attrs.loc[int(basin)].values

    for block_df in clean_blocks:
        if len(block_df) < seq_length:
            continue
            
        start_idx_loc = df_period.index.get_loc(block_df.index[0])
        end_idx_loc = df_period.index.get_loc(block_df.index[-1])
        
        block_dyn_scaled = dyn_data_scaled[start_idx_loc:end_idx_loc+1]
        
        for i in range(len(block_df) - seq_length + 1):
            target_idx_in_block = i + seq_length - 1
            dyn_seq_flat = block_dyn_scaled[i:target_idx_in_block + 1].flatten()
            
            final_features = dyn_seq_flat
            if static_data_scaled is not None:
                final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
            
            X_list.append(final_features)
    
    y_raw = basin_clean_df[q_col].values
    
    if len(X_list) == 0:
        return None, None

    # Targets are the raw values from the clean dataframe
    X_train = np.array(X_list, dtype=np.float32)
    y_train = y_raw[seq_length-1:] # Simple slicing works because we built y from a clean df

    print(f"  Training model for basin {basin} with {len(X_train)} samples")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': run_cfg.get("max_depth", 5),
        'learning_rate': run_cfg.get("learning_rate", 0.05),
        'subsample': run_cfg.get("subsample", 0.8),
        'colsample_bytree': run_cfg.get("colsample_bytree", 0.8),
        'random_state': run_cfg.get("seed", 42),
        'n_jobs': -1, 'verbosity': 0
    }
    
    n_estimators = run_cfg.get("n_estimators", 500)
    basin_model = xgb.XGBRegressor(**params)
    basin_model.fit(X_train, y_train)
    
    return basin_model, basin_stats

def evaluate_basin_adapted_period(basin_model, basin_stats, basin, start_date, end_date, run_cfg, static_attrs=None):
    """Evaluate a basin-specific model over a given period."""
    df = load_data_for_basin(int(basin))
    warmup_start = start_date - pd.DateOffset(days=run_cfg["seq_length"] - 1)
    df_period = df.loc[warmup_start:end_date]

    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'
    
    # Use stats from training to scale evaluation data
    dyn_data_scaled = (df_period[dyn_cols].values - basin_stats['dynamic_means']) / basin_stats['dynamic_stds']
    
    static_data_scaled = None
    if static_attrs is not None and not static_attrs.empty:
        static_data_scaled = static_attrs.loc[int(basin)].values
        
    # ## CRITICAL FIX: Find clean blocks in the evaluation period as well ##
    X_list, y_list, date_list = [], [], []
    seq_length = run_cfg['seq_length']
    
    is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
    clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
    if not isinstance(clumps, list):
        clumps = [clumps] if clumps else []

    for s in clumps:
        start_loc, end_loc = s.start, s.stop
        block_len = end_loc - start_loc
        if block_len < seq_length: continue

        block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
        block_q_raw = df_period[q_col].values[start_loc:end_loc]

        for i in range(block_len - seq_length + 1):
            target_idx_in_block = i + seq_length - 1
            current_date = df_period.index[start_loc + target_idx_in_block]
            if current_date < start_date: continue

            dyn_seq_flat = block_dyn_scaled[i:target_idx_in_block + 1].flatten()
            
            final_features = dyn_seq_flat
            if static_data_scaled is not None:
                final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
            
            X_list.append(final_features)
            y_list.append(block_q_raw[target_idx_in_block])
            date_list.append(current_date)

    if not X_list:
        return None, None, None

    X_eval = np.array(X_list, dtype=np.float32)
    y_eval_raw = np.array(y_list)

    predictions = basin_model.predict(X_eval)
    predictions[predictions < 0] = 0
    
    return predictions, y_eval_raw, date_list

def main():
    parser = argparse.ArgumentParser(description='Adapted XGBoost Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--n_basins', type=int, default=None)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist")
    
    with open(run_dir / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)
    
    print("Adapted XGBoost Evaluation")
    
    basins_to_process = file_num_list
    if args.n_basins is not None:
        basins_to_process = file_num_list[:args.n_basins]
        print(f"Processing only first {args.n_basins} basins for testing")
    
    static_attrs = None
    if not run_cfg.get("no_static", True):
        try:
            db_path = run_dir / "attributes.db"
            static_attrs = load_attributes(db_path, basins=basins_to_process, drop_lat_lon=True)
        except Exception as e:
            print(f"Warning: Could not load static attributes: {e}")
    
    periods = {
        'cal': (GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]),
        'eva': (GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"])
    }
    output_dir = Path("/data0/funato/hsp/0_out/XGB/GB/adapted")

    for period_name, (start_date, end_date) in periods.items():
        print(f"\nEvaluating {period_name.upper()} period ({start_date.date()} to {end_date.date()})")
        
        all_predictions_df = pd.DataFrame()
        all_metrics = []
        
        for basin in tqdm(basins_to_process, desc=f"Processing basins for {period_name}"):
            basin_model, basin_stats = train_basin_model(basin, run_cfg, static_attrs)
            if basin_model is None:
                print(f"Skipping basin {basin} due to insufficient training data.")
                continue

            predictions, targets, dates = evaluate_basin_adapted_period(
                basin_model, basin_stats, basin, start_date, end_date, run_cfg, static_attrs)
            
            if predictions is not None:
                basin_preds_df = pd.DataFrame({'date': dates, 'prediction': predictions})
                basin_preds_df.set_index('date', inplace=True)
                all_predictions_df[basin] = basin_preds_df['prediction']

                metrics = {}
                for benchmark in benchmark_list:
                    metrics[benchmark] = BMK(targets, predictions, benchmark)
                metrics['basin'] = basin
                
                all_metrics.append(metrics)

        if not all_metrics:
            print(f"No valid predictions for {period_name} period")
            continue
            
        # Save Predictions
        pred_output_path = output_dir / "predict" / f"XGB_predict_{period_name}.csv"
        pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        all_predictions_df.sort_index(inplace=True)
        all_predictions_df.to_csv(pred_output_path, index_label='date')
        print(f"Predictions saved to {pred_output_path}")

        # Save Results
        results_df = pd.DataFrame(all_metrics).set_index('basin')
        results_output_path = output_dir / "results" / f"XGB_results_{period_name}.csv"
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_output_path, index_label='basin')
        print(f"Results saved to {results_output_path}")

        print(f"\n{period_name.upper()} Period Summary:")
        print(results_df.describe())

if __name__ == "__main__":
    main()
