"""
Adapted Random Forest Evaluation Script

This script evaluates a trained Random Forest model using an adapted approach where the model
is fine-tuned on each basin individually before making predictions.
For Random Forest, this means training a new model on the training data for each basin.

Usage:
    python eval_adapted.py --run_dir path/to/trained/model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from papercode.datasets import CamelsH5
from papercode.datautils import add_basin_attributes, load_attributes, rescale_features
from papercode.utils import create_h5_files, get_basin_list

# Global Settings (matching main.py)
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
    attribute_size = 196
    output_mean = np.array([3.8526752089320553])
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 396
    attribute_size = 209
    output_mean = np.array([1.838397528976445])

file_num_list = list(map(str, range(1, file_tot_num + 1)))

def file_name(input_num, total_len):
    """Generate zero-padded filename"""
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data_for_basin(file_num):
    """Load data for a specific basin"""
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def normalize_basin_data(data, basin_stats):
    """Normalize data using basin-specific statistics"""
    normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        mean = basin_stats['mean'][i]
        std = basin_stats['std'][i]
        if std > 0:
            normalized[:, i] = (data[:, i] - mean) / std
        else:
            normalized[:, i] = data[:, i] - mean
    return normalized

def denormalize_predictions(predictions, basin_stats, feature_idx=3):
    """Denormalize predictions using basin-specific output statistics"""
    mean = basin_stats['mean'][feature_idx]
    std = basin_stats['std'][feature_idx]
    if std > 0:
        return predictions * std + mean
    else:
        return predictions + mean

def create_sequences_rf(data, seq_length, static_features=None):
    """Create sequences for Random Forest (flatten temporal sequences)"""
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data)):
        # Flatten the sequence (only dynamic features: P, PET, Temp)
        seq = data[i-seq_length:i, :3].flatten()  # Only use first 3 columns (P, PET, Temp)
        
        # Concatenate static features if provided
        if static_features is not None:
            seq = np.concatenate([seq, static_features])
        
        sequences.append(seq)
        targets.append(data[i, 3])  # Assuming discharge is at index 3
        
    return np.array(sequences), np.array(targets)

def train_basin_model(basin, run_cfg, static_attrs=None):
    """Train a Random Forest model specifically for this basin using training data"""
    # Load data for this basin
    df = load_data_for_basin(int(basin))
    
    # Filter by training period with warmup
    train_start = GLOBAL_SETTINGS["train_start"]
    train_end = GLOBAL_SETTINGS["train_end"]
    warmup_start = train_start - pd.DateOffset(days=run_cfg["seq_length"] - 1)
    train_data = df[warmup_start:train_end].copy()
    
    if len(train_data) < run_cfg["seq_length"]:
        return None, None
    
    # Prepare input data (P, PET, Temp, Obs flow for normalization)
    input_cols = ['Precip', 'PET', 'Temp', 'Obs flow']
    input_data = train_data[input_cols].values
    
    # Calculate basin-specific normalization statistics
    basin_stats = {
        'mean': np.nanmean(input_data, axis=0),
        'std': np.nanstd(input_data, axis=0)
    }
    
    # Normalize the data per basin
    input_data_norm = normalize_basin_data(input_data, basin_stats)
    
    # Create sequences (excluding the target variable for input)
    static_features = static_attrs[basin] if static_attrs and basin in static_attrs else None
    sequences, targets = create_sequences_rf(input_data_norm, run_cfg["seq_length"], static_features)
    
    # Only include sequences within the actual training period (not warmup)
    dates = train_data.index[run_cfg["seq_length"]:]
    actual_period_mask = dates >= train_start
    sequences = sequences[actual_period_mask]
    targets = targets[actual_period_mask]
    
    if len(sequences) == 0:
        return None, None
    
    # Denormalize targets for training
    targets_rescaled = denormalize_predictions(targets, basin_stats)
    
    # Train Random Forest model for this basin
    print(f"  Training model for basin {basin} with {len(sequences)} samples")
    basin_model = RandomForestRegressor(
        n_estimators=run_cfg.get("n_estimators", 100),
        max_depth=run_cfg.get("max_depth", 10),
        min_samples_split=run_cfg.get("min_samples_split", 2),
        min_samples_leaf=run_cfg.get("min_samples_leaf", 1),
        random_state=run_cfg.get("seed", 42),
        n_jobs=-1
    )
    
    basin_model.fit(sequences, targets_rescaled)
    
    return basin_model, basin_stats

def evaluate_basin_adapted_period(basin_model, basin, basin_stats, start_date, end_date, run_cfg, static_attrs=None):
    """Evaluate the basin-specific model on a basin for a specific period"""
    # Load data for this basin
    df = load_data_for_basin(int(basin))
    
    # Filter by date range with warmup
    warmup_start = start_date - pd.DateOffset(days=run_cfg["seq_length"] - 1)
    period_data = df[warmup_start:end_date].copy()
    
    if len(period_data) < run_cfg["seq_length"]:
        return None, None, None
    
    # Prepare input data (P, PET, Temp, Obs flow)
    input_cols = ['Precip', 'PET', 'Temp', 'Obs flow']
    input_data = period_data[input_cols].values
    
    # Normalize the data using the basin-specific stats from training
    input_data_norm = normalize_basin_data(input_data, basin_stats)
    
    # Create sequences
    static_features = static_attrs[basin] if static_attrs and basin in static_attrs else None
    sequences, targets = create_sequences_rf(input_data_norm, run_cfg["seq_length"], static_features)
    
    # Get dates for predictions (after warmup period)
    dates = period_data.index[run_cfg["seq_length"]:]
    
    # Only include sequences within the actual period (not warmup)
    actual_period_mask = dates >= start_date
    sequences = sequences[actual_period_mask]
    targets = targets[actual_period_mask]
    dates = dates[actual_period_mask]
    
    if len(sequences) == 0:
        return None, None, None
    
    # Make predictions
    predictions = basin_model.predict(sequences)
    predictions[predictions < 0] = 0
    
    # Get actual targets (denormalized)
    targets_rescaled = denormalize_predictions(targets, basin_stats)
    
    return predictions, targets_rescaled, dates

def get_param_string(run_cfg):
    """Create parameter string for file naming"""
    n_estimators = run_cfg.get("n_estimators", 100)
    max_depth = run_cfg.get("max_depth", 10)
    min_samples_split = run_cfg.get("min_samples_split", 2)
    min_samples_leaf = run_cfg.get("min_samples_leaf", 1)
    max_features = run_cfg.get("max_features", 1.0)
    max_samples = run_cfg.get("max_samples", 1.0)
    seq_length = run_cfg.get("seq_length", 270)
    
    return f"n{n_estimators}_d{max_depth}_s{min_samples_split}_l{min_samples_leaf}_mf{max_features}_ms{max_samples}_w{seq_length}"

def evaluate_adapted(run_cfg, basins, static_attrs=None):
    """Evaluate model using adapted approach (basin-specific training)"""
    print("\n--- Adapted Evaluation (Basin-specific training) ---")
    
    # Define periods
    periods = {
        'cal': (GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]),
        'eva': (GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"])
    }
    
    output_dir = Path("/data0/funato/hsp/0_out/RF/GB/adapted")
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"\nEvaluating {period_name.upper()} period ({start_date.date()} to {end_date.date()})")
        
        # Collect all predictions and targets by basin
        basin_predictions = {}
        basin_dates = {}
        all_results = []
        
        for basin in tqdm(basins, desc=f"Processing basins for {period_name}"):
            try:
                # Train basin-specific model
                basin_model, basin_stats = train_basin_model(basin, run_cfg, static_attrs)
                
                if basin_model is None:
                    print(f"Could not train model for basin {basin}")
                    continue
                
                # Evaluate on the current period
                predictions, targets, dates = evaluate_basin_adapted_period(
                    basin_model, basin, basin_stats, start_date, end_date, run_cfg, static_attrs)
                
                if predictions is not None:
                    basin_predictions[basin] = predictions
                    basin_dates[basin] = dates
                    
                    # Calculate metrics for this basin
                    valid_idx = targets >= 0
                    if np.sum(valid_idx) > 0:
                        obs_valid = targets[valid_idx]
                        pred_valid = predictions[valid_idx]
                        
                        # Calculate benchmarks
                        metrics = {}
                        for benchmark in benchmark_list:
                            metrics[benchmark] = BMK(obs_valid, pred_valid, benchmark)
                        
                        result_row = {'basin': basin}
                        result_row.update(metrics)
                        all_results.append(result_row)
                        
                        print(f"Basin {basin}: NSE = {metrics['NSE']:.4f}")
                
            except Exception as e:
                print(f"Error processing basin {basin}: {e}")
                continue
        
        if not basin_predictions:
            print(f"No valid predictions for {period_name} period")
            continue
        
        # Create prediction matrix (basins x dates)
        all_dates = []
        for dates in basin_dates.values():
            all_dates.extend(dates)
        unique_dates = sorted(list(set(all_dates)))
        
        # Create prediction DataFrame
        pred_matrix = []
        basin_order = []
        
        for basin in sorted(basin_predictions.keys(), key=int):
            basin_order.append(basin)
            basin_preds = basin_predictions[basin]
            basin_dates_list = basin_dates[basin]
            
            # Create row for this basin
            row = {}
            for i, date in enumerate(basin_dates_list):
                row[date.strftime('%Y-%m-%d')] = basin_preds[i]
            
            pred_matrix.append(row)
        
        # Convert to DataFrame with basins as index and dates as columns
        pred_df = pd.DataFrame(pred_matrix, index=basin_order)
        pred_df = pred_df.reindex(columns=sorted(pred_df.columns))
        
        # Save prediction file
        param_string = get_param_string(run_cfg)
        pred_output_path = output_dir / "predict" / f"RF_predict_{param_string}_{period_name}.csv"
        pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_output_path, index_label='basin')
        print(f"Predictions saved to {pred_output_path}")
        
        # Create results DataFrame (basins x benchmarks)
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.set_index('basin')
            
            # Save results file
            results_output_path = output_dir / "results" / f"RF_results_{param_string}_{period_name}.csv"
            results_output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_output_path, index_label='basin')
            print(f"Results saved to {results_output_path}")
            
            # Print summary
            print(f"{period_name.upper()} Period Summary:")
            print(f"Mean NSE: {results_df['NSE'].mean():.4f}")
            print(f"Median NSE: {results_df['NSE'].median():.4f}")
            print(f"NSE > 0.5: {(results_df['NSE'] > 0.5).sum()}/{len(results_df)} basins")

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

def main():
    parser = argparse.ArgumentParser(description='Adapted Random Forest Evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--n_basins', type=int, default=None,
                        help='Number of basins to process (for testing)')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist")
    
    # Load run configuration
    with open(run_dir / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)
    
    run_cfg["run_dir"] = run_dir  # Add run_dir to config
    
    print("Adapted Random Forest Evaluation")
    print(f"Model directory: {run_dir}")
    
    # Limit number of basins for testing if specified
    global file_num_list
    if args.n_basins is not None:
        file_num_list = file_num_list[:args.n_basins]
        print(f"Processing only first {args.n_basins} basins for testing")
    
    # Load static attributes if needed
    static_attrs = None
    if not run_cfg.get("no_static", False):
        try:
            db_path = run_dir / "attributes.db"
            attrs = load_attributes(db_path, basins=file_num_list, 
                                  drop_na=True, loc=GLOBAL_SETTINGS['loc'])
            # Convert to dictionary for easier access
            static_attrs = {}
            for basin in file_num_list:
                if basin in attrs.index:
                    static_attrs[basin] = attrs.loc[basin].values
        except Exception as e:
            print(f"Warning: Could not load static attributes: {e}")
            run_cfg["no_static"] = True
    
    # Evaluate using adapted approach
    evaluate_adapted(run_cfg, file_num_list, static_attrs)
    
    print("\nAdapted evaluation completed successfully!")

if __name__ == "__main__":
    main()