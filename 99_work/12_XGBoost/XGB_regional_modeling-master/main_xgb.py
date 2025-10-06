"""
XGBoost Regional Hydrological Modeling

Adapted from LSTM regional modeling code to use XGBoost instead of LSTM.
XGBoost requires flattened sequences, so we'll flatten the time series data.
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
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_basin_attributes, load_attributes
from papercode.utils import create_h5_files, get_basin_list

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

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

def predict_synthetic_data(model, basins, mean_dyn, std_dyn, cfg):
    """Generate predictions for synthetic data"""
    syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}
    
    for file_num in basins:
        print(f"Processing synthetic data for basin {file_num}...")
        df_current_basin = load_data_for_basin(int(file_num))
        
        for syn_precip_col in syn_precip_cols:
            if syn_precip_col.startswith('syn_peak'):
                end_date_eva_syn = '2001-02-10'
            else:
                end_date_eva_syn = cfg['val_end'].strftime('%Y-%m-%d')
            
            start_date = cfg['train_start'].strftime('%Y-%m-%d')
            df_current = df_current_basin[start_date:end_date_eva_syn].copy()
            
            # Select appropriate temperature and PET columns
            syn_temp_col = 'Temp_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'Temp'
            syn_pet_col = 'PET_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'PET'
            
            # Create input data array
            syn_input_dyn = np.column_stack([
                df_current[syn_precip_col].values,
                df_current[syn_temp_col].values,
                df_current[syn_pet_col].values
            ])
            
            # Remove NaN values
            valid = ~np.isnan(syn_input_dyn).any(axis=1)
            syn_input_clean = syn_input_dyn[valid]
            syn_dates = df_current.index[valid]
            
            if len(syn_input_clean) == 0:
                continue
            
            # Normalize using training statistics
            syn_input_norm = (syn_input_clean - mean_dyn) / std_dyn
            
            # Flatten for XGBoost (270 timesteps * 3 features)
            seq_length = cfg['seq_length']
            X_syn_list = []
            dates_syn_list = []
            
            for i in range(len(syn_input_norm) - seq_length + 1):
                X_syn_list.append(syn_input_norm[i:i+seq_length].flatten())
                dates_syn_list.append(syn_dates[i+seq_length-1])
            
            if len(X_syn_list) == 0:
                continue
                
            X_syn = np.array(X_syn_list)
            
            # Make predictions
            syn_predict = model.predict(X_syn)
            syn_predict[syn_predict < 0] = 0
            
            # Store results
            temp_syn_df = pd.DataFrame({
                'Date': dates_syn_list,
                'prediction': syn_predict
            })
            syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df
    
    return syn_predict_dfs_combined

def save_synthetic_predictions(syn_predict_dfs_combined, output_dir):
    """Save synthetic predictions to CSV files"""
    os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)
    
    for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
        if not basin_preds_dict:
            continue
        
        combined_syn_df = pd.DataFrame()
        for file_num, df_preds in basin_preds_dict.items():
            df_preds_renamed = df_preds.set_index('Date').rename(columns={'prediction': f'file_{file_num}'})
            if combined_syn_df.empty:
                combined_syn_df = df_preds_renamed
            else:
                combined_syn_df = combined_syn_df.join(df_preds_renamed, how='outer')
        
        combined_syn_df.sort_index(inplace=True)
        combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/XGB_predict_{syn_precip_col.replace('syn_', '')}.csv")

def get_args() -> Dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(10000000),
                        help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--cache_data', type=bool, default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static', type=bool, default=True,
                        help="If True, trains XGBoost without static features")
    parser.add_argument('--concat_static', type=bool, default=True,
                        help="Always True for XGBoost - concatenate static features")
    parser.add_argument('--use_mse', type=bool, default=False,
                        help="If True, uses mean squared error as loss function.")
    
    # XGBoost specific arguments
    parser.add_argument('--n_estimators', type=int, default=500,
                        help="Number of gradient boosted trees")
    parser.add_argument('--max_depth', type=int, default=4,
                        help="Maximum tree depth")
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help="Boosting learning rate")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help="Subsample ratio of training instances")
    parser.add_argument('--colsample_bytree', type=float, default=1.0,
                        help="Subsample ratio of columns when constructing each tree")
    
    cfg = vars(parser.parse_args())
    
    # Add global settings
    cfg.update(GLOBAL_SETTINGS)
    
    # Override concat_static to always be True for XGBoost
    cfg["concat_static"] = True
    
    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run"""
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{month}{day}_{hour}{minute}_seed{cfg["seed"]}'
    
    cfg['run_dir'] = Path(__file__).parent / "runs" / run_name
    if not cfg['run_dir'].exists():
        cfg['run_dir'].mkdir(parents=True)
    
    # Create file paths
    cfg['train_dir'] = cfg['run_dir'] / 'data'
    cfg['train_dir'].mkdir(parents=True, exist_ok=True)
    cfg['db_path'] = str(cfg['run_dir'] / 'attributes.db')
    
    # Store config
    with open(cfg['run_dir'] / 'cfg.json', 'w') as fp:
        temp_cfg = cfg.copy()
        temp_cfg['train_start'] = temp_cfg['train_start'].strftime('%d/%m/%Y')
        temp_cfg['train_end'] = temp_cfg['train_end'].strftime('%d/%m/%Y') 
        temp_cfg['val_start'] = temp_cfg['val_start'].strftime('%d/%m/%Y')
        temp_cfg['val_end'] = temp_cfg['val_end'].strftime('%d/%m/%Y')
        temp_cfg['run_dir'] = str(temp_cfg['run_dir'])
        temp_cfg['train_dir'] = str(temp_cfg['train_dir'])
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)
        
    return cfg

def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Prepare training data"""
    add_basin_attributes(db_path=cfg["db_path"], loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)
    
    # Create .h5 files for train data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])
    
    return cfg

def load_data_for_xgb(h5_file: str, basins: List, db_path: str, no_static: bool) -> tuple:
    """Load and prepare data for XGBoost training"""
    print("Loading data for XGBoost...")
    
    # Create dataset to load samples
    ds = CamelsH5(h5_file=h5_file,
                  basins=basins,
                  db_path=db_path,
                  concat_static=True,  # Always concatenate for simplicity
                  cache=False,
                  no_static=no_static)
    
    X_list = []
    y_list = []
    
    # Load all samples and flatten sequences
    for i in tqdm(range(len(ds)), desc="Loading samples"):
        x, y, q_std = ds[i]
        
        # x shape: (seq_length, features)
        # Flatten sequence for XGBoost: (seq_length * features,)
        x_flat = x.flatten()
        
        X_list.append(x_flat)
        y_list.append(y)
    
    X = np.array(X_list)
    y = np.array(y_list).flatten()
    
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    return X, y

def train(cfg):
    """Train XGBoost model"""
    print("Starting XGBoost training...")
    
    # Set random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    
    # Print configuration
    for key, val in cfg.items():
        print(f"{key}: {val}")
    
    basins = file_num_list
    
    # Create folder structure for this run  
    cfg = _setup_run(cfg)
    
    # Prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)
    
    # Load and prepare data for XGBoost
    X, y = load_data_for_xgb(str(cfg["train_file"]), basins, cfg["db_path"], cfg["no_static"])
    
    # Create XGBoost model
    print("Creating XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        random_state=cfg["seed"],
        n_jobs=-1,  # Use all available cores
        verbosity=1
    )
    
    # Train the model
    print("Training XGBoost model...")
    model.fit(X, y)
    
    # Save the model
    model_path = cfg["run_dir"] / "xgb_model.json"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_path = cfg["run_dir"] / "feature_importance.npy"
    np.save(str(importance_path), feature_importance)
    print(f"Feature importance saved to {importance_path}")
    
    # Print some statistics
    train_score = model.score(X, y)
    print(f"Training R² score: {train_score:.4f}")
    
    # Generate synthetic data predictions
    print("\n--- Generating Synthetic Data Predictions ---")
    try:
        # Calculate normalization statistics from training data
        # Assuming X contains flattened sequences of 270 timesteps * 3 features
        seq_length = cfg["seq_length"]
        n_features = 3  # Precip, Temp, PET
        
        # Reshape to get the dynamic features for normalization
        X_reshaped = X[:, :seq_length * n_features].reshape(-1, seq_length, n_features)
        X_dynamic_flat = X_reshaped.reshape(-1, n_features)
        mean_dyn = np.mean(X_dynamic_flat, axis=0)
        std_dyn = np.std(X_dynamic_flat, axis=0)
        std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)
        
        syn_predictions = predict_synthetic_data(model, basins, mean_dyn, std_dyn, cfg)
        save_synthetic_predictions(syn_predictions, str(cfg["run_dir"]))
        print("Synthetic data predictions completed and saved.")
        
    except Exception as e:
        print(f"Error generating synthetic predictions: {e}")
        print("Continuing without synthetic predictions...")

def evaluate(cfg):
    """Evaluate trained XGBoost model"""
    print("XGBoost evaluation not implemented yet")
    pass

if __name__ == "__main__":
    config = get_args()
    
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "evaluate":
        evaluate(config)