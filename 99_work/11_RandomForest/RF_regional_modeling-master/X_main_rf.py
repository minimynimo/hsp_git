"""
Random Forest Regional Hydrological Modeling

Adapted from LSTM regional modeling code to use Random Forest instead of LSTM.
This version always concatenates static features to dynamic features for simplicity.
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm
import h5py

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_basin_attributes, load_attributes
from papercode.utils import create_h5_files, get_basin_list

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 'syn_peak1_rate0',
                   'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_evaluation_predictions(model, basins, cfg):
    """Generate and save predictions for the evaluation period"""
    print("\n--- Generating Evaluation Period Predictions ---")
    try:
        # Create evaluation dataset
        eval_file = cfg["train_dir"] / 'eval_data.h5'
        create_h5_files(out_file=eval_file,
                        basins=basins,
                        dates=[cfg["val_start"], cfg["val_end"]],
                        with_basin_str=True,
                        seq_length=cfg["seq_length"])
        
        eval_ds = CamelsH5(h5_file=str(eval_file),
                          basins=basins,
                          db_path=cfg["db_path"],
                          concat_static=not cfg["no_static"],
                          cache=False,
                          no_static=cfg["no_static"])
        
        X_eval_list = []
        y_eval_list = []
        basin_eval_list = []
        
        # Load evaluation data
        for i in tqdm(range(len(eval_ds)), desc="Loading evaluation data"):
            data = eval_ds[i]
            if cfg["no_static"]:
                x, y, q_std = data
            else:
                if len(data) == 3:
                    x, y, q_std = data
                else:
                    x, attributes, y, q_std = data
            
            # Flatten the sequence dimension for RF
            x_flat = x.numpy().flatten()
            y_val = y.numpy().item()
            
            # Get basin info
            with h5py.File(str(eval_file), 'r') as f:
                basin = f["sample_2_basin"][i].decode("ascii")
            
            X_eval_list.append(x_flat)
            y_eval_list.append(y_val)
            basin_eval_list.append(basin)
        
        X_eval = np.array(X_eval_list)
        y_eval = np.array(y_eval_list)
        
        # Make predictions
        predictions = model.predict(X_eval)
        predictions[predictions < 0] = 0
        
        # Create evaluation results DataFrame
        eval_df = pd.DataFrame({
            'basin': basin_eval_list,
            'prediction': predictions,
            'observation': y_eval
        })
        
        # Save evaluation predictions
        os.makedirs(str(cfg["run_dir"]) + "/predict", exist_ok=True)
        eval_output_path = f"{cfg['run_dir']}/predict/RF_evaluation_predictions.csv"
        eval_df.to_csv(eval_output_path, index=False)
        print(f"Evaluation predictions saved to {eval_output_path}")
        
        # Also save by basin for easier analysis
        for basin in basins:
            basin_data = eval_df[eval_df['basin'] == basin].copy()
            if not basin_data.empty:
                basin_output_path = f"{cfg['run_dir']}/predict/RF_evaluation_basin_{basin}.csv"
                basin_data.to_csv(basin_output_path, index=False)
        
        print("Evaluation period predictions completed and saved.")
        
    except Exception as e:
        print(f"Error generating evaluation predictions: {e}")
        print("Continuing without evaluation predictions...")

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
            
            # Flatten for RF (270 timesteps * 3 features)
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
        combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/RF_predict_{syn_precip_col.replace('syn_', '')}.csv")

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
    parser.add_argument('--no_static', type=str2bool, default=True,
                        help="If True, trains RF without static features")
    parser.add_argument('--concat_static', type=bool, default=True,
                        help="Always True for RF - concatenate static features")
    parser.add_argument('--use_mse', type=bool, default=False,
                        help="If True, uses mean squared error as loss function.")
    
        # Random Forest specific arguments
    parser.add_argument('--n_estimators', type=int, default=100,
                        help="Number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=5,
                        help="Maximum depth of trees")
    parser.add_argument('--max_features', type=float, default=0.7,
                        help="Maximum features to consider for splits")
    parser.add_argument('--max_samples', type=float, default=0.5,
                        help="Maximum fraction of samples for training each tree")
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help="Minimum samples required to split an internal node")
    
    cfg = vars(parser.parse_args())
    
    # Add global settings
    cfg.update(GLOBAL_SETTINGS)
    
    # Override concat_static to always be True for RF
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
        temp_cfg['run_dir'] = str(temp_cfg['run_dir'])  # Convert Path to string
        temp_cfg['train_dir'] = str(temp_cfg['train_dir'])  # Convert Path to string
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

def load_data_for_rf(h5_file: str, basins: List, db_path: str, no_static: bool = False):
    """Load and flatten data for Random Forest training"""
    print("Loading data for Random Forest...")
    
    # Create dataset object
    ds = CamelsH5(h5_file=h5_file,
                  basins=basins,
                  db_path=db_path,
                  concat_static=True,  # Always concatenate for RF
                  cache=False,
                  no_static=no_static)
    
    X_list = []
    y_list = []
    
    # Load all data
    for i in tqdm(range(len(ds)), desc="Loading samples"):
        if no_static:
            x, y, q_std = ds[i]
        else:
            x, y, q_std = ds[i]  # Static features already concatenated
        
        # Flatten the sequence dimension for RF
        # x shape: (seq_length, features) -> flatten to (seq_length * features,)
        x_flat = x.numpy().flatten()
        y_val = y.numpy().item()
        
        X_list.append(x_flat)
        y_list.append(y_val)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    return X, y

def train(cfg):
    """Train Random Forest model"""
    print("Starting Random Forest training...")
    
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
    
    # Load and flatten data for RF
    X, y = load_data_for_rf(str(cfg["train_file"]), basins, cfg["db_path"], cfg["no_static"])
    
    # Create and train Random Forest model
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", 5),
        max_features=cfg.get("max_features", 0.7),
        max_samples=cfg.get("max_samples", 0.5),
        min_samples_split=cfg.get("min_samples_split", 2),
        random_state=cfg["seed"],
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X, y)
    
    # Save the trained model
    model_path = cfg["run_dir"] / "rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Print feature importance (top 20)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop 20 Feature importances:")
        for i in range(min(20, len(importances))):
            print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    # Make predictions and calculate training error
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"\nTraining MSE: {mse:.6f}")
    print(f"Training RMSE: {np.sqrt(mse):.6f}")
    
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
        
        # Generate evaluation period predictions
        save_evaluation_predictions(model, basins, cfg)
        
    except Exception as e:
        print(f"Error generating synthetic predictions: {e}")
        print("Continuing without synthetic predictions...")

def evaluate(cfg):
    """Evaluate trained Random Forest model"""
    print("Random Forest evaluation not implemented yet")
    pass

if __name__ == "__main__":
    config = get_args()
    
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "evaluate":
        evaluate(config)