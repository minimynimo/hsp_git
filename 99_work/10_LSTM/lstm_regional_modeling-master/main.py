"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_basin_attributes, load_attributes,
                                 rescale_features)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

import os

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
                          concat_static=cfg["concat_static"],
                          cache=False,
                          no_static=cfg["no_static"])
        
        eval_loader = DataLoader(eval_ds, batch_size=1024, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_basins = []
        all_dates = []
        
        with torch.no_grad():
            for data in tqdm(eval_loader, desc="Generating evaluation predictions"):
                if cfg["no_static"]:
                    x, y, q_stds = data
                else:
                    x, y, q_stds = data
                
                # Move to device
                device = next(model.parameters()).device
                x = x.to(device)
                
                # Make predictions
                predictions = model(x).cpu().numpy().flatten()
                targets = y.numpy().flatten()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Get basin and date information for each sample
        basin_dates = []
        for i in range(len(eval_ds)):
            # Get basin info from dataset
            with h5py.File(str(eval_file), 'r') as f:
                basin = f["sample_2_basin"][i].decode("ascii")
                basin_dates.append(basin)
        
        # Create evaluation results DataFrame
        eval_df = pd.DataFrame({
            'basin': basin_dates,
            'prediction': all_predictions,
            'observation': all_targets
        })
        
        # Save evaluation predictions
        os.makedirs(str(cfg["run_dir"]) + "/predict", exist_ok=True)
        eval_output_path = f"{cfg['run_dir']}/predict/LSTM_evaluation_predictions.csv"
        eval_df.to_csv(eval_output_path, index=False)
        print(f"Evaluation predictions saved to {eval_output_path}")
        
        # Also save by basin for easier analysis
        for basin in basins:
            basin_data = eval_df[eval_df['basin'] == basin].copy()
            if not basin_data.empty:
                basin_output_path = f"{cfg['run_dir']}/predict/LSTM_evaluation_basin_{basin}.csv"
                basin_data.to_csv(basin_output_path, index=False)
        
        print("Evaluation period predictions completed and saved.")
        
    except Exception as e:
        print(f"Error generating evaluation predictions: {e}")
        print("Continuing without evaluation predictions...")

def save_synthetic_predictions_lstm(syn_predict_dfs_combined, output_dir, model_name):
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
        combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/{model_name}_predict_{syn_precip_col.replace('syn_', '')}.csv")


###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

ver_name = "ver1_1" if GLOBAL_SETTINGS['ver'] == 1 else "ver2_0"

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = f'/data0/funato/hsp/0_out/LSTM/GB/data'
varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

rundir = f'/data0/funato/hsp/0_out/LSTM/GB'
os.makedirs(rundir, exist_ok=True)

if GLOBAL_SETTINGS['loc'] == "JP":
    file_tot_num = 87
    attribute_size = 196
    output_mean = np.array([3.8526752089320553])
elif GLOBAL_SETTINGS['loc'] == "GB":
    file_tot_num = 396
    attribute_size = 209
    output_mean = np.array([1.838397528976445])
file_num_list = list(map(str, range(1, file_tot_num + 1)))

###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness"])
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--cache_data',
                        type=bool,
                        default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static',
                        type=str2bool,
                        default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--concat_static',
                        type=bool,
                        default=True,
                        help="If True, train LSTM with static feats concatenated at each time step. If True, uses standard LSTM otherwise uses EA-LSTM")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=False,
                        help="If True, uses mean squared error as loss function.")
    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).parent / "runs" / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_basin_attributes(db_path=cfg["db_path"], loc=GLOBAL_SETTINGS['loc'], ver_name=ver_name)

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = True,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])


    basins = get_basin_list(file_tot_num=file_tot_num)

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    ds = CamelsH5(h5_file=cfg["train_file"],
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=cfg["concat_static"],
                  cache=cfg["cache_data"],
                  no_static=cfg["no_static"])
    loader = DataLoader(ds,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=cfg["num_workers"])

    # create model and optimizer
    input_size_stat = 0 if cfg["no_static"] else 27
    input_size_dyn = 3 if (cfg["no_static"] or not cfg["concat_static"]) else 30
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))
    
    # Save final model
    final_model_path = cfg["run_dir"] / "lstm_model_final.pt"
    torch.save(model.state_dict(), str(final_model_path))
    print(f"Final LSTM model saved to {final_model_path}")
    
    # Generate synthetic data predictions
    print("\n--- Generating Synthetic Data Predictions ---")
    try:
        syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}
        
        for basin in basins:
            print(f"Processing synthetic data for basin {basin}...")
            df_current_basin = load_data_for_basin(int(basin))
            
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
                
                # Skip if synthetic columns don't exist
                if syn_precip_col not in df_current.columns:
                    print(f"Warning: {syn_precip_col} not found in data for basin {basin}")
                    continue
                    
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
                
                if len(syn_input_clean) < cfg['seq_length']:
                    continue
                
                # Create sequences for LSTM
                X_syn_list = []
                dates_syn_list = []
                
                for i in range(len(syn_input_clean) - cfg['seq_length'] + 1):
                    X_syn_list.append(syn_input_clean[i:i+cfg['seq_length']])
                    dates_syn_list.append(syn_dates[i+cfg['seq_length']-1])
                
                if len(X_syn_list) == 0:
                    continue
                    
                X_syn = torch.tensor(np.array(X_syn_list), dtype=torch.float32)
                
                # Move tensor to the same device as the model
                device = next(model.parameters()).device
                X_syn = X_syn.to(device)
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    if cfg['concat_static']:
                        # Add static features if needed
                        # For simplicity, we'll use dynamic features only
                        syn_predict = model(X_syn).cpu().numpy().flatten()
                    else:
                        syn_predict = model(X_syn).cpu().numpy().flatten()
                    
                syn_predict[syn_predict < 0] = 0
                
                # Store results
                temp_syn_df = pd.DataFrame({
                    'Date': dates_syn_list,
                    'prediction': syn_predict
                })
                syn_predict_dfs_combined[syn_precip_col][basin] = temp_syn_df
        
        # Save synthetic predictions
        save_synthetic_predictions_lstm(syn_predict_dfs_combined, str(cfg["run_dir"]), "LSTM")
        print("Synthetic data predictions completed and saved.")
        
        # Generate evaluation period predictions
        save_evaluation_predictions(model, basins, cfg)
        
    except Exception as e:
        print(f"Error generating synthetic predictions: {e}")
        print("Continuing without synthetic predictions...")


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list(file_tot_num=file_tot_num)

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model
    input_size_stat = 0 if run_cfg["no_static"] else 27
    input_size_dyn = 3 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 30
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"]).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=run_cfg["concat_static"],
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)

        preds, obs = evaluate_basin(model, loader)

        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

        results[basin] = df

    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable='output', loc=GLOBAL_SETTINGS['loc'])
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs


def eval_robustness(user_cfg: Dict):
    """Evaluate model robustness of EA-LSTM

    In this experiment, gaussian noise with increasing scale is added to the static features to
    evaluate the model robustness against pertubations of the static catchment characteristics.
    For each scale, 50 noise vectors are drawn.
    
    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    
    Raises
    ------
    NotImplementedError
        If the run_dir specified points not to a EA-LSTM model folder.
    """
    random.seed(user_cfg["seed"])
    np.random.seed(user_cfg["seed"])

    # fixed settings for this analysis
    n_repetitions = 50
    scales = [0.1 * i for i in range(11)]

    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if run_cfg["concat_static"] or run_cfg["no_static"]:
        raise NotImplementedError("This function is only implemented for EA-LSTM models")

    basins = get_basin_list(file_tot_num=file_tot_num)

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # initialize Model
    model = Model(input_size_dyn=5,
                  input_size_stat=27,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"]).to(DEVICE)
    weight_file = user_cfg["run_dir"] / "model_epoch30.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    overall_results = {}
    # process bar handle
    pbar = tqdm(basins, file=sys.stdout)
    for basin in pbar:
        ds_test = CamelsTXT(
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, num_workers=0)
        basin_results = defaultdict(list)
        step = 1
        for scale in scales:
            for _ in range(1 if scale == 0.0 else n_repetitions):
                noise = np.random.normal(loc=0, scale=scale, size=27).astype(np.float32)
                noise = torch.from_numpy(noise).to(DEVICE)
                nse = eval_with_added_noise(model, loader, noise)
                basin_results[scale].append(nse)
                pbar.set_postfix_str(f"Basin progress: {step}/{(len(scales)-1)*n_repetitions+1}")
                step += 1

        overall_results[basin] = basin_results
    out_file = (Path(__file__).absolute().parent /
                f'results/{user_cfg["run_dir"].name}_model_robustness.p')
    if not out_file.parent.is_dir():
        out_file.parent.mkdir(parents=True)
    with out_file.open("wb") as fp:
        pickle.dump(overall_results, fp)


def eval_with_added_noise(model: torch.nn.Module, loader: DataLoader, noise: torch.Tensor) -> float:
    """Evaluate model on a single basin with added noise

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.
    noise : torch.Tensor
        Tensor containing the noise for this evaluation run.
    
    Returns
    -------
    float
        Nash-Sutcliff-Efficiency of the simulations with added noise.
    """
    model.eval()
    preds, obs = None, None
    with torch.no_grad():
        for x_d, x_s, y in loader:
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            batch_noise = noise.repeat(*x_s.size()[:2], 1)
            x_s = x_s.add(batch_noise)
            y_hat = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = y_hat.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, y_hat.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        obs = obs.numpy()
        preds = rescale_features(preds.numpy(), variable='output', loc=GLOBAL_SETTINGS['loc'])

        # set discharges < 0 to zero
        preds[preds < 0] = 0

        nse = calc_nse(obs[obs >= 0], preds[obs >= 0])
        return nse


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"lstm_no_static_seed{run_cfg['seed']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
