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
import os
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features, SCALER)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.metrics import (calc_nse, calc_kge, calc_rmse, calc_mae, calc_pbias, 
                               calc_lognse, calc_agreement_index)
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 3,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define synthetic precipitation columns for evaluation
syn_precip_cols = ['syn_no_precip',
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

###############
# Prepare run #
###############


def extract_loc_from_path(path_str: str) -> str:
    """Extract location identifier from path string.
    
    Parameters
    ----------
    path_str : str
        Path string to extract location from
        
    Returns
    -------
    str
        Location identifier ("US", "GB", or "JP")
    """
    path_str = str(path_str).upper()
    if 'GB' in path_str or '_GB' in path_str:
        return 'GB'
    elif 'JP' in path_str or '_JP' in path_str:
        return 'JP'
    else:
        return 'US'


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness", "adapt"])
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
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
                        type=bool,
                        default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--concat_static',
                        type=bool,
                        default=False,
                        help="If True, train LSTM with static feats concatenated at each time step")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=True,
                        help="If True, uses mean squared error as loss function.")
    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness", "adapt"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
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
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name
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
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(camels_root=cfg["camels_root"],
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
                 initial_forget_bias: int = 3,
                 dropout: float = 0.0,
                 concat_static: bool = False,
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
            Value of the initial forget gate bias. (default: 3)
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

    loc = extract_loc_from_path(cfg["camels_root"])
    basins = get_basin_list(loc=loc)
    
    # Filter basins to only use those in the basin_ids list
    basin_ids_file = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/small_basins_1000.csv"
    if os.path.exists(basin_ids_file):
        basin_ids_df = pd.read_csv(basin_ids_file, index_col=None, header=0)
        basin_ids = basin_ids_df["basin_id"].tolist()
        # Convert to string format with leading zeros
        basin_ids_str = [str(b).zfill(3) for b in basin_ids]
        # Filter basins to only include those in basin_ids
        basins = [b for b in basins if b in basin_ids_str]
        print(f"Filtered to {len(basins)} basins from basin_ids list")
    else:
        print(f"Basin IDs file not found at {basin_ids_file}, using all basins")

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
    input_size_stat = 0 if cfg["no_static"] else 18
    input_size_dyn = 3 if (cfg["no_static"] or not cfg["concat_static"]) else 21
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


def load_synthetic_data(basin: str, scenario_col: str, dates: List, seq_length: int, 
                       camels_root: PosixPath, means: pd.Series, stds: pd.Series,
                       concat_static: bool, db_path: str) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
    """Load synthetic scenario data for a basin.
    
    Parameters
    ----------
    basin : str
        Basin identifier (e.g., "001")
    scenario_col : str
        Name of the synthetic precipitation column to use
    dates : List
        Start and end date of the period
    seq_length : int
        Length of input sequences
    camels_root : PosixPath
        Path to CAMELS data root
    means : pd.Series
        Attribute means for normalization
    stds : pd.Series
        Attribute stds for normalization
    concat_static : bool
        Whether to concatenate static features
    db_path : str
        Path to attributes database
        
    Returns
    -------
    x : torch.Tensor
        Input sequences
    y : torch.Tensor
        Target values (observations, used for dates but not evaluation)
    dates : pd.DatetimeIndex
        Dates corresponding to predictions
    """
    from papercode.datautils import load_data, normalize_features, reshape_data, load_attributes
    
    # Extract location from camels_root path
    camels_root_str = str(camels_root)
    if 'GB' in camels_root_str or '_GB' in camels_root_str:
        loc = 'GB'
    elif 'JP' in camels_root_str or '_JP' in camels_root_str:
        loc = 'JP'
    else:
        loc = 'US'
    
    # Convert basin to integer file number
    file_num = int(basin)
    
    # Load data from varssim files
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{loc}"
    df = load_data(file_num, varssim_dir, loc)
    
    if df.empty:
        return None, None, None
    
    # Add warmup period
    start_date = dates[0] - pd.DateOffset(days=seq_length - 1)
    end_date = dates[1]
    df = df[start_date:end_date]
    
    # Check if synthetic column exists
    if scenario_col not in df.columns:
        return None, None, None
    
    # Create synthetic data by replacing precipitation
    df_syn = df.copy()
    df_syn['Precip'] = df_syn[scenario_col]
    
    # Use peak temperature and PET for peak scenarios
    if 'peak' in scenario_col:
        if 'Temp_at_peak' in df_syn.columns and 'PET_at_peak' in df_syn.columns:
            df_syn['Temp'] = df_syn['Temp_at_peak']
            df_syn['PET'] = df_syn['PET_at_peak']
    
    # Prepare input data (P, PET, Temp)
    x = np.array([
        df_syn['Precip'].values,
        df_syn['PET'].values,
        df_syn['Temp'].values
    ]).T
    
    # Use observed flow for placeholder (won't be used for metrics, just for structure)
    y = np.array([df_syn['Obs flow'].values]).T
    
    # Normalize and reshape
    x = normalize_features(x, variable='inputs')
    x, y = reshape_data(x, y, seq_length)
    
    # Get dates for predictions (after warmup)
    prediction_dates = df.index[seq_length:]
    
    # Convert to tensors
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    
    # Handle static attributes if needed
    if not concat_static:
        # Load attributes
        attributes = load_attributes(db_path, [basin], drop_lat_lon=True)
        attributes = (attributes - means) / stds
        attributes_tensor = torch.from_numpy(attributes.loc[attributes.index == basin].values.astype(np.float32))
        return x, attributes_tensor, prediction_dates
    else:
        # For concat_static, attributes will be added in the model forward pass
        # We need to return them separately
        attributes = load_attributes(db_path, [basin], drop_lat_lon=True)
        attributes = (attributes - means) / stds
        attributes_tensor = torch.from_numpy(attributes.loc[attributes.index == basin].values.astype(np.float32))
        return x, attributes_tensor, prediction_dates


def evaluate_synthetic_basin_period(model: nn.Module, basin: str, scenario_col: str,
                                    dates: List, seq_length: int, camels_root: PosixPath,
                                    means: pd.Series, stds: pd.Series, concat_static: bool,
                                    db_path: str, no_static: bool) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Evaluate model on synthetic scenario data for a single basin.
    
    Parameters
    ----------
    model : nn.Module
        Trained LSTM/EA-LSTM model
    basin : str
        Basin identifier
    scenario_col : str
        Name of synthetic precipitation column
    dates : List
        Start and end dates
    seq_length : int
        Sequence length
    camels_root : PosixPath
        Path to CAMELS data root
    means : pd.Series
        Attribute means
    stds : pd.Series
        Attribute stds
    concat_static : bool
        Whether to concatenate static features
    db_path : str
        Path to attributes database
    no_static : bool
        Whether model uses static features
        
    Returns
    -------
    predictions : np.ndarray
        Model predictions for the synthetic scenario
    dates : pd.DatetimeIndex
        Dates corresponding to predictions
    """
    result = load_synthetic_data(basin, scenario_col, dates, seq_length, camels_root,
                                 means, stds, concat_static, db_path)
    
    if result[0] is None:
        return None, None
    
    x, attributes, prediction_dates = result
    
    model.eval()
    
    with torch.no_grad():
        x = x.to(DEVICE)
        
        if no_static or concat_static:
            # For LSTM or concatenated static features
            if concat_static:
                # Concatenate static features at each time step
                attributes_expanded = attributes.repeat(x.shape[0], seq_length, 1)
                x = torch.cat([x, attributes_expanded.to(DEVICE)], dim=-1)
            predictions = model(x)[0]
        else:
            # For EA-LSTM with entity-aware input gate
            attributes_expanded = attributes.unsqueeze(0).repeat(x.shape[0], 1, 1).to(DEVICE)
            predictions = model(x, attributes_expanded[:, 0, :])[0]
        
        predictions = predictions.detach().cpu().numpy()
        predictions = rescale_features(predictions, variable='output')
        predictions[predictions < 0] = 0
    
    return predictions.flatten(), prediction_dates


def evaluate_synthetic_scenarios(model: nn.Module, run_cfg: Dict, user_cfg: Dict,
                                 basins: List, means: pd.Series, stds: pd.Series,
                                 db_path: str, loc: str):
    """Evaluate model on all synthetic scenarios.
    
    Parameters
    ----------
    model : nn.Module
        Trained LSTM/EA-LSTM model
    run_cfg : Dict
        Run configuration
    user_cfg : Dict
        User configuration
    basins : List
        List of basin identifiers
    means : pd.Series
        Attribute means
    stds : pd.Series
        Attribute stds
    db_path : str
        Path to attributes database
    loc : str
        Location identifier (US, GB, JP)
    """
    print("\n" + "="*60)
    print("Evaluating Synthetic Scenarios")
    print("="*60)
    
    # Set up output directory
    output_base = f"/data0/funato/hsp2/0_out/LSTM/{loc}/regional"
    output_predict_dir = os.path.join(output_base, "predict", "synthetic")
    os.makedirs(output_predict_dir, exist_ok=True)
    
    # Create hyperparameter string for filenames
    hs = run_cfg["hidden_size"]
    do = run_cfg["dropout"]
    ep = run_cfg["epochs"]
    hp_str = f"hs{hs}_do{do}_e{ep}"
    
    # Evaluation period only for synthetic scenarios
    dates = [GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]]
    date_range = pd.date_range(start=dates[0], end=dates[1])
    
    for scenario_col in syn_precip_cols:
        print(f"\nProcessing scenario: {scenario_col}")
        
        all_predictions = {}
        
        for basin in tqdm(basins, desc=f"Basins for {scenario_col}"):
            try:
                predictions, prediction_dates = evaluate_synthetic_basin_period(
                    model=model,
                    basin=basin,
                    scenario_col=scenario_col,
                    dates=dates,
                    seq_length=run_cfg["seq_length"],
                    camels_root=user_cfg["camels_root"],
                    means=means,
                    stds=stds,
                    concat_static=run_cfg["concat_static"],
                    db_path=db_path,
                    no_static=run_cfg["no_static"]
                )
                
                if predictions is not None:
                    all_predictions[basin] = predictions
                    
            except Exception as e:
                print(f"Error processing basin {basin} for scenario {scenario_col}: {e}")
                continue
        
        if not all_predictions:
            print(f"No valid predictions for scenario {scenario_col}")
            continue
        
        # Save predictions to CSV (basin x dates format)
        predict_filename = f"LSTM_predict_{hp_str}_{scenario_col}_eva.csv"
        predict_filepath = os.path.join(output_predict_dir, predict_filename)
        
        # Create predictions DataFrame with basin as first column (integer)
        predict_df_data = {'basin': [int(b) for b in sorted(all_predictions.keys(), key=int)]}
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            predict_df_data[date_str] = [
                all_predictions[basin][i] if basin in all_predictions and i < len(all_predictions[basin]) else np.nan
                for basin in sorted(all_predictions.keys(), key=int)
            ]
        
        predict_df = pd.DataFrame(predict_df_data)
        predict_df.to_csv(predict_filepath, index=False)
        print(f"Saved synthetic predictions to: {predict_filepath}")
    
    print("\nSynthetic scenario evaluation completed!")


def evaluate(user_cfg: Dict):
    """Evaluate model and save predictions and benchmark results to CSV files.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    loc = extract_loc_from_path(run_cfg["camels_root"])
    basins = get_basin_list(loc=loc)
    
    # Filter basins to only use those in the basin_ids list
    basin_ids_file = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/small_basins_1000.csv"
    if os.path.exists(basin_ids_file):
        basin_ids_df = pd.read_csv(basin_ids_file, index_col=None, header=0)
        basin_ids = basin_ids_df["basin_id"].tolist()
        # Convert to string format with leading zeros
        basin_ids_str = [str(b).zfill(3) for b in basin_ids]
        # Filter basins to only include those in basin_ids
        basins = [b for b in basins if b in basin_ids_str]
        print(f"Filtered to {len(basins)} basins from basin_ids list")
    else:
        print(f"Basin IDs file not found at {basin_ids_file}, using all basins")

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model
    input_size_stat = 0 if run_cfg["no_static"] else 18
    input_size_dyn = 3 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 21
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"]).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    # Create hyperparameter string for filenames
    hs = run_cfg["hidden_size"]
    do = run_cfg["dropout"]
    ep = run_cfg["epochs"]
    hp_str = f"hs{hs}_do{do}_e{ep}"
    
    # Set up output directories
    output_base = f"/data0/funato/hsp2/0_out/LSTM/{loc}/regional"
    output_predict_dir = os.path.join(output_base, "predict")
    output_results_dir = os.path.join(output_base, "results")
    os.makedirs(output_predict_dir, exist_ok=True)
    os.makedirs(output_results_dir, exist_ok=True)
    
    # Process both calibration and evaluation periods
    periods = {
        'cal': [GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]],
        'eva': [GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]]
    }
    
    for period_name, period_dates in periods.items():
        print(f"\n{'='*60}")
        print(f"Processing {period_name.upper()} period: {period_dates[0]} to {period_dates[1]}")
        print(f"{'='*60}")
        
        date_range = pd.date_range(start=period_dates[0], end=period_dates[1])
        
        # Store predictions and metrics for all basins
        all_predictions = {}
        all_metrics = []
        
        for basin in tqdm(basins, desc=f"{period_name.upper()} period"):
            ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                                basin=basin,
                                dates=period_dates,
                                is_train=False,
                                seq_length=run_cfg["seq_length"],
                                with_attributes=True,
                                attribute_means=means,
                                attribute_stds=stds,
                                concat_static=run_cfg["concat_static"],
                                db_path=db_path)
            loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)

            preds, obs = evaluate_basin(model, loader)

            # Store predictions
            all_predictions[basin] = preds.flatten()
            
            # Calculate metrics (only for valid observations)
            valid_mask = obs >= 0
            obs_valid = obs[valid_mask]
            preds_valid = preds[valid_mask]
            
            if len(obs_valid) > 0:
                try:
                    nse = calc_nse(obs_valid, preds_valid)
                    kge = calc_kge(obs_valid, preds_valid)
                    e1 = calc_lognse(obs_valid, preds_valid)
                    ve = calc_pbias(obs_valid, preds_valid)
                    d = calc_agreement_index(obs_valid, preds_valid)
                    rmse = calc_rmse(obs_valid, preds_valid)
                    mae = calc_mae(obs_valid, preds_valid)
                except Exception as e:
                    print(f"Warning: Error calculating metrics for basin {basin}: {e}")
                    nse = kge = e1 = ve = d = rmse = mae = np.nan
            else:
                nse = kge = e1 = ve = d = rmse = mae = np.nan
            
            # Convert basin to integer (remove leading zeros)
            basin_int = int(basin)
            
            all_metrics.append({
                'basin': basin_int,
                'NSE': nse,
                'KGE': kge,
                'E1': e1,
                'VE': ve,
                'D': d,
                'RMSE': rmse,
                'MAE': mae
            })
        
        # Save predictions to CSV
        predict_filename = f"LSTM_predict_{hp_str}_{period_name}.csv"
        predict_filepath = os.path.join(output_predict_dir, predict_filename)
        
        # Create predictions DataFrame with basin as first column (integer)
        predict_df_data = {'basin': [int(b) for b in basins]}
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            predict_df_data[date_str] = [all_predictions[basin][i] if i < len(all_predictions[basin]) else np.nan 
                                         for basin in basins]
        
        predict_df = pd.DataFrame(predict_df_data)
        predict_df.to_csv(predict_filepath, index=False)
        print(f"Saved predictions to: {predict_filepath}")
        
        # Save metrics to CSV
        results_filename = f"LSTM_results_{hp_str}_{period_name}.csv"
        results_filepath = os.path.join(output_results_dir, results_filename)
        
        results_df = pd.DataFrame(all_metrics)
        results_df = results_df.sort_values('basin')
        results_df.to_csv(results_filepath, index=False)
        print(f"Saved results to: {results_filepath}")
        
        # Print summary statistics
        print(f"\n{period_name.upper()} Period Summary:")
        print(f"  Mean NSE: {results_df['NSE'].mean():.3f}")
        print(f"  Mean KGE: {results_df['KGE'].mean():.3f}")
        print(f"  Mean RMSE: {results_df['RMSE'].mean():.3f}")
    
    # Also store results in pickle format (original functionality)
    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins, desc="Creating pickle results"):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
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
    
    # Evaluate synthetic scenarios
    evaluate_synthetic_scenarios(
        model=model,
        run_cfg=run_cfg,
        user_cfg=user_cfg,
        basins=basins,
        means=means,
        stds=stds,
        db_path=db_path,
        loc=loc
    )

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

        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs


############################
# Adaptation Functionality #
############################

class FineTuneDataset(Dataset):
    """Dataset for fine-tuning models on individual basins."""
    
    def __init__(self, basin: str, dates: List, seq_length: int, camels_root: PosixPath,
                 concat_static: bool, db_path: str, attribute_means: pd.Series, 
                 attribute_stds: pd.Series, no_static: bool, global_scalers: Dict = None):
        """Initialize fine-tuning dataset.
        
        Parameters
        ----------
        basin : str
            Basin identifier
        dates : List
            Start and end dates
        seq_length : int
            Sequence length
        camels_root : PosixPath
            Path to CAMELS data root
        concat_static : bool
            Whether to concatenate static features
        db_path : str
            Path to attributes database
        attribute_means : pd.Series
            Attribute means
        attribute_stds : pd.Series
            Attribute stds
        no_static : bool
            Whether model uses no static features
        global_scalers : Dict, optional
            Global scalers matching regional training. If None, uses per-basin scalers.
        """
        self.basin = basin
        self.dates = dates
        self.seq_length = seq_length
        self.camels_root = camels_root
        self.concat_static = concat_static
        self.db_path = db_path
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.no_static = no_static
        self.global_scalers = global_scalers
        self.samples = []
        self.scalers = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare basin-specific data with per-basin normalization."""
        from papercode.datautils import load_data
        
        # Extract location
        camels_root_str = str(self.camels_root)
        if 'GB' in camels_root_str or '_GB' in camels_root_str:
            loc = 'GB'
        elif 'JP' in camels_root_str or '_JP' in camels_root_str:
            loc = 'JP'
        else:
            loc = 'US'
        
        # Load data
        varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{loc}"
        file_num = int(self.basin)
        df = load_data(file_num, varssim_dir, loc)
        
        if df.empty:
            return
        
        # Filter by date range
        df_period = df.loc[self.dates[0]:self.dates[1]]
        
        dyn_cols = ['Precip', 'PET', 'Temp']
        q_col = 'Obs flow'
        
        # Find clean data blocks (no NaN values)
        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []
        
        clean_blocks = []
        for s in clumps:
            clean_blocks.append(df_period.iloc[s])
        
        if not clean_blocks:
            return
        
        # Use global scalers if provided (matching regional training), otherwise fall back to per-basin
        if self.global_scalers is not None:
            self.scalers = self.global_scalers.copy()
        else:
            # Calculate per-basin normalization statistics from clean data
            basin_clean_df = pd.concat(clean_blocks)
            self.scalers = {
                'q_mean': basin_clean_df[q_col].mean(),
                'q_std': basin_clean_df[q_col].std(),
                'dynamic_means': basin_clean_df[dyn_cols].mean().values,
                'dynamic_stds': basin_clean_df[dyn_cols].std().values
            }
        
        # Normalize the full period data using scalers
        dyn_data_scaled = (df_period[dyn_cols].values - self.scalers['dynamic_means']) / self.scalers['dynamic_stds']
        q_data_scaled = (df_period[q_col].values - self.scalers['q_mean']) / self.scalers['q_std']
        
        # Load static features if needed
        static_data_scaled = None
        if not self.no_static:
            attributes = load_attributes(self.db_path, [self.basin], drop_lat_lon=True)
            static_data = attributes.loc[attributes.index == self.basin].values.flatten()
            static_data_scaled = (static_data - self.attribute_means.values) / self.attribute_stds.values
        
        # Create samples from clean blocks
        for block_df in clean_blocks:
            if len(block_df) < self.seq_length:
                continue
            
            start_loc = df_period.index.get_loc(block_df.index[0])
            end_loc = df_period.index.get_loc(block_df.index[-1])
            block_dyn = dyn_data_scaled[start_loc:end_loc + 1]
            block_q = q_data_scaled[start_loc:end_loc + 1]
            
            for i in range(len(block_df) - self.seq_length + 1):
                target_idx = i + self.seq_length - 1
                dyn_seq = block_dyn[i:target_idx + 1]
                
                # Concatenate static features if needed
                if not self.no_static and self.concat_static:
                    static_seq = np.tile(static_data_scaled, (self.seq_length, 1))
                    final_features = np.concatenate([dyn_seq, static_seq], axis=1)
                else:
                    final_features = dyn_seq
                
                sample_date = block_df.index[target_idx]
                
                # Store as (features, target, static_attrs, date)
                if not self.no_static and not self.concat_static:
                    self.samples.append((
                        final_features.astype(np.float32),
                        block_q[target_idx].astype(np.float32),
                        static_data_scaled.astype(np.float32),
                        sample_date
                    ))
                else:
                    self.samples.append((
                        final_features.astype(np.float32),
                        block_q[target_idx].astype(np.float32),
                        None,
                        sample_date
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y, static, date = self.samples[idx]
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor([y])
        
        if static is not None:
            static_tensor = torch.from_numpy(static)
            return x_tensor, static_tensor, y_tensor, date
        else:
            return x_tensor, y_tensor, date


def custom_collate_ft(batch):
    """Custom collate function for fine-tuning that handles dates."""
    if len(batch[0]) == 4:
        # EA-LSTM case: x_d, x_s, y, date
        xs, statics, ys, dates = zip(*batch)
        xs_batch = torch.stack(xs)
        statics_batch = torch.stack(statics)
        ys_batch = torch.stack(ys)
        return xs_batch, statics_batch, ys_batch, list(dates)
    else:
        # LSTM case: x, y, date
        xs, ys, dates = zip(*batch)
        xs_batch = torch.stack(xs)
        ys_batch = torch.stack(ys)
        return xs_batch, ys_batch, list(dates)


def fine_tune_basin_model(base_model: nn.Module, basin: str, run_cfg: Dict, user_cfg: Dict,
                          means: pd.Series, stds: pd.Series, db_path: str) -> Tuple[nn.Module, Dict]:
    """Fine-tune the regional model for a specific basin.
    
    Parameters
    ----------
    base_model : nn.Module
        Pre-trained regional model
    basin : str
        Basin identifier
    run_cfg : Dict
        Run configuration
    user_cfg : Dict
        User configuration
    means : pd.Series
        Attribute means
    stds : pd.Series
        Attribute stds
    db_path : str
        Path to attributes database
        
    Returns
    -------
    basin_model : nn.Module
        Fine-tuned model for this basin
    scalers : Dict
        Global normalization scalers (matching regional training)
    """
    import copy
    
    # Define global scalers matching regional training
    global_scalers = {
        'q_mean': SCALER['output_mean'][0],
        'q_std': SCALER['output_std'][0],
        'dynamic_means': SCALER['input_means'],
        'dynamic_stds': SCALER['input_stds']
    }
    
    # Create fine-tuning dataset using calibration period with global scalers
    ft_dataset = FineTuneDataset(
        basin=basin,
        dates=[GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]],
        seq_length=run_cfg["seq_length"],
        camels_root=user_cfg["camels_root"],
        concat_static=run_cfg["concat_static"],
        db_path=db_path,
        attribute_means=means,
        attribute_stds=stds,
        no_static=run_cfg["no_static"],
        global_scalers=global_scalers
    )
    
    if len(ft_dataset) == 0:
        return None, None
    
    # Create data loader
    ft_loader = DataLoader(ft_dataset, batch_size=256, shuffle=True, 
                          collate_fn=custom_collate_ft, num_workers=2)
    
    # Deep copy the model and prepare for fine-tuning
    basin_model = copy.deepcopy(base_model).to(DEVICE)
    basin_model.train()
    
    # Use lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(basin_model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    
    # Fine-tune for 5 epochs
    for epoch in range(5):
        for data in ft_loader:
            optimizer.zero_grad()
            
            if len(data) == 4:
                # EA-LSTM case
                x_d, x_s, y, _dates = data
                x_d = x_d.to(DEVICE)
                x_s = x_s.to(DEVICE)
                y = y.to(DEVICE)
                predictions = basin_model(x_d, x_s)[0]
            else:
                # LSTM case
                x, y, _dates = data
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                predictions = basin_model(x)[0]
            
            loss = loss_func(predictions, y)
            loss.backward()
            optimizer.step()
    
    basin_model.eval()
    return basin_model, ft_dataset.scalers


def evaluate_adapted_basin(basin_model: nn.Module, basin: str, dates: List,
                           run_cfg: Dict, user_cfg: Dict, means: pd.Series, 
                           stds: pd.Series, db_path: str, scalers: Dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """Evaluate fine-tuned model on a basin.
    
    Parameters
    ----------
    basin_model : nn.Module
        Fine-tuned model
    basin : str
        Basin identifier
    dates : List
        Start and end dates
    run_cfg : Dict
        Run configuration
    user_cfg : Dict
        User configuration
    means : pd.Series
        Attribute means
    stds : pd.Series
        Attribute stds
    db_path : str
        Path to attributes database
    scalers : Dict
        Global normalization scalers (matching regional training)
        
    Returns
    -------
    predictions : np.ndarray
        Model predictions
    observations : np.ndarray
        Observed values
    eval_dates : List
        Dates for predictions
    """
    # Create evaluation dataset with global scalers
    eval_dataset = FineTuneDataset(
        basin=basin,
        dates=dates,
        seq_length=run_cfg["seq_length"],
        camels_root=user_cfg["camels_root"],
        concat_static=run_cfg["concat_static"],
        db_path=db_path,
        attribute_means=means,
        attribute_stds=stds,
        no_static=run_cfg["no_static"],
        global_scalers=scalers
    )
    
    if len(eval_dataset) == 0:
        return None, None, None
    
    eval_loader = DataLoader(eval_dataset, batch_size=2048, shuffle=False,
                            collate_fn=custom_collate_ft, num_workers=4)
    
    predictions = []
    observations = []
    eval_dates = []
    
    with torch.no_grad():
        for data in eval_loader:
            if len(data) == 4:
                # EA-LSTM case
                x_d, x_s, y_scaled, dates_batch = data
                x_d = x_d.to(DEVICE)
                x_s = x_s.to(DEVICE)
                preds_scaled = basin_model(x_d, x_s)[0].cpu().numpy().flatten()
            else:
                # LSTM case
                x, y_scaled, dates_batch = data
                x = x.to(DEVICE)
                preds_scaled = basin_model(x)[0].cpu().numpy().flatten()
            
            # Rescale predictions using basin-specific scalers
            preds_rescaled = (preds_scaled * scalers['q_std']) + scalers['q_mean']
            preds_rescaled[preds_rescaled < 0] = 0
            
            # Rescale observations
            y_numpy = y_scaled.numpy().flatten()
            obs_rescaled = (y_numpy * scalers['q_std']) + scalers['q_mean']
            
            predictions.extend(preds_rescaled)
            observations.extend(obs_rescaled)
            eval_dates.extend(dates_batch)
    
    return np.array(predictions), np.array(observations), eval_dates


def adapt(user_cfg: Dict):
    """Adapt regional model to individual basins through fine-tuning.
    
    This function loads a pre-trained regional model and fine-tunes it
    for each basin individually using that basin's training data.
    The fine-tuned models are then evaluated on both calibration and
    evaluation periods, as well as synthetic scenarios.
    
    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    """
    import pickle
    
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    loc = extract_loc_from_path(run_cfg["camels_root"])
    basins = get_basin_list(loc=loc)
    
    # Filter basins to only use those in the basin_ids list
    basin_ids_file = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/small_basins_1000.csv"
    if os.path.exists(basin_ids_file):
        basin_ids_df = pd.read_csv(basin_ids_file, index_col=None, header=0)
        basin_ids = basin_ids_df["basin_id"].tolist()
        basin_ids_str = [str(b).zfill(3) for b in basin_ids]
        basins = [b for b in basins if b in basin_ids_str]
        print(f"Filtered to {len(basins)} basins from basin_ids list")
    else:
        print(f"Basin IDs file not found at {basin_ids_file}, using all basins")

    # Get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # Create base model
    input_size_stat = 0 if run_cfg["no_static"] else 18
    input_size_dyn = 3 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 21
    base_model = Model(input_size_dyn=input_size_dyn,
                       input_size_stat=input_size_stat,
                       hidden_size=run_cfg["hidden_size"],
                       dropout=run_cfg["dropout"],
                       concat_static=run_cfg["concat_static"],
                       no_static=run_cfg["no_static"]).to(DEVICE)

    # Load trained regional model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt'
    base_model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    # Create hyperparameter string for filenames
    hs = run_cfg["hidden_size"]
    do = run_cfg["dropout"]
    ep = run_cfg["epochs"]
    hp_str = f"hs{hs}_do{do}_e{ep}"
    
    # Set up output directories
    output_base = f"/data0/funato/hsp2/0_out/LSTM/{loc}/adapted"
    output_predict_dir = os.path.join(output_base, "predict")
    output_results_dir = os.path.join(output_base, "results")
    os.makedirs(output_predict_dir, exist_ok=True)
    os.makedirs(output_results_dir, exist_ok=True)
    
    # Directory for saving fine-tuned models
    adapted_models_dir = Path(f"/data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/runs_adapted/{loc}")
    adapted_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Fine-tune models for each basin
    print("\n" + "="*60)
    print("Fine-tuning models for each basin...")
    print("="*60)
    
    basin_models = {}
    basin_scalers = {}
    
    for basin in tqdm(basins, desc="Fine-tuning basins"):
        # Check if already fine-tuned
        model_path = adapted_models_dir / f"basin_{basin}_model.pt"
        scalers_path = adapted_models_dir / f"basin_{basin}_scalers.pkl"
        
        if model_path.exists() and scalers_path.exists():
            # Load existing fine-tuned model
            import copy
            basin_model = copy.deepcopy(base_model)
            basin_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            basin_model.eval()
            
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
        else:
            # Fine-tune and save
            basin_model, scalers = fine_tune_basin_model(
                base_model, basin, run_cfg, user_cfg, means, stds, db_path
            )
            
            if basin_model is not None:
                torch.save(basin_model.state_dict(), model_path)
                with open(scalers_path, 'wb') as f:
                    pickle.dump(scalers, f)
        
        if basin_model is not None:
            basin_models[basin] = basin_model
            basin_scalers[basin] = scalers
    
    print(f"Successfully fine-tuned {len(basin_models)} basins")
    
    # Evaluate on calibration and evaluation periods
    periods = {
        'cal': [GLOBAL_SETTINGS["train_start"], GLOBAL_SETTINGS["train_end"]],
        'eva': [GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]]
    }
    
    for period_name, period_dates in periods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {period_name.upper()} period: {period_dates[0]} to {period_dates[1]}")
        print(f"{'='*60}")
        
        date_range = pd.date_range(start=period_dates[0], end=period_dates[1])
        
        all_predictions = {}
        all_metrics = []
        
        for basin in tqdm(basins, desc=f"{period_name.upper()} period"):
            if basin not in basin_models:
                continue
            
            preds, obs, eval_dates = evaluate_adapted_basin(
                basin_models[basin], basin, period_dates, run_cfg, user_cfg,
                means, stds, db_path, basin_scalers[basin]
            )
            
            if preds is not None:
                # Store predictions
                all_predictions[basin] = preds
                
                # Calculate metrics
                valid_mask = obs >= 0
                obs_valid = obs[valid_mask]
                preds_valid = preds[valid_mask]
                
                if len(obs_valid) > 0:
                    try:
                        nse = calc_nse(obs_valid, preds_valid)
                        kge = calc_kge(obs_valid, preds_valid)
                        e1 = calc_lognse(obs_valid, preds_valid)
                        ve = calc_pbias(obs_valid, preds_valid)
                        d = calc_agreement_index(obs_valid, preds_valid)
                        rmse = calc_rmse(obs_valid, preds_valid)
                        mae = calc_mae(obs_valid, preds_valid)
                    except Exception as e:
                        print(f"Warning: Error calculating metrics for basin {basin}: {e}")
                        nse = kge = e1 = ve = d = rmse = mae = np.nan
                else:
                    nse = kge = e1 = ve = d = rmse = mae = np.nan
                
                basin_int = int(basin)
                all_metrics.append({
                    'basin': basin_int,
                    'NSE': nse,
                    'KGE': kge,
                    'E1': e1,
                    'VE': ve,
                    'D': d,
                    'RMSE': rmse,
                    'MAE': mae
                })
        
        # Save predictions to CSV
        predict_filename = f"LSTM_predict_{hp_str}_{period_name}.csv"
        predict_filepath = os.path.join(output_predict_dir, predict_filename)
        
        predict_df_data = {'basin': [int(b) for b in basins if b in all_predictions]}
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            predict_df_data[date_str] = [
                all_predictions[basin][i] if basin in all_predictions and i < len(all_predictions[basin]) else np.nan
                for basin in basins if basin in all_predictions
            ]
        
        predict_df = pd.DataFrame(predict_df_data)
        predict_df.to_csv(predict_filepath, index=False)
        print(f"Saved predictions to: {predict_filepath}")
        
        # Save metrics to CSV
        if all_metrics:
            results_filename = f"LSTM_results_{hp_str}_{period_name}.csv"
            results_filepath = os.path.join(output_results_dir, results_filename)
            
            results_df = pd.DataFrame(all_metrics)
            results_df = results_df.sort_values('basin')
            results_df.to_csv(results_filepath, index=False)
            print(f"Saved results to: {results_filepath}")
            
            print(f"\n{period_name.upper()} Period Summary:")
            print(f"  Mean NSE: {results_df['NSE'].mean():.3f}")
            print(f"  Mean KGE: {results_df['KGE'].mean():.3f}")
            print(f"  Mean RMSE: {results_df['RMSE'].mean():.3f}")
    
    # Evaluate on synthetic scenarios
    print("\n" + "="*60)
    print("Evaluating Synthetic Scenarios (Adapted Models)")
    print("="*60)
    
    dates = [GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]]
    date_range = pd.date_range(start=dates[0], end=dates[1])
    
    output_synthetic_dir = os.path.join(output_predict_dir, "synthetic")
    os.makedirs(output_synthetic_dir, exist_ok=True)
    
    for scenario_col in syn_precip_cols:
        print(f"\nProcessing scenario: {scenario_col}")
        
        all_predictions = {}
        
        for basin in tqdm(basins, desc=f"Basins for {scenario_col}"):
            if basin not in basin_models:
                continue
            
            try:
                # Load synthetic data using the same approach as evaluation
                result = load_synthetic_data(
                    basin, scenario_col, dates, run_cfg["seq_length"],
                    user_cfg["camels_root"], means, stds,
                    run_cfg["concat_static"], db_path
                )
                
                if result[0] is None:
                    continue
                
                x, attributes, prediction_dates = result
                
                basin_model = basin_models[basin]
                scalers = basin_scalers[basin]
                
                with torch.no_grad():
                    x = x.to(DEVICE)
                    
                    if run_cfg["no_static"] or run_cfg["concat_static"]:
                        if run_cfg["concat_static"]:
                            attributes_expanded = attributes.repeat(x.shape[0], run_cfg["seq_length"], 1)
                            x = torch.cat([x, attributes_expanded.to(DEVICE)], dim=-1)
                        predictions = basin_model(x)[0]
                    else:
                        attributes_expanded = attributes.unsqueeze(0).repeat(x.shape[0], 1, 1).to(DEVICE)
                        predictions = basin_model(x, attributes_expanded[:, 0, :])[0]
                    
                    predictions = predictions.detach().cpu().numpy()
                    
                    # Rescale using basin-specific scalers
                    predictions = (predictions * scalers['q_std']) + scalers['q_mean']
                    predictions[predictions < 0] = 0
                
                all_predictions[basin] = predictions.flatten()
                
            except Exception as e:
                print(f"Error processing basin {basin} for scenario {scenario_col}: {e}")
                continue
        
        if not all_predictions:
            print(f"No valid predictions for scenario {scenario_col}")
            continue
        
        # Save synthetic predictions
        predict_filename = f"LSTM_predict_{hp_str}_{scenario_col}_eva.csv"
        predict_filepath = os.path.join(output_synthetic_dir, predict_filename)
        
        predict_df_data = {'basin': [int(b) for b in sorted(all_predictions.keys(), key=int)]}
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            predict_df_data[date_str] = [
                all_predictions[basin][i] if basin in all_predictions and i < len(all_predictions[basin]) else np.nan
                for basin in sorted(all_predictions.keys(), key=int)
            ]
        
        predict_df = pd.DataFrame(predict_df_data)
        predict_df.to_csv(predict_filepath, index=False)
        print(f"Saved synthetic predictions to: {predict_filepath}")
    
    print("\n" + "="*60)
    print("Adaptation evaluation complete!")
    print("="*60)


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

    loc = extract_loc_from_path(run_cfg["camels_root"])
    basins = get_basin_list(loc=loc)
    
    # Filter basins to only use those in the basin_ids list
    basin_ids_file = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/small_basins_1000.csv"
    if os.path.exists(basin_ids_file):
        basin_ids_df = pd.read_csv(basin_ids_file, index_col=None, header=0)
        basin_ids = basin_ids_df["basin_id"].tolist()
        # Convert to string format with leading zeros
        basin_ids_str = [str(b).zfill(3) for b in basin_ids]
        # Filter basins to only include those in basin_ids
        basins = [b for b in basins if b in basin_ids_str]
        print(f"Filtered to {len(basins)} basins from basin_ids list")
    else:
        print(f"Basin IDs file not found at {basin_ids_file}, using all basins")

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # initialize Model
    model = Model(input_size_dyn=3,
                  input_size_stat=18,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"]).to(DEVICE)
    weight_file = user_cfg["run_dir"] / "model_epoch30.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    overall_results = {}
    # process bar handle
    pbar = tqdm(basins, file=sys.stdout)
    for basin in pbar:
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
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
                noise = np.random.normal(loc=0, scale=scale, size=18).astype(np.float32)
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
        preds = rescale_features(preds.numpy(), variable='output')

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
