from pathlib import Path
from typing import Tuple, List

import gcsfs
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import os

# Globals
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available

@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(self, file_num: str, seq_length: int=365,period: str=None,
                 dates: List=None, means: pd.Series=None, stds: pd.Series=None, varssim_dir: str=None, loc: str=None):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the 
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date 
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.file_num = file_num
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds
        self.varssim_dir = varssim_dir
        self.loc = loc

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load input and output data from text files."""
        df = load_data(self.file_num, self.varssim_dir, self.dates[0],self.dates[1],self.loc)

        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        x = np.array([df['Precip'].values,
                      df['Temp'].values,
                      df['PET'].values]).T #(days, 3)


        y = np.array([df['Obs flow'].values]).T #(days, 1)

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.file_num}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
            
            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            
            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['Precip'],
                              self.means['Temp'],
                              self.means['PET']])
            stds = np.array([self.stds['Precip'],
                             self.stds['Temp'],
                             self.stds['PET']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["Obs flow"]) /
                       self.stds["Obs flow"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['Precip'],
                              self.means['Temp'],
                              self.means['PET']])
            stds = np.array([self.stds['Precip'],
                             self.stds['Temp'],
                             self.stds['PET']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["Obs flow"] +
                       self.means["Obs flow"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds


class Model(nn.Module):
    """Implementation of a single layer LSTM network"""
    
    def __init__(self, hidden_size: int, dropout_rate: float=0.0):
        """Initialize model
        
        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # create required layer
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, 
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.
        
        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.
        
        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred
    

def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        
def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
            
    return torch.cat(obs), torch.cat(preds)
        
def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(obs, torch.Tensor):
        obs = obs.numpy()
    if isinstance(sim, torch.Tensor):
        sim = sim.numpy()

    # Ensure obs and sim have the same length
    min_length = min(len(obs), len(sim))
    obs = obs[:min_length]
    sim = sim[:min_length]

    # Remove invalid values (NaN or infinite)
    valid_mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid_mask]
    sim = sim[valid_mask]

    # Check if there is sufficient data
    if len(obs) == 0 or len(sim) == 0:
        print("Warning: No valid data points available for NSE calculation.")
        return float('nan')

    # Only consider time steps where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # Check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    # Calculate NSE
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    if denominator == 0:
        print("Warning: Denominator is zero during NSE calculation.")
        return float('nan')
    nse_val = 1 - numerator / denominator

    return nse_val


def BMK(obs_data, sim_data, benchmark): 
    # create mask to filter out invalid values for obs_data and sim_data
    valid_mask = np.isfinite(obs_data) & np.isfinite(sim_data)
    obs_data = obs_data[valid_mask]
    sim_data = sim_data[valid_mask]

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        # Handle invalid or zero values in obs_data and sim_data
        obs_data = np.where(np.isnan(obs_data) | np.isinf(obs_data), 1e-6, obs_data)
        sim_data = np.where(np.isnan(sim_data) | np.isinf(sim_data), 1e-6, sim_data)

        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)

        # Avoid division by zero
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave)

        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "logNSE":
        temp_obs_data = np.where((obs_data <= 0) | np.isnan(obs_data), 1e-6, obs_data)
        temp_sim_data = np.where((sim_data <= 0) | np.isnan(sim_data), 1e-6, sim_data)
        obs_ave = np.mean(temp_obs_data)
        numer = np.sum((np.log(temp_sim_data) - np.log(temp_obs_data))**2)
        denom = np.sum((np.log(temp_obs_data) - np.log(obs_ave))**2)
        return 1 - numer / denom
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "Erel":
        obs_ave = np.mean(obs_data)
        temp_obs_data = np.where(obs_data == 0, 1e-6, obs_data)
        numer = np.sum(np.square((temp_obs_data - sim_data) / temp_obs_data))
        denom = np.sum(np.square((obs_data - obs_ave) / obs_ave))
        return 1 - numer / denom
    
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))
    
def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, varssim_dir, start_date_cal, end_date_eva, loc):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df
