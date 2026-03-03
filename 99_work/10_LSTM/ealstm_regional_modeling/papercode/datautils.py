"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit
import os

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

VALID_ATTR = [
    'p_mean', 'pet_mean', 'aridity', 'seasonality', 'frac_snow', 'high_prec_freq',
    'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'ele_mt_smn', 'slp_dg_sav', 'area',
    'for_pc_sse', 'snd_pc_sav', 'slt_pc_sav', 'cly_pc_sav','soc_th_sav', 'swc_pc_syr']

# Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
        'input_means': np.array([3.0267525200819776, 13.110970171836367, 10.81853041139944]),
        'input_stds': np.array([1.085855688036388, 5.04530884159448, 5.45962822094668]),
        'output_mean': np.array([1.6033382557458762]),
        'output_std': np.array([1.6091810953328294])
}

def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files or CSV and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder/file could not be found.
    """
    # Determine location from path
    camels_root_str = str(camels_root).upper()
    if 'GB' in camels_root_str or '_GB' in camels_root_str:
        loc = 'GB'
    elif 'JP' in camels_root_str or '_JP' in camels_root_str:
        loc = 'JP'
    else:
        loc = 'US'
    
    # Try to find combined CSV file first
    combined_csv_path = Path(f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/camels_attribute_combined.csv")
    
    if combined_csv_path.exists():
        # Load from combined CSV file
        print(f"Loading attributes from combined CSV: {combined_csv_path}")
        df = pd.read_csv(combined_csv_path)
        
        # Use File_num as index (basin numbers like 1, 2, 3, ...)
        if 'File_num' in df.columns:
            df = df.set_index('File_num')
            # Convert to 3-digit string format (001, 002, etc.)
            df.index = df.index.astype(str).str.zfill(3)
        else:
            # If no File_num column, use first column
            df = df.set_index(df.columns[0])
            df.index = df.index.astype(str).str.zfill(3)
        
        # Drop gauge_id column if it exists to avoid duplicate column name
        if 'gauge_id' in df.columns:
            df = df.drop('gauge_id', axis=1)
        
        df.index.name = 'gauge_id'
        
    else:
        # Fall back to original CAMELS format
        attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

        if not attributes_path.exists():
            raise RuntimeError(f"Attribute folder not found at {attributes_path} and combined CSV not found at {combined_csv_path}")

        txt_files = attributes_path.glob('camels_*.txt')

        # Read-in attributes into one big dataframe
        df = None
        for f in txt_files:
            df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')

            if df is None:
                df = df_temp.copy()
            else:
                df = pd.concat([df, df_temp], axis=1)

        # convert huc column to double digit strings
        if 'huc_02' in df.columns:
            df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
            df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn, if_exists='replace')

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    drop_lat_lon: bool = True,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    elif VALID_ATTR is not None:
        drop_names = [c for c in df.columns if c not in VALID_ATTR]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_data(file_num, varssim_dir, loc):
    file_path = f"{varssim_dir}/varssim{file_num:03d}.csv"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Skipping this file.")
        return pd.DataFrame()  # Return an empty DataFrame to handle missing files gracefully

    df = pd.read_csv(file_path)
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df

