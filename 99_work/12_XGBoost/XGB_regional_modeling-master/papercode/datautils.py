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

SCALER = {
    "JP": {
        'input_means': np.array([4.749130900644258, 9.637227892152021, 1.9068619713226833]),
        'input_stds': np.array([12.247018935981416, 9.750107710636184, 1.3638978668850177]),
        'output_mean': np.array([3.8526752089320553]),
        'output_std': np.array([7.071976225365726])
    },
    "GB": {
        'input_means': np.array([2.6250822275167094, 9.026790815449628, 3.075126879987041]),
        'input_stds': np.array([4.150642797674723, 5.221306962850592, 2.1611745017309048]),
        'output_mean': np.array([1.838397528976445]),
        'output_std': np.array([3.2740873977972185])
    }
}

# varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"


def add_basin_attributes(db_path: str = None, loc :str=None, ver_name:str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """

    attribute_dir = f'/data0/funato/hsp/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

    if loc == "JP":
        attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
        INVALID_ATTR = [
            'grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio'
        ]
    elif loc == "GB":
        attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8', index_col=0, header=0)
        INVALID_ATTR = [
            'gauge_id','gauge_name','country'
        ]


    # df : file_num, attr1, attr2, attr3, ...
    df = attribute_values
    df = df.drop(columns=[col for col in INVALID_ATTR if col in df.columns])

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into database
        df.to_sql('basin_attributes', conn, if_exists='replace')

    print(f"Successfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    drop_lat_lon: bool = True,
                    keep_features: List = None,
                    loc: str = "GB") -> pd.DataFrame:
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
    if loc == "JP":
        INVALID_ATTR = [
            'grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio'
        ]
    elif loc == "GB":
        INVALID_ATTR = [
            'gauge_id','gauge_name','country'
        ]

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='File_num')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str, loc: str = None) -> np.ndarray:
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
    if loc not in SCALER:
        raise RuntimeError(f"Unknown location {loc}")

    if variable == 'inputs':
        feature = (feature - SCALER[loc]["input_means"]) / SCALER[loc]["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER[loc]["output_mean"]) / SCALER[loc]["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str, loc: str = None) -> np.ndarray:
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
    if loc not in SCALER:
        raise RuntimeError(f"Unknown location {loc}")

    if variable == 'inputs':
        feature = feature * SCALER[loc]["input_stds"] + SCALER[loc]["input_means"]
    elif variable == 'output':
        feature = feature * SCALER[loc]["output_std"] + SCALER[loc]["output_mean"]
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


def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, varssim_dir, loc):
    file_path = f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv"
    
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
