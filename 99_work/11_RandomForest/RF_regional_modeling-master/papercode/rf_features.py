"""
Shared, vectorized dataset construction for the regional Random Forest scripts
(main_rf.py / main_rf_hp.py).

The feature matrix (X, y, sample_info, per-basin scalers) depends only on
(loc, ver, seq_length, train_start, train_end, no_static) -- it does NOT depend
on any Random Forest hyperparameter. main_rf_hp.py is invoked once per
hyperparameter combination (up to 81 times in parallel by run_rf_hp_tuning.sh),
so building this dataset from scratch in every process is the dominant cost of
hyperparameter tuning. This module builds it once and caches it to disk so
every hyperparameter job just loads the cached arrays.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DYN_COLS = ['Precip', 'PET', 'Temp']
Q_COL = 'Obs flow'


def _varssim_path(loc: str, gauge_id: str) -> str:
    return f"/data0/funato/hsp2/2_MERV/varssim_{loc}/varssim_{gauge_id}.csv"


def load_data_for_basin(loc: str, gauge_id: str) -> pd.DataFrame:
    df = pd.read_csv(_varssim_path(loc, gauge_id))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def _clean_blocks(df_period: pd.DataFrame, cols: List[str]) -> List[slice]:
    """Return index slices of contiguous rows with no NaNs in `cols`."""
    is_nan = df_period[cols].isnull().any(axis=1)
    clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
    if not isinstance(clumps, list):
        clumps = [clumps] if clumps else []
    return clumps


def compute_basin_scalers(cfg: Dict, basins: List[str]) -> Dict:
    """Per-basin mean/std of dynamic inputs and streamflow over the training period."""
    basin_scalers = {}
    for basin in basins:
        df = load_data_for_basin(cfg['loc'], basin)
        df_period = df.loc[cfg['train_start']:cfg['train_end']]

        clumps = _clean_blocks(df_period, DYN_COLS + [Q_COL])
        clean_blocks = [df_period.iloc[s] for s in clumps]
        if not clean_blocks:
            continue

        basin_clean_df = pd.concat(clean_blocks)
        basin_scalers[basin] = {
            'q_mean': basin_clean_df[Q_COL].mean(),
            'q_std': basin_clean_df[Q_COL].std(),
            'dynamic_means': basin_clean_df[DYN_COLS].mean().values,
            'dynamic_stds': basin_clean_df[DYN_COLS].std().values,
        }
    return basin_scalers


def _windowed_dynamic_features(block_dyn_scaled: np.ndarray, seq_length: int) -> np.ndarray:
    """Vectorized replacement for the per-sample Python loop + np.concatenate.

    block_dyn_scaled: (block_len, n_dyn_features)
    returns: (block_len - seq_length + 1, seq_length * n_dyn_features), where
    row i is block_dyn_scaled[i:i+seq_length].flatten() -- identical ordering
    to the original nested-loop implementation.
    """
    n_features = block_dyn_scaled.shape[1]
    windows = sliding_window_view(block_dyn_scaled, window_shape=(seq_length, n_features))
    # shape: (n_windows, 1, seq_length, n_features) -> drop the redundant axis
    windows = windows[:, 0, :, :]
    return windows.reshape(windows.shape[0], -1)


def build_feature_matrix(cfg: Dict, basins: List[str], basin_scalers: Dict,
                          attributes: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Vectorized dataset construction. Returns X (float32), y (float32), sample_info (DataFrame)."""
    seq_length = cfg['seq_length']
    no_static = cfg['no_static']

    X_chunks = []
    y_chunks = []
    basin_col_chunks = []
    date_chunks = []

    for basin in basins:
        if basin not in basin_scalers:
            continue

        df = load_data_for_basin(cfg['loc'], basin)
        df_period = df.loc[cfg['train_start']:cfg['train_end']]
        bscaler = basin_scalers[basin]

        dyn_data_scaled = (df_period[DYN_COLS].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
        q_data_scaled = (df_period[Q_COL].values - bscaler['q_mean']) / bscaler['q_std']

        static_data_scaled = None
        if not no_static:
            static_data = attributes.loc[basin].values
            static_data_scaled = (static_data - attributes.attrs['static_means']) / attributes.attrs['static_stds']

        clumps = _clean_blocks(df_period, DYN_COLS + [Q_COL])

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < seq_length:
                continue

            block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
            block_q_scaled = q_data_scaled[start_loc:end_loc]

            dyn_seq_flat_all = _windowed_dynamic_features(block_dyn_scaled, seq_length)
            n_windows = dyn_seq_flat_all.shape[0]

            if not no_static:
                static_tiled = np.tile(static_data_scaled, (n_windows, 1))
                block_X = np.concatenate([dyn_seq_flat_all, static_tiled], axis=1)
            else:
                block_X = dyn_seq_flat_all

            block_y = block_q_scaled[seq_length - 1:]
            block_dates = df_period.index[start_loc + seq_length - 1:end_loc]

            X_chunks.append(block_X.astype(np.float32, copy=False))
            y_chunks.append(block_y.astype(np.float32, copy=False))
            basin_col_chunks.append(np.full(n_windows, basin))
            date_chunks.append(block_dates.values)

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    sample_info = pd.DataFrame({
        'basin': np.concatenate(basin_col_chunks),
        'date': np.concatenate(date_chunks),
    })
    return X, y, sample_info


def _cache_key(cfg: Dict, basins: List[str]) -> str:
    key_fields = {
        'loc': cfg['loc'],
        'ver': cfg['ver'],
        'seq_length': cfg['seq_length'],
        'train_start': cfg['train_start'].strftime('%Y-%m-%d'),
        'train_end': cfg['train_end'].strftime('%Y-%m-%d'),
        'no_static': cfg['no_static'],
        'basins': sorted(basins),
    }
    blob = json.dumps(key_fields, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _atomic_write_bytes(path: Path, write_fn):
    """Write via a temp file in the same directory then os.replace, so concurrent
    readers never see a partially-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.tmp')
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_or_build_dataset(cfg: Dict, basins: List[str], cache_root: Path,
                           attributes: pd.DataFrame = None):
    """Returns (X, y, sample_info, basin_scalers, static_means, static_stds).

    If a cached copy for this exact (loc, ver, seq_length, train period,
    no_static, basin list) already exists, loads it. Otherwise builds it and
    caches it for the next hyperparameter job to reuse.
    """
    cache_root = Path(cache_root)
    key = _cache_key(cfg, basins)
    cache_dir = cache_root / key
    x_path = cache_dir / 'X.npy'
    y_path = cache_dir / 'y.npy'
    info_path = cache_dir / 'sample_info.parquet'
    scalers_path = cache_dir / 'scalers.json'
    done_marker = cache_dir / '_DONE'

    if done_marker.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        sample_info = pd.read_parquet(info_path)
        with open(scalers_path) as fp:
            scalers = json.load(fp)
        basin_scalers = scalers['basin_scalers']
        static_means = np.array(scalers['static_means']) if 'static_means' in scalers else None
        static_stds = np.array(scalers['static_stds']) if 'static_stds' in scalers else None
        return X, y, sample_info, basin_scalers, static_means, static_stds

    print(f"[rf_features] No cache at {cache_dir}, building dataset (this happens once)...")
    basin_scalers = compute_basin_scalers(cfg, basins)

    static_means = static_stds = None
    if not cfg['no_static']:
        static_means = attributes.mean().values
        static_stds = attributes.std().values
        attributes.attrs['static_means'] = static_means
        attributes.attrs['static_stds'] = static_stds

    X, y, sample_info = build_feature_matrix(cfg, basins, basin_scalers, attributes)

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    scalers_out = {'basin_scalers': convert_for_json(basin_scalers)}
    if static_means is not None:
        scalers_out['static_means'] = static_means.tolist()
        scalers_out['static_stds'] = static_stds.tolist()

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_x = str(x_path) + '.tmp.npy'
    np.save(tmp_x, X)
    os.replace(tmp_x, x_path)

    tmp_y = str(y_path) + '.tmp.npy'
    np.save(tmp_y, y)
    os.replace(tmp_y, y_path)

    _atomic_write_bytes(info_path, lambda p: sample_info.to_parquet(p))
    _atomic_write_bytes(scalers_path, lambda p: open(p, 'w').write(json.dumps(scalers_out, indent=2)))

    done_marker.touch()
    print(f"[rf_features] Cached dataset to {cache_dir} (X shape {X.shape})")

    return X, y, sample_info, basin_scalers, static_means, static_stds
