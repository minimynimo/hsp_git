"""
XGBoost Regional Hydrological Model Evaluation

This script evaluates a trained XGBoost model using the same per-basin
scaling pipeline as the training script to ensure consistency.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from papercode.datautils import VALID_ATTR
from papercode.utils import get_basin_list

sys.path.insert(0, "/data0/funato/hsp2/99_work/00_analysis/data")
from ml_predict_long_format import save_regional_predictions_long_format

# ####################### #
# Global Definitions      #
# ####################### #

syn_precip_cols = ['syn_no_precip', 'syn_p_pet',
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

GLOBAL_SETTINGS = {
    'loc': 'US',
    'ver': 2,
    'seq_length': 270,
}

# Per-dataset calibration/evaluation period, matching main_xgb.py's DATASET_SETTINGS. A trained
# run's own cfg.json train_start/train_end/val_start/val_end (present for any run trained after
# --dataset support was added) always takes precedence -- see _run_dates() below.
DATASET_SETTINGS = {
    'camels': {
        'train_start': pd.to_datetime('1987-01-01', format='%Y-%m-%d'),
        'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
        'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
        'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d'),
    },
    'caravan': {
        'train_start': pd.to_datetime('1990-01-01', format='%Y-%m-%d'),
        'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
        'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
        'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d'),
    },
}


def _run_dates(run_cfg):
    dataset = run_cfg.get('dataset', 'camels')
    fallback = DATASET_SETTINGS[dataset]
    return {
        'train_start': pd.to_datetime(run_cfg.get('train_start', fallback['train_start'])),
        'train_end': pd.to_datetime(run_cfg.get('train_end', fallback['train_end'])),
        'val_start': pd.to_datetime(run_cfg.get('val_start', fallback['val_start'])),
        'val_end': pd.to_datetime(run_cfg.get('val_end', fallback['val_end'])),
    }


benchmark_list = ["NSE", "KGE", "E1", "VE", "d", "RMSE", "MAE"]

# ####################### #
# Utility Functions       #
# ####################### #

def load_data_for_basin(gauge_id, dataset='camels'):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def load_attributes(db_path: str, basins: list, drop_lat_lon: bool = True, keep_features: list = None) -> pd.DataFrame:
    # NOTE: this local wrapper used to silently drop the keep_features kwarg that main() passes it
    # (TypeError: unexpected keyword argument), meaning the no_static=False path here had never
    # actually been exercised end-to-end -- same class of bug as TCN's dropped-kwargs wrapper
    # functions, see DEBUG_README.md.
    from papercode.datautils import load_attributes as load_attributes_from_lib
    return load_attributes_from_lib(db_path, basins, drop_lat_lon, keep_features=keep_features)

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
# Evaluation Logic        #
# ####################### #

def evaluate_period(model, basins, cfg, scalers, attributes, start_date, end_date):
    """Evaluate the XGBoost model over a specific period using per-basin scalers."""
    print(f"\nPreparing data for period: {start_date.date()} to {end_date.date()}")
    
    X_list, y_list, basin_list, date_list = [], [], [], []
    dyn_cols = ['Precip', 'PET', 'Temp']
    q_col = 'Obs flow'
    seq_length = cfg['seq_length']

    for basin in tqdm(basins, desc="Processing basins"):
        if basin not in scalers['basin_scalers']:
            continue

        df = load_data_for_basin(basin, cfg.get('dataset', 'camels'))
        warmup_start = start_date - pd.DateOffset(days=seq_length - 1)
        df_period = df.loc[warmup_start:end_date]

        # Use per-basin scaler (must match training convention in main_xgb.py)
        bscaler = scalers['basin_scalers'][basin]
        dyn_data_scaled = (df_period[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']

        q_data_raw = df_period[q_col].values
        
        static_data_scaled = None
        if not cfg["no_static"]:
            static_data = attributes.loc[basin].values
            # Static attributes use global scaler
            static_data_scaled = (static_data - scalers['static_means']) / scalers['static_stds']
        
        is_nan = df_period[dyn_cols + [q_col]].isnull().any(axis=1)
        
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < seq_length: continue

            block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
            block_q_raw = q_data_raw[start_loc:end_loc]
            
            for i in range(block_len - seq_length + 1):
                target_idx_in_block = i + seq_length - 1
                current_date = df_period.index[start_loc + target_idx_in_block]
                if current_date < start_date: continue

                dyn_seq_flat = block_dyn_scaled[i:target_idx_in_block + 1].flatten()
                
                final_features = dyn_seq_flat
                if not cfg["no_static"]:
                    final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
                
                X_list.append(final_features)
                y_list.append(block_q_raw[target_idx_in_block])
                basin_list.append(basin)
                date_list.append(current_date)

    if not X_list:
        print("No data to evaluate for this period.")
        return None, None

    X_eval = np.array(X_list, dtype=np.float32)
    y_eval_raw = np.array(y_list, dtype=np.float32)
    
    print("Making predictions...")
    predictions_scaled = model.predict(X_eval)
    
    # Rescale predictions using the correct per-basin scaler for each sample
    results_df = pd.DataFrame({
        'date': date_list, 'basin': basin_list,
        'pred_scaled': predictions_scaled, 'observation': y_eval_raw
    })

    def rescale_row(row):
        bscaler = scalers['basin_scalers'][row['basin']]
        return (row['pred_scaled'] * bscaler['q_std']) + bscaler['q_mean']

    results_df['prediction'] = results_df.apply(rescale_row, axis=1)
    results_df['prediction'] = results_df['prediction'].clip(lower=0)

    # Compute metrics per basin
    metrics_list = []
    for basin_id, group in results_df.groupby('basin'):
        obs_valid = group['observation'].values
        pred_valid = group['prediction'].values
        basin_metrics = {'basin': basin_id}
        for benchmark in benchmark_list:
            basin_metrics[benchmark] = BMK(obs_valid, pred_valid, benchmark)
        metrics_list.append(basin_metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.set_index('basin')
    return results_df, metrics_df

def evaluate_synthetic_period(model, basins, cfg, scalers, attributes, scenario_col, start_date, end_date):
    """Evaluate the XGBoost model on synthetic data for a specific scenario."""
    print(f"\nProcessing synthetic scenario: {scenario_col}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    X_list, basin_list, date_list = [], [], []
    dyn_cols = ['Precip', 'PET', 'Temp']
    seq_length = cfg['seq_length']

    for basin in tqdm(basins, desc=f"Processing basins for {scenario_col}"):
        if basin not in scalers['basin_scalers']:
            continue

        df = load_data_for_basin(basin, cfg.get('dataset', 'camels'))
        warmup_start = start_date - pd.DateOffset(days=seq_length - 1)
        df_period = df.loc[warmup_start:end_date]

        # Check if synthetic column exists
        if scenario_col not in df_period.columns:
            continue

        # Create synthetic data by replacing precipitation
        df_syn = df_period.copy()
        df_syn['Precip'] = df_syn[scenario_col]

        # Use peak temperature and PET if it's a peak scenario
        if 'peak' in scenario_col:
            if 'Temp_at_peak' in df_syn.columns:
                df_syn['Temp'] = df_syn['Temp_at_peak']
            if 'PET_at_peak' in df_syn.columns:
                df_syn['PET'] = df_syn['PET_at_peak']

        # Use per-basin scaler from original data (must match training convention)
        bscaler = scalers['basin_scalers'][basin]
        dyn_data_scaled = (df_syn[dyn_cols].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']
        
        static_data_scaled = None
        if not cfg["no_static"]:
            static_data = attributes.loc[basin].values
            static_data_scaled = (static_data - scalers['static_means']) / scalers['static_stds']
        
        # Check for NaN in synthetic data
        is_nan = df_syn[dyn_cols].isnull().any(axis=1)
        
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < seq_length:
                continue

            block_dyn_scaled = dyn_data_scaled[start_loc:end_loc]
            
            for i in range(block_len - seq_length + 1):
                target_idx_in_block = i + seq_length - 1
                current_date = df_syn.index[start_loc + target_idx_in_block]
                if current_date < start_date:
                    continue

                dyn_seq_flat = block_dyn_scaled[i:target_idx_in_block + 1].flatten()
                
                final_features = dyn_seq_flat
                if not cfg["no_static"]:
                    final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
                
                X_list.append(final_features)
                basin_list.append(basin)
                date_list.append(current_date)

    if not X_list:
        print(f"No valid data for scenario {scenario_col}")
        return None

    X_eval = np.array(X_list, dtype=np.float32)
    
    print("Making predictions...")
    predictions_scaled = model.predict(X_eval)
    
    # Rescale predictions using per-basin scalers
    results_df = pd.DataFrame({
        'date': date_list,
        'basin': basin_list,
        'pred_scaled': predictions_scaled
    })
    
    def rescale_row(row):
        bscaler = scalers['basin_scalers'][row['basin']]
        return row['pred_scaled'] * bscaler['q_std'] + bscaler['q_mean']

    results_df['prediction'] = results_df.apply(rescale_row, axis=1)
    results_df['prediction'] = results_df['prediction'].clip(lower=0)

    return results_df

def main():
    parser = argparse.ArgumentParser(description='Regional XGBoost Evaluation')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--n_basins', type=int, default=None)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    print("Regional XGBoost Evaluation")
    print(f"Model directory: {run_dir}")
    
    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)
    with open(run_dir / 'scalers.json', 'r') as fp:
        scalers = json.load(fp)
        # basin_scalers is the canonical (per-basin) normalization convention -- see
        # DEBUG_README.md for why per-basin, not global pooled, normalization is required
        # for a regional model. Convert its nested lists back to numpy arrays.
        if 'basin_scalers' in scalers:
            for b_id, b_scaler in scalers['basin_scalers'].items():
                for key, val in b_scaler.items():
                    if isinstance(val, list):
                        b_scaler[key] = np.array(val)
        else:
            raise ValueError(
                f"{run_dir / 'scalers.json'} has no 'basin_scalers' key -- this run was trained "
                "with the old (buggy) global-normalization code path and is not compatible with "
                "this evaluation script's per-basin rescaling. Retrain with the current main_xgb.py."
            )
        # Legacy support only: old runs may have a global_scalers key: convert it too so old
        # run directories don't crash on load, but evaluate_period/evaluate_synthetic_period no
        # longer read from it.
        if 'global_scalers' in scalers:
            for key, val in scalers['global_scalers'].items():
                if isinstance(val, list):
                    scalers['global_scalers'][key] = np.array(val)
        if 'static_means' in scalers and isinstance(scalers['static_means'], list):
            scalers['static_means'] = np.array(scalers['static_means'])
        if 'static_stds' in scalers and isinstance(scalers['static_stds'], list):
            scalers['static_stds'] = np.array(scalers['static_stds'])


    model_file = run_dir / "xgb_model.json"
    if not model_file.exists():
        raise ValueError(f"Model file not found: {model_file}")
    
    print(f"Loading model from: {model_file}")
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    
    basins = get_basin_list(GLOBAL_SETTINGS['loc'], dataset=cfg.get('dataset', 'camels'))
    if args.n_basins:
        basins = basins[:args.n_basins]
        print(f"Processing only first {args.n_basins} basins for testing.")

    attributes = None
    if not cfg.get("no_static", True):
        db_path = str(run_dir / 'attributes.db')
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True, keep_features=VALID_ATTR)

    dates = _run_dates(cfg)
    train_start = dates['train_start']
    train_end = dates['train_end']
    val_start = dates['val_start']
    val_end = dates['val_end']

    periods = {'cal': (train_start, train_end), 'eva': (val_start, val_end)}
    output_base = Path(f"/data0/funato/hsp2/0_out/XGB/{GLOBAL_SETTINGS['loc']}/{cfg.get('dataset', 'camels')}/regional")

    # Run tag: must encode every config dimension that affects the result, not just the ones
    # currently swept in a given hp grid. Without this, two runs that differ only in no_static
    # (or hyperparameters) but share nothing in the filename silently overwrite each other's
    # results -- see DEBUG_README.md for the identical bug found (and fixed) in the LSTM
    # pipeline. This also mirrors the PARAM_STRING run_eval_xgb.sh already computes/prints but
    # never actually passed through to this script's output filenames.
    ns_tag = "ns1" if cfg.get("no_static", True) else "ns0"
    run_tag = (
        f"{ns_tag}_n{cfg.get('n_estimators')}_d{cfg.get('max_depth')}"
        f"_lr{cfg.get('learning_rate')}_sub{cfg.get('subsample')}"
        f"_col{cfg.get('colsample_bytree')}_sl{cfg.get('seq_length')}"
    )
    print(f"Run tag for output filenames: {run_tag}")

    og_frames = []
    for period_name, (start, end) in periods.items():
        predictions_df, metrics_df = evaluate_period(model, basins, cfg, scalers, attributes, start, end)

        if predictions_df is not None and metrics_df is not None:
            og_frames.append(predictions_df)
            pred_dir = output_base / "predict"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"XGB_predict_{run_tag}_{period_name}.csv"

            pivot_df = predictions_df.pivot_table(index='basin', columns='date', values='prediction')
            # Sort basins (gauge_id strings, e.g. leading-zero US ids -- must sort as strings,
            # not int() round-tripped, or the leading zero is lost and the .loc lookup KeyErrors)
            pivot_df = pivot_df.sort_index()
            pivot_df.to_csv(pred_path, index_label='basin')
            print(f"Pivoted predictions for {period_name} period saved to {pred_path}")

            results_dir = output_base / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / f"XGB_results_{run_tag}_{period_name}.csv"
            metrics_df.to_csv(results_path, index_label='basin')
            print(f"Metrics for {period_name} period saved to {results_path}")
            
            print(f"\n--- {period_name.upper()} Period Summary ---")
            print(metrics_df.describe())
            print("\n")
    
    # Evaluate on synthetic scenarios
    print("\n" + "="*60)
    print("Starting Synthetic Scenario Evaluation...")
    print("="*60)
    
    scenario_predictions = {}
    for scenario_col in syn_precip_cols:
        # Use evaluation period for synthetic scenarios
        # For peak scenarios, use shorter period
        syn_end = pd.to_datetime('2001-02-10') if 'peak' in scenario_col else val_end

        predictions_df = evaluate_synthetic_period(
            model, basins, cfg, scalers, attributes, scenario_col, val_start, syn_end)

        if predictions_df is not None:
            pred_dir = output_base / "predict" / "synthetic"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"XGB_predict_{run_tag}_{scenario_col}_eva.csv"

            pivot_df = predictions_df.pivot_table(index='basin', columns='date', values='prediction')
            # Sort basins as strings -- see note above about leading-zero gauge ids.
            pivot_df = pivot_df.sort_index()
            pivot_df.to_csv(pred_path, index_label='basin')
            print(f"Pivoted synthetic predictions for '{scenario_col}' saved to {pred_path}")

            scenario_predictions[scenario_col] = {
                basin: g.set_index('date')['prediction']
                for basin, g in predictions_df.groupby('basin')
            }

    # Write the long-format per-basin prediction files directly (og + all synthetic scenarios),
    # matching reshape_ML_predict.py's old output exactly so that reshape step is no longer
    # needed. The wide cal/eva/synthetic matrices above are still written too, for backward
    # compatibility with anything else that reads them.
    if og_frames:
        og_df = pd.concat(og_frames, axis=0)
        scenario_predictions["og"] = {
            basin: g.set_index('date')['prediction'].sort_index()
            for basin, g in og_df.groupby('basin')
        }
    save_regional_predictions_long_format("XGB", GLOBAL_SETTINGS["loc"], scenario_predictions, basins)

    print("\nRegional XGBoost evaluation completed successfully!")

if __name__ == "__main__":
    main()


