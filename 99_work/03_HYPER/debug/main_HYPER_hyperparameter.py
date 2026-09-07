# Description: Main script for running the ESN model on multiple files.
# Modified for Hyperparameter Analysis to control intrinsic nonlinearity.

# v0: input data not normalized.
# v1: (this file) input data is normalized. 

import matplotlib
matplotlib.use('Agg')
from esn import ESN
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import math
import zarr
import numcodecs

####################
benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

MET_DATA = "MSWEP_MSWX"  # Options: "MSWEP_MSWX", "ERA5land"

spin = 0
nexttime = True
fix = ""
buf = f"{fix}"

reservoir_seed = 42

BMA_TYPE = "kgesoftmax"
if BMA_TYPE not in {"kgesoftmax", "loglikelihood"}:
    raise ValueError(f"Unsupported BMA_TYPE: {BMA_TYPE}")

# --- Full-factorial hyperparameter sweep -----------------------------------
# The lists below are the sweep space; a driver (see
# run_HYPER_hyperparameter_sweep.sh) invokes this script once per combination,
# each run pinned to a single CPU. Running this script directly (no CLI args)
# falls back to the first value of each list, for quick manual testing.
reservoir_size_list = [10, 50, 100, 200, 300, 400, 500, 700]
ridge_param_list = [0.1, 0.01, 0.001]
input_scale_list = [0.1, 1.0, 10.0]
spectral_radius_list = [0.85, 0.90, 0.95, 1.0, 1.05]
adjacency_density_list = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

parser = argparse.ArgumentParser(description="Run a single point of the HYPER hyperparameter sweep.")
parser.add_argument("--reservoir_size", type=int, default=reservoir_size_list[0])
parser.add_argument("--ridge_param", type=float, default=ridge_param_list[0])
parser.add_argument("--input_scale", type=float, default=input_scale_list[0])
parser.add_argument("--spectral_radius", type=float, default=spectral_radius_list[0])
parser.add_argument("--adjacency_density", type=float, default=adjacency_density_list[0])
args = parser.parse_args()

reservoir_size = args.reservoir_size
ridge_param = args.ridge_param
input_scale = args.input_scale
spectral_radius = args.spectral_radius
adjacency_density = args.adjacency_density

output_size = 1
input_size = 3  # Reverted to base 3 dimensions (Precip, Temp, PET)
input_cols = ['Precip', 'Temp', 'PET']

ver = 1
############

sim_dir = f"/data0/funato/97_work/200_dataset/data/MARRMoT/glb_GRDC_{MET_DATA}_nocal/v{ver}"
if MET_DATA == "MSWEP_MSWX":
    grdc_id_list = pd.read_csv(f"/data0/funato/97_work/200_dataset/basin_list/grdc/grdc_station_id_list_small_MSWEP_MSWX_3000.csv", index_col=None, header = 0)['grdc_id'].values.tolist()[::4] # selects every 3rd basin to reduce the number of basins for testing
elif MET_DATA == "ERA5land":
    grdc_id_list = pd.read_csv(f"/data0/funato/97_work/200_dataset/basin_list/grdc/grdc_station_id_list_small_3000.csv", index_col=None, header = 0)['grdc_id'].values.tolist()

start_date_cal = '2000-01-01'
end_date_cal = '2013-12-31'
start_date_eva = '2014-01-01'
end_date_eva = '2020-12-30'

output_base = f'/data0/funato/97_work/_out/97_out_glb{f"_{MET_DATA}" if MET_DATA == "MSWEP_MSWX" else ""}'
# Encode every swept hyperparameter in the folder name so runs never collide.
hp_tag = f'r{reservoir_size}_rr{ridge_param}_is{input_scale}_sr{spectral_radius}_ad{adjacency_density}'
output_dir = f'{output_base}/HYPER/{BMA_TYPE}/hp/{hp_tag}'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/Wout', exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/weights', exist_ok=True)

# Zarr settings
SAVE_ZARR = False
ZARR_BLOCK_BASINS = 50
zarr_compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


param_file_path = os.path.join(output_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n=== FIXED RC PARAMETERS ===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write(f"spin = {spin}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"input_scale = {input_scale}\n")
    param_file.write(f"adjacency_density = {adjacency_density}\n")
    
def load_data(grdc_id):
    sim_file = f"{sim_dir}/grdc_{grdc_id}.csv"
    if not os.path.exists(sim_file):
        return None
    df = pd.read_csv(sim_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)    
    df = df.reindex(full_date_range)
    return df

def zscore_normalize_inputs(df_cal, df_eva, cols):
    """Normalize inputs with calibration-period statistics (no eval leakage)."""
    cal_in = df_cal[cols].astype(float).copy()
    eva_in = df_eva[cols].astype(float).copy()

    means = cal_in.mean(axis=0, skipna=True)
    stds = cal_in.std(axis=0, skipna=True)
    stds = stds.replace(0, np.nan).fillna(1.0)

    cal_z = (cal_in - means) / stds
    eva_z = (eva_in - means) / stds

    # Keep the pipeline numerically stable if any timestamps are missing.
    cal_z = cal_z.fillna(0.0)
    eva_z = eva_z.fillna(0.0)
    return cal_z, eva_z

def BMK(obs_data, sim_data, benchmark):
    obs_data = np.array(obs_data).flatten()
    sim_data = np.array(sim_data).flatten()

    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    if len(obs_data) < 2: return np.nan

    if benchmark == "KGE":
        if np.std(obs_data) == 0 or np.std(sim_data) == 0 or np.mean(obs_data) == 0:
            return np.nan
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2) 
    elif benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.square(obs_data - obs_ave))
        if denom == 0: return np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / denom)
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0: return np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)
    elif benchmark == "VE":
        obs_sum = np.sum(obs_data)
        if obs_sum == 0: return np.nan
        return 1 - np.sum(np.abs(obs_data - sim_data)) / obs_sum
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave))) + 1e-8
        return 1 - numer / denom
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

def BMK(obs_data, sim_data, benchmark):
    obs_data = np.array(obs_data).flatten()
    sim_data = np.array(sim_data).flatten()

    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    if len(obs_data) < 2: return np.nan

    if benchmark == "KGE":
        if np.std(obs_data) == 0 or np.std(sim_data) == 0 or np.mean(obs_data) == 0:
            return np.nan
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2) 
    elif benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.square(obs_data - obs_ave))
        if denom == 0: return np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / denom)
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0: return np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)
    elif benchmark == "VE":
        obs_sum = np.sum(obs_data)
        if obs_sum == 0: return np.nan
        return 1 - np.sum(np.abs(obs_data - sim_data)) / obs_sum
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave))) + 1e-8
        return 1 - numer / denom
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

# Define unique tag for this specific hyperparameter configuration
file_tag = f"_r{reservoir_size}_is{input_scale}_sr{spectral_radius}_rr{ridge_param}{buf}"

# Always build a fresh ESN for this exact hyperparameter combination -- do
# NOT look up/reuse a cached reservoir from another combo or experiment.
# ESN.__init__ seeds via np.random.default_rng(reservoir_seed), so this is
# fully deterministic given (reservoir_size, spectral_radius, input_scale,
# adjacency_density, reservoir_seed): reconstructing here is bit-for-bit
# identical to what a cache would have held, just without the risk of a
# cache key silently omitting one of those hyperparameters (as it previously
# did -- see git history) and returning a reservoir built for a different
# input_scale/adjacency_density than the one actually requested.
reservoir_structure_dir = f"/data0/funato/97_work/_out/0_reservoir_structure{f'_{MET_DATA}' if MET_DATA == 'MSWEP_MSWX' else ''}"
reservoir_tag = f"_r{reservoir_size}_sr{spectral_radius}_is{input_scale}_ad{adjacency_density}"
model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=adjacency_density,
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            reservoir_seed=reservoir_seed)

# Keep a record of the topology used (not read back by anything -- purely
# for later inspection/debugging).
W_in_df = pd.DataFrame(model.W_in_val())
os.makedirs(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "Win"), exist_ok=True)
W_in_df.to_csv(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "Win", f"HYPER_Win{reservoir_tag}.csv"), mode='w', index=False, header=False)

A_val_df = pd.DataFrame(model.A_val())
os.makedirs(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "A"), exist_ok=True)
A_val_df.to_csv(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "A", f"HYPER_A_val{reservoir_tag}.csv"), mode='w', index=False, header=False)

# Save ESN metadata
metadata = {
    'reservoir_size': reservoir_size,
    'spectral_radius': spectral_radius,
    'input_scale': input_scale,
    'adjacency_density': adjacency_density,
    'ridge_param': ridge_param,
    'input_cols': ','.join(input_cols)
}
metadata_df = pd.DataFrame([metadata])
os.makedirs(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "metadata"), exist_ok=True)
metadata_df.to_csv(os.path.join(reservoir_structure_dir, f"seed{reservoir_seed}", "metadata", f"HYPER_metadata{reservoir_tag}_rr{ridge_param}.csv"), index=False)

# =====================================================================
# MAIN PROCESSING LOOP
log_file_path = output_dir + f'/HYPER{file_tag}_log.txt'
if os.path.exists(log_file_path): open(log_file_path, 'w').close()
log_file = open(log_file_path, 'a')

# Load BMA predictions (same for every basin - load once, not per-iteration)
bma_df_cal = pd.read_csv(f'{output_base}/BMA/{BMA_TYPE}/predict/BMA_predict_cal.csv', index_col=0)
bma_df_eva = pd.read_csv(f'{output_base}/BMA/{BMA_TYPE}/predict/BMA_predict_eva.csv', index_col=0)

first_id = True
zarr_initialized = False
block_arrays_cal = {}
block_arrays_eva = {}
for grdc_id in grdc_id_list:
    print(f"Processing grdc_{grdc_id}")
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{grdc_id}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### LOAD DATA ###
    df = load_data(grdc_id)
    # Buffer for basin-list mismatches: skip basins missing a MARRMoT dataset
    # (e.g. corrupt/missing upstream simulation output) or missing from BMA predictions,
    # rather than crashing the whole run.
    if df is None:
        print(f"Skipping grdc_{grdc_id}: no combined MARRMoT dataset found.")
        continue
    if f'grdc_{grdc_id}' not in bma_df_cal.index or f'grdc_{grdc_id}' not in bma_df_eva.index:
        print(f"Skipping grdc_{grdc_id}: not found in BMA predictions.")
        continue

    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    df_cal['Sim flow'] = bma_df_cal.loc[f'grdc_{grdc_id}'].values
    df_eva['Sim flow'] = bma_df_eva.loc[f'grdc_{grdc_id}'].values

    # Calculate Flow Error
    df_cal['Flow Error'] = df_cal['Obs flow'] - df_cal['Sim flow']
    df_eva['Flow Error'] = df_eva['Obs flow'] - df_eva['Sim flow']

    ### Z-SCORE NORMALIZE 3D INPUTS (Precip, Temp, PET) ###
    input_cal_df, input_eva_df = zscore_normalize_inputs(df_cal, df_eva, input_cols)
    
    # Define a directory just for the static basin scalers
    scalers_dir = os.path.join(reservoir_structure_dir, "basin_scalers")
    os.makedirs(scalers_dir, exist_ok=True)
    
    # Create a unique file for THIS basin only
    basin_scaler_file = os.path.join(scalers_dir, f"scaler_grdc_{grdc_id}.csv")
    
    # Only calculate and save if the file doesn't already exist from a previous parallel run
    if not os.path.exists(basin_scaler_file):
        cal_in = df_cal[input_cols].astype(float).copy()
        means = cal_in.mean(axis=0, skipna=True)
        stds = cal_in.std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
        
        # Combine headers and values
        header = ['grdc_id'] + [f"{col}_mean" for col in input_cols] + [f"{col}_std" for col in input_cols]
        scaler_row = [grdc_id] + means.tolist() + stds.tolist()
        
        # Save safely
        pd.DataFrame([header, scaler_row]).to_csv(basin_scaler_file, mode='w', index=False, header=False)

    input_cal = input_cal_df.values.T
    input_eva_full = input_eva_df.values.T
    input_eva = input_eva_full[:, :-1] # Shift for prediction

    obs_error_cal = df_cal['Flow Error'].values.reshape(-1, 1)
    obs_error_eva = df_eva['Flow Error'].values.reshape(-1, 1)

    # target error
    error_target_cal = np.concatenate([obs_error_cal[1:], obs_error_eva[0:1]])
    error_target_eva = obs_error_eva[1:] 

    # Target flow
    target_cal = np.concatenate((df_cal['Obs flow'].values[1:], df_eva['Obs flow'].values[0:1]))
    target_eva = np.concatenate((df_eva['Obs flow'].values[1:],))

    # ==================== CALIBRATION ====================
    Wout, reservoir, reservoir_states_train = model.train(input_cal, target_data=error_target_cal, washout=0, ridge_param=ridge_param, spinup=spin, return_states=True)

    # Predict to get evaluation states as well
    error_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10, return_states=False)
    error_eva, reservoir_states_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10, return_states=True)

    # Save Wout
    Wout_row = np.concatenate(([f'grdc_{grdc_id}'], Wout.flatten()))
    Wout_df = pd.DataFrame([Wout_row])
    if first_id:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout{file_tag}.csv", mode='w', index=False, header=False)
    else:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout{file_tag}.csv", mode='a', index=False, header=False)

    # ==================== SAVE RESERVOIR STATES (Zarr, block-by-basin) ====================
    if SAVE_ZARR:
        cal_len = reservoir_states_train.shape[0]
        eva_len = reservoir_states_eva.shape[0]
        R = reservoir_states_train.shape[1]

        if not zarr_initialized:
            n_basins = len(grdc_id_list)
            basins_per_block = ZARR_BLOCK_BASINS
            blocks = int(math.ceil(n_basins / basins_per_block))
            zarr_base = os.path.join(output_dir, 'reservoir_zarr')
            for b in range(blocks):
                os.makedirs(os.path.join(zarr_base, 'cal'), exist_ok=True)
                os.makedirs(os.path.join(zarr_base, 'eva'), exist_ok=True)
                cal_path = os.path.join(zarr_base, 'cal', f'block_{b}.zarr')
                eva_path = os.path.join(zarr_base, 'eva', f'block_{b}.zarr')
                grp_cal = zarr.open_group(cal_path, mode='a', zarr_format=2)
                grp_eva = zarr.open_group(eva_path, mode='a', zarr_format=2)
                shape_cal = (basins_per_block, cal_len, R)
                shape_eva = (basins_per_block, eva_len, R)
                chunks_cal = (1, cal_len, min(R, 256))
                chunks_eva = (1, eva_len, min(R, 256))
                block_arrays_cal[b] = grp_cal.create_array('states', shape=shape_cal, chunks=chunks_cal, dtype='float32', compressor=zarr_compressor, overwrite=True)
                block_arrays_eva[b] = grp_eva.create_array('states', shape=shape_eva, chunks=chunks_eva, dtype='float32', compressor=zarr_compressor, overwrite=True)
            zarr_initialized = True

        idx = grdc_id_list.index(grdc_id)
        block_idx = idx // ZARR_BLOCK_BASINS
        pos_in_block = idx % ZARR_BLOCK_BASINS
        block_arrays_cal[block_idx][pos_in_block, :cal_len, :] = reservoir_states_train.astype('float32')
        block_arrays_eva[block_idx][pos_in_block, :eva_len, :] = reservoir_states_eva.astype('float32')

    # ==================== PREDICTION ====================
    start_time_pred = datetime.now()
    error_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10, return_states=False)
    error_eva, reservoir_states_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10, return_states=True)
    
    predict_cal = error_cal + np.concatenate([df_cal['Sim flow'].values[1:], df_eva['Sim flow'].values[0:1]])
    predict_eva = error_eva + df_eva['Sim flow'].values[1:]

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    end_time_pred = datetime.now()
    log_file.write(f"pred end: {end_time_pred.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")

    # ==================== SAVE RESULTS ====================
    file_row_cal = [f'grdc_{grdc_id}'] + list(predict_cal.flatten())
    file_row_eva = [f'grdc_{grdc_id}'] + list(predict_eva.flatten())

    if first_id:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal[0]))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva[0]))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        pd.DataFrame([date_row_cal]).to_csv(output_dir + f"/predict/HYPER_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
        pd.DataFrame([date_row_eva]).to_csv(output_dir + f"/predict/HYPER_predict{file_tag}_eva.csv", mode='w', index=False, header=False)
        
        results_header = ['grdc_id'] + benchmark_list
        pd.DataFrame([results_header]).to_csv(output_dir + f'/results/HYPER_results{file_tag}_cal.csv', mode='w', header=False, index=False)
        pd.DataFrame([results_header]).to_csv(output_dir + f'/results/HYPER_results{file_tag}_eva.csv', mode='w', header=False, index=False)

    with open(output_dir + f"/predict/HYPER_predict{file_tag}_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/HYPER_predict{file_tag}_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'grdc_id': grdc_id}
    file_results_eva = {'grdc_id': grdc_id}

    for benchmark in benchmark_list:
        file_results_cal[benchmark] = BMK(target_cal, predict_cal, benchmark)
        file_results_eva[benchmark] = BMK(target_eva, predict_eva, benchmark)

    pd.DataFrame([file_results_cal]).to_csv(output_dir + f'/results/HYPER_results{file_tag}_cal.csv', mode='a', index=False, header=False)
    pd.DataFrame([file_results_eva]).to_csv(output_dir + f'/results/HYPER_results{file_tag}_eva.csv', mode='a', index=False, header=False)

    end_time = datetime.now()
    log_file.write(f"elapsed: {end_time - start_time}\n")

    first_id = False
    del Wout, reservoir, reservoir_states_train, reservoir_states_eva, predict_eva, predict_cal

log_file.write(f"DONE\n")
log_file.close()

# ==========================================
# POST-PROCESSING: Combine Weights
# ==========================================
print("Creating combined weights file...")
bma_weights_file = f'{output_base}/BMA/{BMA_TYPE}/weights/BMA_weights.csv'
wout_file = output_dir + f"/Wout/HYPER_Wout{file_tag}.csv"

if os.path.exists(bma_weights_file) and os.path.exists(wout_file):
    bma_df = pd.read_csv(bma_weights_file, index_col=0)
    bma_df.index = bma_df.index.astype(str).str.replace('grdc_', '').astype(int)
    bma_df.index.name = 'grdc_id'
    bma_df.columns = [f'BMA_{i}' for i in range(1, bma_df.shape[1] + 1)]
    
    wout_df = pd.read_csv(wout_file, header=None)
    wout_df.set_index(0, inplace=True)
    wout_df.index = wout_df.index.astype(str).str.replace('grdc_', '').astype(int)
    wout_df.index.name = 'grdc_id'
    wout_df.columns = [f'RC_{i}' for i in range(1, wout_df.shape[1] + 1)]
    
    common_basins = sorted(list(set(bma_df.index).intersection(set(wout_df.index))))
    
    bma_common = bma_df.loc[common_basins]
    rc_common = wout_df.loc[common_basins]
    combined_df = pd.concat([bma_common, rc_common], axis=1)
    
    combined_weights_file = output_dir + f'/weights/HYPER_weights{file_tag}.csv'
    combined_df.to_csv(combined_weights_file, index=True)
    print(f"Saved combined weights to: {combined_weights_file}")

print("\nHYPER PROCESSING DONE")