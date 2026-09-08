# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
import matplotlib
matplotlib.use('Agg')
from esn_HYPER import ESN
import os
import pandas as pd
import numpy as np
from datetime import datetime

#####################
benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

spin = 1

nexttime = True

fix = ""
#fix = "_nofix" ##
#fix = "_zerofix"##

buf = f"{fix}"

input_size = 3
output_size = 1
reservoir_size = 300
#reservoir_size = 200
spectral_radius = 0.8
ridge_param = 0.1
adjacency_density = 0.1
input_scale = 0.5

washout = 0
#ridge_param = 1.0 

#loc, ver = "GB", 2
loc, ver = "US", 2

# dataset: 'camels' (default) -- built directly from raw CAMELS-US, full 1980-2014 record --
# or 'caravan', the older Caravan-derived extraction (1987-2006 only). Only affects loc='US'.
# See BMA.py's module docstring / ealstm_regional_modeling/main_lstm_regional.py's --dataset
# for the same split, and BMA.py's note that /2_MERV/HYPER/{dataset}/{loc}/ does not currently
# exist for either dataset (this whole pipeline needs its upstream ensemble input rebuilt
# before it can run, independent of this --dataset plumbing).
dataset = "camels"  # "caravan"


file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
#####################
print(file_tag)

ver_name = "ver1_1" if ver == 1 else "ver2_0"

varssim_dir = f"/data0/funato/hsp2/2_MERV/HYPER/{dataset}/{loc}"
reshape_input_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_{loc}"

# Basin list, keyed by gauge_id: /2_MERV/HYPER/{dataset}/{loc}/{scenario}/{ver_name}/ now stores
# its per-basin CSVs as varssim_{gauge_id}.csv (not the old sequential varssim{NNN}.csv), so the
# driver data itself is loaded by gauge_id below. file_num_by_gauge is kept only for the
# handful of things that still use the legacy sequential id: the BMA_predict_*.csv rows (see
# BMA.py) and reshape_input_dir, which is still numbered.
basin_list_csv = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/us_{dataset}_file_num.csv"
basin_df = pd.read_csv(basin_list_csv, dtype={'gauge_id': str})
basins = [g.split('_')[-1] for g in basin_df['gauge_id']]
file_num_by_gauge = dict(zip(basins, basin_df['File_num']))

precip_cols = ['no_precip',
               'peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
               'peak3_rate0', 'peak3_rate10', 'peak3_rate30', 'peak3_rate50',
               'peak5_rate0', 'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

if dataset == "camels":
    start_date_cal = '1981-01-01'
    end_date_cal = '1995-12-31'
    start_date_eva = '1996-01-01'
    end_date_eva = '2010-12-31'
else:  # caravan
    start_date_cal = '1990-01-01'
    end_date_cal = '2000-12-31'
    start_date_eva = '2001-01-01'
    end_date_eva = '2006-12-31'

print(f"[main_HYPER] loc={loc} dataset={dataset!r} "
      f"cal={start_date_cal}..{end_date_cal} eva={start_date_eva}..{end_date_eva}")

output_dir = f'/data0/funato/hsp2/0_out/HYPER/{loc}/{dataset}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict/synthetic', exist_ok=True)

# Final per-basin output, already in the format used by the figure code
# (previously produced by a separate reshape_HYPER_predict.py pass)
pred_discharge_cols = ['og'] + precip_cols
peak_length_list = [1, 3, 5]
rate_list = [0, 10, 30, 50]
reshape_output_dir = output_dir + '/varssim_HYPER'
os.makedirs(reshape_output_dir, exist_ok=True)

param_file_path = os.path.join(output_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write(f"spin = {spin}\n")
    param_file.write("\n===PCA===\n")


# Helper function to format file names
def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

# Input normalization: z-score each driving variable (Precip, Temp, PET) using
# calibration-period statistics only, so evaluation/synthetic scenarios never leak
# their own statistics into the scaler used to train the reservoir.
def compute_input_scaler(df_cal, cols):
    cal_in = df_cal[cols].astype(float)
    means = cal_in.mean(axis=0, skipna=True).values
    stds = cal_in.std(axis=0, skipna=True).values
    stds = np.where(stds == 0, 1.0, stds)
    return means, stds

def normalize_inputs(df, cols, means, stds):
    vals = df[cols].astype(float).values
    z = (vals - means) / stds
    return np.nan_to_num(z, nan=0.0)

# Function to load and preprocess data
def load_data(data_name, gauge_id):
    df = pd.read_csv(f"{varssim_dir}/{data_name}/{ver_name}/varssim_{gauge_id}.csv")

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



def BMK(obs_data, sim_data, benchmark):
    obs_data = np.array(obs_data).flatten()
    sim_data = np.array(sim_data).flatten()

    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    if benchmark == "KGE":
        # Handle edge cases
        if np.std(obs_data) == 0 or np.std(sim_data) == 0:
            return np.nan
        if np.mean(obs_data) == 0:
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
        if denom == 0:
            return np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / denom)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0:
            return np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)
    
    elif benchmark == "VE":
        obs_sum = np.sum(obs_data)
        if obs_sum == 0:
            return np.nan
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

def save_reshaped_output(gauge_id, predictions):
    """Merge this basin's predictions with its base varssim file and write it
    directly in the long-format layout used by the figure code."""
    file_num = file_num_by_gauge[gauge_id]
    # reshape_input_dir (/2_MERV/varssim_{loc}_1993_2006) still uses the legacy sequential
    # numbering, unlike /2_MERV/HYPER/{loc}'s per-scenario dirs.
    base_file = f"{reshape_input_dir}/varssim{file_name(file_num, 3)}.csv"
    if not os.path.exists(base_file):
        print(f"Warning: Base file {base_file} does not exist. Skipping reshape output for gauge {gauge_id}.")
        return

    base_df = pd.read_csv(base_file, parse_dates=['Date'])
    base_df = base_df[['Date', 'Obs flow', 'Precip', 'Temp', 'PET',
                'syn_no_precip', 'Temp_at_peak',
                'PET_at_peak', 'Obs_flow_at_peak',
                'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']].copy()
    base_df.set_index('Date', inplace=True)
    base_df.sort_index(inplace=True)

    precip_col = 'Precip'
    eval_start_date = pd.Timestamp(start_date_eva)
    eval_mask = base_df.index >= eval_start_date

    eval_precip = base_df.loc[eval_mask, precip_col]
    if len(eval_precip) > 0 and not eval_precip.isna().all():
        peak_idx = eval_precip.idxmax()
        peak_value = eval_precip.max()

        # 41-day window around the peak (20 days before, peak day, 20 days after)
        peak_start = max(peak_idx - pd.Timedelta(days=20), eval_start_date, base_df.index[0])
        peak_end = min(peak_idx + pd.Timedelta(days=20), base_df.index[-1])

        source_mask = (base_df.index >= peak_start) & (base_df.index <= peak_end)
        source_precip_values = base_df.loc[source_mask, precip_col].values

        eval_dates = base_df.index[eval_mask]
        num_dest_rows = min(len(source_precip_values), len(eval_dates))
        dest_mask = eval_dates[:num_dest_rows]

        for peak_length in peak_length_list:
            for rate in rate_list:
                column_name = f"syn_peak{peak_length}_rate{rate}"
                base_df[column_name] = base_df[precip_col]
                base_df.loc[eval_mask, column_name] = np.nan

                modified_precip_event = source_precip_values.copy()
                syn_peak_value = peak_value * (1 + rate / 100.0)
                peak_start_index = 10
                peak_end_index = peak_start_index + peak_length
                if peak_end_index <= len(modified_precip_event):
                    modified_precip_event[peak_start_index:peak_end_index] = syn_peak_value

                if len(dest_mask) > 0 and len(modified_precip_event) > 0:
                    base_df.loc[dest_mask, column_name] = modified_precip_event[:num_dest_rows]
    else:
        print(f"Warning: No valid precipitation data found in evaluation period for gauge {gauge_id}")

    for scenario in pred_discharge_cols:
        pred_series = predictions.get(scenario)
        base_df[scenario] = np.nan if pred_series is None else pred_series.reindex(base_df.index).values

    base_df.to_csv(f"{reshape_output_dir}/varssim{file_name(file_num, 3)}.csv", index=True, header=True)

model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=adjacency_density,
            spectral_radius=spectral_radius,
            input_scale=input_scale)
"""
W_in_vals = model.W_in_val()
W_in_df = pd.DataFrame(W_in_vals)
W_in_df.to_csv(output_dir + f"/Win_A/HYPER_Win.csv", mode='a', index=False, header=False)

A_vals = model.A_val()
A_val_df = pd.DataFrame(A_vals)
A_val_df.to_csv(output_dir + f"/Win_A/HYPER_A_val.csv", mode = 'a', index=False, header=False)
"""

if os.path.exists(output_dir + f'/HYPER_log.txt'):
    open(output_dir + f'/HYPER_log.txt', 'w').close()
log_file = open(output_dir + f'/HYPER_log.txt', 'a')

# Prediction timesteps are fixed by the cal/eva date ranges, so compute them
# once up front instead of re-deriving them from file 1's array shapes.
n_cal_days = len(pd.date_range(start=start_date_cal, end=end_date_cal))
n_eva_days = len(pd.date_range(start=start_date_eva, end=end_date_eva))
predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=n_cal_days)
predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=n_eva_days - 1)

results_cal = []
results_eva = []
for gauge_id in basins:
    file_num = file_num_by_gauge[gauge_id]
    print(gauge_id)
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data("varssim_nocal_og", gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Load BMA predictions
    bma_df_cal = pd.read_csv(f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}/predict/BMA_predict_cal.csv', index_col=0)
    bma_df_eva = pd.read_csv(f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}/predict/BMA_predict_eva.csv', index_col=0)

    df_cal['Sim flow'] = bma_df_cal.loc[f'gauge_{gauge_id}'].values
    df_eva['Sim flow'] = bma_df_eva.loc[f'gauge_{gauge_id}'].values

    # Calculate Flow Error
    df_cal['Flow Error'] = df_cal['Obs flow'] - df_cal['Sim flow']
    df_eva['Flow Error'] = df_eva['Obs flow'] - df_eva['Sim flow']

    # For each timestep, Precip,Temp,PET,Obs flow -- z-score normalized per basin
    # using calibration-period statistics (input_means/input_stds reused below for
    # the synthetic scenarios so they stay on the same scale as training).
    input_means, input_stds = compute_input_scaler(df_cal, ['Precip', 'Temp', 'PET'])
    input_cal = normalize_inputs(df_cal, ['Precip', 'Temp', 'PET'], input_means, input_stds).T
    input_eva = normalize_inputs(df_eva, ['Precip', 'Temp', 'PET'], input_means, input_stds).T

    #target_cal = np.concatenate((input_cal[3][1:], input_eva[3][:input_cal.shape[1] - 1])).tolist()

    obs_error_cal = df_cal['Flow Error'].values.reshape(-1, 1)
    obs_error_eva = df_eva['Flow Error'].values.reshape(-1, 1)

    # target error
    error_target_cal = np.concatenate([obs_error_cal[1:], obs_error_eva[0:1]]) #2898
    error_target_eva = obs_error_eva[1:]  #1089

    # Target flow
    target_cal = np.concatenate((df_cal['Obs flow'].values[1:], df_eva['Obs flow'].values[0:1])) #2898
    target_eva = np.concatenate((df_eva['Obs flow'].values[1:],)) #1089

    input_eva = input_eva[:, :-1]

    ##########!!!!
    # CALIBRATION
    # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

    # EVALUATION
    # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW


    Wout, reservoir = model.train(input_cal, target_data=error_target_cal, washout=washout, ridge_param=ridge_param, spinup = spin)

    """
    reservoir_row = np.concatenate(([f'gauge_{gauge_id}'], reservoir.flatten()))
    reservoir_df = pd.DataFrame([reservoir_row])
    if gauge_id == basins[0]:
        reservoir_df.to_csv(output_dir + f"/reservoir/HYPER_reservoir.csv", mode='w', index=False, header=False)
    else:
        reservoir_df.to_csv(output_dir + f"/reservoir/HYPER_reservoir.csv", mode='a', index=False, header=False)

    # Wout (1,R)
    # reservoir (R, )

    Wout_row = np.concatenate(([f'gauge_{gauge_id}'], Wout.flatten()))
    Wout_df = pd.DataFrame([Wout_row])
    if gauge_id == basins[0]:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout.csv", mode='w', index=False, header=False)
    else:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout.csv", mode='a', index=False, header=False)
    """

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")

    start_time_pred = datetime.now()
    start_time_pred_st = start_time_pred.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"pred start: {start_time_pred_st}\n")

    # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
    ### PREDICTIONS OF ERRORS ###
    error_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
    error_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

    # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
    # predict = error_cal + BMAsim
    # predict のtimestep = t+1
    predict_cal = error_cal + np.concatenate([df_cal['Sim flow'].values[1:], df_eva['Sim flow'].values[0:1]])
    predict_eva = error_eva + df_eva['Sim flow'].values[1:]

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    end_time_pred = datetime.now()
    end_time_pred_st = end_time_pred.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"pred end: {end_time_pred_st}\n")
    log_file.write(f"pred elapsed: {end_time_pred - start_time_pred}\n")

    #print(predict_cal)
    #print(predict_eva)
    print(BMK(target_cal.T,predict_cal,"KGE"), BMK(target_eva.T, predict_eva,"KGE"))

    file_row_cal = [f'gauge_{gauge_id}'] + list(predict_cal.flatten())
    file_row_eva = [f'gauge_{gauge_id}'] + list(predict_eva.flatten())

    if gauge_id == basins[0]:
        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/HYPER_predict_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/HYPER_predict_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/HYPER_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/HYPER_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'{benchmark}': BMK(target_cal, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'{benchmark}': BMK(target_eva, predict_eva, benchmark)
        })

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Accumulate this basin's predictions (by scenario) so they can be merged
    # directly into the final long-format output below, without a reshape pass.
    predictions = {
        'og': pd.concat([
            pd.Series(predict_cal.flatten(), index=predict_cal_dates),
            pd.Series(predict_eva.flatten(), index=predict_eva_dates),
        ]).sort_index()
    }

    # run the model for the synthetic data
    for precip_col in precip_cols:
        syn_df = load_data(f"varssim_nocal_{precip_col}", gauge_id)
        if precip_col == 'no_precip':
            syn_precip_col = 'syn_no_precip'
            temp_col = 'Temp'
            pet_col = 'PET'
        else:
            syn_precip_col = f'syn_{precip_col}'
            temp_col = "Temp_at_peak"
            pet_col = "PET_at_peak"
        syn_cal = syn_df[start_date_cal:end_date_cal].copy()
        syn_eva = syn_df[start_date_eva:end_date_eva].copy()   

        # Load BMA predictions for synthetic data
        syn_bma_df_cal = pd.read_csv(f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}/predict/synthetic/BMA_predict_{precip_col}_cal.csv', index_col=0)
        syn_bma_df_eva = pd.read_csv(f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}/predict/synthetic/BMA_predict_{precip_col}_eva.csv', index_col=0)

        syn_cal['Sim flow'] = syn_bma_df_cal.loc[f'gauge_{gauge_id}_cal'].values
        syn_eva['Sim flow'] = syn_bma_df_eva.loc[f'gauge_{gauge_id}_eva'].values

        # Calculate Flow Error for synthetic data
        syn_cal['Flow Error'] = syn_cal['Obs flow'] - syn_cal['Sim flow']
        syn_eva['Flow Error'] = syn_eva['Obs flow'] - syn_eva['Sim flow']   

        # Same means/stds as the 'og' scenario above, so synthetic inputs stay on the
        # scale the reservoir was trained on.
        syn_cols = [syn_precip_col, temp_col, pet_col]
        syn_input_cal = normalize_inputs(syn_cal, syn_cols, input_means, input_stds).T
        syn_input_eva = normalize_inputs(syn_eva, syn_cols, input_means, input_stds).T
        syn_obs_error_cal = syn_cal['Flow Error'].values.reshape(-1, 1)
        syn_obs_error_eva = syn_eva['Flow Error'].values.reshape(-1, 1) 
        syn_error_target_cal = np.concatenate([syn_obs_error_cal[1:], syn_obs_error_eva[0:1]])
        syn_error_target_eva = syn_obs_error_eva[1:]

        syn_input_eva = syn_input_eva[:, :-1]   

        syn_error_cal = model.predict(reservoir, syn_input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        syn_error_eva = model.predict(reservoir, syn_input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        syn_predict_cal = syn_error_cal + np.concatenate([syn_cal['Sim flow'].values[1:], syn_eva['Sim flow'].values[0:1]])
        syn_predict_eva = syn_error_eva + syn_eva['Sim flow'].values[1:]

        syn_predict_cal[syn_predict_cal < 0] = 0
        syn_predict_eva[syn_predict_eva < 0] = 0

        syn_file_row_cal = [f'gauge_{gauge_id}'] + list(syn_predict_cal.flatten())
        syn_file_row_eva = [f'gauge_{gauge_id}'] + list(syn_predict_eva.flatten())

        if gauge_id == basins[0]:
            syn_date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
            syn_date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

            syn_predict_cal_df = pd.DataFrame([syn_date_row_cal])
            syn_predict_eva_df = pd.DataFrame([syn_date_row_eva])

            syn_predict_cal_df.to_csv(output_dir + f"/predict/synthetic/HYPER_predict_{precip_col}_cal.csv", mode='w', index=False, header=False)
            syn_predict_eva_df.to_csv(output_dir + f"/predict/synthetic/HYPER_predict_{precip_col}_eva.csv", mode='w', index=False, header=False)
        with open(output_dir + f"/predict/synthetic/HYPER_predict_{precip_col}_cal.csv", 'a') as file:
            pd.DataFrame([syn_file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/predict/synthetic/HYPER_predict_{precip_col}_eva.csv", 'a') as file:
            pd.DataFrame([syn_file_row_eva]).to_csv(file, header=False, index=False)

        predictions[precip_col] = pd.concat([
            pd.Series(syn_predict_cal.flatten(), index=predict_cal_dates),
            pd.Series(syn_predict_eva.flatten(), index=predict_eva_dates),
        ]).sort_index()

    save_reshaped_output(gauge_id, predictions)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")

    del Wout, reservoir, predict_eva, predict_cal

log_file.write(f"DONE\n")
log_file.close()


df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/HYPER_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/HYPER_results_eva.csv', index=False, header=True)

print(f"HYPER reservoir size {reservoir_size} done")
print(f"HYPER")
print("DONE")