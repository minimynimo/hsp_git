# Description: ESN hyperparameter search for the HYPER bias-correction model.
# can be used for both BMA and model-based bias correction
#
# One process = one hyperparameter combination (mirrors
# ealstm_regional_modeling/main_lstm_regional_hp.py's CLI-per-combo convention), so
# run_hparam_HYPER.sh can dispatch the whole grid across cores in parallel via taskset. Run a
# single combo directly with e.g.:
#   python hparam_main_HYPER.py --reservoir_size 700 --spectral_radius 0.4 --ridge_param 0.001 --input_scale 0.5
import argparse
import matplotlib
matplotlib.use('Agg')
from esn_HYPER import ESN
import os
import pandas as pd
import numpy as np
from datetime import datetime

#####################
benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

nexttime = True

fix = ""
#fix = "_nofix" ##
#fix = "_zerofix"##

buf = f"{fix}"

input_size = 3
output_size = 1


# HP-search-only cal/val split, kept strictly INSIDE each dataset's production calibration
# period -- mirrors ealstm_regional_modeling/main_lstm_regional_hp.py's HP_SPLIT.
#   * 'camels'  -- production calibration is 1981-1995; HP search sub-splits that into
#     1981-1990 (train) / 1991-1995 (val), staying clear of the 1996-2010 held-out period.
#   * 'caravan' -- production calibration is 1990-2000 (main_HYPER.py's own dates); HP search
#     sub-splits that into 1990-1998 (train) / 1999-2000 (val), staying clear of the 2001-2006
#     held-out period. Unchanged from before this script supported multiple datasets.
HP_SPLIT = {
    'camels': {
        'start_date_cal': '1981-01-01', 'end_date_cal': '1990-12-31',
        'start_date_eva': '1991-01-01', 'end_date_eva': '1995-12-31',
    },
    'caravan': {
        'start_date_cal': '1990-01-01', 'end_date_cal': '1998-12-31',
        'start_date_eva': '1999-01-01', 'end_date_eva': '2000-12-31',
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="ESN hyperparameter search (HYPER bias-correction model)")
    parser.add_argument('--loc', type=str, default='US', choices=['US'])
    # 'camels' (default): built directly from raw CAMELS-US, full 1980-2014 record. 'caravan':
    # older Caravan-derived extraction (1987-2006 only). See HP_SPLIT above.
    parser.add_argument('--dataset', type=str, default='camels', choices=['camels', 'caravan'])
    parser.add_argument('--reservoir_size', type=int, default=700)
    parser.add_argument('--spectral_radius', type=float, default=0.4)
    parser.add_argument('--ridge_param', type=float, default=0.001)
    parser.add_argument('--input_scale', type=float, default=0.5)
    parser.add_argument('--spinup', type=int, default=1)
    parser.add_argument('--adjacency_density', type=float, default=0.0006)
    return parser.parse_args()


args = get_args()
loc = args.loc
dataset = args.dataset
reservoir_size = args.reservoir_size
spectral_radius = args.spectral_radius
ridge_param = args.ridge_param
input_scale = args.input_scale
spinup = args.spinup
adjacency_density = args.adjacency_density
washout = 0
ver = 2

start_date_cal = HP_SPLIT[dataset]['start_date_cal']
end_date_cal = HP_SPLIT[dataset]['end_date_cal']
start_date_eva = HP_SPLIT[dataset]['start_date_eva']
end_date_eva = HP_SPLIT[dataset]['end_date_eva']

print(f"[hparam_main_HYPER] loc={loc} dataset={dataset!r} "
      f"train={start_date_cal}..{end_date_cal} val={start_date_eva}..{end_date_eva}")

ver_name = "ver1_1" if ver == 1 else "ver2_0"

# /2_MERV/HYPER/{dataset}/{loc}/varssim_nocal_og/{ver_name}/ stores its per-basin CSVs as
# varssim_{gauge_id}.csv (the same directory main_HYPER.py reads its "og" input from).
varssim_dir = f"/data0/funato/hsp2/2_MERV/HYPER/{dataset}/{loc}/varssim_nocal_og/{ver_name}"

# Basin list, keyed by gauge_id (matches ealstm_regional_modeling/papercode/utils.py's
# get_basin_list): read from us_{dataset}_file_num.csv's gauge_id column, stripped of the
# "camels_" prefix.
basin_list_csv = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/us_{dataset}_file_num.csv"
basin_df = pd.read_csv(basin_list_csv, dtype={'gauge_id': str})
basins = [g.split('_')[-1] for g in basin_df['gauge_id']]

file_tag = f"_r{reservoir_size}_s{spinup}_sr{spectral_radius}_rr{ridge_param}_ad{adjacency_density}"
#####################
print(file_tag)

output_dir = f'/data0/funato/hsp2/0_out/HYPER/{loc}/{dataset}/HYPER_hp/spin{spinup}/reservoir{reservoir_size}/HYPER{file_tag}_is{input_scale}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/Win_A', exist_ok=True)
os.makedirs(output_dir + '/Wout', exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/reservoir', exist_ok=True)


def load_data(gauge_id):
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)

    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)

    return df


def BMK(obs_data,sim_data,benchmark):
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
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


model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=adjacency_density,
            spectral_radius=spectral_radius,
            input_scale=input_scale)

W_in_vals = model.W_in_val()
W_in_df = pd.DataFrame(W_in_vals)
W_in_df.to_csv(output_dir + f"/Win_A/HYPER_Win{file_tag}.csv", mode='a', index=False, header=False)

A_vals = model.A_val()
A_val_df = pd.DataFrame(A_vals)
A_val_df.to_csv(output_dir + f"/Win_A/HYPER_A_val{file_tag}.csv", mode = 'a', index=False, header=False)

log_path = output_dir + f'/HYPER{file_tag}_log.txt'
if os.path.exists(log_path):
    with open(log_path) as existing_log:
        if existing_log.read().rstrip().endswith("DONE"):
            print(f"[HP] Skipping: a completed run already exists at {log_path}")
    open(log_path, 'w').close()
log_file = open(log_path, 'a')

results_cal = []
results_eva = []
for gauge_id in basins:
    print(gauge_id)
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # loading the sim flow data for the BMA and 44 MARRMoT models
    # BMA_predict_cal.csv covers BMA.py's own (dataset-dependent) cal period, which spans
    # both our train and val windows here (HP_SPLIT is a sub-split of it) -- reindex by date
    # rather than assuming a train/val file split.
    bma_df_cal = pd.read_csv(f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}/predict/BMA_predict_cal.csv', index_col=0)
    bma_row = bma_df_cal.loc[f'gauge_{gauge_id}']
    bma_row.index = pd.to_datetime(bma_row.index)

    df_cal['Sim flow'] = bma_row.reindex(df_cal.index).values
    df_eva['Sim flow'] = bma_row.reindex(df_eva.index).values

    # Calculate Flow Error
    df_cal['Flow Error'] = df_cal['Obs flow'] - df_cal['Sim flow']
    df_eva['Flow Error'] = df_eva['Obs flow'] - df_eva['Sim flow']

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
    input_eva = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

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


    Wout, reservoir = model.train(input_cal, target_data=error_target_cal, washout=washout, ridge_param=ridge_param, spinup = spinup)

    # reservoir_row = np.concatenate(([f'gauge_{gauge_id}'], reservoir.flatten()))
    # reservoir_df = pd.DataFrame([reservoir_row])
    # if gauge_id == basins[0]:
    #     reservoir_df.to_csv(output_dir + f"/reservoir/HYPER_reservoir{file_tag}.csv", mode='w', index=False, header=False)
    # else:
    #     reservoir_df.to_csv(output_dir + f"/reservoir/HYPER_reservoir{file_tag}.csv", mode='a', index=False, header=False)

    # Wout (1,R)
    # reservoir (R, )

    Wout_row = np.concatenate(([f'gauge_{gauge_id}'], Wout.flatten()))
    Wout_df = pd.DataFrame([Wout_row])
    if gauge_id == basins[0]:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout{file_tag}.csv", mode='w', index=False, header=False)
    else:
        Wout_df.to_csv(output_dir + f"/Wout/HYPER_Wout{file_tag}.csv", mode='a', index=False, header=False)

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

    #print(predict_cal)
    #print(predict_eva)
    print(BMK(target_cal.T,predict_cal,"KGE"), BMK(target_eva.T, predict_eva,"KGE"))

    file_row_cal = [f'gauge_{gauge_id}'] + list(predict_cal.flatten())
    file_row_eva = [f'gauge_{gauge_id}'] + list(predict_eva.flatten())

    if gauge_id == basins[0]:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal[0]))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva[0]))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/HYPER_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/HYPER_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/HYPER_predict{file_tag}_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/HYPER_predict{file_tag}_eva.csv", 'a') as file:
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

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")

    del Wout, reservoir, predict_eva, predict_cal

log_file.write(f"DONE\n")
log_file.close()


df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/HYPER_results{file_tag}_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/HYPER_results{file_tag}_eva.csv', index=False, header=True)

print(f"HYPER{file_tag} reservoir size {reservoir_size} done")
print(f"HYPER{file_tag}")
print("DONE")
