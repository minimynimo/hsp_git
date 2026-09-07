# based on https://github.com/urgedata/pythondata/blob/master/rf_timeseries/Random%20Forest%20for%20Time%20Series%20Forecasting.ipynb?utm_source=chatgpt.com
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF_regional_modeling-master"))
from papercode.utils import get_basin_list
from rf_indiv_resume import load_completed_basins, ensure_predict_header

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


#loc, ver = "JP", 2
# loc, ver = "GB", 2
loc, ver = "US", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   


if loc == "JP" and ver == 2:
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_GB"
elif loc == "US" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_US"

gauge_id_list = get_basin_list(loc)

# HP-search cal/val split -- must NOT touch the real 2001-01-01..2006-12-31 test period.
start_date_cal = '1990-01-01'
end_date_cal = '1998-12-31'
start_date_eva = '1999-01-01'
end_date_eva = '2000-12-31'

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser(description='Random Forest Individual Basin Modeling with Hyperparameter Tuning')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of trees (None for unlimited)')
parser.add_argument('--min_samples_leaf', type=int, default=5, help='Minimum samples required at leaf node')
parser.add_argument('--max_features', type=str, default='sqrt', help='Max features: sqrt, 1/3, or float')
args = parser.parse_args()

# Set hyperparameters from arguments
n_estimators = args.n_estimators
max_depth = args.max_depth
min_samples_leaf = args.min_samples_leaf

# Handle max_features
if args.max_features == 'sqrt':
    max_features = 'sqrt'
elif args.max_features == '1/3':
    max_features = 1.0/3.0
else:
    max_features = float(args.max_features)

# Fixed hyperparameters as per specifications
criterion = 'squared_error'
bootstrap = True
min_samples_split = 2

# Create output directory with hyperparameter settings
hp_str = f"n{n_estimators}_d{max_depth if max_depth is not None else 'None'}_l{min_samples_leaf}_f{args.max_features}"
output_dir = f"/data0/funato/hsp2/0_out/RF/{loc}/individual/hp/{hp_str}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)


def load_data(gauge_id):
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    
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
    

results_cal_path = output_dir + f'/results/RF_results_cal.csv'
results_eva_path = output_dir + f'/results/RF_results_eva.csv'
predict_cal_path = output_dir + f"/predict/RF_predict_cal.csv"
predict_eva_path = output_dir + f"/predict/RF_predict_eva.csv"

completed_basins, results_cal, results_eva = load_completed_basins(results_cal_path, results_eva_path)

n_days_cal = len(pd.date_range(start=start_date_cal, end=end_date_cal)) - 1
n_days_eva = len(pd.date_range(start=start_date_eva, end=end_date_eva)) - 1
ensure_predict_header(predict_cal_path, pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=n_days_cal))
ensure_predict_header(predict_eva_path, pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=n_days_eva))

log_file = open(output_dir + f'/RF_log.txt', 'a')
if completed_basins:
    log_file.write(f"RESUME: {len(completed_basins)} basins already completed, skipping them.\n")
    log_file.flush()
    print(f"Resuming: {len(completed_basins)} basins already completed, skipping them.")

for gauge_id in gauge_id_list:
    if str(gauge_id) in completed_basins:
        continue

    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.hstack([
        df_cal['Precip'].shift(1).values.reshape(-1, 1),
        df_cal['Temp'].shift(1).values.reshape(-1, 1),
        df_cal['PET'].shift(1).values.reshape(-1, 1)])# Observed - Simulation
    input_eva = np.hstack([
        df_eva['Precip'].shift(1).values.reshape(-1, 1),
        df_eva['Temp'].shift(1).values.reshape(-1, 1),
        df_eva['PET'].shift(1).values.reshape(-1, 1)]) # Observed - Simulation

    # shift(1) leaves a leading NaN row; align input and target on the same [1:] slice so
    # Precip/Temp/PET at day k-1 pairs with Obs flow at day k (was previously input[:-1]
    # vs target[1:], a 2-day misalignment plus a stray NaN row -- see DEBUG_README.md §16.13).
    input_cal = input_cal[1:]
    input_eva = input_eva[1:]

    target_cal = df_cal['Obs flow'].values[1:]
    target_eva = df_eva['Obs flow'].values[1:]

    scaler = StandardScaler()
    input_cal = scaler.fit_transform(input_cal)
    input_eva = scaler.transform(input_eva)

    # A handful of basins (e.g. 03066000, 12141300) have a full missing year inside the
    # calibration or evaluation period after reindex -- StandardScaler passes NaN through
    # silently, but RandomForestRegressor rejects NaN in both X and y at fit/predict time.
    # Fit and predict only on finite rows; leave NaN placeholders elsewhere so predict_cal/eva
    # keep the same length as the date-range header written by ensure_predict_header (BMK()
    # already NaN-masks for metrics).
    valid_cal = ~(np.isnan(target_cal) | np.isnan(input_cal).any(axis=1))
    valid_eva = ~(np.isnan(target_eva) | np.isnan(input_eva).any(axis=1))

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        min_samples_split=min_samples_split,
        oob_score=True,
        random_state=42,
        n_jobs=1  # Use single core as specified
    )

    rf.fit(input_cal[valid_cal], target_cal[valid_cal])

    predict_cal = np.full(len(target_cal), np.nan)
    predict_eva = np.full(len(target_eva), np.nan)
    predict_cal[valid_cal] = rf.predict(input_cal[valid_cal])
    predict_eva[valid_eva] = rf.predict(input_eva[valid_eva])

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0


    file_row_cal = [f'basin_{gauge_id}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'basin_{gauge_id}_eva'] + list(predict_eva.flatten())

    with open(predict_cal_path, 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(predict_eva_path, 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'RF_{benchmark}_cal': BMK(target_cal, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'RF_{benchmark}_eva': BMK(target_eva, predict_eva, benchmark)
        })  
    
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results to CSV after each file
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(results_cal_path, index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(results_eva_path, index=False, header=True)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")
    log_file.write(f"Results saved for basin {gauge_id}\n")
    print(f"Results saved for basin {gauge_id}")

log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/RF_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/RF_results_eva.csv', index=False, header=True)


print("DONE")
