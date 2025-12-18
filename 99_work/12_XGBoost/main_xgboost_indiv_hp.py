# based on https://github.com/xbeat/Machine-Learning/blob/main/XGBoost%20Regression%20with%20Python.md?utm_source=chatgpt.com
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from datetime import datetime

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

if loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num+1))
#file_list = list(range(20,30))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

# Parse hyperparameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100, help='Number of boosting rounds (I: 100/300)')
parser.add_argument('--max_depth', type=int, default=3, help='Maximum tree depth (I: 3/4)')
parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate (I: 0.05/0.1)')
args = parser.parse_args()

n_estimators = args.n_estimators
max_depth = args.max_depth
learning_rate = args.learning_rate

# Fixed hyperparameters
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 1

hp_str = f"n{n_estimators}_d{max_depth}_lr{learning_rate}"
output_dir = f"/data0/funato/hsp/0_out/XGB/{loc}/individual/hp/{hp_str}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)


def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
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

def BMK(obs_data,sim_data,benchmark):   
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask] 
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
    

if os.path.exists(output_dir + f'/XGB_{hp_str}_log.txt'):
    open(output_dir + f'/XGB_{hp_str}_log.txt', 'w').close()
log_file = open(output_dir + f'/XGB_{hp_str}_log.txt', 'a')

results_cal = []
results_eva = []

for file_num in file_list:
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(file_num)

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

    input_cal = input_cal[:-1] #2898
    input_eva = input_eva[:-1] #1089

    scaler = StandardScaler()
    input_cal = scaler.fit_transform(input_cal)
    input_eva = scaler.transform(input_eva)

    target_cal = df_cal['Obs flow'].values[1:] #2898
    target_eva = df_eva['Obs flow'].values[1:] #1089


    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=42
    )

    
    xgb_model.fit(input_cal, target_cal)
    
    predict_cal=xgb_model.predict(input_cal)
    predict_eva=xgb_model.predict(input_eva)

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    print(f"file_num: {file_num}, NSE_cal: {BMK(target_cal, predict_cal, 'NSE')}, NSE_eva: {BMK(target_eva, predict_eva, 'NSE')}")


    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

    if file_num == 1:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/XGB_predict_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/XGB_predict_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/XGB_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/XGB_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'XGB_{benchmark}_cal': BMK(target_cal, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'XGB_{benchmark}_eva': BMK(target_eva, predict_eva, benchmark)
        })  
    
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results to CSV after each file
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/XGB_results_cal.csv', index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/XGB_results_eva.csv', index=False, header=True)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")
    log_file.write(f"Results saved for file {file_num}\n")
    print(f"Results saved for file {file_num}")

log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/XGB_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/XGB_results_eva.csv', index=False, header=True)


print("DONE")
