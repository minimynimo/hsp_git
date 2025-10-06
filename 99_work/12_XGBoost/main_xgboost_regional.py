import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
import xgboost as xgb

from datetime import datetime


benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

# loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 'syn_peak1_rate0',
                   'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

dynamic_features = ["Precip", "Temp", "PET"]

if loc == "JP" and ver == 2:
    file_tot_num = 87
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    # Adjusted file_tot_num to 20 as in the RF script for consistency. Change if needed.
    file_tot_num = 424
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num + 1))
# file_list = list(range(20,22)) # Uncomment for quick test

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

# XGBoost Parameters
n_estimators = 500
learning_rate = 0.1
max_depth = 4

output_dir = f"/data0/funato/hsp/0_out/XGB/{loc}/regional"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)


# Clean attribute values and normalize
# Static attributes removed - using only dynamic features

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
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    df = df.reindex(full_date_range)
    return df

def BMK(obs_data, sim_data, benchmark):
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

# Initialize lists and dicts for aggregated data
all_input_cal_raw = []
all_target_cal_raw = []
all_input_eva_raw = []
all_target_eva_raw = []

basin_cal_info = {}
basin_eva_info = {}

start_time_total = datetime.now()
start_time_total_st = start_time_total.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file = open(output_dir + f'/XGB_log.txt', 'w')
log_file.write(f"Regional training start: {start_time_total_st}\n")
log_file.flush()

print("Collecting and preprocessing data from all basins...")
for file_idx, file_num in enumerate(file_list):
    log_file.write(f"  Processing FILE_{file_num}\n")
    log_file.flush()

    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    input_cal_dynamic_shifted_raw = np.hstack([
        df_cal['Precip'].shift(1).values.reshape(-1, 1),
        df_cal['Temp'].shift(1).values.reshape(-1, 1),
        df_cal['PET'].shift(1).values.reshape(-1, 1)
    ])
    
    input_cal_dynamic_aligned = input_cal_dynamic_shifted_raw[1:]
    target_cal_aligned = df_cal['Obs flow'].values[1:]

    valid_cal_rows = ~np.isnan(input_cal_dynamic_aligned).any(axis=1) & ~np.isnan(target_cal_aligned)
    
    input_cal_cleaned_dynamic = input_cal_dynamic_aligned[valid_cal_rows]
    target_cal_cleaned = target_cal_aligned[valid_cal_rows]

    current_cal_len = len(input_cal_cleaned_dynamic)
    basin_cal_info[file_num] = {'start_idx': len(all_input_cal_raw), 'length': current_cal_len, 'dates': df_cal.index[1:][valid_cal_rows]}
    all_input_cal_raw.extend(input_cal_cleaned_dynamic)
    all_target_cal_raw.extend(target_cal_cleaned)

    input_eva_dynamic_shifted_raw = np.hstack([
        df_eva['Precip'].shift(1).values.reshape(-1, 1),
        df_eva['Temp'].shift(1).values.reshape(-1, 1),
        df_eva['PET'].shift(1).values.reshape(-1, 1)
    ])
    
    input_eva_dynamic_aligned = input_eva_dynamic_shifted_raw[1:]
    target_eva_aligned = df_eva['Obs flow'].values[1:]

    valid_eva_rows = ~np.isnan(input_eva_dynamic_aligned).any(axis=1) & ~np.isnan(target_eva_aligned)

    input_eva_cleaned_dynamic = input_eva_dynamic_aligned[valid_eva_rows]
    target_eva_cleaned = target_eva_aligned[valid_eva_rows]

    current_eva_len = len(input_eva_cleaned_dynamic)
    basin_eva_info[file_num] = {'start_idx': len(all_input_eva_raw), 'length': current_eva_len, 'dates': df_eva.index[1:][valid_eva_rows]}
    all_input_eva_raw.extend(input_eva_cleaned_dynamic)
    all_target_eva_raw.extend(target_eva_cleaned)

# Convert lists to stacked NumPy arrays
all_input_cal_stacked = np.array(all_input_cal_raw)
all_target_cal_stacked = np.array(all_target_cal_raw)
all_input_eva_stacked = np.array(all_input_eva_raw)
all_target_eva_stacked = np.array(all_target_eva_raw)


print("Normalizing dynamic features using aggregated calibration data...")
mean_dynamic_features = np.mean(all_input_cal_stacked[:, :3], axis=0)
std_dynamic_features = np.std(all_input_cal_stacked[:, :3], axis=0)
std_dynamic_features = np.where(std_dynamic_features == 0, 1e-6, std_dynamic_features)

all_input_cal_normalized_dynamic = (all_input_cal_stacked[:, :3] - mean_dynamic_features) / std_dynamic_features
all_input_eva_normalized_dynamic = (all_input_eva_stacked[:, :3] - mean_dynamic_features) / std_dynamic_features

X_train_final_list = []
X_eval_final_list = []

print("Combining dynamic and static features for all basins...")
current_cal_idx = 0
current_eva_idx = 0
for file_num in file_list:
    cal_len = basin_cal_info[file_num]['length']
    eva_len = basin_eva_info[file_num]['length']

    basin_cal_dynamic = all_input_cal_normalized_dynamic[current_cal_idx : current_cal_idx + cal_len]
    basin_eva_dynamic = all_input_eva_normalized_dynamic[current_eva_idx : current_eva_idx + eva_len]

    X_train_final_list.append(basin_cal_dynamic)  # Only dynamic features
    X_eval_final_list.append(basin_eva_dynamic)  # Only dynamic features

    current_cal_idx += cal_len
    current_eva_idx += eva_len

X_train_final = np.vstack(X_train_final_list)
X_eval_final = np.vstack(X_eval_final_list)

print("Training the regional XGBoost model...")
all_feature_names = dynamic_features  # Only dynamic features

assert X_train_final.shape[1] == len(all_feature_names), "Feature mismatch"

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_final, all_target_cal_stacked)
print("Regional XGBoost model trained.")

predict_cal_aggregated = xgb_model.predict(X_train_final)
predict_eva_aggregated = xgb_model.predict(X_eval_final)

predict_cal_aggregated[predict_cal_aggregated < 0] = 0
predict_eva_aggregated[predict_eva_aggregated < 0] = 0



print("Saving per-basin predictions, results, and evaluating synthetic data...")
results_cal = []
results_eva = []
syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}

# Reset indices for slicing predictions
current_cal_idx = 0
current_eva_idx = 0

for file_num in file_list:
    # --- Standard Predictions & Results ---
    cal_info = basin_cal_info[file_num]
    eva_info = basin_eva_info[file_num]
    
    basin_predict_cal = predict_cal_aggregated[current_cal_idx : current_cal_idx + cal_info['length']]
    basin_target_cal = all_target_cal_stacked[current_cal_idx : current_cal_idx + cal_info['length']]
    
    basin_predict_eva = predict_eva_aggregated[current_eva_idx : current_eva_idx + eva_info['length']]
    basin_target_eva = all_target_eva_stacked[current_eva_idx : current_eva_idx + eva_info['length']]

    # Save daily predictions
    pd.DataFrame({'Date': cal_info['dates'], 'Prediction': basin_predict_cal}).to_csv(f"{output_dir}/predict/og/XGB_predict_cal_basin_{file_num}.csv", index=False)
    pd.DataFrame({'Date': eva_info['dates'], 'Prediction': basin_predict_eva}).to_csv(f"{output_dir}/predict/og/XGB_predict_eva_basin_{file_num}.csv", index=False)

    # Calculate and store per-basin benchmark results
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}
    for benchmark in benchmark_list:
        file_results_cal[f'{benchmark}'] = BMK(basin_target_cal, basin_predict_cal, benchmark)
        file_results_eva[f'{benchmark}'] = BMK(basin_target_eva, basin_predict_eva, benchmark)
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Update indices for next basin
    current_cal_idx += cal_info['length']
    current_eva_idx += eva_info['length']

    # --- Synthetic Data Evaluation ---
    df_current_basin = load_data(file_num)
    df_cal_current = df_current_basin[start_date_cal:end_date_cal].copy()

    for syn_precip_col in syn_precip_cols:
        if syn_precip_col.startswith('syn_peak'):
            end_date_eva_syn = '2001-02-10'
        else:
            end_date_eva_syn = end_date_eva

        df_current = df_current_basin[start_date_cal:end_date_eva_syn].copy()
        
        syn_temp_col = 'Temp_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'Temp'
        syn_pet_col = 'PET_at_peak' if syn_precip_col not in ['syn_no_precip', 'syn_p_pet'] else 'PET'

        syn_input_dyn = np.hstack([df_current[syn_precip_col].shift(1).values.reshape(-1, 1), df_current[syn_temp_col].shift(1).values.reshape(-1, 1), df_current[syn_pet_col].shift(1).values.reshape(-1, 1)])[1:]
        
        valid = ~np.isnan(syn_input_dyn).any(axis=1)
        syn_input_clean = syn_input_dyn[valid]
        syn_dates = df_current.index[1:][valid]

        syn_input_norm = (syn_input_clean - mean_dynamic_features) / std_dynamic_features

        syn_input_final = syn_input_norm  # Only dynamic features

        syn_predict = xgb_model.predict(syn_input_final)
        syn_predict[syn_predict < 0] = 0

        temp_syn_df = pd.DataFrame({'Date': syn_dates, 'prediction': syn_predict})
        syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df

# --- Final Save Operations ---
for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
    combined_syn_df = pd.DataFrame()
    for file_num, df_preds in basin_preds_dict.items():
        df_preds = df_preds.rename(columns={'prediction': f'file_{file_num}'})
        df_preds = df_preds.set_index('Date')
        if combined_syn_df.empty:
            combined_syn_df = df_preds
        else:
            combined_syn_df = combined_syn_df.join(df_preds, how='outer')
    if syn_precip_col.startswith('syn_peak'):
        end_date_syn_final = '2001-02-10'
        combined_syn_df = combined_syn_df.loc[:end_date_syn_final]
    
    combined_syn_df.sort_index(inplace=True)
    combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/XGB_predict_{syn_precip_col.replace('syn_','')}.csv")

df_results_cal = pd.DataFrame(results_cal).set_index('file_num')
df_results_cal.to_csv(output_dir + f'/results/XGB_results_cal.csv')

df_results_eva = pd.DataFrame(results_eva).set_index('file_num')
df_results_eva.to_csv(output_dir + f'/results/XGB_results_eva.csv')

end_time_total = datetime.now()
end_time_total_st = end_time_total.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Regional training end: {end_time_total_st}\n")
log_file.write(f"Total elapsed: {end_time_total - start_time_total}\n")
log_file.close()

print("DONE")
