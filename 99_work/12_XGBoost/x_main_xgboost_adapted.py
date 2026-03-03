import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
import xgboost as xgb
import joblib

from datetime import datetime

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

# loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

attribute_dir = f'/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 
                   'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0','syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0','syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

dynamic_features = ["Precip", "Temp", "PET"]

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no', 'river', 'station', 'WaterArea', 'ForestArea', 'ForestAreaRatio', 'WaterAreaRatio', 'lat_org', 'long_org', 'land_GolfCourse', 'land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding='UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    # Adjusted file_tot_num to 20 as in the RF script for consistency. Change if needed.
    file_tot_num = 424
    columns_drop = ['gauge_id']
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding='UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim"

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

output_dir = f"/data0/funato/hsp2/0_out/XGB/{loc}/adapted_424"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)

# Clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns])

cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped

for col in attribute_values_numeric.columns:
    if attribute_values_numeric[col].dtype == 'int64':
        attribute_values_numeric[col] = attribute_values_numeric[col].astype('float64')

attribute_mean = attribute_values_numeric.mean()
attribute_std = attribute_values_numeric.std()
attribute_values_normalized = attribute_values_dropped.copy()
attribute_values_normalized.loc[:, attribute_values_numeric.columns] = (attribute_values_numeric - attribute_mean) / attribute_std

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

def load_attribute_values_for_basin(basin_dynamic_data, file_num):
    attribute_row = attribute_values_normalized[attribute_values_normalized.index == file_num]
    attribute_list = attribute_row.iloc[0]
    attribute_list = attribute_list.drop('File_num') if 'File_num' in attribute_list.index else attribute_list
    static_attributes_array = np.array([attribute_list.values] * basin_dynamic_data.shape[0])
    combined_data = np.hstack((basin_dynamic_data, static_attributes_array))
    return combined_data

def BMK(obs_data, sim_data, benchmark):
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

    X_train_final_list.append(load_attribute_values_for_basin(basin_cal_dynamic, file_num))
    X_eval_final_list.append(load_attribute_values_for_basin(basin_eva_dynamic, file_num))

    current_cal_idx += cal_len
    current_eva_idx += eva_len

X_train_final = np.vstack(X_train_final_list)
X_eval_final = np.vstack(X_eval_final_list)

# Define feature names (for consistency and potential debugging)
all_feature_names = dynamic_features + [col for col in attribute_values_normalized.columns if col != 'File_num']

# Try to load pre-trained regional model from shap_regional directory
regional_model_path = f"/data0/funato/hsp2/0_out/XGB/{loc}/shap/regional_424/XGB_model.pkl"
if os.path.exists(regional_model_path):
    print(f"Loading pre-trained regional model from {regional_model_path}...")
    xgb_model = joblib.load(regional_model_path)
    print("Pre-trained regional XGBoost model loaded successfully.")
else:
    print("No pre-trained model found. Training a new regional XGBoost model...")
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

print("Fine-tuning XGBoost model for each basin and generating predictions...")
results_cal = []
results_eva = []
syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}

for file_num in file_list:
    print(f"Fine-tuning and evaluating basin {file_num}...")
    start_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, start: {start_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
    
    # Get this basin's data
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Prepare calibration data for fine-tuning
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
    
    # Prepare evaluation data
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

    # Create basin-specific fine-tuned model if sufficient data available
    if len(input_cal_cleaned_dynamic) > 0 and len(target_cal_cleaned) > 0:
        # Normalize features
        input_cal_normalized_dynamic = (input_cal_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features
        input_eva_normalized_dynamic = (input_eva_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features
        
        # Add static attributes
        X_cal_basin_final = load_attribute_values_for_basin(input_cal_normalized_dynamic, file_num)
        X_eva_basin_final = load_attribute_values_for_basin(input_eva_normalized_dynamic, file_num)
        
        # Fine-tune: Train a new XGBoost model with basin-specific data
        # Using fewer trees for fine-tuning to prevent overfitting
        xgb_basin = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=max(50, n_estimators//10),  # Fewer trees for fine-tuning
            max_depth=max_depth,
            learning_rate=learning_rate * 2,  # Higher learning rate for fine-tuning
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Initialize with regional model weights if possible, then fine-tune
        try:
            # Copy the regional model and continue training (warm start approach)
            import copy
            xgb_basin = copy.deepcopy(xgb_model)
            xgb_basin.n_estimators = max(50, n_estimators//10)
            
            # Additional training on basin-specific data
            xgb_basin.fit(X_cal_basin_final, target_cal_cleaned, 
                         xgb_model=xgb_basin.get_booster(),  # Continue from regional model
                         verbose=False)
        except:
            # Fallback: train new model on basin data
            xgb_basin.fit(X_cal_basin_final, target_cal_cleaned)
        
        # Ensemble prediction: combine regional model with basin-specific model
        predict_cal_regional = xgb_model.predict(X_cal_basin_final)
        predict_cal_basin = xgb_basin.predict(X_cal_basin_final)
        predict_cal_combined = 0.7 * predict_cal_regional + 0.3 * predict_cal_basin  # Weighted ensemble
        
        predict_eva_regional = xgb_model.predict(X_eva_basin_final)
        predict_eva_basin = xgb_basin.predict(X_eva_basin_final)
        predict_eva_combined = 0.7 * predict_eva_regional + 0.3 * predict_eva_basin  # Weighted ensemble
        
    else:
        # Fallback to regional model only
        if len(input_cal_cleaned_dynamic) > 0:
            input_cal_normalized_dynamic = (input_cal_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features
            X_cal_basin_final = load_attribute_values_for_basin(input_cal_normalized_dynamic, file_num)
            predict_cal_combined = xgb_model.predict(X_cal_basin_final)
        else:
            predict_cal_combined = np.array([])
            
        if len(input_eva_cleaned_dynamic) > 0:
            input_eva_normalized_dynamic = (input_eva_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features
            X_eva_basin_final = load_attribute_values_for_basin(input_eva_normalized_dynamic, file_num)
            predict_eva_combined = xgb_model.predict(X_eva_basin_final)
        else:
            predict_eva_combined = np.array([])
    
    predict_cal_combined[predict_cal_combined < 0] = 0
    predict_eva_combined[predict_eva_combined < 0] = 0

    # Save individual basin predictions
    if len(predict_cal_combined) > 0:
        cal_dates = df_cal.index[1:][valid_cal_rows]
        pd.DataFrame({'Date': cal_dates, 'Observed': target_cal_cleaned, 'Predicted': predict_cal_combined}).to_csv(f"{output_dir}/predict/og/XGB_predict_cal_basin_{file_num}.csv", index=False)

    if len(predict_eva_combined) > 0:
        eva_dates = df_eva.index[1:][valid_eva_rows]
        pd.DataFrame({'Date': eva_dates, 'Observed': target_eva_cleaned, 'Predicted': predict_eva_combined}).to_csv(f"{output_dir}/predict/og/XGB_predict_eva_basin_{file_num}.csv", index=False)

    # Calculate benchmarks
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal[f'XGB_{benchmark}_cal'] = BMK(target_cal_cleaned, predict_cal_combined, benchmark) if len(predict_cal_combined) > 0 else np.nan
        file_results_eva[f'XGB_{benchmark}_eva'] = BMK(target_eva_cleaned, predict_eva_combined, benchmark) if len(predict_eva_combined) > 0 else np.nan

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # --- Synthetic Data Evaluation with fine-tuned model ---
    df_current_basin = load_data(file_num)

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

        if len(syn_input_clean) > 0:
            syn_input_norm = (syn_input_clean - mean_dynamic_features) / std_dynamic_features
            syn_input_final = load_attribute_values_for_basin(syn_input_norm, file_num)

            # Use ensemble prediction if basin-specific model exists, otherwise regional model
            syn_predict_regional = xgb_model.predict(syn_input_final)
            if len(input_cal_cleaned_dynamic) > 0 and len(target_cal_cleaned) > 0:
                try:
                    syn_predict_basin = xgb_basin.predict(syn_input_final)
                    syn_predict = 0.7 * syn_predict_regional + 0.3 * syn_predict_basin
                except:
                    syn_predict = syn_predict_regional
            else:
                syn_predict = syn_predict_regional
                
            syn_predict[syn_predict < 0] = 0

            temp_syn_df = pd.DataFrame({'Date': syn_dates, 'prediction': syn_predict})
            syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df

    # Save results incrementally after each basin
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/XGB_results_cal.csv', index=False, header=True)
    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/XGB_results_eva.csv', index=False, header=True)

    end_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, end: {end_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}, elapsed: {end_time_basin - start_time_basin}\n")
    log_file.write(f"Results saved for file {file_num}\n")
    print(f"Results saved for file {file_num}")
    log_file.flush()

# --- Final Save Operations ---
for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
    if not basin_preds_dict: continue
    
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
    
    # Transpose to have basins as rows (index) and dates as columns (like regional model format)
    combined_syn_df_transposed = combined_syn_df.T
    # Clean up the index by removing 'file_' prefix to just have basin numbers
    combined_syn_df_transposed.index = combined_syn_df_transposed.index.str.replace('file_', '')
    combined_syn_df_transposed.index.name = 'basin'
    
    combined_syn_df_transposed.to_csv(f"{output_dir}/predict/synthetic/XGB_predict_{syn_precip_col.replace('syn_','')}.csv")

end_time_total = datetime.now()
end_time_total_st = end_time_total.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Regional training end: {end_time_total_st}\n")
log_file.write(f"Total elapsed: {end_time_total - start_time_total}\n")
log_file.close()

print("DONE")
