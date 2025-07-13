import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

import shap

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

# loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

attribute_dir = f'/data0/funato/hsp/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 'syn_peak1_rate0',
                   'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

dynamic_features = ["Precip", "Temp", "PET"]

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no', 'river', 'station', 'WaterArea', 'ForestArea', 'ForestAreaRatio', 'WaterAreaRatio', 'lat_org', 'long_org', 'land_GolfCourse', 'land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding='UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    columns_drop = ['gauge_id']
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding='UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num + 1))
# file_list = list(range(20,21)) # Uncomment for quick test with one file

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31' # Original end date for standard evaluation

max_depth = 5
max_features = 0.7
n_estimators = 100
max_samples = 0.5
min_samples_split = 2

output_dir = f"/data0/funato/hsp/0_out/11_RandomForest/{loc}/shap/regional"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)

output_dir_shap = f"0_out/1_fig/shap/11_RandomForest/regional"
os.makedirs(output_dir_shap, exist_ok=True)

# Clean attribute values and normalize (global normalization for static attributes)
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
    if len(obs_data) == 0 or len(sim_data) == 0: return np.nan
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

all_input_cal_raw = []
all_target_cal_raw = []
all_input_eva_raw = []
all_target_eva_raw = []

basin_cal_info = {}
basin_eva_info = {}

start_time_total = datetime.now()
start_time_total_st = start_time_total.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file = open(output_dir + f'/RF_log.txt', 'w')
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
    if current_cal_len > 0:
        basin_cal_info[file_num] = {'start_idx': len(np.vstack(all_input_cal_raw)) if all_input_cal_raw else 0, 'length': current_cal_len, 'dates': df_cal.index[1:][valid_cal_rows]}
        all_input_cal_raw.append(input_cal_cleaned_dynamic)
        all_target_cal_raw.append(target_cal_cleaned)

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
    if current_eva_len > 0:
        basin_eva_info[file_num] = {'start_idx': len(np.vstack(all_input_eva_raw)) if all_input_eva_raw else 0, 'length': current_eva_len, 'dates': df_eva.index[1:][valid_eva_rows]}
        all_input_eva_raw.append(input_eva_cleaned_dynamic)
        all_target_eva_raw.append(target_eva_cleaned)

all_input_cal_stacked = np.vstack(all_input_cal_raw)
all_target_cal_stacked = np.hstack(all_target_cal_raw)
all_input_eva_stacked = np.vstack(all_input_eva_raw)
all_target_eva_stacked = np.hstack(all_target_eva_raw)

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
for file_num in file_list:
    if file_num in basin_cal_info:
        cal_len = basin_cal_info[file_num]['length']
        basin_cal_dynamic = all_input_cal_normalized_dynamic[current_cal_idx : current_cal_idx + cal_len]
        X_train_final_list.append(load_attribute_values_for_basin(basin_cal_dynamic, file_num))
        current_cal_idx += cal_len

current_eva_idx = 0
for file_num in file_list:
    if file_num in basin_eva_info:
        eva_len = basin_eva_info[file_num]['length']
        basin_eva_dynamic = all_input_eva_normalized_dynamic[current_eva_idx : current_eva_idx + eva_len]
        X_eval_final_list.append(load_attribute_values_for_basin(basin_eva_dynamic, file_num))
        current_eva_idx += eva_len

X_train_final = np.vstack(X_train_final_list)
X_eval_final = np.vstack(X_eval_final_list)

print("Training the regional Random Forest model...")
all_feature_names = dynamic_features + [col for col in attribute_values_normalized.columns if col != 'File_num']

assert X_train_final.shape[1] == len(all_feature_names), \
    f"Number of features ({X_train_final.shape[1]}) in X_train_final does not match all_feature_names length ({len(all_feature_names)})"

rf = RandomForestRegressor(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators, min_samples_split=min_samples_split, oob_score=True, random_state=42, n_jobs=-1)

rf.fit(X_train_final, all_target_cal_stacked)
print("Regional Random Forest model trained.")

predict_cal_aggregated = rf.predict(X_train_final)
predict_eva_aggregated = rf.predict(X_eval_final)

predict_cal_aggregated[predict_cal_aggregated < 0] = 0
predict_eva_aggregated[predict_eva_aggregated < 0] = 0

print("Performing SHAP analysis...")
# For performance, let's explain a subset of the training data
explain_size = 5000
if X_train_final.shape[0] > explain_size:
    explain_indices = np.random.choice(X_train_final.shape[0], explain_size, replace=False)
    X_explain = X_train_final[explain_indices]
    X_explain_df = pd.DataFrame(X_explain, columns=all_feature_names)
else:
    X_explain_df = pd.DataFrame(X_train_final, columns=all_feature_names)

# Use TreeExplainer for Random Forest, which is much faster
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_explain_df)

# --- Save SHAP results for later analysis ---
print("Saving SHAP objects for later analysis...")
shap_results_dir = os.path.join(output_dir, "shap_objects")
os.makedirs(shap_results_dir, exist_ok=True)

# Save the SHAP values (as numpy array)
np.save(os.path.join(shap_results_dir, "shap_values.npy"), shap_values.values)

# Save the expected value (base value)
np.save(os.path.join(shap_results_dir, "expected_value.npy"), shap_values.base_values)

# Save the corresponding feature data
X_explain_df.to_csv(os.path.join(shap_results_dir, "X_features.csv"), index=False)

print("SHAP objects saved.")

shap.summary_plot(shap_values, X_explain_df, plot_type="bar", show=False)
plt.savefig(f"{output_dir_shap}/RF_shap_regional_summary_bar.png")
plt.close()

shap.summary_plot(shap_values, X_explain_df, show=False)
plt.savefig(f"{output_dir_shap}/RF_shap_regional_summary.png")
plt.close()

# For dependence plots, we still use the numpy array of shap values
shap_values_array = shap_values.values

for i in range(min(3, X_explain_df.shape[1])):
    shap.dependence_plot(i, shap_values_array, X_explain_df, show=False)
    plt.savefig(f"{output_dir_shap}/RF_shap_regional_dependence_{X_explain_df.columns[i]}.png")
    plt.close()

for i in range(min(3, X_explain_df.shape[0])):
    shap.save_html(
        f"{output_dir_shap}/force_plot_regional_sample_{i}.html",
        shap.force_plot(
            shap_values.base_values[i],
            shap_values.values[i],
            X_explain_df.iloc[i],
            feature_names=all_feature_names
        )
    )
print("SHAP analysis complete.")

print("Saving per-basin predictions and results...")
results_cal = []
results_eva = []

predict_cal_dfs = {}
predict_eva_dfs = {}
syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}

current_cal_idx = 0
current_eva_idx = 0
for file_num in file_list:
    if file_num in basin_cal_info:
        cal_info = basin_cal_info[file_num]
        basin_predict_cal = predict_cal_aggregated[current_cal_idx : current_cal_idx + cal_info['length']]
        basin_target_cal = all_target_cal_stacked[current_cal_idx : current_cal_idx + cal_info['length']]
        current_cal_idx += cal_info['length']
        
        file_results_cal = {'file_num': file_num}
        for benchmark in benchmark_list:
            file_results_cal[f'RF_{benchmark}_cal'] = BMK(basin_target_cal, basin_predict_cal, benchmark)
        results_cal.append(file_results_cal)
        predict_cal_dfs[file_num] = pd.DataFrame({'Date': cal_info['dates'], 'Prediction': basin_predict_cal})

    if file_num in basin_eva_info:
        eva_info = basin_eva_info[file_num]
        basin_predict_eva = predict_eva_aggregated[current_eva_idx : current_eva_idx + eva_info['length']]
        basin_target_eva = all_target_eva_stacked[current_eva_idx : current_eva_idx + eva_info['length']]
        current_eva_idx += eva_info['length']

        file_results_eva = {'file_num': file_num}
        for benchmark in benchmark_list:
            file_results_eva[f'RF_{benchmark}_eva'] = BMK(basin_target_eva, basin_predict_eva, benchmark)
        results_eva.append(file_results_eva)
        predict_eva_dfs[file_num] = pd.DataFrame({'Date': eva_info['dates'], 'Prediction': basin_predict_eva})

    # --- Synthetic Data Evaluation ---
    df_current_basin = load_data(file_num)
    df_cal_current = df_current_basin[start_date_cal:end_date_cal].copy()
    
    for syn_precip_col in syn_precip_cols:
        if syn_precip_col.startswith('syn_peak'):
            end_date_eva_syn = '2001-02-10'
        else:
            end_date_eva_syn = end_date_eva

        df_eva_current = df_current_basin[start_date_eva:end_date_eva_syn].copy()

        syn_temp_col = 'syn_Temp' if 'peak' in syn_precip_col else 'Temp'
        syn_pet_col = 'syn_PET' if 'peak' in syn_precip_col else 'PET'

        syn_input_cal_dyn = np.hstack([df_cal_current[syn_precip_col].shift(1).values.reshape(-1, 1), df_cal_current[syn_temp_col].shift(1).values.reshape(-1, 1), df_cal_current[syn_pet_col].shift(1).values.reshape(-1, 1)])[1:]
        syn_input_eva_dyn = np.hstack([df_eva_current[syn_precip_col].shift(1).values.reshape(-1, 1), df_eva_current[syn_temp_col].shift(1).values.reshape(-1, 1), df_eva_current[syn_pet_col].shift(1).values.reshape(-1, 1)])[1:]
        
        valid_cal = ~np.isnan(syn_input_cal_dyn).any(axis=1)
        valid_eva = ~np.isnan(syn_input_eva_dyn).any(axis=1)

        syn_input_cal_clean = syn_input_cal_dyn[valid_cal]
        syn_input_eva_clean = syn_input_eva_dyn[valid_eva]

        if len(syn_input_cal_clean) > 0:
            syn_input_cal_normalized = (syn_input_cal_clean - mean_dynamic_features) / std_dynamic_features
            syn_input_cal_final = load_attribute_values_for_basin(syn_input_cal_normalized, file_num)
            syn_predict_cal = rf.predict(syn_input_cal_final)
            syn_predict_cal[syn_predict_cal < 0] = 0
            syn_cal_dates_cleaned = df_cal_current.index[1:][valid_cal]
        else:
            syn_predict_cal = np.array([])
            syn_cal_dates_cleaned = pd.to_datetime([])

        if len(syn_input_eva_clean) > 0:
            syn_input_eva_normalized = (syn_input_eva_clean - mean_dynamic_features) / std_dynamic_features
            syn_input_eva_final = load_attribute_values_for_basin(syn_input_eva_normalized, file_num)
            syn_predict_eva = rf.predict(syn_input_eva_final)
            syn_predict_eva[syn_predict_eva < 0] = 0
            syn_eva_dates_cleaned = df_eva_current.index[1:][valid_eva]
        else:
            syn_predict_eva = np.array([])
            syn_eva_dates_cleaned = pd.to_datetime([])

        combined_syn_predictions = np.concatenate((syn_predict_cal, syn_predict_eva))
        combined_syn_dates = pd.to_datetime(list(syn_cal_dates_cleaned) + list(syn_eva_dates_cleaned))

        if len(combined_syn_dates) > 0:
            temp_syn_df = pd.DataFrame({'Date': combined_syn_dates, f'prediction': combined_syn_predictions}).set_index('Date')
            syn_predict_dfs_combined.setdefault(syn_precip_col, {})[file_num] = temp_syn_df

for file_num, df_preds in predict_cal_dfs.items():
    df_preds.to_csv(f"{output_dir}/predict/RF_predict_cal_basin_{file_num}.csv", index=False, header=True)

for file_num, df_preds in predict_eva_dfs.items():
    df_preds.to_csv(f"{output_dir}/predict/RF_predict_eva_basin_{file_num}.csv", index=False, header=True)

os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)
for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
    if not basin_preds_dict: continue
    
    combined_syn_df = pd.DataFrame()
    for file_num in file_list:
        if file_num in basin_preds_dict:
            df_preds = basin_preds_dict[file_num]
            df_preds = df_preds.rename(columns={'prediction': f'file_{file_num}'})
            if combined_syn_df.empty:
                combined_syn_df = df_preds
            else:
                combined_syn_df = combined_syn_df.join(df_preds, how='outer')

    if not combined_syn_df.empty:
        if syn_precip_col.startswith('syn_peak'):
            end_date_syn_final = '2001-02-10'
            combined_syn_df = combined_syn_df.loc[:end_date_syn_final]
        
        combined_syn_df.sort_index(inplace=True)
        combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/RF_syn_predict_{syn_precip_col}.csv", index=True, header=True)

df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/RF_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/RF_results_eva.csv', index=False, header=True)

end_time_total = datetime.now()
end_time_total_st = end_time_total.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Regional training end: {end_time_total_st}\n")
log_file.write(f"Total elapsed: {end_time_total - start_time_total}\n")
log_file.close()

print("DONE")
