import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from datetime import datetime

import shap

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

attribute_dir = f'/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

syn_precip_cols = ['syn_no_precip','syn_p_pet','syn_peak1_rate0',
                    'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                    'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                    'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    columns_drop = ['gauge_id']
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim"

#file_list = list(range(1, file_tot_num+1))
file_list = list(range(20,30))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

max_depth = 15
max_features = 0.7
n_estimators = 100
max_samples = 0.5
min_samples_split = 2

output_dir = f"/data0/funato/hsp2/0_out/RF/{loc}/shap"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True) # New directory for synthetic predictions

output_dir_shap = f"0_out/1_fig/shap/RF/{loc}"
os.makedirs(output_dir_shap, exist_ok=True)


# clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns])

# Convert all columns except 'File_num' to numeric, coercing errors to NaN
cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped

# Explicitly cast columns to float64 before normalization
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
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def load_attribute_values(input_data, file_num):
    attribute_row = attribute_values_normalized[attribute_values_normalized.index == file_num]
    attribute_list = attribute_row.iloc[0]
    attribute_list = attribute_list.drop('File_num') if 'File_num' in attribute_list.index else attribute_list
    static_attributes = np.array([attribute_list.values] * input_data.shape[0])
    input_data = np.hstack((input_data, static_attributes))
    return input_data

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
    

if os.path.exists(output_dir + f'/RF_log.txt'):
    open(output_dir + f'/RF_log.txt', 'w').close()
log_file = open(output_dir + f'/RF_log.txt', 'a')

results_cal = []
results_eva = []

dynamic_features = ["Precip", "Temp", "PET"]
static_features = attribute_values_normalized.columns.tolist()
# This will be used for plotting labels, but the model will receive a NumPy array.
all_feature_names = dynamic_features + [col for col in attribute_values_normalized.columns if col != 'File_num']

# Dictionary to store synthetic prediction dataframes, keyed by syn_precip_col
synthetic_predictions_dfs = {}

FirstFile = True
for file_num in file_list:
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Create dynamic input features for calibration
    temp_input_cal_dynamic = np.hstack([
        df_cal['Precip'].shift(1).values.reshape(-1, 1),
        df_cal['Temp'].shift(1).values.reshape(-1, 1),
        df_cal['PET'].shift(1).values.reshape(-1, 1)
    ])

    # Create dynamic input features for evaluation
    temp_input_eva_dynamic = np.hstack([
        df_eva['Precip'].shift(1).values.reshape(-1, 1),
        df_eva['Temp'].shift(1).values.reshape(-1, 1),
        df_eva['PET'].shift(1).values.reshape(-1, 1)
    ])

    temp_input_cal_dynamic = temp_input_cal_dynamic[:-1]
    temp_input_eva_dynamic = temp_input_eva_dynamic[:-1]

    valid_cal_rows = ~np.isnan(temp_input_cal_dynamic).any(axis=1)
    valid_eva_rows = ~np.isnan(temp_input_eva_dynamic).any(axis=1)

    input_cal_cleaned_dynamic = temp_input_cal_dynamic[valid_cal_rows]
    input_eva_cleaned_dynamic = temp_input_eva_dynamic[valid_eva_rows]

    mean_dynamic_features = np.mean(input_cal_cleaned_dynamic, axis=0)
    std_dynamic_features = np.std(input_cal_cleaned_dynamic, axis=0)
    std_dynamic_features = np.where(std_dynamic_features == 0, 1e-6, std_dynamic_features)

    input_cal_normalized_dynamic = (input_cal_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features
    input_eva_normalized_dynamic = (input_eva_cleaned_dynamic - mean_dynamic_features) / std_dynamic_features

    # X_train_final and X_eval_final will be NumPy arrays
    X_train_final = load_attribute_values(input_cal_normalized_dynamic, file_num)
    X_eval_final = load_attribute_values(input_eva_normalized_dynamic, file_num)

    target_cal_final = df_cal['Obs flow'].values[1:][valid_cal_rows]
    target_eva_final = df_eva['Obs flow'].values[1:][valid_eva_rows]

    assert X_train_final.shape[1] == len(all_feature_names), \
        f"Number of features ({X_train_final.shape[1]}) in X_train_final does not match all_feature_names length ({len(all_feature_names)})"

    # Convert to DataFrame *only for SHAP if necessary for plotting with names*
    # However, the model will be fitted on the NumPy array X_train_final
    # and the explainer will receive a NumPy array.
    # We will create X_train_final_df and X_eval_final_df *after* model fitting
    # just for the purpose of SHAP plotting which benefits from named columns.

    rf = RandomForestRegressor(max_depth=max_depth, max_features = max_features, n_estimators=n_estimators, min_samples_split=min_samples_split, oob_score=True, random_state=42, n_jobs=-1)
    
    # --- IMPORTANT CHANGE: Fit the model with the NumPy array ---
    rf.fit(X_train_final, target_cal_final) 
    
    predict_cal=rf.predict(X_train_final)
    predict_eva=rf.predict(X_eval_final)

    ### SHAP values ###
    # Now, convert to DataFrame for SHAP explanation and plotting if desired
    # This ensures consistency for the explainer (NumPy in, NumPy out for core calculations)
    # but still allows nice plotting with feature names.
    X_train_final_df_for_shap = pd.DataFrame(X_train_final, columns=all_feature_names)

    # SHAP Analysis for RandomForestRegressor
    explainer = shap.Explainer(rf, X_train_final_df_for_shap) # Pass the DataFrame for background
    shap_values = explainer(X_train_final_df_for_shap,check_additivity=False) # Pass the DataFrame for explanation

    # Debug prints (keep these, they are useful)
    if hasattr(rf, 'n_features_in_'):
        print(f"Model was trained with {rf.n_features_in_} features.")
        assert rf.n_features_in_ == X_train_final_df_for_shap.shape[1], "Mismatch in number of features between model and SHAP data."

    sample_index = 0
    sample_data = X_train_final_df_for_shap.iloc[[sample_index]]
    model_prediction = rf.predict(sample_data)[0] # Predict with DataFrame
    print(f"Model prediction for sample {sample_index}: {model_prediction}")

    explainer_debug = shap.Explainer(rf, X_train_final_df_for_shap)
    shap_values_debug = explainer_debug(sample_data)
    sum_of_shap_values = shap_values_debug.values.sum()
    expected_value = explainer_debug.expected_value
    print(f"Sum of SHAP values + expected_value for sample {sample_index}: {sum_of_shap_values + expected_value}")
    print(f"Difference: {(sum_of_shap_values + expected_value) - model_prediction}")

    # --- SHAP Plotting (using the DataFrame with names) ---
    shap.summary_plot(shap_values, X_train_final_df_for_shap, plot_type="bar", show=False)
    plt.savefig(f"{output_dir_shap}/RF_shap_eva_file_{file_num}_summary_bar.png")
    plt.close()

    shap.summary_plot(shap_values, X_train_final_df_for_shap, show=False)
    plt.savefig(f"{output_dir_shap}/RF_shap_eva_file_{file_num}_summary.png")
    plt.close()

    if isinstance(shap_values, shap.Explanation):
        shap_values_array = shap_values.values
    else:
        shap_values_array = shap_values

    shap.dependence_plot(0, shap_values_array, X_train_final_df_for_shap, show=False)
    plt.savefig(f"{output_dir_shap}/RF_shap_eva_file_{file_num}_dependence_0.png")
    plt.close()

    i = 0
    shap.save_html(f"{output_dir_shap}/force_plot_file_{file_num}.html", shap.force_plot(explainer.expected_value, shap_values_array[i], X_train_final_df_for_shap.iloc[i]))

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0


    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

    if FirstFile:
        FirstFile = False
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/RF_predict_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/RF_predict_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/RF_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/RF_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal.update({
            f'RF_{benchmark}_cal': BMK(target_cal_final, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'RF_{benchmark}_eva': BMK(target_eva_final, predict_eva, benchmark)
        })  
    
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    ## evaluate using synthetic precipitation with the already calibrated model
    full_prediction_dates = pd.date_range(
        start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), 
        end=pd.to_datetime(end_date_eva) + pd.DateOffset(days=1)
    )

    for syn_precip_col in syn_precip_cols:
        syn_input_cal = np.hstack([
            df_cal[syn_precip_col].shift(1).values.reshape(-1, 1),
            df_cal['Temp'].shift(1).values.reshape(-1, 1),
            df_cal['PET'].shift(1).values.reshape(-1, 1)])
        syn_input_eva = np.hstack([
            df_eva[syn_precip_col].shift(1).values.reshape(-1, 1),
            df_eva['Temp'].shift(1).values.reshape(-1, 1),
            df_eva['PET'].shift(1).values.reshape(-1, 1)])

        # Apply slicing for shifted data
        syn_input_cal_shifted = syn_input_cal[:-1]
        syn_input_eva_shifted = syn_input_eva[:-1]

        # Get original dates *before* NaN removal for alignment
        # These are the dates corresponding to the *shifted* input arrays
        original_cal_dates_shifted = pd.date_range(
            start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1),
            periods=len(syn_input_cal_shifted) # Use length of shifted data
        )
        original_eva_dates_shifted = pd.date_range(
            start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1),
            periods=len(syn_input_eva_shifted) # Use length of shifted data
        )

        # Identify rows with NaNs in the *shifted* data
        valid_indices_cal_shifted = ~np.isnan(syn_input_cal_shifted).any(axis=1)
        valid_indices_eva_shifted = ~np.isnan(syn_input_eva_shifted).any(axis=1)

        # Clean the input data using the valid indices
        syn_input_cal_clean = syn_input_cal_shifted[valid_indices_cal_shifted]
        syn_input_eva_clean = syn_input_eva_shifted[valid_indices_eva_shifted]

        # Calculate normalization parameters (mean and std) from the *cleaned* calibration dynamic data
        # Ensure mean_dynamic_features and std_dynamic_features are available from the main loop
        # (they should be, as they are defined globally relative to this loop)
        syn_input_cal_normalized = (syn_input_cal_clean - mean_dynamic_features) / std_dynamic_features
        syn_input_eva_normalized = (syn_input_eva_clean - mean_dynamic_features) / std_dynamic_features

        syn_input_cal_final = load_attribute_values(syn_input_cal_normalized, file_num)
        syn_input_eva_final = load_attribute_values(syn_input_eva_normalized, file_num)

        # Now, apply the boolean mask to the original shifted dates
        syn_cal_dates_cleaned = original_cal_dates_shifted[valid_indices_cal_shifted]
        syn_eva_dates_cleaned = original_eva_dates_shifted[valid_indices_eva_shifted]

        syn_predict_cal = rf.predict(syn_input_cal_final)
        syn_predict_eva = rf.predict(syn_input_eva_final)

        syn_predict_cal[syn_predict_cal < 0] = 0
        syn_predict_eva[syn_predict_eva < 0] = 0

        combined_syn_predictions = np.concatenate((syn_predict_cal, syn_predict_eva))
        combined_syn_dates = pd.to_datetime(list(syn_cal_dates_cleaned) + list(syn_eva_dates_cleaned))
        
        temp_syn_df = pd.DataFrame({
            'Date': combined_syn_dates,
            f'file_{file_num}_prediction': combined_syn_predictions
        })
        temp_syn_df.set_index('Date', inplace=True)

        if syn_precip_col not in synthetic_predictions_dfs:
            all_dates_for_syn_file = pd.date_range(
                start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), 
                end=pd.to_datetime(end_date_eva) + pd.DateOffset(days=1)
            )
            synthetic_predictions_dfs[syn_precip_col] = pd.DataFrame(index=all_dates_for_syn_file)
            synthetic_predictions_dfs[syn_precip_col].index.name = 'Date'
        
        synthetic_predictions_dfs[syn_precip_col] = synthetic_predictions_dfs[syn_precip_col].merge(
            temp_syn_df, 
            left_index=True, 
            right_index=True, 
            how='left'
        )

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")

log_file.write(f"DONE\n")
log_file.close()

for syn_precip_col, df_syn_preds in synthetic_predictions_dfs.items():
    df_syn_preds.to_csv(f"{output_dir}/predict/synthetic/RF_syn_predict_{syn_precip_col}.csv", index=True, header=True)

df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/RF_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/RF_results_eva.csv', index=False, header=True)


print("DONE")