# based on https://github.com/urgedata/pythondata/blob/master/rf_timeseries/Random%20Forest%20for%20Time%20Series%20Forecasting.ipynb?utm_source=chatgpt.com
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb

from sklearn.preprocessing import StandardScaler

from datetime import datetime

import shap

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

n_estimators = 500
learning_rate = 0.1

loc, ver = "JP", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

file_list = list(range(1, file_tot_num+1))
#file_list = list(range(20,22))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

max_depth = 15
max_features = 0.7
n_estimators = 100
max_samples = 0.5
min_samples_split = 2

output_dir = f"/data0/funato/hsp/0_out/12_XGBoost/{loc}/shap"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)

output_dir_shap = f"0_out/1_fig/shap/12_XGBoost/regional"
os.makedirs(output_dir_shap, exist_ok=True)


# clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns]) # 87,193
#print(attribute_values_dropped.head())

# Convert all columns except 'File_num' to numeric, coercing errors to NaN
cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped
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
    # input_data ; 2190, 3
    attribute_row = attribute_values_normalized[attribute_values_normalized.index == file_num]
    attribute_list = attribute_row.iloc[0] # 194,
    attribute_list = attribute_list.drop('File_num') if 'File_num' in attribute_list.index else attribute_list #193
    static_attributes = np.array([attribute_list.values] * input_data.shape[0])
    input_data = np.hstack((input_data, static_attributes))
    return input_data # 2190, 196

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
    

if os.path.exists(output_dir + f'/XGB_log.txt'):
    open(output_dir + f'/XGB_log.txt', 'w').close()
log_file = open(output_dir + f'/XGB_log.txt', 'a')

results_cal = []
results_eva = []

dynamic_features = ["Precip", "Temp", "PET"]
static_features = attribute_values_normalized.columns.tolist()
feature_names = dynamic_features + static_features

all_input_cal = []
all_input_eva = []

all_target_cal = []
all_target_eva = []

all_input_cal = []
all_target_cal = []

all_input_eva = []
all_target_eva = []

start_time = datetime.now()
start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"start: {start_time_st}\n")
log_file.flush()

for file_num in file_list:
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Extract dynamic features as before
    input_cal_dyn = np.hstack([
        df_cal['Precip'].shift(1).values.reshape(-1,1),
        df_cal['Temp'].shift(1).values.reshape(-1,1),
        df_cal['PET'].shift(1).values.reshape(-1,1)
    ])
    input_eva_dyn = np.hstack([
        df_eva['Precip'].shift(1).values.reshape(-1,1),
        df_eva['Temp'].shift(1).values.reshape(-1,1),
        df_eva['PET'].shift(1).values.reshape(-1,1)
    ])

    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1)

    input_cal_dyn = input_cal_dyn[mask_cal]
    input_eva_dyn = input_eva_dyn[mask_eva]

    # Load static features once for this basin
    static_features = attribute_values_normalized.loc[file_num]  # or however you index basins
    # Repeat static features for every timestep
    static_array_cal = np.tile(static_features.values, (input_cal_dyn.shape[0], 1))
    static_array_eva = np.tile(static_features.values, (input_eva_dyn.shape[0], 1))

    # Combine dynamic + static features
    input_cal_full = np.hstack([input_cal_dyn, static_array_cal])
    input_eva_full = np.hstack([input_eva_dyn, static_array_eva])

    # Append to global lists
    all_input_cal.append(input_cal_full)
    all_input_eva.append(input_eva_full)

    # Targets
    target_cal = df_cal['Obs flow'].values
    target_cal = target_cal[mask_cal]  # Apply mask to targets

    # Debugging: Check the dimensions of target_cal and mask_cal
    #print("Length of target_cal:", len(target_cal))
    #print("Length of mask_cal:", len(mask_cal))

    # Apply the mask
    all_target_cal.append(target_cal)

    target_eva = df_eva['Obs flow'].values
    target_eva = target_eva[mask_eva]  # Apply mask to targets
    all_target_eva.append(target_eva)

# Stack all basins together
all_input_cal = np.vstack(all_input_cal)
all_target_cal = np.hstack(all_target_cal)

all_input_eva = np.vstack(all_input_eva)
all_target_eva = np.hstack(all_target_eva)

all_input_cal = np.vstack(all_input_cal)
all_input_eva = np.vstack(all_input_eva)

all_target_cal = np.hstack(all_target_cal)
all_target_eva = np.hstack(all_target_eva)

mean_dyn = np.mean(all_input_cal[:, :3], axis=0)

# Debugging: Check the type and content of all_input_cal
#print("Type of all_input_cal:", type(all_input_cal))
#print("Shape of all_input_cal:", all_input_cal.shape)
#print("First few rows of all_input_cal[:, :3]:", all_input_cal[:, :3])

all_input_cal = np.array(all_input_cal, dtype=float)

# Compute standard deviation
std_dyn = np.std(all_input_cal[:, :3], axis=0)

# Normalize dynamic features
all_input_cal[:, :3] = (all_input_cal[:, :3] - mean_dyn) / std_dyn
all_input_eva[:, :3] = (all_input_eva[:, :3] - mean_dyn) / std_dyn


xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=n_estimators,
    max_depth=4,              # allow slightly more complexity
    learning_rate=learning_rate,       # slightly faster learning
    subsample=0.8,            
    colsample_bytree=0.8,
    reg_alpha=0.5,            # slightly less L1 regularization
    reg_lambda=0.5            # slightly less L2 regularization
)


# Ensure target_cal is correctly aligned with all_input_cal
all_input_cal = np.vstack(all_input_cal)  # Combine all input data
all_target_cal = np.hstack(all_target_cal)  # Combine all target data

# Debugging: Check alignment of input and target data
print("Shape of all_input_cal:", all_input_cal.shape)
print("Shape of all_target_cal:", all_target_cal.shape)

if all_input_cal.shape[0] != all_target_cal.shape[0]:
    raise ValueError(f"Mismatch between input rows ({all_input_cal.shape[0]}) and target rows ({all_target_cal.shape[0]}).")

# Train the model
xgb_model.fit(all_input_cal, all_target_cal)

predict_cal=xgb_model.predict(all_input_cal)
predict_eva=xgb_model.predict(all_input_eva)

### SHAP values ###
# SHAP Analysis for RandomForestRegressor
all_input_cal_df = pd.DataFrame(all_input_cal, columns=feature_names)

explainer = shap.Explainer(xgb_model, all_input_cal_df)
shap_values = explainer(all_input_cal_df, check_additivity=False)

# Debugging: Check the shape and columns of all_input_cal_df
print("Shape of all_input_cal_df:", all_input_cal_df.shape)
print("Columns of all_input_cal_df:", all_input_cal_df.columns)

# Debugging: Check the shape of all_input_cal
print("Shape of all_input_cal:", all_input_cal.shape)
print("Model expects features:", xgb_model.n_features_in_)

shap.summary_plot(shap_values, all_input_cal_df, plot_type="bar", show=False)
plt.savefig(f"{output_dir_shap}/XGB_shap_eva_file_regional_summary_bar.png")

shap.summary_plot(shap_values, all_input_cal_df, show=False)
plt.savefig(f"{output_dir_shap}/XGB_shap_eva_file_summary.png")

# Ensure shap_values is in the correct format
if isinstance(shap_values, shap.Explanation):
    shap_values = shap_values.values

shap.dependence_plot(0, shap_values, all_input_cal_df, show=False)
plt.savefig(f"{output_dir_shap}/XGB_shap_eva_file_dependence_0.png")

shap.dependence_plot(1, shap_values, all_input_cal_df, show=False)
plt.savefig(f"{output_dir_shap}/XGB_shap_eva_file_dependence_1.png")

shap.dependence_plot(2, shap_values, all_input_cal_df, show=False)
plt.savefig(f"{output_dir_shap}/XGB_shap_eva_file_dependence_2.png")


#shap.force_plot(shap_values.base_values[i], shap_values.values[i], input_cal.iloc[i])
# Ensure the base value is passed as the first parameter
#shap.save_html("force_plot.html", shap.force_plot(explainer.expected_value, shap_values[i]))


# Debug print statements to check shapes
print("Input shape:", all_input_cal.shape)
print("SHAP values shape:", shap_values.shape)
print("Explainer expected value:", explainer.expected_value)

# Ensure the base value is passed as the first parameter and feature names are included
for i in range(3):
    shap.save_html(
        f"{output_dir_shap}/force_plot_{i}.html",
        shap.force_plot(
            explainer.expected_value, 
            shap_values[i], 
            all_input_cal_df.iloc[i], 
            feature_names=all_input_cal_df.columns
        )
    )

predict_cal[predict_cal < 0] = 0
predict_eva[predict_eva < 0] = 0


file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

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
        f'XGB_{benchmark}_cal': BMK(all_target_cal, predict_cal, benchmark)
    })
    file_results_eva.update({
        f'XGB_{benchmark}_eva': BMK(all_target_eva, predict_eva, benchmark)
    })  

results_cal.append(file_results_cal)
results_eva.append(file_results_eva)


end_time = datetime.now()
end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"end: {end_time_st}\n")
log_file.write(f"elapsed: {end_time - start_time}\n")

log_file.write(f"DONE\n")
log_file.close()


df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/XGB_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/XGB_results_eva.csv', index=False, header=True)


print("DONE")
