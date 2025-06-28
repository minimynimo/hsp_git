import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from tcn import TCN
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import shap

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


loc, ver = "JP", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

file_list = list(range(1, file_tot_num+1))
#file_list = list(range(20,23))

input_dim = 3 + len(attribute_values.columns) - len(columns_drop)  # 3 meteorological + 193 attributes

timesteps, input_dim = 365, input_dim
epochs = 50 # Changed from 5 to 50 for better training potential
batch_size = 64
nb_filters = 128
kernel_size = 4
nb_stacks = 1


start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp/0_out/13_TCN/{loc}/shap"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True) # New directory for hydrographs

output_dir_shap = f"0_out/1_fig/shap/13_TCN/regional"
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


early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)


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
    
def create_sequences(data, targets, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

if os.path.exists(output_dir + f'/TCN_log.txt'):
    open(output_dir + f'/TCN_log.txt', 'w').close()
log_file = open(output_dir + f'/TCN_log.txt', 'a')


results_cal = []
results_eva = []

dynamic_features = ["Precip", "Temp", "PET"]
static_features = attribute_values_normalized.columns.tolist()
feature_names = dynamic_features + static_features

# Initialize lists to store data for all basins
all_input_cal_list = []
all_input_eva_list = []

all_target_cal_list = []
all_target_eva_list = []

start_time = datetime.now()
start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"start: {start_time_st}\n")
log_file.flush()

# --- Collecting Data for all Basins (for combined training) ---
# Also store individual basin data for later plotting
individual_basin_data = {}

for file_num in file_list:
    start_time_basin = datetime.now() # Use a different variable name to avoid confusion with overall start_time
    start_time_st_basin = start_time_basin.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time_st_basin}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal_dyn = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)])
    input_eva_dyn = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)])
    
    # Ensure masks consider both dynamic inputs and target
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    input_cal_dyn = input_cal_dyn[mask_cal]
    input_eva_dyn = input_eva_dyn[mask_eva]

    # Load static features once for this basin
    static_features_for_basin = attribute_values_normalized.loc[file_num]
    # Repeat static features for every timestep
    static_array_cal = np.tile(static_features_for_basin.values, (input_cal_dyn.shape[0], 1))
    static_array_eva = np.tile(static_features_for_basin.values, (input_eva_dyn.shape[0], 1))

    # Combine dynamic + static features
    input_cal_full = np.hstack([input_cal_dyn, static_array_cal])
    input_eva_full = np.hstack([input_eva_dyn, static_array_eva])

    # Append to global lists
    all_input_cal_list.append(input_cal_full)
    all_input_eva_list.append(input_eva_full)

    # Targets
    target_cal = df_cal['Obs flow'].values
    target_cal = target_cal[mask_cal]  # Apply mask to targets
    all_target_cal_list.append(target_cal)

    target_eva = df_eva['Obs flow'].values
    target_eva = target_eva[mask_eva]  # Apply mask to targets
    all_target_eva_list.append(target_eva)

    # Store original dates for individual plotting (before sequence creation)
    dates_cal_basin = df_cal.index[mask_cal] # Keep as pandas DatetimeIndex for now
    dates_eva_basin = df_eva.index[mask_eva] # Keep as pandas DatetimeIndex for now

    individual_basin_data[file_num] = {
        'input_cal_full': input_cal_full,
        'target_cal': target_cal,
        'dates_cal': dates_cal_basin,
        'input_eva_full': input_eva_full,
        'target_eva': target_eva,
        'dates_eva': dates_eva_basin,
    }


# Stack all basins together for combined training
all_input_cal = np.vstack(all_input_cal_list)
all_target_cal = np.hstack(all_target_cal_list)

all_input_eva = np.vstack(all_input_eva_list)
all_target_eva = np.hstack(all_target_eva_list)

all_input_cal = np.array(all_input_cal, dtype=float)

# Compute standard deviation for normalization of dynamic features
mean_dyn = np.mean(all_input_cal[:, :3], axis=0)
std_dyn = np.std(all_input_cal[:, :3], axis=0)

# Normalize dynamic features
all_input_cal[:, :3] = (all_input_cal[:, :3] - mean_dyn) / std_dyn
all_input_eva[:, :3] = (all_input_eva[:, :3] - mean_dyn) / std_dyn


# Create sequences for combined data
input_cal_seq, target_cal_seq = create_sequences(all_input_cal, all_target_cal, timesteps)
input_eva_seq, target_eva_seq = create_sequences(all_input_eva, all_target_eva, timesteps)

model_in = Input(shape=(timesteps, all_input_cal.shape[1])) # (365, 196)

# TCN block 
model_out = TCN(
    nb_filters=nb_filters,
    kernel_size=kernel_size,
    nb_stacks=nb_stacks,
    return_sequences=False
)(model_in)

# Output
model_out = Dense(1, activation='linear')(model_out)

# Build and compile
TCN_model = Model(inputs=model_in, outputs=model_out)
TCN_model.compile(optimizer='adam', loss='mse')


# Fit the model
TCN_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

predict_cal = TCN_model.predict(input_cal_seq)
predict_eva = TCN_model.predict(input_eva_seq)

### SHAP Analysis (only on a subset of the combined calibration data) ###
# Background data (e.g., 50 samples)
background = input_cal_seq[np.random.choice(input_cal_seq.shape[0], 50, replace=False)]

# Samples to explain (e.g., 10-100)
X_sample = input_cal_seq[:100]

explainer = shap.GradientExplainer((TCN_model.input, TCN_model.output), background)
shap_values = explainer.shap_values(X_sample)

shap_array = np.stack(shap_values, axis=0)  # Shape: (100, 365, 196, 1)
shap_vals_mean = shap_array.mean(axis=1)  # Shape: (100, 196, 1)
shap_vals_mean = shap_vals_mean.squeeze(axis=-1)  # Remove the last dimension, now shape: (100, 196)

# Average input over time for visualization
X_sample_mean = X_sample.mean(axis=1)  # Shape: (samples, features)

X_sample_df = pd.DataFrame(X_sample_mean, columns=feature_names)

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_mean, X_sample_df, show=False)
plt.tight_layout()
plt.savefig(f"{output_dir_shap}/TCN_shap_summary_beeswarm.png")
plt.close()

# bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_mean, X_sample_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{output_dir_shap}/TCN_shap_summary_bar.png")
plt.close()


# SHAP dependence plots (top 3 features based on summary plot)
# This will plot for the first 3 features in your feature_names list.
# You might want to get the actual top 3 features from shap_vals_mean and plot those.
for i in range(min(3, len(feature_names))): # Plot for up to top 3 features
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(i, shap_vals_mean, X_sample_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/TCN_shap_dependence_{feature_names[i].replace('/', '_')}.png") # Sanitize feature name for filename
    plt.close()

base_values = TCN_model.predict(X_sample).flatten() # Ensure base_values is 1D

# Force plots (top 3 examples)
for i in range(min(3, X_sample.shape[0])): # Plot for up to top 3 samples
    shap.save_html(
        f"{output_dir_shap}/force_plot_sample_{i}.html",
        shap.force_plot(
            base_value=base_values[i],
            shap_values=shap_vals_mean[i],
            features=X_sample_df.iloc[i],
            feature_names=feature_names
        )
    )

# Debug info
print("SHAP input sample shape:", X_sample.shape)
print("SHAP values mean shape:", shap_vals_mean.shape)

predict_cal[predict_cal < 0] = 0
predict_eva[predict_eva < 0] = 0

print("target_cal_seq", target_cal_seq.shape, "predict_cal", predict_cal.shape)
print("target_eva_seq", target_eva_seq.shape, "predict_eva", predict_eva.shape)

# --- Plotting Combined Hydrographs ---
# Determine the correct dates for the combined predictions
# The first prediction corresponds to the (timesteps)-th day of the combined data.
# Note: Ensure these dates align with the *actual* start of the sequences.
# The original df_cal and df_eva dates already start correctly.
# The create_sequences function effectively shifts the series forward by 'timesteps'.
# So, the first predicted value corresponds to the (timesteps + 1)th day of the *original*
# data period (if the data starts from day 1).
# We are aligning with target_cal_seq, which is already correctly trimmed by create_sequences.

# The `dates_cal_combined` and `dates_eva_combined` should be generated based on the length
# of `target_cal_seq` and `target_eva_seq` and their respective start dates.
# The `create_sequences` function uses `data[i+timesteps]`, meaning the first target `y[0]` corresponds
# to `data[timesteps]`. So the first date for the target sequence should be `start_date_cal + timesteps days`.

# We need to reconstruct the combined dates by taking the original `df_cal` and `df_eva` indices,
# applying the masks, and then taking the slice starting from `timesteps`.

# Reconstruct combined original dates (before sequence creation)
all_dates_cal_orig_flat = np.hstack([d.to_numpy() for d in [v['dates_cal'] for k, v in individual_basin_data.items()]])
all_dates_eva_orig_flat = np.hstack([d.to_numpy() for d in [v['dates_eva'] for k, v in individual_basin_data.items()]])


dates_cal_combined = all_dates_cal_orig_flat[timesteps:]
dates_eva_combined = all_dates_eva_orig_flat[timesteps:]


# Plot for combined Calibration
plt.figure(figsize=(15, 6))
# Convert dates to NumPy array for plotting
plt.plot(dates_cal_combined, target_cal_seq, label='Observed (Combined)', color='blue', alpha=0.7)
plt.plot(dates_cal_combined, predict_cal, label='Predicted (Combined)', color='red', alpha=0.7)
plt.title('Hydrograph: Combined Calibration Period')
plt.xlabel('Date')
plt.ylabel('Observed Flow')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hydrographs", "combined_hydrograph_cal.png"))
plt.close()

# Plot for combined Evaluation
plt.figure(figsize=(15, 6))
# Convert dates to NumPy array for plotting
plt.plot(dates_eva_combined, target_eva_seq, label='Observed (Combined)', color='blue', alpha=0.7)
plt.plot(dates_eva_combined, predict_eva, label='Predicted (Combined)', color='red', alpha=0.7)
plt.title('Hydrograph: Combined Evaluation Period')
plt.xlabel('Date')
plt.ylabel('Observed Flow')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hydrographs", "combined_hydrograph_eva.png"))
plt.close()

# --- Saving combined predictions to CSV ---
# Create a single DataFrame for calibration predictions
df_predict_cal_combined = pd.DataFrame({
    'Date': dates_cal_combined,
    'Observed': target_cal_seq.flatten(),
    'Predicted': predict_cal.flatten()
})
df_predict_cal_combined.to_csv(output_dir + f"/predict/TCN_predict_cal_combined.csv", index=False)

# Create a single DataFrame for evaluation predictions
df_predict_eva_combined = pd.DataFrame({
    'Date': dates_eva_combined,
    'Observed': target_eva_seq.flatten(),
    'Predicted': predict_eva.flatten()
})
df_predict_eva_combined.to_csv(output_dir + f"/predict/TCN_predict_eva_combined.csv", index=False)


# --- Re-calculating individual basin predictions and plotting ---
# This part is crucial because the model was trained on combined data.
# To get individual basin predictions, you need to:
# 1. Take the original, unscaled dynamic data for each basin.
# 2. Scale it using the mean/std from the *combined* training data.
# 3. Combine with normalized static data for that basin.
# 4. Create sequences for that basin's data.
# 5. Predict using the trained model.

for file_num in file_list:
    basin_data = individual_basin_data[file_num]

    # --- Calibration Data for current basin ---
    input_cal_full_basin = basin_data['input_cal_full'].copy()
    target_cal_basin = basin_data['target_cal']
    dates_cal_basin_orig = basin_data['dates_cal'] # This is already a pandas DatetimeIndex

    # Normalize dynamic features using the *combined* mean/std
    input_cal_full_basin[:, :3] = (input_cal_full_basin[:, :3] - mean_dyn) / std_dyn
    
    input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_full_basin, target_cal_basin, timesteps)
    predict_cal_basin = TCN_model.predict(input_cal_seq_basin).flatten()
    predict_cal_basin[predict_cal_basin < 0] = 0

    # Adjust dates for the predictions based on timesteps and convert to numpy
    dates_cal_plot = dates_cal_basin_orig[timesteps:].to_numpy()

    # Plot for individual basin Calibration
    plt.figure(figsize=(15, 6))
    plt.plot(dates_cal_plot, target_cal_seq_basin, label='Observed', color='blue', alpha=0.7)
    plt.plot(dates_cal_plot, predict_cal_basin, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Hydrograph (Calibration) - File {file_num}')
    plt.xlabel('Date')
    plt.ylabel('Observed Flow')
    plt.legend()
    plt.grid(True)
    # Add accuracy textbox (NSE)
    nse_cal = BMK(target_cal_seq_basin, predict_cal_basin, 'NSE')
    plt.text(0.01, 0.98, f'NSE = {nse_cal:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hydrographs", f"hydrograph_cal_file_{file_num}.png"))
    plt.close()

    # --- Evaluation Data for current basin ---
    input_eva_full_basin = basin_data['input_eva_full'].copy()
    target_eva_basin = basin_data['target_eva']
    dates_eva_basin_orig = basin_data['dates_eva'] # This is already a pandas DatetimeIndex

    # Normalize dynamic features using the *combined* mean/std
    input_eva_full_basin[:, :3] = (input_eva_full_basin[:, :3] - mean_dyn) / std_dyn

    input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_full_basin, target_eva_basin, timesteps)
    predict_eva_basin = TCN_model.predict(input_eva_seq_basin).flatten()
    predict_eva_basin[predict_eva_basin < 0] = 0

    # Adjust dates for the predictions based on timesteps and convert to numpy
    dates_eva_plot = dates_eva_basin_orig[timesteps:].to_numpy()

    # Plot for individual basin Evaluation
    plt.figure(figsize=(15, 6))
    plt.plot(dates_eva_plot, target_eva_seq_basin, label='Observed', color='blue', alpha=0.7)
    plt.plot(dates_eva_plot, predict_eva_basin, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Hydrograph (Evaluation) - File {file_num}')
    plt.xlabel('Date')
    plt.ylabel('Observed Flow')
    plt.legend()
    plt.grid(True)
    # Add accuracy textbox (NSE)
    nse_eva = BMK(target_eva_seq_basin, predict_eva_basin, 'NSE')
    plt.text(0.01, 0.98, f'NSE = {nse_eva:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hydrographs", f"hydrograph_eva_file_{file_num}.png"))
    plt.close()

    # --- Calculate and store benchmark results for individual basins ---
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal.update({
            f'TCN_{benchmark}_cal': BMK(target_cal_seq_basin, predict_cal_basin, benchmark)
        })
        file_results_eva.update({
            f'TCN_{benchmark}_eva': BMK(target_eva_seq_basin, predict_eva_basin, benchmark)
        })  

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    print(f"file_num: {file_num}, NSE_cal: {BMK(target_cal_seq_basin, predict_cal_basin, 'NSE'):.4f}, NSE_eva: {BMK(target_eva_seq_basin, predict_eva_basin, 'NSE'):.4f}")

    end_time_basin = datetime.now()
    end_time_st_basin = end_time_basin.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st_basin}\n")
    log_file.write(f"elapsed: {end_time_basin - start_time_basin}\n")


log_file.write(f"DONE\n")
log_file.close()


df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)


print("DONE")