import os
# Disable GPU before importing TensorFlow - must be set before TF initialization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors

import tensorflow as tf

# Verify GPU is disabled
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Warning: GPU detected even after disabling. This shouldn't happen.")
else:
    print("GPU disabled. Using CPU for all computations.")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np

from keras.layers import Dense, Input, LSTM, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from datetime import datetime

import shap
import sys

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

attribute_dir = f'/data0/funato/hsp/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 
                   'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0','syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0','syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

if loc == "JP" and ver == 2:
    file_tot_num = 87
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424 # Using 20 for consistency with other models
    columns_drop = ['gauge_id']
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num+1))

# --- Updated LSTM parameters ---
epochs = 50 
hidden_size = 20 # Number of LSTM cells (lstm_units)
learning_rate = 1e-3 # Learning rate used to update the weights
window_size = 60 # Length of the meteorological record provided to the network (timesteps)
batch_size = 16 #512 # Use larger batch size if GPU is available
dropout_rate = 0.1 # Dropout rate of the final fully connected Layer [0.0, 1.0]
timesteps = window_size # For clarity in the rest of the script

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

# --- Updated output directories for LSTM ---
output_dir = f"/data0/funato/hsp/0_out/LSTM/{loc}/shap/regional_424"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)

output_dir_shap = f"0_out/1_fig/shap/LSTM/{loc}/regional_424"
os.makedirs(output_dir_shap, exist_ok=True)


# clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns])

cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped
attribute_mean = attribute_values_numeric.mean()
attribute_std = attribute_values_numeric.std()
attribute_values_normalized = attribute_values_dropped.copy()
attribute_values_normalized.loc[:, attribute_values_numeric.columns] = (attribute_values_numeric - attribute_mean) / attribute_std

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Increased patience

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

def BMK(obs_data,sim_data,benchmark):
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

# --- Updated log file for LSTM ---
if os.path.exists(output_dir + f'/LSTM_log.txt'):
    open(output_dir + f'/LSTM_log.txt', 'w').close()
log_file = open(output_dir + f'/LSTM_log.txt', 'a')

dynamic_features = ["Precip", "Temp", "PET"]
static_feature_names = [col for col in attribute_values_normalized.columns if col != 'File_num']
feature_names = dynamic_features + static_feature_names

all_input_cal_list, all_target_cal_list = [], []
all_input_eva_list, all_target_eva_list = [], []
individual_basin_data = {}

start_time = datetime.now()
log_file.write(f"start: {start_time.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
log_file.flush()

print("--- Collecting Data for all Basins ---")
for file_num in file_list:
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    input_cal_dyn = df_cal[dynamic_features].values
    input_eva_dyn = df_eva[dynamic_features].values
    
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    static_features_for_basin = attribute_values_normalized.loc[file_num].drop('File_num', errors='ignore')
    static_array_cal = np.tile(static_features_for_basin.values, (np.sum(mask_cal), 1))
    static_array_eva = np.tile(static_features_for_basin.values, (np.sum(mask_eva), 1))

    input_cal_full = np.hstack([input_cal_dyn[mask_cal], static_array_cal])
    input_eva_full = np.hstack([input_eva_dyn[mask_eva], static_array_eva])

    all_input_cal_list.append(input_cal_full)
    all_input_eva_list.append(input_eva_full)

    target_cal = df_cal['Obs flow'].values[mask_cal]
    target_eva = df_eva['Obs flow'].values[mask_eva]
    all_target_cal_list.append(target_cal)
    all_target_eva_list.append(target_eva)

    individual_basin_data[file_num] = {
        'input_cal_full': input_cal_full, 'target_cal': target_cal, 'dates_cal': df_cal.index[mask_cal],
        'input_eva_full': input_eva_full, 'target_eva': target_eva, 'dates_eva': df_eva.index[mask_eva],
    }

all_input_cal = np.vstack(all_input_cal_list)
all_target_cal = np.hstack(all_target_cal_list)
all_input_eva = np.vstack(all_input_eva_list)
all_target_eva = np.hstack(all_target_eva_list)

print("--- Normalizing Data and Creating Sequences ---")
mean_dyn = np.mean(all_input_cal[:, :3], axis=0)
std_dyn = np.std(all_input_cal[:, :3], axis=0)
std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)

all_input_cal[:, :3] = (all_input_cal[:, :3] - mean_dyn) / std_dyn
all_input_eva[:, :3] = (all_input_eva[:, :3] - mean_dyn) / std_dyn

input_cal_seq, target_cal_seq = create_sequences(all_input_cal, all_target_cal, timesteps)
input_eva_seq, target_eva_seq = create_sequences(all_input_eva, all_target_eva, timesteps)

print(f"Shape of training sequences: {input_cal_seq.shape}")
print(f"Shape of training targets: {target_cal_seq.shape}")

# Adjust dataset size based on available hardware
max_training_samples = 500000 
if input_cal_seq.shape[0] > max_training_samples:
    print(f"Dataset too large ({input_cal_seq.shape[0]} samples). Reducing to {max_training_samples} samples.")
    
    # Randomly sample a subset
    indices = np.random.choice(input_cal_seq.shape[0], max_training_samples, replace=False)
    input_cal_seq = input_cal_seq[indices]
    target_cal_seq = target_cal_seq[indices]
    
    print(f"Reduced training sequences shape: {input_cal_seq.shape}")
    print(f"Reduced training targets shape: {target_cal_seq.shape}")

# Also reduce evaluation set if too large
max_eval_samples = 200000 #if gpus else 50000
if input_eva_seq.shape[0] > max_eval_samples:
    print(f"Evaluation dataset too large ({input_eva_seq.shape[0]} samples). Reducing to {max_eval_samples} samples.")
    indices_eva = np.random.choice(input_eva_seq.shape[0], max_eval_samples, replace=False)
    input_eva_seq = input_eva_seq[indices_eva]
    target_eva_seq = target_eva_seq[indices_eva]

print("--- Building and Training LSTM Model ---")
model_in = Input(shape=(timesteps, all_input_cal.shape[1]))
# Removed recurrent_dropout to avoid potential issues
model_out = LSTM(hidden_size, return_sequences=False)(model_in)
model_out = Dropout(dropout_rate)(model_out)
model_out = Dense(1, activation='linear')(model_out)
LSTM_model = Model(inputs=model_in, outputs=model_out)

optimizer = Adam(learning_rate=learning_rate)
LSTM_model.compile(optimizer=optimizer, loss='mse')

# Train the model (GPU will be used if available)
hardware_type = "GPU" if gpus else "CPU"
print(f"Training LSTM model on {hardware_type}...")
LSTM_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

print("Training completed.")

# Save the trained model
model_save_path = os.path.join(output_dir, "LSTM_model.h5")
LSTM_model.save(model_save_path)
log_file.write(f"Model saved to: {model_save_path}\n")
log_file.flush()
print(f"Model saved to: {model_save_path}")

print("--- Performing SHAP Analysis using KernelExplainer ---")
# Use a subset of the training data as background for the explainer
# KernelExplainer is slow, so we use a very small background set, summarized by k-means.
background_size = 50
if input_cal_seq.shape[0] > background_size:
    background_indices = np.random.choice(input_cal_seq.shape[0], background_size, replace=False)
    background = input_cal_seq[background_indices]
else:
    background = input_cal_seq

# Use a small subset of the evaluation data to explain
explain_size = 100
if input_eva_seq.shape[0] > explain_size:
    explain_indices = np.random.choice(input_eva_seq.shape[0], explain_size, replace=False)
    X_sample = input_eva_seq[explain_indices]
else:
    X_sample = input_eva_seq

if X_sample.shape[0] > 0:
    # --- FIX: Switched to KernelExplainer ---
    # We need a function that can take a 2D array and predict, as KernelExplainer works with 2D data.
    # The function will expand the dimensions back to the 3D sequence format for the LSTM.
    def predict_fn(x):
        # x has shape (n_samples, n_features)
        # We need to expand it to (n_samples, timesteps, n_features)
        # We'll do this by tiling the 2D sample across the time dimension
        x_3d = np.tile(x[:, np.newaxis, :], (1, timesteps, 1))
        return LSTM_model.predict(x_3d, batch_size=batch_size)

    # We summarize the background data by taking the mean over the time axis
    background_summary = np.mean(background, axis=1)
    
    print("Initializing SHAP KernelExplainer (this may be slow)...")
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    
    # We explain the time-series-averaged samples
    X_sample_summary = np.mean(X_sample, axis=1)
    
    print("Calculating SHAP values with KernelExplainer...")
    # The output, shap_values, will be 2D directly.
    shap_values = explainer.shap_values(X_sample_summary)
    
    # Squeeze the shap_values array to remove the trailing dimension if it exists
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(axis=2)
    
    X_sample_df = pd.DataFrame(X_sample_summary, columns=feature_names)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Features DataFrame shape: {X_sample_df.shape}")

    # --- Save SHAP results for later analysis ---
    print("Saving SHAP objects for later analysis...")
    shap_results_dir = os.path.join(output_dir, "shap_objects")
    os.makedirs(shap_results_dir, exist_ok=True)
    
    # Save the SHAP values
    np.save(os.path.join(shap_results_dir, "shap_values.npy"), shap_values)
    
    # Save the expected value (base value)
    np.save(os.path.join(shap_results_dir, "expected_value.npy"), explainer.expected_value)
    
    # Save the corresponding feature data
    X_sample_df.to_csv(os.path.join(shap_results_dir, "X_sample.csv"))
    
    print("SHAP objects saved.")

    # Summary Plot (Beeswarm)
    print("Generating SHAP summary plots...")
    shap.summary_plot(shap_values, X_sample_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/LSTM_shap_summary_beeswarm.png")
    plt.close()

    # Summary Plot (Bar)
    shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/LSTM_shap_summary_bar.png")
    plt.close()
    
    # Force Plots
    print("Generating SHAP force plots...")
    for i in range(min(5, X_sample_summary.shape[0])):
        force_plot = shap.force_plot(explainer.expected_value, shap_values[i,:], X_sample_df.iloc[i,:])
        shap.save_html(f"{output_dir_shap}/LSTM_force_plot_sample_{i}.html", force_plot)

print("--- Evaluating Model and Generating Outputs ---")
results_cal, results_eva = [], []
syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}

# Use smaller batch size for prediction to save memory
prediction_batch_size = min(batch_size, 16)

for file_num in file_list:
    basin_data = individual_basin_data[file_num]
    start_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, start: {start_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
    
    # --- Standard Prediction and Evaluation ---
    input_cal_full_basin = basin_data['input_cal_full'].copy()
    input_cal_full_basin[:, :3] = (input_cal_full_basin[:, :3] - mean_dyn) / std_dyn
    input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_full_basin, basin_data['target_cal'], timesteps)
    
    predict_cal_basin = LSTM_model.predict(input_cal_seq_basin, batch_size=prediction_batch_size).flatten() if input_cal_seq_basin.shape[0] > 0 else np.array([])
    predict_cal_basin[predict_cal_basin < 0] = 0

    input_eva_full_basin = basin_data['input_eva_full'].copy()
    input_eva_full_basin[:, :3] = (input_eva_full_basin[:, :3] - mean_dyn) / std_dyn
    input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_full_basin, basin_data['target_eva'], timesteps)
    
    predict_eva_basin = LSTM_model.predict(input_eva_seq_basin, batch_size=prediction_batch_size).flatten() if input_eva_seq_basin.shape[0] > 0 else np.array([])
    predict_eva_basin[predict_eva_basin < 0] = 0

    dates_predict_cal = basin_data['dates_cal'][timesteps:]
    dates_predict_eva = basin_data['dates_eva'][timesteps:]
    
    pd.DataFrame({
        'Date': dates_predict_cal, 'Observed': target_cal_seq_basin, 'Predicted': predict_cal_basin
    }).to_csv(f"{output_dir}/predict/og/LSTM_predict_cal_basin_{file_num}.csv", index=False)

    pd.DataFrame({
        'Date': dates_predict_eva, 'Observed': target_eva_seq_basin, 'Predicted': predict_eva_basin
    }).to_csv(f"{output_dir}/predict/og/LSTM_predict_eva_basin_{file_num}.csv", index=False)
    
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal[f'LSTM_{benchmark}_cal'] = BMK(target_cal_seq_basin, predict_cal_basin, benchmark)
        file_results_eva[f'LSTM_{benchmark}_eva'] = BMK(target_eva_seq_basin, predict_eva_basin, benchmark)

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # --- Synthetic Data Evaluation ---
    df_current_basin = load_data(file_num)
    static_features_for_basin = attribute_values_normalized.loc[file_num].drop('File_num', errors='ignore')

    for syn_precip_col in syn_precip_cols:
        if syn_precip_col.startswith('syn_peak'):
            end_date_eva_syn = '2001-02-10'
        else:
            end_date_eva_syn = end_date_eva

        df_syn = df_current_basin[start_date_cal:end_date_eva_syn].copy()
        
        syn_temp_col = 'Temp_at_peak' if 'peak' in syn_precip_col else 'Temp'
        syn_pet_col = 'PET_at_peak' if 'peak' in syn_precip_col else 'PET'
        syn_obs_flow_col = 'Obs_flow_at_peak' if 'peak' in syn_precip_col else 'Obs flow'
        syn_dyn_cols = [syn_precip_col, syn_temp_col, syn_pet_col]
        
        input_dyn_syn = df_syn[syn_dyn_cols].values
        mask_syn = ~np.isnan(input_dyn_syn).any(axis=1) & ~np.isnan(df_syn[syn_obs_flow_col].values)
        static_array_syn = np.tile(static_features_for_basin.values, (np.sum(mask_syn), 1))
        input_syn = np.hstack([input_dyn_syn[mask_syn], static_array_syn])
        target_syn = df_syn[syn_obs_flow_col].values[mask_syn]
        
        input_syn[:, :3] = (input_syn[:, :3] - mean_dyn) / std_dyn
        
        input_seq_syn, _ = create_sequences(input_syn, target_syn, timesteps)
        
        predict_syn = LSTM_model.predict(input_seq_syn, batch_size=prediction_batch_size).flatten()
        predict_syn[predict_syn < 0] = 0
        
        dates_syn = df_syn.index[mask_syn]
        dates_predict_syn = dates_syn[timesteps:]
        
        temp_syn_df = pd.DataFrame({'Date': dates_predict_syn, 'prediction': predict_syn}).set_index('Date')
        syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df

    # Clear memory after each basin to prevent accumulation
    try:
        import gc
        gc.collect()
    except Exception as e:
        pass

    end_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, end: {end_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}, elapsed: {end_time_basin - start_time_basin}\n")
    log_file.flush()

print("--- Saving Final Results ---")
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/LSTM_results_cal.csv', index=False, header=True)
df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/LSTM_results_eva.csv', index=False, header=True)

for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
    if not basin_preds_dict: 
        continue
    
    combined_syn_df = pd.DataFrame()
    for file_num, df_preds in basin_preds_dict.items():
        df_preds = df_preds.rename(columns={'prediction': f'file_{file_num}'})
        if combined_syn_df.empty:
            combined_syn_df = df_preds
        else:
            combined_syn_df = combined_syn_df.join(df_preds, how='outer')

    if syn_precip_col.startswith('syn_peak'):
        end_date_syn_final = '2001-02-10'
        combined_syn_df = combined_syn_df.loc[:end_date_syn_final]
    
    combined_syn_df.sort_index(inplace=True)
    
    # Transpose to get basins as rows (index) and dates as columns
    combined_syn_df_transposed = combined_syn_df.T
    
    # Clean up basin names (remove 'file_' prefix)
    combined_syn_df_transposed.index = combined_syn_df_transposed.index.str.replace('file_', '')
    combined_syn_df_transposed.index.name = 'basin'
    
    # Save with correct format
    combined_syn_df_transposed.to_csv(f"{output_dir}/predict/synthetic/LSTM_predict_{syn_precip_col.replace('syn_','')}.csv")
    log_file.write(f"Saved {syn_precip_col} with shape: {combined_syn_df_transposed.shape} (basins × dates)\n")

log_file.write("DONE\n")
log_file.close()

print("DONE")
