import os
# Configure GPU usage - will use GPU if available, fall back to CPU if not
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors
# Uncomment the line below if you encounter CUDA compilation issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import tensorflow as tf

# Check for GPU availability and configure memory growth to prevent memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs to prevent memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected and configured: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU")
else:
    print("No GPU detected. Using CPU.")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np

from keras.layers import Dense, Input, LSTM, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam

from datetime import datetime

import sys

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip', 'syn_p_pet', 'syn_peak1_rate0',
                   'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

if loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424 # Using 20 for consistency with other models
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num+1))

# --- Updated LSTM parameters ---
epochs = 30 
hidden_size = 256 # Number of LSTM cells (lstm_units)
learning_rate = 1e-3 # Initial learning rate (will be scheduled: 0.001 -> 5e-4 at epoch 20 -> 1e-4 at epoch 25)
window_size = 365 # Length of the meteorological record provided to the network (timesteps)
batch_size = 256 #512 # Use larger batch size if GPU is available
dropout_rate = 0.4 # Dropout rate of the final fully connected Layer [0.0, 1.0]
timesteps = window_size # For clarity in the rest of the script

# Dynamically adjust parameters based on GPU availability
hardware_type = "GPU" if gpus else "CPU"
if not gpus:
    # Further reduce parameters for CPU training
    batch_size = 16
    hidden_size = 64
    print(f"Adjusted for CPU: batch_size={batch_size}, hidden_size={hidden_size}")
else:
    print(f"GPU training: batch_size={batch_size}, hidden_size={hidden_size}")

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

# --- Updated output directories for LSTM ---
output_dir = f"/data0/funato/hsp/0_out/LSTM/{loc}/regional"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Increased patience

# Learning rate schedule function
def lr_schedule(epoch, lr):
    """
    Learning rate schedule:
    - Initial: 0.001 (1e-3)
    - Epoch 20: 0.0005 (5e-4)
    - Epoch 25: 0.0001 (1e-4)
    """
    if epoch < 20:
        return 1e-3
    elif epoch < 25:
        return 5e-4
    else:
        return 1e-4

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

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

all_input_cal_list, all_target_cal_list = [], []
all_input_eva_list, all_target_eva_list = [], []
individual_basin_data = {}

start_time = datetime.now()
log_file.write(f"start: {start_time.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
log_file.flush()

print("--- Collecting Data for all Basins ---")
all_input_cal_seqs, all_target_cal_seqs = [], []
all_input_eva_seqs, all_target_eva_seqs = [], []
individual_basin_data = {}

# Collect all input data first to compute global normalization parameters
all_input_for_norm = []

for file_num in file_list:
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    input_cal_dyn = df_cal[dynamic_features].values
    input_eva_dyn = df_eva[dynamic_features].values
    
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    # Store clean data for this basin
    input_cal_clean = input_cal_dyn[mask_cal]
    input_eva_clean = input_eva_dyn[mask_eva]
    target_cal_clean = df_cal['Obs flow'].values[mask_cal]
    target_eva_clean = df_eva['Obs flow'].values[mask_eva]

    # Collect all input data for normalization (concatenate for computing global stats)
    all_input_for_norm.append(input_cal_clean)

    # Store individual basin data for later processing
    individual_basin_data[file_num] = {
        'input_cal_clean': input_cal_clean, 'target_cal_clean': target_cal_clean, 'dates_cal': df_cal.index[mask_cal],
        'input_eva_clean': input_eva_clean, 'target_eva_clean': target_eva_clean, 'dates_eva': df_eva.index[mask_eva],
    }

# Compute global normalization parameters from all basins' calibration data
all_input_for_norm_combined = np.concatenate(all_input_for_norm, axis=0)
mean_dyn = np.mean(all_input_for_norm_combined, axis=0)  # Shape: (3,) for [Precip, Temp, PET]
std_dyn = np.std(all_input_for_norm_combined, axis=0)
std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)

print("--- Creating Sequences from Each Basin Separately ---")
# Now create sequences from each basin separately, then combine
for file_num in file_list:
    basin_data = individual_basin_data[file_num]
    
    # Normalize this basin's data using global parameters
    input_cal_norm = (basin_data['input_cal_clean'] - mean_dyn) / std_dyn
    input_eva_norm = (basin_data['input_eva_clean'] - mean_dyn) / std_dyn
    
    # Create sequences within this basin only
    if len(input_cal_norm) > timesteps:
        input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_norm, basin_data['target_cal_clean'], timesteps)
        all_input_cal_seqs.append(input_cal_seq_basin)
        all_target_cal_seqs.append(target_cal_seq_basin)
    
    if len(input_eva_norm) > timesteps:
        input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_norm, basin_data['target_eva_clean'], timesteps)
        all_input_eva_seqs.append(input_eva_seq_basin)
        all_target_eva_seqs.append(target_eva_seq_basin)

# Combine all valid sequences from all basins
input_cal_seq = np.concatenate(all_input_cal_seqs, axis=0) if all_input_cal_seqs else np.array([])
target_cal_seq = np.concatenate(all_target_cal_seqs, axis=0) if all_target_cal_seqs else np.array([])
input_eva_seq = np.concatenate(all_input_eva_seqs, axis=0) if all_input_eva_seqs else np.array([])
target_eva_seq = np.concatenate(all_target_eva_seqs, axis=0) if all_target_eva_seqs else np.array([])

print(f"Shape of training sequences: {input_cal_seq.shape}")
print(f"Shape of training targets: {target_cal_seq.shape}")
print(f"Total sequences from {len(file_list)} basins: {len(all_input_cal_seqs)} basins contributed sequences")


# Set prediction batch size to prevent OOM during inference
prediction_batch_size = 32  # Use smaller batch size for prediction to avoid OOM

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
model_in = Input(shape=(timesteps, len(dynamic_features)))  # Shape: (timesteps, 3) for dynamic features
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
LSTM_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

print("Training completed.")


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
    # Use the stored clean data and apply same normalization as training
    input_cal_basin = basin_data['input_cal_clean'].copy()
    input_cal_basin_normalized = (input_cal_basin - mean_dyn) / std_dyn
    input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_basin_normalized, basin_data['target_cal_clean'], timesteps)
    
    predict_cal_basin = LSTM_model.predict(input_cal_seq_basin, batch_size=prediction_batch_size).flatten() if input_cal_seq_basin.shape[0] > 0 else np.array([])
    predict_cal_basin[predict_cal_basin < 0] = 0

    input_eva_basin = basin_data['input_eva_clean'].copy()
    input_eva_basin_normalized = (input_eva_basin - mean_dyn) / std_dyn
    input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_basin_normalized, basin_data['target_eva_clean'], timesteps)
    
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
        file_results_cal[f'{benchmark}'] = BMK(target_cal_seq_basin, predict_cal_basin, benchmark)
        file_results_eva[f'{benchmark}'] = BMK(target_eva_seq_basin, predict_eva_basin, benchmark)

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # --- Synthetic Data Evaluation ---
    df_current_basin = load_data(file_num)

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
        input_syn = input_dyn_syn[mask_syn]  # Only dynamic features, shape: (n_samples, 3)
        target_syn = df_syn[syn_obs_flow_col].values[mask_syn]
        
        # Apply same normalization as training data
        input_syn_normalized = (input_syn - mean_dyn) / std_dyn
        
        input_seq_syn, _ = create_sequences(input_syn_normalized, target_syn, timesteps)
        
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
    if not basin_preds_dict: continue
    
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
    combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/LSTM_predict_{syn_precip_col.replace('syn_','')}.csv")

log_file.write("DONE\n")
log_file.close()

print("DONE")
