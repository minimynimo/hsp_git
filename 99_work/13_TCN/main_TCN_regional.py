import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid CUDA compilation issues

from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from tcn import TCN
from datetime import datetime

import sys
import tensorflow as tf

# Force CPU usage to avoid CUDA compilation issues
tf.config.set_visible_devices([], 'GPU')

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip',
                    'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                    'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                    'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

if loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

file_list = list(range(1, file_tot_num+1))

# TCN parameters
timesteps = 25
epochs = 50
batch_size = 32 #64
nb_filters = 128
kernel_size = 4
nb_stacks = 3

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp/0_out/TCN_temp/{loc}/regional"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)


# clean attribute values and normalize
# Static attributes removed - using only dynamic features

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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

if os.path.exists(output_dir + f'/TCN_log.txt'):
    open(output_dir + f'/TCN_log.txt', 'w').close()
log_file = open(output_dir + f'/TCN_log.txt', 'a')

dynamic_features = ["Precip", "Temp", "PET"]
# Feature names (only dynamic features)
feature_names = dynamic_features

all_input_cal_list, all_target_cal_list = [], []
all_input_eva_list, all_target_eva_list = [], []
individual_basin_data = {}

start_time = datetime.now()
log_file.write(f"start: {start_time.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
log_file.flush()

print("--- Collecting Data for all Basins ---")
log_file.write("Collecting data for all basins...\n")
for file_num in file_list:
    log_file.write(f"Processing basin {file_num}/{file_tot_num}\n")
    log_file.flush()
    
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    input_cal_dyn = df_cal[dynamic_features].values
    input_eva_dyn = df_eva[dynamic_features].values
    
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    input_cal_full = input_cal_dyn[mask_cal]  # Only dynamic features
    input_eva_full = input_eva_dyn[mask_eva]  # Only dynamic features

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
log_file.write("Normalizing data and creating sequences...\n")
log_file.flush()
mean_dyn = np.mean(all_input_cal[:, :3], axis=0)
std_dyn = np.std(all_input_cal[:, :3], axis=0)
std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)

all_input_cal[:, :3] = (all_input_cal[:, :3] - mean_dyn) / std_dyn
all_input_eva[:, :3] = (all_input_eva[:, :3] - mean_dyn) / std_dyn

input_cal_seq, target_cal_seq = create_sequences(all_input_cal, all_target_cal, timesteps)
input_eva_seq, target_eva_seq = create_sequences(all_input_eva, all_target_eva, timesteps)

print(f"Shape of training sequences: {input_cal_seq.shape}")
print(f"Shape of training targets: {target_cal_seq.shape}")

print("--- Building and Training TCN Model ---")
model_in = Input(shape=(timesteps, all_input_cal.shape[1]))
model_out = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, return_sequences=False)(model_in)
model_out = Dense(1, activation='linear')(model_out)
TCN_model = Model(inputs=model_in, outputs=model_out)
TCN_model.compile(optimizer='adam', loss='mse')

TCN_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

log_file.flush()
print("--- Evaluating Model and Generating Outputs ---")
log_file.write("Evaluating model and generating outputs...\n")
log_file.flush()    
results_cal, results_eva = [], []
syn_predict_dfs_combined = {col: {} for col in syn_precip_cols}

for file_num in file_list:
    basin_data = individual_basin_data[file_num]
    start_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, start: {start_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
    
    # --- Standard Prediction and Evaluation ---
    input_cal_full_basin = basin_data['input_cal_full'].copy()
    input_cal_full_basin[:, :3] = (input_cal_full_basin[:, :3] - mean_dyn) / std_dyn
    input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_full_basin, basin_data['target_cal'], timesteps)
    predict_cal_basin = TCN_model.predict(input_cal_seq_basin, batch_size=batch_size).flatten()
    predict_cal_basin[predict_cal_basin < 0] = 0

    input_eva_full_basin = basin_data['input_eva_full'].copy()
    input_eva_full_basin[:, :3] = (input_eva_full_basin[:, :3] - mean_dyn) / std_dyn
    input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_full_basin, basin_data['target_eva'], timesteps)
    predict_eva_basin = TCN_model.predict(input_eva_seq_basin, batch_size=batch_size).flatten()
    predict_eva_basin[predict_eva_basin < 0] = 0

    # Get correct dates for predictions
    dates_predict_cal = basin_data['dates_cal'][timesteps:]
    dates_predict_eva = basin_data['dates_eva'][timesteps:]
    
    # Save daily predictions for standard cal/eva
    pd.DataFrame({
        'Date': dates_predict_cal,
        'Observed': target_cal_seq_basin,
        'Predicted': predict_cal_basin
    }).to_csv(f"{output_dir}/predict/og/TCN_predict_cal_basin_{file_num}.csv", index=False)

    pd.DataFrame({
        'Date': dates_predict_eva,
        'Observed': target_eva_seq_basin,
        'Predicted': predict_eva_basin
    }).to_csv(f"{output_dir}/predict/og/TCN_predict_eva_basin_{file_num}.csv", index=False)
    
    # Calculate and store benchmark results
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
        log_file.write(f"Processing synthetic column: {syn_precip_col}\n")
        log_file.flush()
        if syn_precip_col.startswith('syn_peak'):
            end_date_eva_syn = '2001-02-10'
        else:
            end_date_eva_syn = end_date_eva
        log_file.write(f"Using end date for synthetic evaluation: {end_date_eva_syn}\n")
        log_file.flush()

        df_syn = df_current_basin[start_date_cal:end_date_eva_syn].copy()
        
        syn_temp_col = 'Temp_at_peak' if 'peak' in syn_precip_col else 'Temp'
        syn_pet_col = 'PET_at_peak' if 'peak' in syn_precip_col else 'PET'
        syn_flow_col = 'Obs_flow_at_peak' if 'peak' in syn_precip_col else 'Obs flow'
        syn_dyn_cols = [syn_precip_col, syn_temp_col, syn_pet_col]

        log_file.write(f"Using dynamic columns: {syn_dyn_cols}, with size {len(syn_dyn_cols)}\n")
        
        input_dyn_syn = df_syn[syn_dyn_cols].values
        log_file.write(f"Input dynamic shape: {input_dyn_syn.shape}\n")
        log_file.flush()
        mask_syn = ~np.isnan(input_dyn_syn).any(axis=1) & ~np.isnan(df_syn[syn_flow_col].values)
        input_syn = input_dyn_syn[mask_syn]  # Only dynamic features
        target_syn = df_syn[syn_flow_col].values[mask_syn]
        
        input_syn[:, :3] = (input_syn[:, :3] - mean_dyn) / std_dyn
        
        input_seq_syn, _ = create_sequences(input_syn, target_syn, timesteps)
        log_file.write(f"Input sequence shape for synthetic {syn_precip_col}: {input_seq_syn.shape}\n")
        log_file.flush()
        
        if input_seq_syn.shape[0] > 0:
            predict_syn = TCN_model.predict(input_seq_syn, batch_size=batch_size).flatten()
            predict_syn[predict_syn < 0] = 0
            
            dates_syn = df_syn.index[mask_syn]
            dates_predict_syn = dates_syn[timesteps:]
            
            temp_syn_df = pd.DataFrame({'Date': dates_predict_syn, 'prediction': predict_syn}).set_index('Date')
            syn_predict_dfs_combined[syn_precip_col][file_num] = temp_syn_df

        # save the dimension sze of temp_syn_df
        log_file.write(f"Shape of synthetic predictions for {syn_precip_col}: {temp_syn_df.shape}\n")

    end_time_basin = datetime.now()
    log_file.write(f"FILE_{file_num}, end: {end_time_basin.strftime('%a %b %d %I:%M:%S %p JST %Y')}, elapsed: {end_time_basin - start_time_basin}\n")
    log_file.flush()

print("--- Saving Final Results ---")
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)
df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)

for syn_precip_col, basin_preds_dict in syn_predict_dfs_combined.items():
    log_file.write(f"Processing synthetic column: {syn_precip_col}\n")

    if not basin_preds_dict: continue
    
    combined_syn_df = pd.DataFrame()
    for file_num, df_preds in basin_preds_dict.items():
        df_preds = df_preds.rename(columns={'prediction': f'file_{file_num}'})
        log_file.write(f"  Adding predictions for file {file_num}, shape: {df_preds.shape}\n")
        if combined_syn_df.empty:
            combined_syn_df = df_preds
        else:
            combined_syn_df = combined_syn_df.join(df_preds, how='outer')

    if syn_precip_col.startswith('syn_peak'):
        end_date_syn_final = '2001-02-10'
        combined_syn_df = combined_syn_df.loc[:end_date_syn_final]
    else:
        end_date_syn_final = end_date_eva
        combined_syn_df = combined_syn_df.loc[:end_date_syn_final]
    
    combined_syn_df.sort_index(inplace=True)
    combined_syn_df.to_csv(f"{output_dir}/predict/synthetic/TCN_predict_{syn_precip_col.replace('syn_','')}.csv")

log_file.write("DONE\n")
log_file.close()

print("DONE")
