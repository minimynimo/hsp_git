import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os

# --- GPU CONFIGURATION CHANGES START ---
# Move the fallback flag to XLA_FLAGS where it belongs
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 2 = warnings/errors, 3 = errors
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Select GPU 0

import tensorflow as tf

# Check for GPUs and set memory growth to avoid allocating all memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected and configured: {gpus}")
    except RuntimeError as e:
        print(f"GPU Error: {e}")
else:
    print("No GPU detected. Running on CPU.")
# --- GPU CONFIGURATION CHANGES END ---

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from tcn import TCN
from sklearn.preprocessing import StandardScaler
from datetime import datetime

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


#loc, ver = "JP", 2
#loc, ver = "GB", 2
loc, ver = "US", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   


if loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_GB"
elif loc == "US" and ver == 2:
    file_tot_num = 614
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_US"

file_list = list(range(1, file_tot_num+1))

input_dim = 3

timesteps = 30
epochs = 100 
batch_size = 32
nb_filters = 64
kernel_size = 4
nb_stacks = 1
dropout_rate = 0.2

param_tag = f"f{nb_filters}_k{kernel_size}_d{dropout_rate}_w{timesteps}"

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp2/0_out/TCN/{loc}/individual/{param_tag}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True) 

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
    # Ensure data are flat for calculations if they come in (N, 1) shape
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

    # Handle cases where obs_data might be all zero or very close to zero
    if np.sum(obs_data) == 0 and benchmark not in ["RMSE", "MAE"]:
        return np.nan 

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        # Avoid division by zero if obs_data is constant
        if np.sum(np.square(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan 
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        
        # Guard against zero standard deviation for constant series
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        # Guard against zero mean for very small values
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
        
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        if np.sum(np.abs(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "VE":
        if np.sum(obs_data) == 0:
             return 1.0 if np.all(obs_data == sim_data) else np.nan 
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return 1.0 if numer == 0 else np.nan
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

# results_cal and results_eva will store dicts for each file_num
results_cal = []
results_eva = []

start_time_overall = datetime.now()
start_time_overall_st = start_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall start: {start_time_overall_st}\n")
log_file.flush()


# --- Individual Training Loop ---
for file_num in file_list:
    start_time_basin = datetime.now()
    start_time_st_basin = start_time_basin.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{file_num} - Individual Training\n")
    log_file.write(f"start: {start_time_st_basin}\n")
    log_file.flush()

    ### LOAD DATA FOR CURRENT BASIN ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Prepare dynamic features
    input_cal_dyn = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)])
    input_eva_dyn = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)])
    
    # Apply masks to remove NaN values
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    input_cal = input_cal_dyn[mask_cal]
    input_eva = input_eva_dyn[mask_eva]

    # Targets (raw discharge values)
    target_cal_raw = df_cal['Obs flow'].values[mask_cal]
    target_eva_raw = df_eva['Obs flow'].values[mask_eva]

    # Normalize dynamic features SEPARATELY
    scaler_dyn = StandardScaler()
    input_cal = scaler_dyn.fit_transform(input_cal)
    input_eva = scaler_dyn.transform(input_eva)

    # Normalize discharge SEPARATELY (CRITICAL for deep learning models)
    scaler_q = StandardScaler()
    target_cal = scaler_q.fit_transform(target_cal_raw.reshape(-1, 1)).flatten()
    target_eva = scaler_q.transform(target_eva_raw.reshape(-1, 1)).flatten()

    # Store discharge statistics for denormalization
    discharge_mean = scaler_q.mean_[0]
    discharge_std = scaler_q.scale_[0]

    # Create sequences
    input_cal_seq, target_cal_seq = create_sequences(input_cal, target_cal, timesteps)
    input_eva_seq, target_eva_seq = create_sequences(input_eva, target_eva, timesteps)

    # Skip if insufficient data
    if len(input_cal_seq) == 0 or len(input_eva_seq) == 0:
        log_file.write(f"Insufficient data for basin {file_num}, skipping\n")
        continue

    ### BUILD AND TRAIN MODEL FOR THIS BASIN ###
    model_in = Input(shape=(timesteps, input_cal.shape[1]))
    
    # TCN block 
    model_out = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        return_sequences=False,
        dropout_rate=dropout_rate
    )(model_in)

    # Output layer
    model_out = Dense(1, activation='linear')(model_out)

    # Build and compile model
    TCN_model = Model(inputs=model_in, outputs=model_out)
    TCN_model.compile(optimizer='adam', loss='mse')

    # Train the model on this basin's data
    TCN_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    ### MAKE PREDICTIONS ###
    predict_cal_basin = TCN_model.predict(input_cal_seq, batch_size=batch_size, verbose=0).flatten()
    predict_eva_basin = TCN_model.predict(input_eva_seq, batch_size=batch_size, verbose=0).flatten()
    
    # Denormalize predictions back to original scale
    predict_cal_basin = predict_cal_basin * discharge_std + discharge_mean
    predict_eva_basin = predict_eva_basin * discharge_std + discharge_mean
    
    predict_cal_basin[predict_cal_basin < 0] = 0
    predict_eva_basin[predict_eva_basin < 0] = 0
    
    # Denormalize targets for benchmark calculation
    target_cal_denorm = target_cal_seq * discharge_std + discharge_mean
    target_eva_denorm = target_eva_seq * discharge_std + discharge_mean

    ### SAVE PREDICTIONS ###
    # Prepare CSV data
    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal_basin)
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva_basin)

    # Write headers for first file
    if file_num == file_list[0]:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=timesteps+1), periods=len(predict_cal_basin))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=timesteps+1), periods=len(predict_eva_basin))

        date_row_cal = ['Date'] + [str(date.date()) for date in predict_cal_dates]
        date_row_eva = ['Date'] + [str(date.date()) for date in predict_eva_dates]

        with open(output_dir + f"/predict/TCN_predict_{param_tag}_cal.csv", 'w') as file:
            pd.DataFrame([date_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/predict/TCN_predict_{param_tag}_eva.csv", 'w') as file:
            pd.DataFrame([date_row_eva]).to_csv(file, header=False, index=False)

    # Append predictions
    with open(output_dir + f"/predict/TCN_predict_{param_tag}_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/TCN_predict_{param_tag}_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    ### CALCULATE BENCHMARKS ###
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal.update({
            f'TCN_{benchmark}_cal': BMK(target_cal_denorm, predict_cal_basin, benchmark)
        })
        file_results_eva.update({
            f'TCN_{benchmark}_eva': BMK(target_eva_denorm, predict_eva_basin, benchmark)
        })  

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results incrementally
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/TCN_results_{param_tag}_cal.csv', index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/TCN_results_{param_tag}_eva.csv', index=False, header=True)

    print(f"file_num: {file_num}, NSE_cal: {BMK(target_cal_denorm, predict_cal_basin, 'NSE'):.4f}, NSE_eva: {BMK(target_eva_denorm, predict_eva_basin, 'NSE'):.4f}")

    end_time_basin = datetime.now()
    log_file.write(f"end: {end_time_basin}\n")
    log_file.write(f"elapsed: {end_time_basin - start_time_basin}\n")
    log_file.write(f"Results saved for file {file_num}\n")
    print(f"Results saved for file {file_num}")
    log_file.flush()

end_time_overall = datetime.now()
end_time_overall_st = end_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall end: {end_time_overall_st}\n")
log_file.write(f"Overall elapsed: {end_time_overall - start_time_overall}\n")
log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/TCN_results_{param_tag}_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_{param_tag}_eva.csv', index=False, header=True)

print("DONE")