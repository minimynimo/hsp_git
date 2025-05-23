# based on https://github.com/DevonFulcher/CryptoPricePredictor?utm_source=chatgpt.com
import pandas as pd
import matplotlib.pyplot as plt
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

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

batch_size, timesteps, input_dim = 32, 365, 3
epochs = 100

loc, ver = "JP", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

if loc == "JP" and ver == 2:
    file_tot_num = 87
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

file_list = list(range(1, file_tot_num+1))
#file_list = list(range(20,30))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp/0_out/13_TCN/{loc}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)

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

for file_num in file_list:
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)])# Observed - Simulation
    input_eva = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)]) # Observed - Simulation

    input_cal = input_cal[:-1] #2898
    input_eva = input_eva[:-1] #1089

    scaler = StandardScaler()
    input_cal = scaler.fit_transform(input_cal)
    input_eva = scaler.transform(input_eva)

    target_cal = df_cal['Obs flow'].values[1:] #2898
    target_eva = df_eva['Obs flow'].values[1:] #1089

    model_in = Input(shape=(timesteps, input_cal.shape[1]))

    input_cal_seq, target_cal_seq = create_sequences(input_cal, target_cal, timesteps)
    input_eva_seq, target_eva_seq = create_sequences(input_eva, target_eva, timesteps)

    # Remove samples with NaN or Inf in input or target
    def clean_sequences(X, y):
        mask = ~(
            np.isnan(X).any(axis=(1,2)) | np.isinf(X).any(axis=(1,2)) |
            np.isnan(y) | np.isinf(y)
        )
        return X[mask], y[mask]

    input_cal_seq, target_cal_seq = clean_sequences(input_cal_seq, target_cal_seq)
    input_eva_seq, target_eva_seq = clean_sequences(input_eva_seq, target_eva_seq)

    
    model_out = TCN(
        return_sequences=False
    )(model_in)
    model_out = Dense(1, activation='linear')(model_out)

    TCN_model = Model(inputs=[model_in], outputs=[model_out])
    TCN_model.compile(optimizer='adam', loss='mse')

    TCN_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

    predict_cal = TCN_model.predict(input_cal_seq)
    predict_eva = TCN_model.predict(input_eva_seq)

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    print("target_cal_seq", target_cal_seq.shape, "predict_cal", predict_cal.shape)
    print("target_eva_seq", target_eva_seq.shape, "predict_eva", predict_eva.shape)

    print(f"file_num: {file_num}, NSE_cal: {BMK(target_cal_seq, predict_cal, 'NSE')}, NSE_eva: {BMK(target_eva_seq, predict_eva, 'NSE')}")


    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

    if file_num == 1:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/TCN_predict_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/TCN_predict_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/TCN_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/TCN_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'TCN_{benchmark}_cal': BMK(target_cal_seq, predict_cal.flatten(), benchmark)
        })
        file_results_eva.update({
            f'TCN_{benchmark}_eva': BMK(target_eva_seq, predict_eva.flatten(), benchmark)
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
df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)


print("DONE")
