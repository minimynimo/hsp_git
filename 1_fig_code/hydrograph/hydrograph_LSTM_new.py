import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "US", 2
#loc, ver = "US", 2
#loc, ver = "AUS", 2
#loc, ver = "GB", 2
loc,ver = "JP", 2

fs = 14

LSTM_file_path = f"/data0/funato/0_out/99_out/{loc}/LSTM"
BC_file_path = f"/data0/funato/0_out/99_out/{loc}/BC_700_0.001"



ce_list = ["cal", "eva"]

### Define the time period###
start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

num_epochs = 200 # Number of training epochs
hidden_size = 20 # Number of LSTM cells
learning_rate = 1e-3 # Learning rate used to update the weights
window_size = 365 # Length of the meteorological record provided to the network
batch_size = 512 # Number of samples in each batch
dropout_rate = 0.1 # Dropout rate of the final fully connected Layer [0.0, 1.0]
file_tag = f"_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}"


output_dir = f'/data0/funato/0_out/0_fig/{loc}/hydrograph/LSTMvsBC'
os.makedirs(output_dir, exist_ok=True)


ver_name = "ver1_1" if ver == 1 else "ver2_0"   
if loc == "JP" and ver == 1:
    file_tot_num  = 135
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"
elif loc == "US" and ver == 1:
    file_tot_num = 669
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "US" and ver == 2:
    file_tot_num = 667
    arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"
elif loc == "AUS" and ver == 2:
    file_tot_num = 84
elif loc == "GB" and ver == 2:
    file_tot_num = 396
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

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


input_LSTM_file_cal = f"{LSTM_file_path}/predict/LSTM_predict_h20_lr0.001_e200_w365_b512_d0.1_cal.csv"
input_LSTM_file_eva = f"{LSTM_file_path}/predict/LSTM_predict_h20_lr0.001_e200_w365_b512_d0.1_eva.csv"

input_BC_file_cal = f"{BC_file_path}/predict/BC_predict_r700_s1_sr0.4_rr0.001_bma_cal.csv"
input_BC_file_eva = f"{BC_file_path}/predict/BC_predict_r700_s1_sr0.4_rr0.001_bma_eva.csv"


df_predict_LSTM_cal = pd.read_csv(input_LSTM_file_cal, index_col=0, header=0)
df_predict_LSTM_eva = pd.read_csv(input_LSTM_file_eva, index_col=0, header=0)

df_predict_BC_cal = pd.read_csv(input_BC_file_cal, index_col=0, header=0)
df_predict_BC_eva = pd.read_csv(input_BC_file_eva, index_col=0, header=0)

for file_num in range(1, file_tot_num + 1): 
    if file_num < 10 or file_num in list(range(76,83)) or file_num == 488 or file_num == 431:
        df_obs = load_data(file_num)
        
        df_obs_cal = df_obs[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_obs_eva = df_obs[start_date_eva:end_date_eva].copy()

        target_cal = np.concatenate((df_obs_cal['Obs flow'].values[1:], df_obs_eva['Obs flow'].values[0:1])) #2898
        target_eva = np.concatenate((df_obs_eva['Obs flow'].values[1:],)) #1089

        # Ensure only the first row is selected if there are duplicates
        predict_LSTM_cal_row = df_predict_LSTM_cal.loc[f'file_{file_num}_cal']
        predict_LSTM_cal_values = predict_LSTM_cal_row.values[0] if isinstance(predict_LSTM_cal_row, pd.DataFrame) else predict_LSTM_cal_row.values
        predict_LSTM_eva_row = df_predict_LSTM_eva.loc[f'file_{file_num}_eva']
        predict_LSTM_eva_values = predict_LSTM_eva_row.values[0] if isinstance(predict_LSTM_eva_row, pd.DataFrame) else predict_LSTM_eva_row.values

        predict_BC_cal_row = df_predict_BC_cal.loc[f'file_{file_num}_cal']
        predict_BC_cal_values = predict_BC_cal_row.values[0] if isinstance(predict_BC_cal_row, pd.DataFrame) else predict_BC_cal_row.values
        predict_BC_eva_row = df_predict_BC_eva.loc[f'file_{file_num}_eva']
        predict_BC_eva_values = predict_BC_eva_row.values[0] if isinstance(predict_BC_eva_row, pd.DataFrame) else predict_BC_eva_row.values 

        predict_LSTM_cal = np.concatenate((predict_LSTM_cal_values[1:], predict_LSTM_eva_values[0:1]))  # 2898
        predict_LSTM_eva = np.concatenate((predict_LSTM_eva_values[1:],))  # 1089

        predict_BC_cal = np.concatenate((predict_BC_cal_values[1:], predict_BC_eva_values[0:1]))  # 2898
        predict_BC_eva = np.concatenate((predict_BC_eva_values[1:],))  # 1089

        target_cal = target_cal[-len(predict_LSTM_cal):]
        target_eva = target_eva[-len(predict_LSTM_eva):]

        predict_BC_cal = predict_BC_cal[-len(predict_LSTM_cal):]
        predict_BC_eva = predict_BC_eva[-len(predict_LSTM_eva):]

        # Plot hydrographs for calibration and evaluation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Calibration hydrograph
        axes[0].plot(target_cal, label='Obs', color='dimgray')
        axes[0].plot(predict_BC_cal, label='BC', color='crimson', lw = 0.8)
        axes[0].plot(predict_LSTM_cal, label='LSTM', color='royalblue', lw = 0.8)
        axes[0].set_title(f'Calibration Hydrograph (File {file_num})', fontsize=fs)
        axes[0].set_xlabel('Time Step', fontsize=fs)
        axes[0].set_ylabel('Flow', fontsize=fs)
        axes[0].legend(fontsize=fs, loc='upper right')

        # Calculate NSE values for calibration
        nse_bc_cal = BMK(target_cal, predict_BC_cal, "NSE")
        nse_lstm_cal = BMK(target_cal, predict_LSTM_cal, "NSE")

        # Add NSE values to the calibration plot
        axes[0].text(0.05, 0.95, f'NSE BC: {nse_bc_cal:.2f}\nNSE LSTM: {nse_lstm_cal:.2f}', 
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # Evaluation hydrograph
        axes[1].plot(target_eva, label='Obs', color='dimgray')
        axes[1].plot(predict_BC_eva, label='BC', color='crimson', lw = 0.8)
        axes[1].plot(predict_LSTM_eva, label='LSTM', color='royalblue', lw = 0.8)
        axes[1].set_title(f'Evaluation Hydrograph (File {file_num})', fontsize=fs)
        axes[1].set_xlabel('Time Step', fontsize=fs)
        axes[1].set_ylabel('Flow', fontsize=fs)
        axes[1].legend(fontsize=fs, loc='upper right')

        # Calculate NSE values for evaluation
        nse_bc_eva = BMK(target_eva, predict_BC_eva, "NSE")
        nse_lstm_eva = BMK(target_eva, predict_LSTM_eva, "NSE")

        # Add NSE values to the evaluation plot
        axes[1].text(0.05, 0.95, f'NSE BC: {nse_bc_eva:.2f}\nNSE LSTM: {nse_lstm_eva:.2f}', 
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/hydrograph_{file_num}.png")
        plt.close()
    
print("DONE")
