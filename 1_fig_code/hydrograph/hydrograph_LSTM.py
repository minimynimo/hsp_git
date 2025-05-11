import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define parameters
input_size = 3
hidden_size = 20 # 20 (kratzert et al., 2018)
output_size = 1
batch_first = True
learning_rate = 0.001 # (kratzert et al., 2018)
num_epochs = 200 # 200 was tested but 50 was optimal in (kratzert et al., 2018)
window_size = 365  # Number of past days to use for prediction usually from 20 to 60 #30 # 365 (kratzert et al., 2018)
batch_size = 512 # 512 (kratzert et al., 2018) # larger batch size for faster training and more stable learning but may generalize worse 
dropout = 0.1  # 0.1 in (kratzert et al., 2018)
# sample size = 2892 = 2922 - window_size

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'


loc, ver = "JP", 2
#loc, ver = "US", 2

base_dir = f'/data0/funato/0_out/99_out/{loc}/LSTM/predict'

file_tag = f"_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout}"
output_dir = f'/data0/funato/0_out/0_fig/{loc}/LSTM/hydrograph/LSTM{file_tag}'
os.makedirs(output_dir, exist_ok=True)

if loc == "JP" and ver == 1:
    file_tot_num = 135
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"
elif loc == "US" and ver == 1:
    file_tot_num = 669
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "US" and ver == 2:
    file_tot_num = 667
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"

def visualize_predictions(dates_cal, obs_cal, pred_cal, dates_eva, obs_eva, pred_eva, file_num):
    plt.figure(figsize=(14, 7))
    
    # Plot calibration period
    plt.plot(dates_cal, obs_cal, label='Observed (Calibration)', color='blue')
    plt.plot(dates_cal, pred_cal, label='Predicted (Calibration)', color='cyan')
    
    # Plot evaluation period
    plt.plot(dates_eva, obs_eva, label='Observed (Evaluation)', color='green')
    plt.plot(dates_eva, pred_eva, label='Predicted (Evaluation)', color='red')
    
    # Add vertical line to separate calibration and evaluation periods
    plt.axvline(x=dates_eva[0], color='black', linestyle='--', label='Calibration/Evaluation Split')
    
    plt.xlabel('Date')
    plt.ylabel('Flow')
    #plt.ylim(0,1)
    plt.title('Observed vs Predicted Flow')
    plt.legend()
    plt.savefig(f'{output_dir}/LSTM_hydrograph_file{file_num}.png')
    plt.close()

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, varssim_dir, loc):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    df = df.reindex(full_date_range)
    
    return df

# Example usage
if __name__ == "__main__":
    predict_cal_df = pd.read_csv(f'{base_dir}/LSTM_predict{file_tag}_cal.csv', index_col=0)
    predict_eva_df = pd.read_csv(f'{base_dir}/LSTM_predict{file_tag}_eva.csv', index_col=0)

    # Extract dates and values
    dates_cal = pd.to_datetime(predict_cal_df.columns) #2921
    dates_eva = pd.to_datetime(predict_eva_df.columns) #2190


    for file_num in range(1, file_tot_num + 1):
        # Load data
        df = load_data(file_num, varssim_dir, loc)
        
        df_cal = df[start_date_cal:end_date_cal].copy()
        df_eva = df[start_date_eva:end_date_eva].copy()
        
        obs_cal = df_cal['Obs flow'].values.reshape(-1, 1).astype(float)
        obs_eva = df_eva['Obs flow'].values.reshape(-1, 1)
        obs_cal = np.concatenate((df_cal['Obs flow'].values[1:],))
        obs_eva = np.concatenate((df_eva['Obs flow'].values[1:],))

        pred_cal = predict_cal_df.loc[f'file_{file_num}_cal'].values.astype(float).flatten()
        pred_eva = predict_eva_df.loc[f'file_{file_num}_eva'].values.astype(float).flatten()

        visualize_predictions(dates_cal, obs_cal, pred_cal, dates_eva, obs_eva, pred_eva, file_num)

    print("Done")