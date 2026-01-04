import pandas as pd
import numpy as no
import os
import matplotlib.pyplot as plt

# plot the results comparing the epoch to the prediction performance

epochs_list = [10,20,30,40,50,60,70,80,90,100]

output_dir = "/data0/funato/hsp/0_out/1_fig_all_temp/benchmark/hyperparameter_analysis/LSTM/GB/epoch_analysis"
os.makedirs(output_dir, exist_ok=True)


## INDIVIDUAL
plot_df = pd.DataFrame()
for epochs in epochs_list:
    input_dir = f"/data0/funato/hsp/0_out/LSTM/GB/individual/results/epoch_{epochs}/LSTM_results_h32_lr0.001_e100_w270_b256_d0.2_ep{epochs}_eva.csv"
    df = pd.read_csv(input_dir)
    # ensure RMSE is a DataFrame so we can add an epochs column
    col = df[['RMSE']].copy()
    # get median RMSE across all basins
    median_mae = col.median().to_frame().T
    temp_df = median_mae.reset_index(drop=True)
    temp_df['epochs'] = epochs
    plot_df = pd.concat([plot_df, temp_df], axis=0, ignore_index=True)


plt.figure(figsize=(10, 6))
plt.plot(plot_df['epochs'], plot_df['RMSE'], marker='o')
plt.title('LSTM Prediction Performance vs. Number of Epochs', fontsize=16)
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(epochs_list)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'LSTM_RMSE_vs_Epochs_individual.png'), dpi=300)
plt.close() 

## REGIONAL
epochs_list = [5, 10,15,20,25,30]
plot_df = pd.DataFrame()
for epochs in epochs_list:
    input_path = f"/data0/funato/hsp/0_out/LSTM/GB/regional/results/LSTM_results_hs256_do0.2_e{epochs}_cal.csv"
    df = pd.read_csv(input_path)
    # ensure RMSE is a DataFrame so we can add an epochs column
    col = df[['RMSE']].copy()
    # get median RMSE across all basins
    median_mae = col.median().to_frame().T
    temp_df = median_mae.reset_index(drop=True)
    temp_df['epochs'] = epochs
    plot_df = pd.concat([plot_df, temp_df], axis=0, ignore_index=True)      

plt.figure(figsize=(10, 6)) 
plt.plot(plot_df['epochs'], plot_df['RMSE'], marker='o')
plt.title('LSTM Prediction Performance vs. Number of Epochs', fontsize=16)
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(epochs_list)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'LSTM_RMSE_vs_Epochs_regional.png'), dpi=300)
plt.close() 
