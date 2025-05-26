import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fs = 23

#BcProx_path_cal = "/data0/funato/0_out/99_out/JP/BC_PUB_kfold_200_0.001/test_basin/results/BC_PUB_results_r200_s1_sr0.4_rr0.001_eva.csv"
BcProx_path_val = "/data0/funato/0_out/99_out/JP/BC_PUB_kfold_200_0.001/test_basin/results/BC_PUB_results_r200_s1_sr0.4_rr0.001_eva.csv"
BcReg_path_val = "/data0/funato/0_out/99_out/JP/BC-PCA-lasso_PUB_kfold_200_0.001/test_basin/results/BC-PCA-lasso_PUB_results_rev_PC2_eva.csv"
LSTM_path_val = "/data0/funato/0_out/99_out/JP/LSTM_PUB/results/LSTM_PUB_results_eva.csv"

# Load data
BcProx_data = pd.read_csv(BcProx_path_val)
BcReg_data = pd.read_csv(BcReg_path_val)
LSTM_data = pd.read_csv(LSTM_path_val)

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]
os.makedirs('/data0/funato/0_out/0_fig/JP/benchmark/kfold', exist_ok=True)



# Extract relevant columns (assuming the evaluation metric is in a column named 'value')
for benchmark in benchmark_list:
    BcProx_values = BcProx_data[f'BC_PUB_r200_s1_sr0.4_rr0.001_{benchmark}_eva']
    BcReg_values = BcReg_data[f'BC-PCA-lasso_r200_s1_sr0.4_rr0.001_{benchmark}_eva']
    LSTM_values = LSTM_data[f'LSTM_PUB_{benchmark}_eva']

    # Combine data for box plot
    data = [BcProx_values, BcReg_values, LSTM_values]
    labels = ['BcProx', 'BcReg', 'LSTM']

    # Create box plot
    plt.figure(figsize=(9, 6))
    box = plt.boxplot(data, labels=labels, patch_artist=True)

    # Set colors for each box
    colors = ['#b39ddb', '#e57373', '#7cc7bc']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set median line color to black
    for median in box['medians']:
        median.set_color('black')

    plt.ylabel(f'{benchmark}', fontsize=fs)
    plt.grid(axis='y')
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    if benchmark in ["NSE", "KGE", "E1", "VE"]:
        plt.ylim(-2, 1)  # Adjust y-axis limits as needed
    elif benchmark in ["RMSE", "MAE"]:
        plt.ylim(0, 10)
    plt.tight_layout()

    # Save or show the plot
    plt.savefig(f'/data0/funato/0_out/0_fig/JP/benchmark/kfold/kfold_results_{benchmark}.png')




