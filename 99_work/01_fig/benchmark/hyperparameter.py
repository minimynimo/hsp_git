import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

input_dir = "/data0/funato/hsp/0_out/LSTM/GB/individual_hyperparam/results"
output_dir = "/data0/funato/hsp/0_out/1_fig/hyperparam/LSTM"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ['KGE', 'NSE', 'RMSE', 'MAE']

file_prefix = "LSTM_results_"

plot_df = pd.DataFrame()
for file in os.listdir(input_dir):
    if file.startswith(file_prefix) and file.endswith("_eva.csv"):
        filepath = os.path.join(input_dir, file)
        df = pd.read_csv(filepath)
        
        # Extract hyperparameter settings from filename
        params_part = file[len(file_prefix):-8]  # Remove prefix and _eva.csv
        params = params_part.split('_')
        param_dict = {}
        for i in range(0, len(params), 2):
            param_dict[params[i]] = params[i+1]

        df['Params'] = str(param_dict)
        plot_df = pd.concat([plot_df, df], ignore_index=True)

benchmark_limit = {
    'KGE': (0, 1),
    'NSE': (0, 1),
    'RMSE': (0, None),
    'MAE': (0, None)
}


# the x axis represent the param setting index, the y axis represent the benchmark value in that setting as box plots. Each benchmark is shown in a different png. 
for benchmark in benchmark_list:
    plt.figure(figsize=(10, 9))
    data_to_plot = []
    labels = []
    
    for params_str, group in plot_df.groupby('Params'):
        data_to_plot.append(group[benchmark].values)
        labels.append(params_str)
    
    plt.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(benchmark)
    plt.title(f"LSTM Hyperparameter Tuning - {benchmark}")
    plt.tight_layout()
    plt.ylim(benchmark_limit[benchmark])
    plt.grid()
    
    # Save plot
    output_filepath = os.path.join(output_dir, f"LSTM_hyperparam_{benchmark}.png")
    plt.savefig(output_filepath)
    plt.close()
