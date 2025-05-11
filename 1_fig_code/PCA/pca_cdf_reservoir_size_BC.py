import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

loc = "JP"
#loc = "US"
model_type = "BC"
fs = 20

spectral_radius = 0.4
ridge_param = 1.0

#reservoir_size_list = [5,10,20,50,100,150,200,250,300,400,500,700]
#reservoir_size_list = [5,10,50,100,200,300,400,500,700]
reservoir_size_list = [10,100,200,300,500,700,1000]
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

benchmark_lim = {
    'KGE': {'min': 0, 'max': 1},
    'NSE': {'min': 0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

output_dir = f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/cdf/combined"

def read_data(reservoir_size,benchmark):
    file_tag = f"r{reservoir_size}_s1_sr{spectral_radius}_rr{ridge_param}"
    data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}_hp/spin1/reservoir{reservoir_size}/{model_type}_r{reservoir_size}_s1_sr{spectral_radius}_rr{ridge_param}_is0.5_a0.1/results/{model_type}_results_{file_tag}_bma_eva.csv"
    df = pd.read_csv(data_folder)
    return df[f"{model_type}_{file_tag}_bma_{benchmark}_eva"]

def plot_cdf(data, benchmark):
    plt.figure(figsize=(10,8))
    colors = cm.turbo(np.linspace(0, 1, len(reservoir_size_list)))
    for i, column in enumerate(data.columns):
        sorted_data = np.sort(data[column])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, label=f"r{reservoir_size_list[i]}", color=colors[i])
    
    title = f"{benchmark}"
    plt.title(title, fontsize=fs)
    plt.ylabel('CDF', fontsize=fs)
    plt.xlabel(benchmark, fontsize=fs)
    plt.xticks(fontsize=fs*0.8)
    plt.yticks(fontsize=fs*0.8) 
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(benchmark_lim[benchmark]['min'], benchmark_lim[benchmark]['max'])
    plt.legend(loc='upper left', fontsize=fs)
    plt.tight_layout(rect=[0, 0, 1, 1])

def save_plot(output_dir, benchmark):
    filename = f"{model_type}_results_{benchmark}.png"
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

os.makedirs(output_dir, exist_ok=True)
data = pd.DataFrame()
for benchmark in benchmark_list:
    for reservoir_size in reservoir_size_list:
        data[f"r{reservoir_size}"] = read_data(reservoir_size, benchmark)
    plot_cdf(data, benchmark)
    save_plot(output_dir, benchmark)

print("DONE")
