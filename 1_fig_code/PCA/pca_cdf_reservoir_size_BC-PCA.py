import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

loc = "JP"
#loc = "GB"
#loc = "US"  
model_type = "BC-PCA-lasso"
fs = 20
n_components = 5

ridge_param = 1.0

spectral_radius = 0.4

#reservoir_size_list = [5,10,20,50,100,150,200,250,300,400,500,700]
reservoir_size_list = [5,10,50,100,200,300,500,700]
reservoir_size_list = [10,100,200,300,500,700,1000]
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

"""benchmark_lim = {
    'KGE': {'min': 0.2, 'max': 1},
    'NSE': {'min': 0.2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}"""

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

def create_output_dir(pc_indiv, benchmark, pc=None):
    if pc_indiv:
        return f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/cdf/PC{pc}"
    else:
        return f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/cdf/combined"

def read_data(pc_indiv, reservoir_size, pc, benchmark):
    file_tag = f"r{reservoir_size}_s1_sr{spectral_radius}_rr{ridge_param}"
    #data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}/{model_type}_{reservoir_size}_{ridge_param}/results/{model_type}_results_{file_tag}_rev_PC{pc}_eva.csv"
    data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}_hp/spin1/reservoir{reservoir_size}/BC-PCA-lasso_r{reservoir_size}_s1_sr{spectral_radius}_rr{ridge_param}_is0.5_a0.1/results/BC-PCA-lasso_results_r{reservoir_size}_s1_sr{spectral_radius}_rr{ridge_param}_rev_PC{pc}_eva.csv"
    df = pd.read_csv(data_folder)
    return df[f"{model_type}_{file_tag}_{benchmark}_eva"]

def plot_cdf(data, benchmark, pc_indiv, pc=None):
    plt.figure(figsize=(10,8) if pc_indiv else (15,9))
    colors = cm.turbo(np.linspace(0, 1, len(data.columns)))
    markers = ['o', '^', 'x', 's', 'D', 'v', 'p', 'P', '*', 'X']
    for i, column in enumerate(data.columns):
        sorted_data = np.sort(data[column])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        color_index = i // n_components
        alpha = 0.5 + (i % n_components) * 0.1  # Light to dark gradient
        if color_index >= len(colors):
            color_index = len(colors) - 1
        if pc_indiv:
            markers = markers[:len(reservoir_size_list)]
            marker_index = i % len(markers)
            plt.plot(sorted_data, yvals, label=f"r{reservoir_size_list[i]}", color=colors[i])
        else:
            marker_index = i % len(markers)
            plt.plot(sorted_data, yvals, label=column, color=colors[i], marker=markers[marker_index], markersize=7, markevery=10, alpha=alpha)
    title = f"{benchmark} PC{pc}" if pc_indiv else f"{benchmark} Combined PCs"
    plt.title(title, fontsize=fs)
    plt.ylabel('CDF', fontsize=fs)
    plt.xlabel(benchmark, fontsize=fs)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(benchmark_lim[benchmark]['min'], benchmark_lim[benchmark]['max'])
    plt.xticks(fontsize=fs*0.8)
    plt.yticks(fontsize=fs*0.8) 
    if pc_indiv:
        plt.legend(loc='upper left', fontsize=fs* 0.7)
    else:
        plt.legend(loc='upper left', ncol=2, fontsize=fs* 0.7)
    plt.tight_layout(rect=[0, 0, 1, 1])

def save_plot(output_dir, benchmark, pc_indiv, pc=None):
    filename = f"{model_type}_results_PC{pc}_{benchmark}.png" if pc_indiv else f"{model_type}_results_combined_{benchmark}.png"
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

for pc_indiv in [True, False]:
    if pc_indiv:
        for benchmark in benchmark_list:
            for pc in range(1, n_components +1):
                output_dir = create_output_dir(pc_indiv, benchmark, pc)
                os.makedirs(output_dir, exist_ok=True)
                data = pd.DataFrame()
                for reservoir_size in reservoir_size_list:
                    data[f"r{reservoir_size}"] = read_data(pc_indiv, reservoir_size, pc, benchmark)
                plot_cdf(data, benchmark, pc_indiv, pc)
                save_plot(output_dir, benchmark, pc_indiv, pc)
    else:
        for pc in range(1, n_components +1):
            for benchmark in benchmark_list:
                output_dir = create_output_dir(pc_indiv, benchmark)
                os.makedirs(output_dir, exist_ok=True)
                data = pd.DataFrame()
                for reservoir_size in reservoir_size_list:
                    for pc in range(1, n_components +1):
                        data[f"{reservoir_size} PC{pc}"] = read_data(pc_indiv, reservoir_size, pc, benchmark).values

            plot_cdf(data, benchmark, pc_indiv)
            save_plot(output_dir, benchmark, pc_indiv)

print(f"Output saved to {output_dir}")

print("DONE")
