import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#loc = "JP"
loc = "US"
model_type = "BC"
fs = 20

#reservoir_size_list = [5,10,15,20,25,30,50,60,70,80,90,100,150,200,250,300,350,400,500,700]
reservoir_size_list = [5,10,20,50,100,150,200,250,300,400,500,700]

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]

benchmark_lim = {
    'KGE': {'min': -2, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

output_dir = f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/box/"
os.makedirs(output_dir, exist_ok=True)
for benchmark in benchmark_list:
    data = pd.DataFrame()
    for reservoir_size in reservoir_size_list:
        file_tag = f"r{reservoir_size}_s1_sr0.4_rr0.001"
        data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}/{model_type}_{reservoir_size}/results/{model_type}_results_{file_tag}_bma_eva.csv"

        df = pd.read_csv(data_folder)

        data[f"r{reservoir_size}"] = df[f"{model_type}_{file_tag}_bma_{benchmark}_eva"]

    #plot boxplot per reservoir size  
    plt.figure(figsize=(10,6))
    data.boxplot()
    plt.title(f"{benchmark}", fontsize=fs)
    plt.ylabel(benchmark, fontsize=fs)
    plt.xlabel("Reservoir Size", fontsize=fs)
    plt.xticks(rotation=90, fontsize=fs*0.8)
    plt.yticks(fontsize=fs*0.8)
    plt.grid(True)

    plt.ylim(benchmark_lim[benchmark]['min'], benchmark_lim[benchmark]['max'])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_results_{benchmark}.png")
    plt.close()

print("DONE")


