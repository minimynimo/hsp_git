import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#loc = "JP"
loc = "US"
loc = "GB"

model_type = "BC-PCA-lasso"
fs = 20

#reservoir_size_list = [5,10,15,20,25,30,50,70,100,150,200,250,300,400,500,700]
reservoir_size_list = [5,10,20,50,100,150,200,250,300,400,500,700]
reservoir_size_list = [5,100,200,300,500,700]

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

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

for pc_indiv in [True, False]:
    for benchmark in benchmark_list:
        for pc in range(1,3):
            output_dir = f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/box/PC{pc}"
            os.makedirs(output_dir, exist_ok=True)
            data = pd.DataFrame()
            for reservoir_size in reservoir_size_list:
                file_tag = f"r{reservoir_size}_s1_sr0.4_rr0.001"
                data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}/{model_type}_{reservoir_size}/results/{model_type}_results_{file_tag}_rev_PC{pc}_eva.csv"
                
                df = pd.read_csv(data_folder)

                data[f"r{reservoir_size}"] = df[f"{model_type}_{file_tag}_{benchmark}_eva"]

            #plot boxplot per reservoir size  
            plt.figure(figsize=(10,6))
            data.boxplot()
            plt.title(f"{benchmark} PC{pc}", fontsize=fs)
            plt.ylabel(benchmark, fontsize=fs)
            plt.xlabel("Reservoir Size", fontsize=fs)
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.ylim(benchmark_lim[benchmark]['min'], benchmark_lim[benchmark]['max'])

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_type}_results_PC{pc}_{benchmark}.png")
            plt.close()
else:
    for benchmark in benchmark_list:
        output_dir = f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}/results/box/combined"
        os.makedirs(output_dir, exist_ok=True)
        data = pd.DataFrame()
        
        for reservoir_size in reservoir_size_list:
            file_tag = f"r{reservoir_size}_s1_sr0.4_rr0.001"
            for pc in range(1, 6):
                data_folder = f"/data0/funato/0_out/99_out/{loc}/{model_type}/{model_type}_{reservoir_size}/results/{model_type}_results_{file_tag}_rev_PC{pc}_eva.csv"
                
                df = pd.read_csv(data_folder)
                data[f"{reservoir_size}_PC{pc}"] = df[f"{model_type}_{file_tag}_{benchmark}_eva"].values

        # Plot boxplot per reservoir size  
        plt.figure(figsize=(12, 6))
        
        box = data.boxplot(patch_artist=True)  # Patch artist=True for color fill

        # Define box colors per reservoir size (5 PCs per size)
        box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightyellow', 'lightpink', 'lightgrey', 'lightgoldenrodyellow', 'lightsteelblue', 'lightseagreen', 'lightsalmon', 'lightblue']
        markers = ['o', 's', 'D', '^', 'v']

        # Apply colors and markers to special boxes
        for i, patch in enumerate(box.artists):
            reservoir_index = i // 5
            patch.set_facecolor(box_colors[reservoir_index])
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
            for j in range(5):
                line = box['medians'][i*5 + j]
                line.set_marker(markers[j])
                line.set_markerfacecolor('black')
                line.set_markeredgecolor('black')

        plt.title(f"{benchmark} Combined PCs", fontsize=fs)
        plt.ylabel(benchmark, fontsize=fs)
        plt.xlabel("Reservoir Size", fontsize=fs)
        plt.xticks(
            ticks=[i for i in range(1, len(data.columns) + 1)],
            labels=[f"{reservoir_size}_PC{pc}" for reservoir_size in reservoir_size_list for pc in range(1, 6)],
            rotation=90
        )
        plt.grid(True)
        plt.ylim(benchmark_lim[benchmark]['min'], benchmark_lim[benchmark]['max'])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type}_results_combined_{benchmark}.png")
        plt.close()

print("DONE")


