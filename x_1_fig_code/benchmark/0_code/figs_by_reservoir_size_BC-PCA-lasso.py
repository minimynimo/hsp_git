import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import matplotlib.pyplot as plt

n_components = 5
reservoir_size_list = [100,300,500,700]
loc = "JP"
# loc = "US"
fs = 15
benchmark_list = ["KGE", "NSE", "logNSE", "E1", "VE", "d", "RMSE", "MAE"]



for pc_n in range(1,n_components +1):
    output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA-lasso/reservoir_size/box/PC{pc_n}'
    os.makedirs(output_dir, exist_ok=True)
    for benchmark in benchmark_list:    
        plot_df = pd.DataFrame()
        for rsize in reservoir_size_list:
            result_file = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{rsize}/results/BC-PCA-lasso_results_r{rsize}_s1_sr0.4_rr0.001_rev_PC{pc_n}_eva.csv')
            region_data = result_file[f'BC-PCA-lasso_r{rsize}_s1_sr0.4_rr0.001_{benchmark}_eva']
            plot_df = pd.concat([plot_df, region_data.rename(f'R {rsize}')], axis=1)

        fig, ax = plt.subplots(figsize=(6, 8))
        plot_df.boxplot(ax=ax)
        ax.set_title(f'PC{pc_n} {benchmark}', fontsize=fs)
        ax.set_xlabel('Reservoir Size', fontsize=fs)
        ax.set_ylabel(benchmark, fontsize=fs)
        ax.tick_params(labelsize=fs)
        #plt.xticks(rotation=90)
        plt.xlabel('Reservoir Size', fontsize=fs)

        if benchmark == "KGE":
            ymin, ymax = -10, 1
        elif benchmark == "NSE":
            ymin, ymax = -10, 1
        elif benchmark == "logNSE":
            ymin, ymax = -1, 1
        elif benchmark == "E1":
            ymin, ymax = -10, 1
        elif benchmark == "VE":
            ymin, ymax = -10, 1
        elif benchmark == "d":
            ymin, ymax = -10, 1
        elif benchmark == "RMSE":
            ymin, ymax = 0, 20
        elif benchmark == "MAE":
            ymin, ymax = 0, 20
        else:
            ymin, ymax = plot_df.min().min(), plot_df.max().max()
        
        ax.set_ylim(ymin, ymax)

        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/PC{pc_n}_{benchmark}.png')
        plt.close()

print("DONE")
