import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

n_components = 2
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

fs = 22

buf = ""

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
loc, ver = "US", 2
#loc, ver = "AUS", 2

#input_scale_list = [0.1,0.5,1.0,1.5,2.0] #5
input_scale_list = [0.5]
#alpha_list = [0.1,0.5,1.0,1.5] #4
alpha_list = [0.1]

spin_list = [1] #2
reservoir_size_list = [200,300] #6 10,100, ,1000

spectral_radius_list = [0.4] #6
ridge_param_list = [0.0001,0.001,0.01,0.1,1.0] #6  ,10

output_dir = f"/data0/funato/0_out/0_fig/{loc}/benchmark/BC_hyperparam/"
os.makedirs(output_dir, exist_ok=True)

def plot_box(data, benchmark, spin,  reservoir_size = None, buf = None):
    plt.figure(figsize=(15, 12))
    data.boxplot()
    plt.title(f'Box Plot of {benchmark} for Different Hyperparameters')
    plt.xlabel('Hyperparameter Combinations')
    plt.ylabel(f'{benchmark} Value')
    plt.xticks(rotation=90)
    plt.ylim(benchmark_box[benchmark]['min'], benchmark_box[benchmark]['max'])
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if buf == '_spin':
        output_path = f'{output_dir}/box/by{buf}/BC_hp_{benchmark}_boxplot.png'
    else:
        if reservoir_size is None:
            output_path = f'{output_dir}/box/spin{spin}/by{buf}/BC_hp_{benchmark}_boxplot.png'
        else:
            output_path = f'{output_dir}/box/spin{spin}/R{reservoir_size}/BC_hp_{benchmark}_boxplot.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_cdf(data, benchmark, spin,  col, reservoir_size = None, buf = None, color = False):
    data = pd.DataFrame(data)
    
    if color:
        group_colors = cm.Spectral(np.linspace(0, 1, col))
        colors = np.array([group_colors[i // (data.shape[1] // col)] for i in range(len(data.columns))])
    else:
        colors = cm.turbo(np.linspace(0, 1, len(data.columns)))

    for i, column in enumerate(data.columns):
        sorted_data = np.sort(data[column])
        yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))

        plt.plot(sorted_data, yvals, label=column, color=colors[i])
    
    plt.title(f'Cumulative Distribution Function (CDF)\n{benchmark} for Different Hyperparameters', fontsize=fs)
    plt.xlabel(f'{benchmark} Value', fontsize=fs)
    plt.ylabel('Cumulative Probability', fontsize=fs)
    plt.xticks(fontsize=fs*0.8)
    plt.yticks(fontsize=fs*0.8)
    plt.grid(True)
    plt.xlim(benchmark_cdf[benchmark]['min'], benchmark_cdf[benchmark]['max'])
    plt.ylim(0, 1)
    plt.legend(loc='lower right', fontsize='small', ncol=col)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    if buf == '_spin':
        output_path = f'{output_dir}/cdf/by{buf}/BC_hp_{benchmark}_cdf.png'
    else:
        if reservoir_size is None:
            output_path = f'{output_dir}/cdf/spin{spin}/by{buf}/BC_hp_{benchmark}_cdf.png'
        else:
            output_path = f'{output_dir}/cdf/spin{spin}/R{reservoir_size}/BC_hp_{benchmark}_cdf.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

benchmark_box = {
    'KGE': {'min': -2, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 15},
    'MAE': {'min': 0, 'max': 15}
}

benchmark_cdf = {
    'KGE': {'min': 0, 'max': 1},
    'NSE': {'min': 0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

for benchmark in benchmark_list:
    for input_scale in input_scale_list:
        for alpha in alpha_list:
            plot_df_spin = pd.DataFrame()
            for spin in spin_list:
                plot_df_ridge = pd.DataFrame()
                plot_df_reservoir = pd.DataFrame()
                # ridge_param
                buf = "_ridge_param"
                for ridge_param in ridge_param_list:
                    for reservoir_size in reservoir_size_list:
                        plot_df_mini = pd.DataFrame()
                        for spectral_radius in spectral_radius_list:
                            file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                            input_dir = f'/data0/funato/0_out/99_out/{loc}/BC_hp/spin{spin}/reservoir{reservoir_size}/BC{file_tag}_is{input_scale}_a{alpha}'
                            results_df = pd.read_csv(f'{input_dir}/results/BC_results{file_tag}_bma_eva.csv', index_col=0, header=0)
                            result_column = results_df[f'BC{file_tag}_bma_{benchmark}_eva']

                            plot_df_mini[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'] = result_column
                            if spectral_radius == 0.4:
                                plot_df_ridge[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'] = result_column

                        plot_box(plot_df_mini, benchmark, spin,  reservoir_size, buf = buf)
                        plt.figure(figsize=(12, 12))
                        plot_cdf(plot_df_mini, benchmark, spin,  3, reservoir_size, buf = buf)

                plot_box(plot_df_ridge, benchmark, spin,  buf = buf)
                plt.figure(figsize=(12, 12))
                plot_cdf(plot_df_ridge, benchmark, spin, col = len(ridge_param_list), buf = buf)

                # reservoir_size
                buf = "_reservoir_size"
                for reservoir_size in reservoir_size_list:
                    for spectral_radius in spectral_radius_list:
                        plot_df_mini = pd.DataFrame()
                        for ridge_param in ridge_param_list:
                            file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                            input_dir = f'/data0/funato/0_out/99_out/{loc}/BC_hp/spin{spin}/reservoir{reservoir_size}/BC{file_tag}_is{input_scale}_a{alpha}'
                            results_df = pd.read_csv(f'{input_dir}/results/BC_results{file_tag}_bma_eva.csv', index_col=0, header=0)
                            result_column = results_df[f'BC{file_tag}_bma_{benchmark}_eva']

                            plot_df_mini[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'] = result_column
                            if spectral_radius == 0.4:
                                plot_df_reservoir[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'] = result_column
                                plot_df_spin[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'] = result_column

                plot_box(plot_df_reservoir, benchmark, spin,buf = buf)
                plt.figure(figsize=(9, 9))
                plot_cdf(plot_df_reservoir, benchmark, spin, col = 1,buf = buf)

            buf = "_spin"
            plot_box(plot_df_spin, benchmark, spin, buf = buf)
            plt.figure(figsize=(9, 9))
            plot_cdf(plot_df_spin, benchmark, spin, col = len(spin_list),buf = buf, color = True)
    

print('Done')
