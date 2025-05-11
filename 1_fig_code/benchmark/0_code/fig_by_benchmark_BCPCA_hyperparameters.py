import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

n_components = 3
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

fs = 22

buf = ""

non_arid_files = True

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc, ver = "US", 1
#loc, ver = "US", 2
#loc, ver = "GB",2

if loc != "US":
    non_arid_files = False
if non_arid_files:
    non_arid_dir = "non_arid/"
else:
    non_arid_dir = ""

#input_scale_list = [0.1,0.5,1.0,1.5,2.0] #5
input_scale_list = [0.5]
#alpha_list = [0.1,0.5,1.0,1.5] #4
alpha_list = [0.1, 0.01, 0.001]
spin_list = [1] #2
reservoir_size_list = [10,100,200,300,500,700] #6 ,1000

#spectral_radius_list = [0.1,0.2,0.4,0.6,0.8,1.0] #6
spectral_radius_list = [0.1,0.4,1.0]
ridge_param_list = [0.001,0.01,0.1,1.0] #6 10 #0.0001,

output_dir = f"/data0/funato/0_out/0_fig/{loc}/benchmark/BCPCA_hyperparam/{non_arid_dir}"
os.makedirs(output_dir, exist_ok=True)

def plot_box(data, benchmark, spin, pc, reservoir_size = None, buf = None):
    plt.figure(figsize=(26, 12))
    data.boxplot()
    plt.title(f'Box Plot of {benchmark} for Different Hyperparameters')
    plt.xlabel('Hyperparameter Combinations')
    plt.ylabel(f'{benchmark} Value')
    plt.xticks(rotation=90)
    plt.ylim(benchmark_box[benchmark]['min'], benchmark_box[benchmark]['max'])
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if buf == '_spin':
        output_path = f'{output_dir}/box/by{buf}/BCPCA_hp_{benchmark}_boxplot_PC{pc}.png'
    else:
        if reservoir_size is None:
            output_path = f'{output_dir}/box/spin{spin}/by{buf}/BCPCA_hp_{benchmark}_boxplot_PC{pc}.png'
        else:
            output_path = f'{output_dir}/box/spin{spin}/R{reservoir_size}/BCPCA_hp_{benchmark}_boxplot_PC{pc}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_cdf(data, benchmark, spin, pc, col, reservoir_size = None, buf = None, color = False):
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
    
    plt.title(f'Cumulative Distribution Function (CDF) of {benchmark} for Different Hyperparameters', fontsize=fs)
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
        output_path = f'{output_dir}/cdf/by{buf}/BCPCA_hp_{benchmark}_cdf.png'
    else:
        if reservoir_size is None:
            output_path = f'{output_dir}/cdf/spin{spin}/by{buf}/BCPCA_hp_{benchmark}_cdf_PC{pc}.png'
        else:
            output_path = f'{output_dir}/cdf/spin{spin}/R{reservoir_size}/BCPCA_hp_{benchmark}_cdf_PC{pc}.png'
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

for benchmark in benchmark_list:
    for input_scale in input_scale_list:
        alpha = 0.1
        plot_df_spin = pd.DataFrame()
        for spin in spin_list:
            plot_df_ridge = pd.DataFrame()
            plot_df_reservoir = pd.DataFrame()
            plot_df_reservioir_spectral = pd.DataFrame()
            # ridge_param
            buf = "_ridge_param"
            for ridge_param in ridge_param_list:
                for reservoir_size in reservoir_size_list:
                    plot_df_mini = pd.DataFrame()
                    for spectral_radius in spectral_radius_list:
                        for pc in range(1, n_components+1):
                            file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                            input_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_hp/{non_arid_dir}spin{spin}/reservoir{reservoir_size}/BC-PCA-lasso{file_tag}_is{input_scale}_a{alpha}'
                            results_df = pd.read_csv(f'{input_dir}/results/BC-PCA-lasso_results{file_tag}_rev_PC{pc}_eva.csv', index_col=0, header=0)
                            result_column = results_df[f'BC-PCA-lasso{file_tag}_{benchmark}_eva']

                            plot_df_mini[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column
                            if spectral_radius == 0.4:
                                plot_df_ridge[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column

                            if ridge_param == 1.0 and spectral_radius == 0.4:
                                plot_df_reservioir_spectral[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column

                plot_box(plot_df_mini, benchmark, spin, pc, reservoir_size, buf = buf)
                plt.figure(figsize=(12, 12))
                plot_cdf(plot_df_mini, benchmark, spin, pc, 3, reservoir_size, buf = buf)
                plt.tight_layout(rect=[0, 0, 1, 1])

            plot_box(plot_df_ridge, benchmark, spin, pc, buf = buf)
            plt.figure(figsize=(12, 12))
            plot_cdf(plot_df_ridge, benchmark, spin, pc,col = len(ridge_param_list), buf = buf)
            plt.tight_layout(rect=[0, 0, 1, 1])

            buf = "_reservoir_spectral"
            plot_box(plot_df_reservioir_spectral, benchmark, spin, pc, buf=buf)
            plt.figure(figsize=(12, 12))
            plot_cdf(plot_df_reservioir_spectral, benchmark, spin, pc, col=len(spectral_radius_list), buf=buf)
            plt.tight_layout(rect=[0, 0, 1, 1])

            for pc in range(1, n_components+1):
                # reservoir_size
                buf = "_reservoir_size"
                for reservoir_size in reservoir_size_list:
                    for ridge_param in ridge_param_list:
                        plot_df_mini = pd.DataFrame()
                        for spectral_radius in spectral_radius_list:
                            file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                            input_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_hp/{non_arid_dir}/spin{spin}/reservoir{reservoir_size}/BC-PCA-lasso{file_tag}_is{input_scale}_a{alpha}'
                            results_df = pd.read_csv(f'{input_dir}/results/BC-PCA-lasso_results{file_tag}_rev_PC{pc}_eva.csv', index_col=0, header=0)
                            result_column = results_df[f'BC-PCA-lasso{file_tag}_{benchmark}_eva']

                            plot_df_mini[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column
                            if spectral_radius == 0.4 and ridge_param == 1.0:
                                plot_df_reservoir[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column
                                plot_df_spin[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column

                plot_box(plot_df_reservoir, benchmark, spin, pc,buf = buf)
                plt.figure(figsize=(9, 9))
                plot_cdf(plot_df_reservoir, benchmark, spin, pc,col = 1,buf = buf)
                plt.tight_layout(rect=[0, 0, 1, 1])

                # New plots for comparison of ridge param and spectral radius for each spin and reservoir size
                for reservoir_size in reservoir_size_list:
                    for spin in spin_list:
                        plot_df_comparison = pd.DataFrame()
                        for ridge_param in ridge_param_list:
                            for spectral_radius in spectral_radius_list:
                                file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                                input_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_hp/{non_arid_dir}spin{spin}/reservoir{reservoir_size}/BC-PCA-lasso{file_tag}_is{input_scale}_a{alpha}'
                                results_df = pd.read_csv(f'{input_dir}/results/BC-PCA-lasso_results{file_tag}_rev_PC{pc}_eva.csv', index_col=0, header=0)
                                result_column = results_df[f'BC-PCA-lasso{file_tag}_{benchmark}_eva']

                                plot_df_comparison[f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_pc{pc}'] = result_column

                        buf = f"_comparison_r{reservoir_size}_s{spin}"
                        plot_box(plot_df_comparison, benchmark, spin, pc, reservoir_size, buf = buf)
                        plt.figure(figsize=(12, 12))
                        plot_cdf(plot_df_comparison, benchmark, spin, pc, col=len(ridge_param_list), reservoir_size=reservoir_size, buf=buf)
                        plt.tight_layout(rect=[0, 0, 1, 1])



        buf = "_spin"
        #plot_box(plot_df_spin, benchmark, spin, pc,buf = buf)
        #plt.figure(figsize=(9, 9))
        #plot_cdf(plot_df_spin, benchmark, spin, pc,col = len(spin_list),buf = buf, color = True)
        #plt.tight_layout(rect=[0, 0, 1, 1])

        spectral_radius = 0.4
        plot_df_alpha = pd.DataFrame()
        for reservoir_size in reservoir_size_list:
            plot_df_alpha_by_res = pd.DataFrame()
            for alpha in alpha_list:
                for ridge_param in ridge_param_list:
                    for pc in range(1, n_components+1):
                        file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
                        input_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_hp/{non_arid_dir}spin{spin}/reservoir{reservoir_size}/BC-PCA-lasso{file_tag}_is{input_scale}_a{alpha}'
                        results_df = pd.read_csv(f'{input_dir}/results/BC-PCA-lasso_results{file_tag}_rev_PC{pc}_eva.csv', index_col=0, header=0)
                        result_column = results_df[f'BC-PCA-lasso{file_tag}_{benchmark}_eva']

                        if ridge_param == 1.0:
                            plot_df_alpha_by_res[f'a{alpha}_rr{ridge_param}_pc{pc}'] = result_column
                            if reservoir_size in [200,300,500,700]:
                                plot_df_alpha[f'r{reservoir_size}_a{alpha}_rr{ridge_param}_pc{pc}'] = result_column

            buf = f"_alpha_by_r{reservoir_size}"
            plot_box(plot_df_alpha_by_res, benchmark, spin, pc, buf = buf)
            plt.figure(figsize=(12, 12))
            plot_cdf(plot_df_alpha_by_res, benchmark, spin, pc, col=len(reservoir_size_list), buf = buf)
            plt.tight_layout(rect=[0, 0, 1, 1])
        
        buf = "_alpha"
        plot_box(plot_df_alpha, benchmark, spin, pc, buf = buf)
        plt.figure(figsize=(12, 12))    
        plot_cdf(plot_df_alpha, benchmark, spin, pc, col=len(reservoir_size_list), buf = buf)
        plt.tight_layout(rect=[0, 0, 1, 1])

print('Done')
