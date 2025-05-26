# Creates box, line, and point plots for the Spatial vs Regression PUB benchmark
# The norm for all plots of PUB can be shown for the training and testing data
# The difference between the following can be shown for the training and testing data:
### RCBC-BMA and BMA
### BC and BMA
### RCBC-BMA and BC
### BC and PCA 

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#########
str_stp_tag = False

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc = "US"

nocal_tag = ''
n_components = 3
reservoir_size = 200
#########
file_tag = f'r{reservoir_size}_s1_sr0.4_rr0.001'


ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87

fs = 20 # 6 for paper

def create_box_plot(ax, plot_df, benchmark, train_test):
    train_test_name = 'Training Basin' if train_test == 'train' else 'Testing Basin'
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    if 'norm' in calc_mode:
        ax.set_title(f'Boxplot for {benchmark} {calc_mode}\n\n{train_test_name}', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Boxplot for {benchmark} {calc_mode}\n\n{train_test_name}', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

def create_line_plot(ax, plot_df, benchmark, train_test):
    train_test_name = 'Training Basin' if train_test == 'train' else 'Testing Basin'

    positions = range(1, len(plot_df.columns) + 1) 
    lw = 0.4
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='gray')
    
    if 'norm' in calc_mode:
        ax.set_title(f'Line plot for {benchmark} {calc_mode}\n\n{train_test_name}', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Line plot for {benchmark} {calc_mode}\n\n{train_test_name}', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)

def create_point_plot(ax, plot_df, benchmark, train_test):
    train_test_name = 'Training Basin' if train_test == 'train' else 'Testing Basin'

    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')

    if 'norm' in calc_mode:
        ax.set_title(f'Point plot for {benchmark} {calc_mode}\n\n{train_test_name}')
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Point plot for {benchmark} {calc_mode}\n\n{train_test_name}')
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)

def load_BC_PUB_data():
    BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_kfold_{reservoir_size}/test_basin/results/BC_PUB_results_r{reservoir_size}_s1_sr0.4_rr0.001_eva.csv'
    BC_PUB_df = pd.read_csv(BC_PUB_data)
    BC_PUB_column_data = np.array(BC_PUB_df[f'BC_PUB_r{reservoir_size}_s1_sr0.4_rr0.001_{benchmark}_eva'].tolist())
    return BC_PUB_column_data

def load_BMA_PUB_data():
    BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB_kfold/test_basin/results/BMA_PUB_results_eva.csv'
    BMA_PUB_df = pd.read_csv(BMA_PUB_data)
    BMA_PUB_column_data = np.array(BMA_PUB_df[f'BMA_PUB_{benchmark}_eva'].tolist())
    return BMA_PUB_column_data
    
def load_BC_PCA_lasso_PUB_data(pc):
    BC_PCA_lasso_result_data_dir = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_kfold_{reservoir_size}/test_basin/results/BC-PCA-lasso_PUB_results_rev_PC{pc}_eva.csv')
    BC_PCA_lasso_PUB_column_data = BC_PCA_lasso_result_data_dir[f'BC-PCA-lasso_r{reservoir_size}_s1_sr0.4_rr0.001_{benchmark}_eva']
    return BC_PCA_lasso_PUB_column_data

def safe_min(sequence, default=float('inf')):
    return np.min(sequence) if len(sequence) > 0 else default

def safe_max(sequence, default=float('-inf')):
    return np.max(sequence) if len(sequence) > 0 else default

def safe_subtract(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    return arr1[:min_len] - arr2[:min_len]

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/Spatial_PUB{nocal_tag}_kfold'
os.makedirs(os.path.join(output_dir, 'box', 'norm'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'line', 'norm'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point', 'norm'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box', 'dif'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'line', 'dif'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point', 'dif'), exist_ok=True)

calc_mode_to_name = {
    'BMA_norm': 'BMA_PUB',
    'BC_norm': 'BC_PUB',
    'PCA_norm': 'BC-PCA-lasso_PUB',
    'BC_BMA_dif': 'dif_BC & BMA',
    'BC_PCA_dif': 'dif_PCA & BC'
}

"""
calc_mode_to_name = {
    'PCA_norm': 'BC-PCA-lasso_PUB',
    'BC_PCA_dif': 'dif_PCA & BC'
}"""

benchmark_limits_norm = {
    'KGE': {'min': -2, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 20}
}
benchmark_limits_dif = {
    'KGE': {'min': -3, 'max': 3},
    'NSE': {'min': -3, 'max': 3},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -3, 'max': 3},
    'VE': {'min': -2, 'max': 2},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': -10, 'max': 10},
    'MAE': {'min': -3, 'max': 3}
}

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]

for benchmark in benchmark_list:
    buf = f''

    for calc_type in ['norm', 'dif']:
        calc_mode_list = [mode for mode in calc_mode_to_name.keys() if calc_type in mode]

        plot_train_df = pd.DataFrame() 
        plot_test_df = pd.DataFrame() 

        for calc_mode in calc_mode_list:
            calc_name = calc_mode_to_name[calc_mode]
            # Create a figure with 1 subplot for each plot type
            x_size = 12 #26
            y_size = 15
            fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))
            fig_line, ax_line = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))
            fig_point, ax_point = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))    

            global_y_min, global_y_max = float('inf'), float('-inf')

            # for median color in PCA
            median_color_train = ['orange'] * 2
            median_color_test = []

            og_BMA_data = f'/data0/funato/0_out/99_out/{loc}/BMA/results/BMA_results_eva.csv'
            og_BMA_df = pd.read_csv(og_BMA_data)
            og_BMA_column = f'BMA_{benchmark}_eva' 
            og_BMA_data = og_BMA_df[og_BMA_column]

            og_BC_data = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}/results/BC_results_{file_tag}_bma_eva.csv'
            og_BC_df = pd.read_csv(og_BC_data)
            og_BC_column = f'BC_{file_tag}_bma_{benchmark}_eva'
            og_BC_data = og_BC_df[og_BC_column]

            og_BC_PCA_lasso_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}/results/BC_results_{file_tag}_eva.csv'
            og_BC_PCA_lasso_df = pd.read_csv(og_BC_PCA_lasso_data)
            og_BC_PCA_lasso_column = f'BC-og_{file_tag}_{benchmark}_eva'
            og_BC_PCA_lasso_data = og_BC_PCA_lasso_df[og_BC_PCA_lasso_column]

            if 'norm' in calc_mode:
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    'BMA' : og_BMA_data,
                    'BC' : og_BC_data
                })], axis=1)
            elif calc_mode == 'BC_BMA_dif':
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    'BC - BMA' : og_BC_data - og_BMA_data
                })], axis=1)
            elif calc_mode == 'BC_PCA':
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    'BC - PCA' : og_BC_data - og_BC_PCA_lasso_data
                })], axis=1)


            BMA_PUB_column_test_data = load_BMA_PUB_data()
            BC_PUB_column_test_data = load_BC_PUB_data()
            
            if calc_mode == 'BMA_norm':
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    f'BMA-PUB' : BMA_PUB_column_test_data
                })], axis=1)
                global_y_min = min(global_y_min, safe_min(BMA_PUB_column_test_data))
                global_y_max = max(global_y_max, safe_max(BMA_PUB_column_test_data))

            elif calc_mode == 'BC_norm':
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    f'BC-PUB' : BC_PUB_column_test_data
                })], axis=1)
                global_y_min = min(global_y_min, safe_min(BC_PUB_column_test_data))
                global_y_max = max(global_y_max, safe_max(BC_PUB_column_test_data))

            elif calc_mode == 'PCA_norm':
                for pc in range(1, n_components + 1):
                    median_color_test.append('red' if pc == 1 else 'blue' if pc == 2 else 'green')
                    BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc)
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'BCPCA-PUB_PC{pc}' : BC_PCA_lasso_PUB_column_test_data
                    })], axis=1)
                    global_y_min = min(global_y_min, safe_min(BC_PCA_lasso_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(BC_PCA_lasso_PUB_column_test_data))

            elif calc_mode == 'BC_BMA_dif':
                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                    f'(BC-PUB) - (BMA-PUB)' : safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)
                })], axis=1)
                global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))
                global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))

            elif calc_mode == 'BC_PCA_dif':
                for pc in range(1, n_components + 1):
                    BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc)
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'(BC-PUB) - (BCPCA-PUB_PC{pc})' : safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)
                    })], axis=1)
                    global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))
                    global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))

        create_box_plot(ax_box, plot_test_df, benchmark, 'test')
        create_line_plot(ax_line, plot_test_df, benchmark, 'test')
        create_point_plot(ax_point, plot_test_df, benchmark, 'test')

        # Set the same y-axis limits for subplots
        all_axes = [ax_box, ax_line, ax_point]

        for ax in all_axes:
            if benchmark in benchmark_limits_norm and 'norm' in calc_mode:
                ax.set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
            elif benchmark in benchmark_limits_dif and 'dif' in calc_mode:
                ax.set_ylim(benchmark_limits_dif[benchmark]['min'], benchmark_limits_dif[benchmark]['max'])

        for ax in all_axes:
            for label in ax.get_xticklabels():
                label.set_rotation(90)

        # median color for PCA
        if calc_mode == 'PCA_norm':
            for ax in [ax_point]:
                for i, pc in enumerate(range(1, n_components + 1)):
                    ax.get_lines()[i].set_color(median_color_test[i])
                    ax.get_lines()[i].set_label(f'PC{pc}')
                ax.legend()
            
        fig_box.tight_layout()
        fig_line.tight_layout()
        fig_point.tight_layout()


        box_output_path = os.path.join(output_dir, f'box/{calc_type}/{benchmark}_boxplot{buf}_eva.jpg')
        line_output_path = os.path.join(output_dir, f'line/{calc_type}/{benchmark}_lineplot{buf}_eva.jpg')
        point_output_path = os.path.join(output_dir, f'point/{calc_type}/{benchmark}_pointplot{buf}_eva.jpg')

        fig_box.savefig(box_output_path)
        fig_line.savefig(line_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        plt.close(fig_line)
        plt.close(fig_point)

        print(f'{benchmark} {nocal_tag}{calc_type} DONE!')
print(f'saved to {output_dir}')

print("DONE!")
