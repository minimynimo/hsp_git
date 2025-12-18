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
train_int_tag = False

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc = "US"

nocal_tag = ''
n_components = 2
reservoir_size = 200

samples = 100
#########
file_tag = f'r{reservoir_size}_s1_sr0.4_rr0.001'


ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    train_basin_int_list = [30,20,15,10,5,3,2]
elif loc == "US" and ver == 1:
    file_tot_num = 671
    start_val_list = [336]
    #step_val_list = list(range(2,337))
    step_val_list = [2,5,10,21,42,84,168,337]
elif loc == "US" and ver == 2:
    file_tot_num = 667
    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    train_basin_int_list = [300,150,100,50,25,10,5,3]

cal_eva = ['test'] #'train', 

benchmark_list = ["KGE","NSE","logNSE","E1", "VE", "d","RMSE","MAE"]

fs = 17 # 6 for paper

def create_box_plot(ax, plot_df, benchmark, train_test):
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    if 'norm' in calc_mode:
        ax.set_title(f'Boxplot for {benchmark} {calc_mode} {train_test}\n', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Boxplot for {benchmark} {calc_mode} {train_test}\n', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

def create_line_plot(ax, plot_df, benchmark, train_test):
    positions = range(1, len(plot_df.columns) + 1) 
    lw = 0.4
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='gray')
    
    if 'norm' in calc_mode:
        ax.set_title(f'Line plot for {benchmark} {calc_mode} {train_test}\n', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Line plot for {benchmark} {calc_mode} {train_test}\n', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)

def create_point_plot(ax, plot_df, benchmark, train_test):
    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')

    if 'norm' in calc_mode:
        ax.set_title(f'Point plot for {benchmark} {calc_mode} {train_test}\n')
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif 'dif' in calc_mode:
        ax.set_title(f'Point plot for {benchmark} {calc_mode} {train_test}\n')
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)

def load_BC_PUB_data(train_basin_int, benchmark):
    BC_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_random_{reservoir_size}/Train{train_basin_int}/test_basin/results/BC_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
        BC_PUB_df = pd.read_csv(BC_PUB_data, index_col=0)
        BC_PUB_df = BC_PUB_df.sort_values(by='file_num')
        BC_PUB_df = pd.DataFrame(BC_PUB_df.values)

        if BC_PUB_sum_data is None:
            BC_PUB_sum_data = BC_PUB_df.copy()
        else:
            BC_PUB_sum_data = BC_PUB_sum_data.add(BC_PUB_df, fill_value=0)
    
    BC_PUB_mean_data = BC_PUB_sum_data.copy()
    BC_PUB_mean_data = BC_PUB_mean_data.divide(samples)

    BC_PUB_mean_data.columns = benchmark_list

    BC_PUB_mean_data.index.name = 'file_num'

    BC_PUB_mean_data = BC_PUB_mean_data[[benchmark]]
    BC_PUB_mean_data.index = test_basins_list  # Ensure index is set

    return BC_PUB_mean_data

def load_BMA_PUB_data(train_basin_int, benchmark):
    BMA_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB_random/Train{train_basin_int}/test_basin/results/BMA_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
        BMA_PUB_df = pd.read_csv(BMA_PUB_data, index_col=0)
        BMA_PUB_df = BMA_PUB_df.sort_values(by='file_num')
        BMA_PUB_df = pd.DataFrame(BMA_PUB_df.values)

        if BMA_PUB_sum_data is None:
            BMA_PUB_sum_data = BMA_PUB_df.copy()
        else:
            BMA_PUB_sum_data = BMA_PUB_sum_data.add(BMA_PUB_df, fill_value=0)
    
    BMA_PUB_mean_data = BMA_PUB_sum_data.copy()
    BMA_PUB_mean_data = BMA_PUB_mean_data.divide(samples)

    BMA_PUB_mean_data.columns = benchmark_list

    BMA_PUB_mean_data.index.name = 'file_num'

    BMA_PUB_mean_data = BMA_PUB_mean_data[[benchmark]]
    BMA_PUB_mean_data.index = test_basins_list  # Ensure index is set

    return BMA_PUB_mean_data
    
def load_RCBC_BMA_PUB_data(train_basin_int, benchmark):
    RCBC_BMA_PUB_sum_data = None
    for sample in range(1, samples + 1):
        RCBC_BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_PUB_random_{reservoir_size}/Train{train_basin_int}/test_basin/results/RCBC-BMA_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
        RCBC_BMA_PUB_df = pd.read_csv(RCBC_BMA_PUB_data, index_col=0)
        RCBC_BMA_PUB_df = RCBC_BMA_PUB_df.sort_values(by='file_num')
        RCBC_BMA_PUB_df = pd.DataFrame(RCBC_BMA_PUB_df.values)

        if RCBC_BMA_PUB_sum_data is None:
            RCBC_BMA_PUB_sum_data = RCBC_BMA_PUB_df.copy()
        else:
            RCBC_BMA_PUB_sum_data = RCBC_BMA_PUB_sum_data.add(RCBC_BMA_PUB_df, fill_value=0)
    
    RCBC_BMA_PUB_mean_data = RCBC_BMA_PUB_sum_data.copy()
    RCBC_BMA_PUB_mean_data = RCBC_BMA_PUB_mean_data.divide(samples)

    RCBC_BMA_PUB_mean_data.index = test_basins_list
    RCBC_BMA_PUB_mean_data.columns = benchmark_list

    RCBC_BMA_PUB_mean_data.index.name = 'file_num'

    RCBC_BMA_PUB_mean_data = RCBC_BMA_PUB_mean_data[[benchmark]]
    RCBC_BMA_PUB_mean_data.index = test_basins_list  # Ensure index is set

    return RCBC_BMA_PUB_mean_data

def load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark):
    BC_PCA_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BC_PCA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_random_distributed_{reservoir_size}/Train{train_basin_int}/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train{train_basin_int}_sample{sample}_rev_PC{pc}_eva.csv'
        BC_PCA_PUB_df = pd.read_csv(BC_PCA_PUB_data, index_col=0)
        BC_PCA_PUB_df = BC_PCA_PUB_df.sort_values(by='file_num')
        BC_PCA_PUB_df = pd.DataFrame(BC_PCA_PUB_df.values)

        if BC_PCA_PUB_sum_data is None:
            BC_PCA_PUB_sum_data = BC_PCA_PUB_df.copy()
        else:
            BC_PCA_PUB_sum_data = BC_PCA_PUB_sum_data.add(BC_PCA_PUB_df, fill_value=0)
    
    BC_PCA_PUB_mean_data = BC_PCA_PUB_sum_data.copy()
    BC_PCA_PUB_mean_data = BC_PCA_PUB_mean_data.divide(samples)

    BC_PCA_PUB_mean_data.index = test_basins_list
    BC_PCA_PUB_mean_data.columns = benchmark_list

    BC_PCA_PUB_mean_data.index.name = 'file_num'

    BC_PCA_PUB_mean_data = BC_PCA_PUB_mean_data[[benchmark]]
    BC_PCA_PUB_mean_data.index = test_basins_list  # Ensure index is set

    return BC_PCA_PUB_mean_data

def safe_min(sequence, default=float('inf')):
    return np.min(sequence) if len(sequence) > 0 else default

def safe_max(sequence, default=float('-inf')):
    return np.max(sequence) if len(sequence) > 0 else default

def safe_subtract(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    return arr1[:min_len] - arr2[:min_len]

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/Spatial_PUB_random{nocal_tag}/'
os.makedirs(output_dir, exist_ok=True)

calc_mode_to_name = {
    #'BMA_norm': 'BMA_PUB',
    #'BC_norm': 'BC_PUB',
    #'RCBC_norm': 'RCBC-BMA_PUB',
    'PCA_norm': 'BC-PCA-lasso_PUB',
    #'RCBC_BMA_dif': 'dif_RCBC-BMA & BMA',
    'BC_BMA_dif': 'dif_BC & BMA',
    #'RCBC_BC_dif': 'dif_RCBC-BMA & BC',
    'BC_PCA_dif': 'dif_PCA & BC'
}

"""
calc_mode_to_name = {
    'PCA_norm': 'BC-PCA-lasso_PUB',
    'BC_PCA_dif': 'dif_PCA & BC'
}"""

benchmark_limits_norm = {
    'KGE': {'min': -5, 'max': 1},
    'NSE': {'min': -5, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 20}
}
benchmark_limits_dif = {
    'KGE': {'min': -5, 'max': 5},
    'NSE': {'min': -1, 'max':   1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -3, 'max': 3},
    'VE': {'min': -2, 'max': 2},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': -10, 'max': 10},
    'MAE': {'min': -3, 'max': 3}
}

for calc_mode in calc_mode_to_name.keys():
    calc_name = calc_mode_to_name[calc_mode]

    output_dir_calc_mode = os.path.join(output_dir, calc_name)
    os.makedirs(output_dir_calc_mode, exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'box'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'line'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'point'), exist_ok=True)

    buf = f''
    
    for benchmark in benchmark_list:
        # Create a figure with 2 subplots: one for cal and one for eva
        x_size = 8 #26
        y_size = 6
        fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))
        fig_line, ax_line = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))
        fig_point, ax_point = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))    

        global_y_min, global_y_max = float('inf'), float('-inf')

        plot_train_df = pd.DataFrame() 
        plot_test_df = pd.DataFrame() 

        # for median color in PCA
        median_color_train = ['orange'] * 2
        median_color_test = []

        for ce in cal_eva:
            if ce == 'train':
                og_BMA_data = f'/data0/funato/0_out/99_out/{loc}/BMA/results/BMA_results_eva.csv'
                og_BMA_df = pd.read_csv(og_BMA_data)
                og_BMA_column = f'BMA_{benchmark}_eva' 
                og_BMA_data = og_BMA_df[og_BMA_column]

                og_BC_data = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}/results/BC_results_{file_tag}_bma_eva.csv'
                og_BC_df = pd.read_csv(og_BC_data)
                og_BC_column = f'BC_{file_tag}_bma_{benchmark}_eva'
                og_BC_data = og_BC_df[og_BC_column]

                if 'RCBC' in calc_mode:
                    og_RCBC_data = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA/results/RCBC-BMA_results_{file_tag}_eva.csv'
                    og_RCBC_df = pd.read_csv(og_RCBC_data)
                    og_RCBC_column = f'RCBC-BMA_{file_tag}_{benchmark}_eva'
                    og_RCBC_data = og_RCBC_df[og_RCBC_column]

                og_BC_PCA_lasso_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}/results/BC_results_{file_tag}_eva.csv'
                og_BC_PCA_lasso_df = pd.read_csv(og_BC_PCA_lasso_data)
                og_BC_PCA_lasso_column = f'BC-og_{file_tag}_{benchmark}_eva'
                og_BC_PCA_lasso_data = og_BC_PCA_lasso_df[og_BC_PCA_lasso_column]

                if 'norm' in calc_mode:
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BMA' : og_BMA_data,
                        'BC' : og_BC_data
                    })], axis=1)
                elif calc_mode == 'RCBC_BMA_dif':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'RCBC- BMA' : og_RCBC_data - og_BMA_data
                    })], axis=1)
                elif calc_mode == 'BC_BMA_dif':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BC - BMA' : og_BC_data - og_BMA_data
                    })], axis=1)
                elif calc_mode == 'RCBC_BC_dif':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'RCBC - BC' : og_RCBC_data - og_BC_data
                    })], axis=1)
                elif calc_mode == 'BC_PCA':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BC - PCA' : og_BC_data - og_BC_PCA_lasso_data
                    })], axis=1)
                
                elif calc_mode == 'ratio':
                    calc_og_value = []
                    for i in range(len(og_BC_data)):
                        calc_og_value.append(og_BC_data[i] / og_BMA_data[i] * 100.0 if og_BC_data[i] > og_BMA_data[i] else -100.0 * og_BC_data[i] / og_BMA_data[i])
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BC / BMA' : calc_og_value    
                    })], axis=1)

            # for both train and test
            for train_basin_int in train_basin_int_list:
                if train_int_tag:
                    train_int_tag_name = f'Train{train_basin_int}_'
                else:
                    train_int_tag_name = f''

                BMA_PUB_column_train_data = load_BMA_PUB_data(train_basin_int, benchmark)
                BMA_PUB_column_test_data = load_BMA_PUB_data(train_basin_int, benchmark)

                BC_PUB_column_train_data = load_BC_PUB_data(train_basin_int, benchmark)
                BC_PUB_column_test_data = load_BC_PUB_data(train_basin_int, benchmark)
                
                if calc_mode == 'RCBC_norm' or calc_mode == 'RCBC_BMA_dif' or calc_mode == 'RCBC_BC_dif':
                    RCBC_BMA_PUB_column_train_data = load_RCBC_BMA_PUB_data(train_basin_int, benchmark)
                    RCBC_BMA_PUB_column_test_data = load_RCBC_BMA_PUB_data(train_basin_int, benchmark)
                if calc_mode == 'RCBC_norm':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': RCBC_BMA_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(RCBC_BMA_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(RCBC_BMA_PUB_column_test_data))

                elif calc_mode == 'BMA_norm':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': BMA_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(BMA_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(BMA_PUB_column_test_data))

                elif calc_mode == 'BC_norm':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': BC_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(BC_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(BC_PUB_column_test_data))

                elif calc_mode == 'PCA_norm':
                    for pc in range(1, n_components + 1):
                        median_color_test.append('red' if pc == 1 else 'blue' if pc == 2 else 'green')
                        BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            f'{train_int_tag_name}{train_basin_int}_PC{pc}': BC_PCA_lasso_PUB_column_test_data.values.flatten()
                        }, index=test_basins_list)], axis=1)
                        global_y_min = min(global_y_min, safe_min(BC_PCA_lasso_PUB_column_test_data))
                        global_y_max = max(global_y_max, safe_max(BC_PCA_lasso_PUB_column_test_data))

                elif calc_mode == 'RCBC_BMA_dif':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': safe_subtract(RCBC_BMA_PUB_column_test_data.values.flatten(), BMA_PUB_column_test_data.values.flatten())
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_test_data, BMA_PUB_column_test_data)))
                    global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_test_data, BMA_PUB_column_test_data))) 

                elif calc_mode == 'RCBC_BC_dif':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': safe_subtract(RCBC_BMA_PUB_column_test_data.values.flatten(), BC_PUB_column_test_data.values.flatten())
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_test_data, BC_PUB_column_test_data)))
                    global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_test_data, BC_PUB_column_test_data)))

                elif calc_mode == 'BC_BMA_dif':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': safe_subtract(BC_PUB_column_test_data.values.flatten(), BMA_PUB_column_test_data.values.flatten())
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))
                    global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))

                elif calc_mode == 'BC_PCA_dif':
                    for pc in range(1, n_components + 1):
                        BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            f'{train_int_tag_name}{train_basin_int}_PC{pc}': safe_subtract(BC_PUB_column_test_data.values.flatten(), BC_PCA_lasso_PUB_column_test_data.values.flatten())
                        }, index=test_basins_list)], axis=1)
                        global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))
                        global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))

        #create_box_plot(axes_box[0], plot_train_df, benchmark, 'train')
        #create_line_plot(axes_line[0], plot_train_df, benchmark, 'train')
        #create_point_plot(axes_point[0], plot_train_df, benchmark, 'train')

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
            for i, pc in enumerate(range(1, n_components + 1)):
                ax_point.get_lines()[i].set_color(median_color_train[i])
                ax_point.get_lines()[i].set_label(f'PC{pc}')
            ax_point.legend()
            

        fig_box.tight_layout()
        fig_line.tight_layout()
        fig_point.tight_layout()


        box_output_path = os.path.join(output_dir_calc_mode, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
        line_output_path = os.path.join(output_dir_calc_mode, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
        point_output_path = os.path.join(output_dir_calc_mode, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

        fig_box.savefig(box_output_path)
        fig_line.savefig(line_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        plt.close(fig_line)
        plt.close(fig_point)

        print(f'{benchmark} {nocal_tag} {calc_mode} DONE!')
    print(f'saved to {output_dir_calc_mode}')

print("DONE!")
