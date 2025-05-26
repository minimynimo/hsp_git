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
file_tag = 'r700_s1_sr0.4_rr0.001'
str_stp_tag = False

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc = "US"

nocal_tag = ''
n_components = 2
reservoir_size = 700
#########

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50]
    Train_basin_num_list = [67,33,17,9,7,5,5,3,3]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    test_basins_list = [77,84,24,34,40,45,70,11,4] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #40
    #step_val_list = list(range(2, start_val_list[0] + 1))
    #step_val_list = [2,4,8,15,20,25,30,40] # IF test basins eight; 39, 20, 10, 5, 4, 3, 2
    step_val_list = [2,4,8,10,15,17,30] # IF test basins seventeen; 35,18,9,7,5,4,3
    Train_basin_num_list = [39,20,10,5,4,3,3,2]
nums = len(step_val_list)

cal_eva = ['train', 'test']

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

def load_BC_PUB_data(start_val, step_val, train_test):
    BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_{reservoir_size}/start{start_val}/{train_test}_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
    BC_PUB_df = pd.read_csv(BC_PUB_data)
    BC_PUB_column_data = np.array(BC_PUB_df[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        BC_PUB_column_data = BC_PUB_column_data[BC_PUB_df['file_num'].isin(test_basins_list)]
    return BC_PUB_column_data

def load_BMA_PUB_data(start_val, step_val, train_test):
    BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB/start{start_val}/{train_test}_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
    BMA_PUB_df = pd.read_csv(BMA_PUB_data)
    BMA_PUB_column_data = np.array(BMA_PUB_df[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        BMA_PUB_column_data = BMA_PUB_column_data[BMA_PUB_df['file_num'].isin(test_basins_list)]
    return BMA_PUB_column_data
    
def load_RCBC_BMA_PUB_data(start_val, step_val, train_test):
    RCBC_BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_PUB_{reservoir_size}/start{start_val}/{train_test}_basin/results/RCBC-BMA_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
    RCBC_BMA_PUB_df = pd.read_csv(RCBC_BMA_PUB_data)
    RCBC_BMA_PUB_column_data =  np.array(RCBC_BMA_PUB_df[f'RCBC-BMA_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        RCBC_BMA_PUB_column_data = RCBC_BMA_PUB_column_data[RCBC_BMA_PUB_df['file_num'].isin(test_basins_list)]
    return RCBC_BMA_PUB_column_data

def load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, train_test):
    BC_PCA_lasso_result_data_dir = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}/start{start_val}/step{step_val}/{train_test}_basin/results/BC-PCA-lasso_PUB_results_{file_tag}_start{start_val}_step{step_val}_rev_PC{pc}_eva.csv')
    BC_PCA_lasso_PUB_column_data = BC_PCA_lasso_result_data_dir[f'BC-PCA-lasso_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva']
    if train_test == 'test':
        BC_PCA_lasso_PUB_column_data = BC_PCA_lasso_PUB_column_data[BC_PCA_lasso_result_data_dir['file_num'].isin(test_basins_list)]
    return BC_PCA_lasso_PUB_column_data

def safe_min(sequence, default=float('inf')):
    return np.min(sequence) if len(sequence) > 0 else default

def safe_max(sequence, default=float('-inf')):
    return np.max(sequence) if len(sequence) > 0 else default

def safe_subtract(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    return arr1[:min_len] - arr2[:min_len]

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/Spatial_PUB{nocal_tag}/start{start_val_list[0]}/n{len(step_val_list)}'
os.makedirs(output_dir, exist_ok=True)

calc_mode_to_name = {
    'BMA_norm': 'BMA_PUB',
    'BC_norm': 'BC_PUB',
    'RCBC_norm': 'RCBC-BMA_PUB',
    'PCA_norm': 'BC-PCA-lasso_PUB',
    'RCBC_BMA_dif': 'dif_RCBC-BMA & BMA',
    'BC_BMA_dif': 'dif_BC & BMA',
    'RCBC_BC_dif': 'dif_RCBC-BMA & BC',
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
        x_size = 15 #26
        y_size = 6
        fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))
        fig_line, axes_line = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))
        fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))    

        global_y_min, global_y_max = float('inf'), float('-inf')


        plot_train_df = pd.DataFrame() 
        plot_test_df = pd.DataFrame() 
        for start_val in start_val_list:
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

                    og_RCBC_data = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_{reservoir_size}/results/RCBC-BMA_results_{file_tag}_eva.csv'
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

                for step_val, train_basin_int in zip(step_val_list, Train_basin_num_list):
                    if str_stp_tag:
                        step_val_tag_name = f'start{start_val}_step{step_val}_'
                    else:
                        step_val_tag_name = f''

                    BMA_PUB_column_train_data = load_BMA_PUB_data(start_val, step_val, 'train')
                    BMA_PUB_column_test_data = load_BMA_PUB_data(start_val, step_val, 'test')

                    BC_PUB_column_train_data = load_BC_PUB_data(start_val, step_val, 'train')
                    BC_PUB_column_test_data = load_BC_PUB_data(start_val, step_val, 'test')
                    
                    if calc_mode == 'RCBC_norm' or calc_mode == 'RCBC_BMA_dif' or calc_mode == 'RCBC_BC_dif':
                        RCBC_BMA_PUB_column_train_data = load_RCBC_BMA_PUB_data(start_val, step_val, 'train')
                        RCBC_BMA_PUB_column_test_data = load_RCBC_BMA_PUB_data(start_val, step_val, 'test')

                    if calc_mode == 'RCBC_norm':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : RCBC_BMA_PUB_column_train_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(RCBC_BMA_PUB_column_train_data))
                            global_y_max = max(global_y_max, safe_max(RCBC_BMA_PUB_column_train_data))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : RCBC_BMA_PUB_column_test_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(RCBC_BMA_PUB_column_test_data))
                            global_y_max = max(global_y_max, safe_max(RCBC_BMA_PUB_column_test_data))

                    elif calc_mode == 'BMA_norm':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : BMA_PUB_column_train_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(BMA_PUB_column_train_data))
                            global_y_max = max(global_y_max, safe_max(BMA_PUB_column_train_data))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : BMA_PUB_column_test_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(BMA_PUB_column_test_data))
                            global_y_max = max(global_y_max, safe_max(BMA_PUB_column_test_data))

                    elif calc_mode == 'BC_norm':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : BC_PUB_column_train_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(BC_PUB_column_train_data))
                            global_y_max = max(global_y_max, safe_max(BC_PUB_column_train_data))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : BC_PUB_column_test_data
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(BC_PUB_column_test_data))
                            global_y_max = max(global_y_max, safe_max(BC_PUB_column_test_data))

                    elif calc_mode == 'PCA_norm':
                        if ce == 'train':
                            for pc in range(1, n_components + 1):
                                median_color_train.append('red' if pc == 1 else 'blue' if pc == 2 else 'green')
                                BC_PCA_lasso_PUB_column_train_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, 'train')
                                plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                    f'{step_val_tag_name}{train_basin_int}_PC{pc}' : BC_PCA_lasso_PUB_column_train_data
                                })], axis=1)
                                global_y_min = min(global_y_min, safe_min(BC_PCA_lasso_PUB_column_train_data))
                                global_y_max = max(global_y_max, safe_max(BC_PCA_lasso_PUB_column_train_data))
                        elif ce == 'test':
                            for pc in range(1, n_components + 1):
                                median_color_test.append('red' if pc == 1 else 'blue' if pc == 2 else 'green')
                                BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, 'test')
                                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                    f'{step_val_tag_name}{train_basin_int}_PC{pc}' : BC_PCA_lasso_PUB_column_test_data
                                })], axis=1)
                                global_y_min = min(global_y_min, safe_min(BC_PCA_lasso_PUB_column_test_data))
                                global_y_max = max(global_y_max, safe_max(BC_PCA_lasso_PUB_column_test_data))

                    elif calc_mode == 'RCBC_BMA_dif':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(RCBC_BMA_PUB_column_train_data, BMA_PUB_column_train_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_train_data, BMA_PUB_column_train_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_train_data, BMA_PUB_column_train_data)))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(RCBC_BMA_PUB_column_test_data, BMA_PUB_column_test_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_test_data, BMA_PUB_column_test_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_test_data, BMA_PUB_column_test_data))) 
 
                    elif calc_mode == 'RCBC_BC_dif':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(RCBC_BMA_PUB_column_train_data, BC_PUB_column_train_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_train_data, BC_PUB_column_train_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_train_data, BC_PUB_column_train_data)))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(RCBC_BMA_PUB_column_test_data, BC_PUB_column_test_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(RCBC_BMA_PUB_column_test_data, BC_PUB_column_test_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(RCBC_BMA_PUB_column_test_data, BC_PUB_column_test_data)))

                    elif calc_mode == 'BC_BMA_dif':
                        if ce == 'train':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(BC_PUB_column_train_data, BMA_PUB_column_train_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_train_data, BMA_PUB_column_train_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_train_data, BMA_PUB_column_train_data)))
                        elif ce == 'test':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{train_basin_int}' : safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)
                            })], axis=1)
                            global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))
                            global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))

                    elif calc_mode == 'BC_PCA_dif':
                        if ce == 'train':
                            for pc in range(1, n_components + 1):
                                BC_PCA_lasso_PUB_column_train_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, 'train')
                                plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                    f'{step_val_tag_name}{train_basin_int}_PC{pc}' : safe_subtract(BC_PUB_column_train_data, BC_PCA_lasso_PUB_column_train_data)
                                })], axis=1)
                                global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_train_data, BC_PCA_lasso_PUB_column_train_data)))
                                global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_train_data, BC_PCA_lasso_PUB_column_train_data)))
                        elif ce == 'test':
                            for pc in range(1, n_components + 1):
                                BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, 'test')
                                plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                    f'{step_val_tag_name}{train_basin_int}_PC{pc}' : safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)
                                })], axis=1)
                                global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))
                                global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BC_PCA_lasso_PUB_column_test_data)))

            create_box_plot(axes_box[0], plot_train_df, benchmark, 'train')
            create_line_plot(axes_line[0], plot_train_df, benchmark, 'train')
            create_point_plot(axes_point[0], plot_train_df, benchmark, 'train')

            create_box_plot(axes_box[1], plot_test_df, benchmark, 'test')
            create_line_plot(axes_line[1], plot_test_df, benchmark, 'test')
            create_point_plot(axes_point[1], plot_test_df, benchmark, 'test')

            # Set the same y-axis limits for subplots
            all_axes = [axes_box, axes_line, axes_point]

            for ax_group in [axes_box, axes_line, axes_point]:
                for ax in ax_group:
                    if benchmark in benchmark_limits_norm and 'norm' in calc_mode:
                        ax.set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
                    elif benchmark in benchmark_limits_dif and 'dif' in calc_mode:
                        ax.set_ylim(benchmark_limits_dif[benchmark]['min'], benchmark_limits_dif[benchmark]['max'])

            for ax_group in all_axes:
                for ax in ax_group:
                    for label in ax.get_xticklabels():
                        label.set_rotation(90)

            # median color for PCA
            if calc_mode == 'PCA_norm':
                for ax in axes_point:
                    for i, pc in enumerate(range(1, n_components + 1)):
                        ax.get_lines()[i].set_color(median_color_train[i])
                        ax.get_lines()[i].set_label(f'PC{pc}')
                    ax.legend()
                

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
