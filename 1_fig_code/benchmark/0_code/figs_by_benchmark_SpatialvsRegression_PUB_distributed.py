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
from matplotlib.lines import Line2D
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#########
train_int_tag = False

non_arid_files = True
BmaProx_cdf = False
LSTM_cdf = True
gauged_BC = True

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc, ver = "US", 2
#loc, ver = "AUS", 2
#loc, ver = "GB", 2
if loc != "US":
    non_arid_files = False

nocal_tag = ''
n_components = 3
spin = 1
spectral_radius = 0.4

samples = 100

fs = 27 # 6 for paper

# size for norm and dif
x_size = 10 #26
y_size = 6
x_size_cdf =16
y_size_cdf = 12
#########


ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    train_basin_int_list = [30,20,15,10,5,3]
    ridge_param = 1.0
elif loc == "US" and ver == 2:
    file_tot_num = 667
    reservoir_size = 200

    if non_arid_files:
        arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
        arid_file_list = pd.read_csv(arid_file_list)
        arid_file_list = arid_file_list['File_num'].tolist()
        arid_file_list.sort()
        non_arid_file_list = [i for i in range(1,file_tot_num+1) if i not in arid_file_list]
        test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
        test_basins_list = [basin for basin in test_basins_list if basin in non_arid_file_list]
    else:
        test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    if non_arid_files:
        train_basin_int_list = [200,150,100,75,50,25,10] #
    else:
        train_basin_int_list = [300,200,150,100,50,25,10,5,3]
    ridge_param = 1.0
elif loc == "AUS" and ver == 2:
    file_tot_num = 84
    reservoir_size = 200

    test_basins_list = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82] # 17 basins
    train_basin_int_list = [30,20,15,10,5,3,2]
    ridge_param = 1.0
elif loc == "GB" and ver == 2:
    file_tot_num = 396
    reservoir_size = 300

    test_basins_list = list(range(4, file_tot_num, 10))
    train_basin_int_list = [170,130,90,60,40,20,10]
    ridge_param = 1.0
else:
    print("Invalid loc and ver")
    exit()


file_tag = f'r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}'

if non_arid_files:
    non_arid_buf = '_non_arid'
else:
    non_arid_buf = ''

cal_eva = ['test'] #'train', 

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

def create_box_plot(ax, plot_df, benchmark, train_test):
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-'), 
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
    lw = 0.7
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

def create_cdf_plot(ax, plot_df, benchmark, train_test):
    markers_list = ['D', 'o', 's', '^', 'v', 'x', '*']
    markers = markers_list[:len(train_basin_int_list)]
    
    # Define base colors and gradients
    base_colors = {
        'BMA': '#3CB371',
        'BC': 'royalblue',
        'PCA1': 'crimson',
        'PCA2': 'blue',
        'PCA3': 'green',
        'LSTM': 'orange',
        'Gauged BC': 'Black'   # Dark gray for Gauged BC
    }
    color_gradients = {
        'BMA': plt.cm.Greens,
        'BC': plt.cm.Purples,
        'PCA1': plt.cm.Reds,
        'PCA2': plt.cm.Blues,
        'PCA3': plt.cm.Wistia,
        'LSTM': plt.cm.Greens,
    }
    
    reset_interval = len(train_basin_int_list)
    legend_entries = {'BcProx': [], 'Gauged BC': [], 'BcReg': [], 'LSTM':[]}  # Separate lists for BcProx, Gauged BC, and BcReg

    # Loop through columns in plot_df
    for i, column in enumerate(plot_df.columns):
        sorted_data = np.sort(plot_df[column])  # Ensure data is sorted for CDF
        cdf = np.linspace(0, 1, len(sorted_data))  # Ensure proper CDF scaling
        
        # Identify color key
        if 'Reg' in column and 'PC1' in column:
            color_key = 'PCA1'
        elif 'Reg' in column and 'PC2' in column:
            color_key = 'PCA2'
        elif 'Reg' in column and 'PC3' in column:
            color_key = 'PCA3'
        elif 'Reg' in column:
            color_key = 'PCA1'
        elif 'Bma' in column:
            color_key = 'BMA'
        elif 'Gauged BC' in column:
            color_key = 'Gauged BC'
        elif 'LSTM' in column:
            color_key = 'LSTM'
        else:
            color_key = 'BC'

        # Assign color and marker
        if color_key in color_gradients:
            color = color_gradients[color_key](0.2 + 0.8 * (1 - (i % reset_interval) / reset_interval))
        else:
            color = base_colors[color_key]
        marker = '' if column == 'Gauged BC' else markers[i % len(markers)]

        if column == 'Gauged BC':  
            lw_cdf = 3.5
            linestyle = '-'
        else:
            lw_cdf = 1.5
            linestyle = '-'

        # Plot CDF
        line = ax.plot(sorted_data, cdf, label=column, marker=marker, color=color,
                       markersize=9, linewidth=lw_cdf, linestyle=linestyle)[0]

        # Group legend entries into BcProx, Gauged BC, and BcReg categories
        if 'Prox' in column:
            legend_entries['BcProx'].append((line, column.split()[1]))
        elif 'Gauged BC' in column:
            legend_entries['Gauged BC'].append((line, 'Gauged BC'))
        elif 'Reg' in column:
            legend_entries['BcReg'].append((line, column.split()[1]))
        elif 'LSTM' in column:
            legend_entries['LSTM'].append((line, column.split()[1]))

    
    ax.set_title(f'CDF plot for {benchmark} {train_test}\n', fontsize=fs)
    ax.set_xlabel(f'{benchmark}', fontsize=fs)
    ax.set_ylabel('CDF', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.grid(True)

    # Create legend for BcProx, then Gauged BC, then BcReg with their respective titles
    handles = []
    labels = []

    # Add BcProx group title
    handles.append(Line2D([0], [0], color='none', label='BcProx', linestyle=''))
    labels.append('BcProx')
    # Add lines for BcProx
    for line, label in legend_entries['BcProx']:
        handles.append(line)
        labels.append(label)
        
    # Add Gauged BC
    for line, label in legend_entries['Gauged BC']:
        handles.append(line)
        labels.append(label)
        
    # Add BcReg group title
    handles.append(Line2D([0], [0], color='none', label='BcReg', linestyle=''))
    labels.append('BcReg')
    # Add lines for BcReg
    for line, label in legend_entries['BcReg']:
        handles.append(line)
        labels.append(label)

    # Add LSTM group title
    handles.append(Line2D([0], [0], color='none', label='LSTM', linestyle=''))
    labels.append('LSTM')
    # Add lines for LSTM
    for line, label in legend_entries['LSTM']:
        handles.append(line)
        labels.append(label)

    # Add legend with the specified order and group titles
    ax.legend(handles, labels, ncol=3, fontsize=fs * 0.8, loc='upper left')


def load_BC_PUB_data(train_basin_int, benchmark):
    BC_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_random_distributed_{reservoir_size}_{ridge_param}{non_arid_buf}/Train{train_basin_int}/test_basin/results/BC_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
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
        BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB_random_distributed{non_arid_buf}/Train{train_basin_int}/test_basin/results/BMA_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
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
    
def load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark):
    BC_PCA_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BC_PCA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_random_distributed_{reservoir_size}_{ridge_param}{non_arid_buf}/Train{train_basin_int}/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train{train_basin_int}_sample{sample}_rev_PC{pc}_eva.csv'
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

def load_BC_data(benchmark, ce):
    BC_data = f'/data0/funato/0_out/99_out/{loc}/BC_700_0.001/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv'
    BC_df = pd.read_csv(BC_data)
    BC_column = f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva'
    BC_data = BC_df[BC_column]

    return BC_data

def load_LSTM_PUB_data(train_basin_int, benchmark):
    ## use /data0/funato/99_work/92_LSTM_PUB/benchmark_random.py for calculating benchmark mean
    LSTM_PUB_data = f'/data0/funato/0_out/99_out/{loc}/LSTM_PUB_random/ensemble/Train{train_basin_int}/test_basin/results/LSTM_PUB_results_eva.csv'
    LSTM_PUB_df = pd.read_csv(LSTM_PUB_data, index_col=0)
    # Return the actual column of data instead of a string
    LSTM_PUB_df_column = LSTM_PUB_df[f'LSTM_PUB_{benchmark}_eva']
    return LSTM_PUB_df_column

def safe_min(sequence, default=float('inf')):
    return np.min(sequence) if len(sequence) > 0 else default

def safe_max(sequence, default=float('-inf')):
    return np.max(sequence) if len(sequence) > 0 else default

def safe_subtract(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    return arr1[:min_len] - arr2[:min_len]

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/Spatial_PUB_random_distributed_{reservoir_size}_{ridge_param}{non_arid_buf}{nocal_tag}/'
os.makedirs(output_dir, exist_ok=True)
if BmaProx_cdf:
    os.makedirs(os.path.join(output_dir, 'cdf'), exist_ok=True)
else:
    if gauged_BC:
        os.makedirs(os.path.join(output_dir, 'cdf_woBmaProx_BCgauged'), exist_ok=True)
        if LSTM_cdf:
            os.makedirs(os.path.join(output_dir, 'cdf_woBmaProx_BCgauged_LSTM'), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_dir, 'cdf_woBmaProx'), exist_ok=True)


calc_mode_to_name = {
    #'BMA_norm': 'BmaProx',
    'BC_norm': 'BcProx',
    'PCA_norm': 'BcReg',
    'LSTM_norm': 'LSTM',
    #'BC_BMA_dif': 'dif_BcProx & BmaProx',
    #'PCA_BC_dif': 'dif_BcReg & BcProx'
}


benchmark_limits_norm = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -3, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}
benchmark_limits_dif = {
    'KGE': {'min': -5, 'max': 5},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -3, 'max': 3},
    'VE': {'min': -2, 'max': 2},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': -10, 'max': 10},
    'MAE': {'min': -3, 'max': 3}
}
benchmark_limits_cdf = {
    'KGE': {'min': -0.5, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}


for benchmark in benchmark_list:
    plot_test_df_cdf = pd.DataFrame()

    for calc_mode in calc_mode_to_name.keys():
        calc_name = calc_mode_to_name[calc_mode]

        output_dir_calc_mode = os.path.join(output_dir, calc_name)
        os.makedirs(output_dir_calc_mode, exist_ok=True)
        os.makedirs(os.path.join(output_dir_calc_mode, 'box'), exist_ok=True)
        os.makedirs(os.path.join(output_dir_calc_mode, 'line'), exist_ok=True)
        os.makedirs(os.path.join(output_dir_calc_mode, 'point'), exist_ok=True)

        buf = f''
        
        # Create a figure with 2 subplots: one for cal and one for eva
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

                og_BC_data = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}/results/BC_results_{file_tag}_bma_eva.csv'
                og_BC_df = pd.read_csv(og_BC_data)
                og_BC_column = f'BC_{file_tag}_bma_{benchmark}_eva'
                og_BC_data = og_BC_df[og_BC_column]

                og_BC_PCA_lasso_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}_{ridge_param}/results/BC_results_{file_tag}_eva.csv'
                og_BC_PCA_lasso_df = pd.read_csv(og_BC_PCA_lasso_data)
                og_BC_PCA_lasso_column = f'BC-og_{file_tag}_{benchmark}_eva'
                og_BC_PCA_lasso_data = og_BC_PCA_lasso_df[og_BC_PCA_lasso_column]

                if 'norm' in calc_mode:
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BMA' : og_BMA_data,
                        'BC' : og_BC_data
                    })], axis=1)
                elif calc_mode == 'BC_BMA_dif':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'BC - BMA' : og_BC_data - og_BMA_data
                    })], axis=1)
                elif calc_mode == 'PCA_BC_dif':
                    plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                        'PCA - BC' : og_BC_PCA_lasso_data - og_BC_data
                    })], axis=1)

            # for both train and test
            for train_basin_int in train_basin_int_list:
                if train_int_tag:
                    train_int_tag_name = f'Train{train_basin_int}_'
                else:
                    train_int_tag_name = f''

                if 'BMA' in calc_mode:
                    BMA_PUB_column_train_data = load_BMA_PUB_data(train_basin_int, benchmark)
                    BMA_PUB_column_test_data = load_BMA_PUB_data(train_basin_int, benchmark)

                if 'BC' in calc_mode:
                    BC_PUB_column_train_data = load_BC_PUB_data(train_basin_int, benchmark)
                    BC_PUB_column_test_data = load_BC_PUB_data(train_basin_int, benchmark)

                if calc_mode == 'BMA_norm':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': BMA_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(BMA_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(BMA_PUB_column_test_data))
                    if BmaProx_cdf:
                        plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                            f'{calc_name} {train_int_tag_name}{train_basin_int}': BMA_PUB_column_test_data.values.flatten()
                        }, index=test_basins_list)], axis=1)

                elif calc_mode == 'BC_norm':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': BC_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(BC_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(BC_PUB_column_test_data))
                    plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                        f'{calc_name} {train_int_tag_name}{train_basin_int}': BC_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)

                elif calc_mode == 'PCA_norm':
                    for pc in range(1, n_components + 1):
                        median_color_test.append('red' if pc == 1 else 'blue' if pc == 2 else 'green')
                        BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark)
                        if pc == 3:
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{train_int_tag_name}{train_basin_int}': BC_PCA_lasso_PUB_column_test_data.values.flatten()
                            }, index=test_basins_list)], axis=1)
                        else:
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{train_int_tag_name}{train_basin_int} PC1~{pc}': BC_PCA_lasso_PUB_column_test_data.values.flatten()
                            }, index=test_basins_list)], axis=1)                            
                        global_y_min = min(global_y_min, safe_min(BC_PCA_lasso_PUB_column_test_data))
                        global_y_max = max(global_y_max, safe_max(BC_PCA_lasso_PUB_column_test_data))
                        if pc ==3:
                            plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                                f'{calc_name} {train_int_tag_name}{train_basin_int}': BC_PCA_lasso_PUB_column_test_data.values.flatten()
                            }, index=test_basins_list)], axis=1)

                elif calc_mode == 'LSTM_norm':  
                    LSTM_PUB_column_test_data = load_LSTM_PUB_data(train_basin_int, benchmark)
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({  
                        f'{train_int_tag_name}{train_basin_int}': LSTM_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(LSTM_PUB_column_test_data))
                    global_y_max = max(global_y_max, safe_max(LSTM_PUB_column_test_data))
                    plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                        f'{calc_name} {train_int_tag_name}{train_basin_int}': LSTM_PUB_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)

                elif calc_mode == 'BC_BMA_dif':
                    plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                        f'{train_int_tag_name}{train_basin_int}': safe_subtract(BC_PUB_column_test_data.values.flatten(), BMA_PUB_column_test_data.values.flatten())
                    }, index=test_basins_list)], axis=1)
                    global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))
                    global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PUB_column_test_data, BMA_PUB_column_test_data)))

                elif calc_mode == 'PCA_BC_dif':
                    for pc in range(1, n_components + 1):
                        BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            f'{train_int_tag_name}{train_basin_int}_PC{pc}': safe_subtract(BC_PCA_lasso_PUB_column_test_data.values.flatten(), BC_PUB_column_test_data.values.flatten())
                        }, index=test_basins_list)], axis=1)
                        global_y_min = min(global_y_min, safe_min(safe_subtract(BC_PCA_lasso_PUB_column_test_data,BC_PUB_column_test_data)))
                        global_y_max = max(global_y_max, safe_max(safe_subtract(BC_PCA_lasso_PUB_column_test_data, BC_PUB_column_test_data)))

        # Create box, line, and point plots for the test data   
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
        """if calc_mode == 'PCA_norm':
            for i, pc in enumerate(range(1, n_components + 1)):
                if i < len(ax_point.get_lines()):
                    ax_point.get_lines()[i].set_color(median_color_train[i])
                    ax_point.get_lines()[i].set_label(f'PC{pc}')
            ax_point.legend()"""
            
        fig_box.tight_layout()
        fig_line.tight_layout()
        fig_point.tight_layout()

        box_output_path = os.path.join(output_dir_calc_mode, f'box/{benchmark}_boxplot{buf}_cal_eva.png')
        line_output_path = os.path.join(output_dir_calc_mode, f'line/{benchmark}_lineplot{buf}_cal_eva.png')
        point_output_path = os.path.join(output_dir_calc_mode, f'point/{benchmark}_pointplot{buf}_cal_eva.png')

        fig_box.savefig(box_output_path)
        fig_line.savefig(line_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        plt.close(fig_line)
        plt.close(fig_point)

        print(f'{benchmark} {nocal_tag} {calc_mode} DONE!')

    # Sort plot_test_df_cdf to group PCA columns by PC1 to PC3 and sort the numbers before _PC from largest to smallest
    pca_columns = [col for col in plot_test_df_cdf.columns if 'PCA' in col]
    non_pca_columns = [col for col in plot_test_df_cdf.columns if 'PCA' not in col]
    pca_columns_sorted = sorted(pca_columns, key=lambda x: (x.split('_PC')[-1], -int(x.split('_PC')[0].split()[-1])))
    plot_test_df_cdf = plot_test_df_cdf[non_pca_columns + pca_columns_sorted]

    if gauged_BC:
        BC_data = load_BC_data(benchmark, ce)
        plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
            f'Gauged BC': BC_data
        }, index=test_basins_list)], axis=1)

    fig_cdf, ax_cdf = plt.subplots(nrows=1, ncols=1, figsize=(x_size_cdf, y_size_cdf))
    create_cdf_plot(ax_cdf, plot_test_df_cdf, benchmark,'test')
    # Set xlim and ylim for CDF plot
    if benchmark in benchmark_limits_cdf:
        ax_cdf.set_xlim(benchmark_limits_cdf[benchmark]['min'], benchmark_limits_cdf[benchmark]['max'])
    ax_cdf.set_ylim(0, 1)
    fig_cdf.tight_layout()

    if BmaProx_cdf:
        cdf_output_path = os.path.join(output_dir, f'cdf/{benchmark}_cdf{buf}_cal_eva.png')
    else:
        if gauged_BC:
            cdf_output_path = os.path.join(output_dir, f'cdf_woBmaProx_BCgauged/{benchmark}_cdf{buf}_cal_eva.png')
            if LSTM_cdf:
                cdf_output_path = os.path.join(output_dir, f'cdf_woBmaProx_BCgauged_LSTM/{benchmark}_cdf{buf}_cal_eva.png')
        else:
            cdf_output_path = os.path.join(output_dir, f'cdf_woBmaProx/{benchmark}_cdf{buf}_cal_eva.png')
    fig_cdf.savefig(cdf_output_path)
    plt.close(fig_cdf)

    print(f'saved to {output_dir_calc_mode}')

print("DONE!")
