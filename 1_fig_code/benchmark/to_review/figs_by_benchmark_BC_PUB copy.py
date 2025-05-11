# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use BC_PUB to compare the performance of the trained and tested basins for different start and step values
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import warnings

warnings.filterwarnings('ignore')

#########
start_val_list = [68]
#step_val_list = list(range(2, 69))
#step_val_list = [2,10,20,30,40,50,60,68]
step_val_list = [2,4,10,20,30,40,50] # Train: 67, 27, 13, 7, 5, 3, 3
Train_basin_num_list = [67, 33, 13, 7, 5, 3, 3]
#step_val_list = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,68]
nums = len(step_val_list)
file_tag = 'r700_s1_sr0.4_rr0.001'
RCnocal = True
BMAnocal = True
str_stp_tag = False
#########
end_val = 136

#cal_eva = ['train', 'test']
cal_eva = ['test']

if RCnocal:
    nocal_tag = '_RCnocal'
else:
    nocal_tag = ''
if BMAnocal:    
    nocal_tag += '_BMAnocal'


fs = 17 # 6 for paper

def create_box_plot(ax, plot_df, benchmark, train_test):
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    if calc_mode == 'norm':
        ax.set_title(f'Boxplot for {benchmark} {nocal_tag} {train_test}\n', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif calc_mode == 'dif':
        ax.set_title(f'Boxplot for {benchmark} {nocal_tag} {train_test}\n(BC - BMA)\n', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark} of BC  - {benchmark} of BMA', fontsize = fs)
    else: # calc_mode == 'ratio'
        ax.set_title(f'Boxplot for {benchmark} {nocal_tag} {train_test}\n(BC / BMA [%])\n', fontsize = fs)
        ax.axhline(y=1, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark} of BC  / {benchmark} of BMA [%]', fontsize = fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.set_xlabel('# Trained Basins', fontsize=fs)

def create_line_plot(ax, plot_df, benchmark, train_test):
    positions = range(1, len(plot_df.columns) + 1) 
    lw = 0.4
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='gray')
    
    if calc_mode == 'norm':
        ax.set_title(f'Line plot for {benchmark} {nocal_tag} {train_test}\n', fontsize = fs)
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif calc_mode == 'dif':
        ax.set_title(f'Line plot for {benchmark} {nocal_tag} {train_test}\n(BC - BMA)\n', fontsize = fs)
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark} of BC  - {benchmark} of BMA', fontsize = fs)
    else: # calc_mode == 'ratio'
        ax.set_title(f'Line plot for {benchmark} {nocal_tag} {train_test}\n(BC / BMA [%])\n', fontsize = fs)
        ax.axhline(y=1, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark} of BC / {benchmark} of BMA [%]', fontsize = fs)
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)
    ax.set_xlabel('# Trained Basins', fontsize = fs)  # Add x-axis title


def create_point_plot(ax, plot_df, benchmark, train_test):
    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')

    if calc_mode == 'norm':
        ax.set_title(f'Point plot for {benchmark} {nocal_tag} {train_test}\n')
        ax.set_ylabel(f'{benchmark}', fontsize = fs)
    elif calc_mode == 'dif':
        ax.set_title(f'Point plot for {benchmark} {nocal_tag} {train_test}\n(BC - BMA)\n')
        ax.axhline(y=0, color='dimgray', linewidth=1.0)
        ax.set_ylabel(f'{benchmark} of BC - {benchmark} of BMA', fontsize = fs)
    else: # calc_mode == 'ratio'
        ax.set_title(f'Point plot for {benchmark} {nocal_tag} {train_test}\n(BC / BMA)\n')
        ax.set_ylabel(f'{benchmark} of BC  / {benchmark} of BMA [%]')
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.xaxis.set_tick_params(labelsize = fs)
    ax.set_xlabel('# Trained Basins', fontsize = fs)  # Add x-axis title


output_dir = f'/data0/funato/0_out/0_fig/benchmark/BC_PUB{nocal_tag}/start{start_val_list[0]}/n{len(step_val_list)}'
os.makedirs(output_dir, exist_ok=True)

for calc_mode in ['dif', 'norm', 'ratio']:
    output_dir_calc_mode = os.path.join(output_dir, calc_mode)
    os.makedirs(output_dir_calc_mode, exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'box'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'line'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_calc_mode, 'point'), exist_ok=True)

    buf = f'_BC_PUB{nocal_tag}_{nums}'
    
    benchmark_list = ["KGE","NSE","logNSE","E1", "VE", "d","RMSE","MAE"]
            
    for benchmark in benchmark_list:
        # Create a figure with 2 subplots: one for cal and one for eva
        x_size = 14#26
        y_size = 7
        fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))
        fig_line, axes_line = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))
        fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(x_size, y_size))    

        global_y_min, global_y_max = float('inf'), float('-inf')


        plot_train_df = pd.DataFrame() 
        plot_test_df = pd.DataFrame() 
        for start_val in start_val_list:
            for ce in cal_eva:
                if ce == 'train' or ce == 'test':
                    og_BMA_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BMA_nocal/results/BMA_results_eva.csv'
                    og_BMA_df = pd.read_csv(og_BMA_data)
                    og_BMA_column = f'BMA_{benchmark}_eva' 
                    og_BMA_data = og_BMA_df[og_BMA_column]

                    og_BC_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BC_700_nocal/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv'
                    og_BC_df = pd.read_csv(og_BC_data)
                    og_BC_column = f'BC_{file_tag}_bma_{benchmark}_eva'
                    og_BC_data = og_BC_df[og_BC_column]

                    if calc_mode == 'norm':
                        plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                            #'OG_BMA' : og_BMA_data,
                            'OG_BC' : og_BC_data
                        })], axis=1)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            #'OG_BMA' : og_BMA_data,
                            'OG_BC' : og_BC_data
                        })], axis=1)
                    elif calc_mode == 'dif':
                        plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                            'OG_BC' : og_BC_data - og_BMA_data
                        })], axis=1)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            'OG_BC' : og_BC_data - og_BMA_data
                        })], axis=1)
                    else: # calc_mode == 'ratio'
                        calc_og_value = []
                        for i in range(len(og_BC_data)):
                            calc_og_value.append(og_BC_data[i] / og_BMA_data[i] * 100.0 if og_BC_data[i] > og_BMA_data[i] else -100.0 * og_BC_data[i] / og_BMA_data[i])
                        plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                            'OG_BC / OG_BMA' : calc_og_value    
                        })], axis=1)
                        plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                            'OG_BC / OG_BMA' : calc_og_value    
                        })], axis=1)

                for step_val in step_val_list:
                    if str_stp_tag:
                        step_val_tag_name = f'start{start_val}_step{step_val}_'
                    else:
                        step_val_tag_name = f''
                    BMA_PUB_train_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BMA_PUB_nocal/start{start_val}/train_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
                    BMA_PUB_train_df = pd.read_csv(BMA_PUB_train_data)
                    BMA_PUB_column_train_data =  np.array(BMA_PUB_train_df[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                    BMA_PUB_data_train_len = len(BMA_PUB_column_train_data)

                    BMA_PUB_test_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BMA_PUB_nocal/start{start_val}/test_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
                    BMA_PUB_test_df = pd.read_csv(BMA_PUB_test_data)
                    BMA_PUB_column_test_data =  np.array(BMA_PUB_test_df[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                    BMA_PUB_data_test_len = len(BMA_PUB_column_test_data)
                    
                    BC_PUB_train_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BC_PUB_RCnocal_BMAnocal/start68/test_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
                    BC_PUB_train_df = pd.read_csv(BC_PUB_train_data)
                    BC_PUB_column_train_data = np.array(BC_PUB_train_df[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                    BC_PUB_data_train_len = len(BC_PUB_column_train_data)
                    
                    BC_PUB_test_data = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BC_PUB_RCnocal_BMAnocal/start{start_val}/test_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
                    BC_PUB_test_df = pd.read_csv(BC_PUB_test_data)
                    BC_PUB_column_test_data =  np.array(BC_PUB_test_df[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                    BC_PUB_data_test_len = len(BC_PUB_column_test_data)
                    
                    if ce  == 'train':
                        if calc_mode == 'norm':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : BC_PUB_column_train_data
                            })], axis=1)
                            global_y_min = min(global_y_min, min(BC_PUB_column_train_data))
                            global_y_max = max(global_y_max, max(BC_PUB_column_train_data))
                        elif calc_mode == 'dif':
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : BC_PUB_column_train_data - BMA_PUB_column_train_data
                            })], axis=1)
                            global_y_min = min(global_y_min, min(BC_PUB_column_train_data-BMA_PUB_column_train_data))
                            global_y_max = max(global_y_max, max(BC_PUB_column_train_data-BMA_PUB_column_train_data))
                        else: # calc_mode == 'ratio'
                            BMA_PUB_data_train_len = len(BMA_PUB_column_train_data)
                            calc_train_value = []
                            for i in range(len(BC_PUB_column_train_data)):
                                calc_train_value.append(BC_PUB_column_train_data[i] / BMA_PUB_column_train_data[i] * 100.0 if BC_PUB_column_train_data[i] > BMA_PUB_column_train_data[i] else -100.0 * BMA_PUB_column_train_data[i] / BC_PUB_column_train_data[i]) 
                            plot_train_df = pd.concat([plot_train_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : calc_train_value
                            })], axis=1)
                            global_y_min = min(global_y_min, min(calc_train_value))
                            global_y_max = max(global_y_max, max(calc_train_value))

                    elif ce == 'test':
                        if calc_mode == 'norm':
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : BC_PUB_column_test_data
                            })], axis=1)
                            global_y_min = min(global_y_min, min(BC_PUB_column_test_data))
                            global_y_max = max(global_y_max, max(BC_PUB_column_test_data))
                        elif calc_mode == 'dif':
                            BMA_PUB_data_test_len = len(BMA_PUB_column_test_data)
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : BC_PUB_column_test_data - BMA_PUB_column_test_data
                            })], axis=1)
                            global_y_min = min(global_y_min, min(BC_PUB_column_test_data-BMA_PUB_column_test_data))
                            global_y_max = max(global_y_max, max(BC_PUB_column_test_data-BMA_PUB_column_test_data)) 
                        else: # calc_mode == 'ratio'
                            calc_test_value = []
                            for i in range(len(BC_PUB_column_test_data)):
                                calc_test_value.append(BC_PUB_column_test_data[i] / BMA_PUB_column_test_data[i] * 100.0 if BC_PUB_column_test_data[i] > BMA_PUB_column_test_data[i] else -100.0 * BMA_PUB_column_test_data[i] / BC_PUB_column_test_data[i])
                            plot_test_df = pd.concat([plot_test_df, pd.DataFrame({
                                f'{step_val_tag_name}{135-BC_PUB_data_test_len}' : calc_test_value
                            })], axis=1)
                            global_y_min = min(global_y_min, min(calc_test_value))
                            global_y_max = max(global_y_max, max(calc_test_value))                                

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
                    if calc_mode == 'norm':
                        if benchmark in ['KGE', 'NSE', 'logNSE', 'E1', 'VE', 'd']:
                            global_y_max = 1
                        elif benchmark in ['RMSE', 'MAE']:
                            global_y_min = 0
                        if benchmark == 'KGE':
                            global_y_min = -1.5
                        elif benchmark == 'NSE':
                            global_y_min = -5
                            #global_y_min = -3
                        elif benchmark == 'logNSE':
                            global_y_min = -150
                        elif benchmark == 'E1':
                            global_y_min = -2
                        elif benchmark == 'VE':
                            global_y_min = -2
                        elif benchmark == 'd':
                            global_y_max = 1
                        elif benchmark == 'RMSE':
                            global_y_min = 0
                        elif benchmark == 'MAE':
                            global_y_min = 0

                    elif calc_mode == 'dif':
                        if benchmark =='KGE':
                            global_y_max = 2
                            global_y_min = -3
                        elif benchmark == 'NSE':
                            global_y_max = 2
                            global_y_min = -5
                            #global_y_min = -5
                        elif benchmark == 'logNSE':
                            global_y_max = 2000
                            global_y_min = -200
                        elif benchmark == 'E1':
                            global_y_max = 2
                            global_y_min = -3
                        elif benchmark == 'VE':
                            global_y_max = 2
                            global_y_min = -3
                        elif benchmark == 'd':
                            global_y_max = 0.75
                            global_y_min = -0.75
                        elif benchmark == 'RMSE':
                            glo1bal_y_max = 6
                            global_y_min = -3
                        elif benchmark == 'MAE':
                            global_y_max = 4
                            global_y_min = -2
                        global_y_min = math.floor(global_y_min)
                        global_y_max = math.ceil(global_y_max)
                        
                    else: # calc_mode == 'ratio'
                        if benchmark == 'KGE':
                            global_y_max = 1000
                            global_y_min = -1000
                        elif benchmark == 'NSE':
                            global_y_max = 1000
                            global_y_min = -1000
                        elif benchmark == 'logNSE':
                            global_y_max = 10000
                            global_y_min = -10000
                        elif benchmark == 'E1':
                            global_y_max = 1000
                            global_y_min = -1000
                        elif benchmark == 'VE':
                            global_y_max = 1000
                            global_y_min = -1000
                        elif benchmark == 'd':
                            global_y_max = 300
                            global_y_min = -200
                        elif benchmark == 'RMSE':
                            global_y_max = 500
                            global_y_min = -500
                        elif benchmark == 'MAE':
                            global_y_max = 500
                            global_y_min = -500
                        global_y_min = math.floor(global_y_min)
                        global_y_max = math.ceil(global_y_max)

                    ax.set_ylim(global_y_min, global_y_max)


            for ax_group in all_axes:
                for ax in ax_group:
                    for label in ax.get_xticklabels():
                        label.set_rotation(90)

            fig_box.tight_layout()
            fig_line.tight_layout()
            fig_point.tight_layout()


            if benchmark == 'logNSE':
                box_output_path = os.path.join(output_dir_calc_mode, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
                line_output_path = os.path.join(output_dir_calc_mode, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
                point_output_path = os.path.join(output_dir_calc_mode, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')
            else:
                box_output_path = os.path.join(output_dir_calc_mode, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
                line_output_path = os.path.join(output_dir_calc_mode, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
                point_output_path = os.path.join(output_dir_calc_mode, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

            fig_box.savefig(box_output_path)
            fig_line.savefig(line_output_path)
            fig_point.savefig(point_output_path)

            plt.close(fig_box)
            plt.close(fig_line)
            plt.close(fig_point)

print("DONE!")
