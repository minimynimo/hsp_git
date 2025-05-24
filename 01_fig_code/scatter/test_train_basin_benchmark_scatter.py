import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import PowerNorm
import matplotlib.gridspec as gridspec

# BMA のtrain, test の関係の散布図を示す
#          x - y
# 1. BMA train vs BMA test
#
#### BCandBMA
# BCのtrain、test、BMA（RCなし）の関係の散布図を示す
#          x - y
# 1. BC train vs BC test
# 2. BMA train vs BMA test
# 2. BMA test vs BC test
#
#### RCBC-BMAandBMA
# RCBC-BMAのtrain, test, BMA(RCなし)の関係の散布図を示す
#                  x - y
# 1. RCBC-BMA train vs RCBC-BMA test
# 2. BMA train vs BMA test
# 3. BMA test vs RCBC-BMA test
#
#### BCandRCBC-BMA
# BC のtrain, test　とRCBC-BMAのtrain, testの関係の散布図を示す
#          x - y
# 1. BC train vs RCBC-BMA train
# 2. BC test vs RCBC-BMA test

plot_exp_list = ['BCandRCBC-BMA','BCandBMA','RCBC-BMAandBMA']


#####
start_val_list = [68]
#step_val_list = list(range(2, 34)) ### change to limit only where the distance is similar or maybe only a single step
#step_val_list = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,68]
step_val_list = [2,10,20,30,40,50,60,68]
nums = len(step_val_list)
file_tag = 'r700_s1_sr0.4_rr0.001'
RCnocal = True
BMAnocal = True
#####

end_val = 136

if RCnocal:
    nocal_tag = '_RCnocal'
else:
    nocal_tag = ''
if BMAnocal:    
    nocal_tag += '_BMAnocal'

output_dir = f'/data0/funato/0_out/0_fig/scatter/test_train_basin_benchmark_scatter/start{start_val_list[0]}'
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]
#benchmark_list = ["RMSE","MAE"]

for plot_exp in plot_exp_list:
    for benchmark in benchmark_list:
        print(f'plotting {plot_exp} {benchmark}')
        BC_PUB_train_plot_df = pd.DataFrame() ##the values used for each test basins
        BC_PUB_test_plot_df = pd.DataFrame()
        BMA_PUB_train_plot_df = pd.DataFrame()
        BMA_PUB_test_plot_df = pd.DataFrame()
        RCBC_BMA_PUB_train_plot_df = pd.DataFrame()
        RCBC_BMA_PUB_test_plot_df = pd.DataFrame()

        PUB_test_train_distance = pd.DataFrame()
        for start_val in start_val_list:
            for step_val in step_val_list:
                ## load the data
                BC_PUB_train_data = pd.read_csv(f'/data0/funato/0_out/BC_PUB{nocal_tag}/start{start_val}/train_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv')
                BC_PUB_test_data = pd.read_csv(f'/data0/funato/0_out/BC_PUB{nocal_tag}/start{start_val}/test_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv')
                BMA_PUB_train_data = pd.read_csv(f'/data0/funato/0_out/BMA_PUB_nocal/start{start_val}/train_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv')
                BMA_PUB_test_data = pd.read_csv(f'/data0/funato/0_out/BMA_PUB_nocal/start{start_val}/test_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv')
                RCBC_BMA_PUB_train_data = pd.read_csv(f'/data0/funato/0_out/RCBC-BMA_PUB{nocal_tag}l/start{start_val}/train_basin/results/RCBC-BMA_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv')
                RCBC_BMA_PUB_test_data = pd.read_csv(f'/data0/funato/0_out/RCBC-BMA_PUB{nocal_tag}/start{start_val}/test_basin/results/RCBC-BMA_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv')

                ## get the test data
                BC_PUB_test_plot_df = BC_PUB_test_plot_df.append(BC_PUB_test_data[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                BMA_PUB_test_plot_df = BMA_PUB_test_plot_df.append(BMA_PUB_test_data[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
                RCBC_BMA_PUB_test_plot_df = RCBC_BMA_PUB_test_plot_df.append(RCBC_BMA_PUB_test_data[f'RCBC-BMA_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())


                ## get the train data
                closest_basin_data = pd.read_csv(f'/data0/funato/3_gis_data/closest/start{start_val}/closest_basins_start{start_val}_step{step_val}.csv')
                for filenum in BC_PUB_test_data['file_num']:
                    closest_basin_num = closest_basin_data.loc[closest_basin_data['Basin'] == filenum, 'ClosestBasin'].values[0]

                    BC_PUB_closest_basin_train_data = BC_PUB_train_data[BC_PUB_train_data['file_num'] == closest_basin_num]
                    BC_PUB_train_plot_df = pd.concat([BC_PUB_train_plot_df, BC_PUB_closest_basin_train_data[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva']])
                    BMA_PUB_closest_basin_train_data = BMA_PUB_train_data[BMA_PUB_train_data['file_num'] == closest_basin_num]
                    BMA_PUB_train_plot_df = pd.concat([BMA_PUB_train_plot_df, BMA_PUB_closest_basin_train_data[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva']])
                    RCBC_BMA_PUB_closest_basin_train_data = RCBC_BMA_PUB_train_data[RCBC_BMA_PUB_train_data['file_num'] == closest_basin_num]
                    RCBC_BMA_PUB_train_plot_df = pd.concat([RCBC_BMA_PUB_train_plot_df, RCBC_BMA_PUB_closest_basin_train_data[f'RCBC-BMA_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva']])

                    distance = closest_basin_data.loc[closest_basin_data['Basin'] == filenum, 'Distance'].values[0]
                    PUB_test_train_distance = PUB_test_train_distance.append(pd.Series([distance]), ignore_index=True)


        PUB_test_train_distance = PUB_test_train_distance/1000

        if plot_exp in ['BCandBMA', 'RCBC-BMAandBMA']:
            fig = plt.figure(figsize=(22, 6))
            gs = gridspec.GridSpec(1, 6, width_ratios=[1, 0.2, 1, 0.2, 1.0,0.05])
        elif plot_exp == 'BCandRCBC-BMA':
            fig = plt.figure(figsize=(15, 6))
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.2, 1, 0.05])
        plt.subplots_adjust(wspace=0.15) 

        alpha_val = 1
        fs = 14
        tick_fs = 12
        gamma = 0.8
        linthresh_val = 0.4  # Adjust this value to control the linear range near zero

        # BC train vs BC test
        def plot_scatter(ax, x_data, y_data, distance_data, title, xlabel, ylabel, benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val):
            ax.scatter(x_data, y_data, c=distance_data, cmap='inferno_r', alpha=alpha_val, norm=PowerNorm(gamma=gamma))
            ax.set_title(title, fontsize=fs)
            ax.set_xlabel(xlabel, fontsize=fs)
            ax.set_ylabel(ylabel, fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=tick_fs)
            if benchmark not in ['RMSE', 'MAE']:
                ax.set_xscale('symlog', linthresh=linthresh_val)
                ax.set_yscale('symlog', linthresh=linthresh_val)
            else:
                ax.set_xscale('log')
                ax.set_yscale('log')

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot (gs[2])
        if plot_exp == 'BCandBMA':
            plot_scatter(ax1, BC_PUB_train_plot_df, BC_PUB_test_plot_df, PUB_test_train_distance, 
                        f'BC train vs BC test ({benchmark})\n', f'BC train ({benchmark})', f'BC test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)

            plot_scatter(ax2, BMA_PUB_train_plot_df, BMA_PUB_test_plot_df, PUB_test_train_distance, 
                        f'BMA train vs BMA test ({benchmark})\n', f'BMA train ({benchmark})', f'BMA test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
            
            ax3 = fig.add_subplot(gs[4])
            plot_scatter(ax3, BMA_PUB_test_plot_df, BC_PUB_test_plot_df, PUB_test_train_distance, 
                        f'BMA test vs BC test ({benchmark})\n', f'BMA test ({benchmark})', f'BC test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
        
        elif plot_exp == 'RCBC-BMAandBMA':
            plot_scatter(ax1, RCBC_BMA_PUB_train_plot_df, RCBC_BMA_PUB_test_plot_df, PUB_test_train_distance, 
                        f'RCBC-BMA train vs RCBC-BMA test ({benchmark})\n', f'RCBC-BMA train ({benchmark})', f'RCBC-BMA test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
            plot_scatter(ax2, BMA_PUB_train_plot_df, BMA_PUB_test_plot_df, PUB_test_train_distance, 
                        f'BMA train vs BMA test ({benchmark})\n', f'BMA train ({benchmark})', f'BMA test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
            ax3 = fig.add_subplot(gs[4])
            plot_scatter(ax3, BMA_PUB_test_plot_df, RCBC_BMA_PUB_test_plot_df, PUB_test_train_distance, 
                        f'BMA test vs RCBC-BMA test ({benchmark})\n', f'BMA test ({benchmark})', f'RCBC-BMA test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
            
        elif plot_exp == 'BCandRCBC-BMA':
            plot_scatter(ax1, BC_PUB_train_plot_df, RCBC_BMA_PUB_train_plot_df, PUB_test_train_distance,
                        f'BC train vs RCBC-BMA train ({benchmark})\n', f'BC train ({benchmark})', f'RCBC-BMA train ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)
            plot_scatter(ax2, BC_PUB_test_plot_df, RCBC_BMA_PUB_test_plot_df, PUB_test_train_distance,
                        f'BC test vs RCBC-BMA test ({benchmark})\n', f'BC test ({benchmark})', f'RCBC-BMA test ({benchmark})', 
                        benchmark, alpha_val, fs, tick_fs, gamma, linthresh_val)

        if plot_exp in ['BCandBMA', 'RCBC-BMAandBMA']:
            cbar_ax = fig.add_subplot(gs[5])
        elif plot_exp == 'BCandRCBC-BMA':
            cbar_ax = fig.add_subplot(gs[3])
        scatter = ax1.collections[0]  # Assuming ax1 contains the scatter plot
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        scatter.set_clim(0, scatter.get_clim()[1])
        cbar.set_label('Distance [$km$]', fontsize=fs)
        cbar.ax.tick_params(labelsize=tick_fs)


        all_data = pd.concat([BC_PUB_train_plot_df, BC_PUB_test_plot_df, BMA_PUB_train_plot_df, BMA_PUB_test_plot_df])
        min_val = all_data.min().min()*0.99
        max_val = all_data.max().max()*1.01

        if benchmark in ['KGE', 'NSE', 'logNSE', 'E1', 'Erel', 'VE', 'd']:
            max_val = 1
            if benchmark == 'NSE':
                min_val = -15
        elif benchmark in ['RMSE', 'MAE']:
            min_val = 0
            #max_val = 30

        # Add diagonal line y = x
        if plot_exp in ['BCandBMA', 'RCBC-BMAandBMA']:
            ax_group = [ax1, ax2, ax3]
        elif plot_exp == 'BCandRCBC-BMA':
            ax_group = [ax1, ax2]
        for ax in ax_group:
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),  # min of both axes
                max(ax.get_xlim()[1], ax.get_ylim()[1]),  # max of both axes
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            if benchmark == ['NSE','KGE'] and ax != ax3:
                ax.set_xlim(-0.1,1.0)
                ax.set_ylim(lims)
            else:
                ax.set_xlim(lims)
                ax.set_ylim(lims)
            ax.set_aspect('auto')

        #plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        output_dir_step = os.path.join(output_dir, f'step{min(step_val_list)}to{max(step_val_list)}_n{len(step_val_list)}/{plot_exp}')
        os.makedirs(output_dir_step, exist_ok=True)
        plt.savefig(os.path.join(output_dir_step, f'{plot_exp}_test_train_basin_{benchmark}_step{min(step_val_list)}to{max(step_val_list)}_scatter.jpg'))
        plt.close()

        print(f'{plot_exp} {benchmark} Done!')

print('Done!')