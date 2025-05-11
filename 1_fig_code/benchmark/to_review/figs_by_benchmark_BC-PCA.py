import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

########
fs = 12  

## for BC-PUB_RCnocal_BMAnocal
start_val_list = [68]
step_val_list = [2, 5, 10, 20, 30, 40, 50, 60]
reservoir_size = 700
loc ,ver= "JP", 2
if loc == "JP" and ver == 2:
    file_tot_num = 87
    test_basins_list = [77,84,24,34,40,45,70,11,4] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #40
    #step_val_list = list(range(2, start_val_list[0] + 1))
    step_val_list = [ 2, 4, 8,15,20,25, 30, 40] # 39, 20, 10, 5, 4, 3, 2
    Train_basin_num_list = [39, 20, 10,5,4, 3,3,2]
########
end_val = 136

### TODO ADD RESULTS OF BC-PUB_nocal ####

benchmark_list = ["KGE", "NSE", "logNSE", "E1", "VE", "d", "RMSE", "MAE"]
PCA_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}/results'
PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_{reservoir_size}/start36/test_basin/results'

og_eva = pd.read_csv(os.path.join(PCA_result_data_dir, 'BC_results_r700_s1_sr0.4_rr0.001_eva.csv'))

def load_pc_data(pc_num):
    return pd.read_csv(os.path.join(PCA_result_data_dir, f'BC-PCA-lasso_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_num}_eva.csv'))

def load_BC_PUB_data(start_val, step_val):
    return pd.read_csv(os.path.join(PUB_result_data_dir, f'BC_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

benchmark_limits_norm = {
    'KGE': {'min': -4, 'max': 1},
    'NSE': {'min': -4, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 20}
}

for PC_only in [True]: #, False
    if PC_only:
        num_pcs = 5
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA/box'
    else:
        num_pcs = 3
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA/box_comp'
    os.makedirs(output_dir, exist_ok=True)


    for benchmark in benchmark_list:
        data = [og_eva[f'BC-og_r700_s1_sr0.4_rr0.001_{benchmark}_eva']] 

        if PC_only == False:    
            box_colors = ['silver'] + ['#a6bddb'] * (len(start_val_list) * len(step_val_list)) + ['#fb8072'] * num_pcs
            for start_val in start_val_list:
                for step_val in step_val_list:
                    pub_data = load_BC_PUB_data(start_val, step_val)
                    data.append(pub_data[f'BC_PUB_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'])
        else:
            box_colors = ['silver'] + ['#fb8072'] * num_pcs

        for pc in range(1, num_pcs + 1):
            pc_data = load_pc_data(pc)
            data.append(pc_data[f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_{benchmark}_eva'])
         

        plt.figure(figsize=(7, 7))

        if PC_only:
            box = plt.boxplot(data, patch_artist=True, tick_labels=['og_eva'] + [f'PC{pc}' for pc in range(1, num_pcs + 1)])
        else:
            box = plt.boxplot(data, patch_artist=True, tick_labels=['og_eva'] + [f'BC_PUB_sr{start_val}_st{step_val}' for start_val in start_val_list for step_val in step_val_list] + [f'PC{pc}' for pc in range(1, num_pcs + 1)])
        
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)

        for median in box['medians']:
            median.set_color('black')
            median.set_linewidth(2)   

        plt.title(f'Box Plot for {benchmark}', fontsize=fs)
        plt.xlabel('Evaluated values', fontsize=fs, labelpad=10)
        plt.ylabel(benchmark, fontsize=fs)
        plt.xticks(fontsize=fs, rotation=90)
        plt.yticks(fontsize=fs)
        plt.grid()
        plt.tight_layout()

        plt.ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])

        
        plt.savefig(os.path.join(output_dir, f'{benchmark}_boxplot_BC-PCA_eva.jpg'))
        plt.close()
print('Done')