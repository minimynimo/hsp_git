import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# compare BC og with BC-PUB based on spatial proximity and BC-PCA-lasso based on PC1-PC5

########
fs = 17

## for RCBC-PUB


#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc, ver = "US", 1
#loc, ver = "US", 2

num_pcs = 2 # max 5
file_tag = ''
scatter = False

reservoir_size = 700
########

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)]
    #step_val_list = list(range(2, start_val_list[0] + 1))
    step_val_list = [2,4,8,15,20,25,30,40] # 39, 20, 10, 5, 4, 3, 2 
    Train_basin_num_list = [39, 20, 10, 5, 4, 3, 3, 2 ]
elif loc == "US" and ver == 1:
    file_tot_num = 671
    start_val_list = [336]
    step_val_list = [2,5,10,21,42,84,168,337]
elif loc == "US" and ver == 2:
    file_tot_num = 667
    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #301
    step_val_list = [2,4,6,12,24,47,84,125,149,168,277] #### Train basin num: 299,150,100,50,25,13,7,5,4,3,3
    Train_basin_num_list = [299,150,100,50,25,13,7,5,4,3,3]
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

# BC og, BC pub, BC pca, RCBC og, RCBC pub, RCBC pca
bc_rcbc_list = []
file_tag = ''

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"] # "logNSE",

# for non pca predictions
BC_og_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv')
og_BMA_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/results/BMA_results_eva.csv')
AVE_df = pd.read_csv(f'/data0/funato/0_out/99_out/JP/AVE/results/AVE_results_eva.csv')

def load_BMA_PUB_data(start_val, step_val, BMA_PUB_result_data_dir):
    bma_pub_data = pd.read_csv(os.path.join(BMA_PUB_result_data_dir, f'BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'))
    filtered_bma_pub_data = bma_pub_data[bma_pub_data['file_num'].isin(test_basins_list)]
    return filtered_bma_pub_data


def load_BC_PUB_data(start_val, step_val, BC_PUB_result_data_dir):
    return pd.read_csv(os.path.join(BC_PUB_result_data_dir, f'BC_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

def load_BC_PCA_lasso_PUB_data(start_val, step_val, pc_num, BC_PCA_lasso_PUB_result_data_dir, benchmark):
    pca_pub_data = pd.read_csv(os.path.join(BC_PCA_lasso_PUB_result_data_dir, f'BC-PCA-lasso_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_rev_PC{pc_num}_eva.csv'))
    """
    pca_pub_data = pca_pub_data[f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva']
    filtered_pca_pub_data = pca_pub_data[pca_pub_data['file_num'].isin(test_basins_list)]
    """
    column_name = f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'
    filtered_pca_pub_data = pca_pub_data[pca_pub_data['file_num'].isin(test_basins_list)]
    filtered_pca_pub_data = filtered_pca_pub_data[['file_num', column_name]]
    filtered_pca_pub_data = pd.DataFrame(filtered_pca_pub_data[column_name].values, index=filtered_pca_pub_data['file_num'], columns=[column_name])
    print(filtered_pca_pub_data)

    return filtered_pca_pub_data

benchmark_limits_norm = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}
benchmark_limits_dif = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -2, 'max': 2},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 2},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': -4, 'max': 4},
    'MAE': {'min': -3, 'max': 3}
}

for pc in range(1, num_pcs + 1):
    print(f'PC{pc}')
    output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA_PUB/PC{pc}'
    os.makedirs(output_dir, exist_ok=True)
    for benchmark in benchmark_list:
        # box_plot
        BMA_pub_data = []
        BC_pub_data = []
        BC_PCA_pub_data = []
        box_colors = ['silver']
        # add BC_og_eva data
        og_eva = BC_og_df
        tick_labels = []

        # og-BC
        #og_BC_column = f'BC{file_tag}_bma_{benchmark}_eva'
        og_BC_column = f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva'
        og_BC_data = BC_og_df[og_BC_column]

        #og-BMA
        og_BMA_column = f'BMA_{benchmark}_eva' 
        og_BMA_data = og_BMA_df[og_BMA_column]

        #simple average
        AVE_column = f'AVE_{benchmark}_eva'
        AVE_data = AVE_df[AVE_column]

        for start_val in start_val_list:
            BMA_PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB/start{start_val}/test_basin/results'
            BC_PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC_PUB/start{start_val}/test_basin/results'
            BC_PCA_lasso_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/results'

            #BMA_PUB
            for step_val in step_val_list:
                bma_pub_data = load_BMA_PUB_data(start_val, step_val, BMA_PUB_result_data_dir)
                BMA_pub_data.append(bma_pub_data[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].values) 

            #BC_PUB
            for step_val in step_val_list:
                bc_pub_data = load_BC_PUB_data(start_val, step_val, BC_PUB_result_data_dir)
                filtered_bc_pub_data = bc_pub_data[bc_pub_data['file_num'].isin(test_basins_list)]
                BC_pub_data.append(filtered_bc_pub_data[f'BC_PUB_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'].values)

            #BC_PCA_PUB
            i = 0
            for step_val in step_val_list:
                BC_PCA_lasso_PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB/start{start_val}/step{step_val}/test_basin/results'
                pca_pub_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, BC_PCA_lasso_PUB_result_data_dir, benchmark)
                BC_PCA_pub_data.append(pca_pub_data.values.flatten())
                box_colors.extend(['#74c476']) # color: green
                tick_labels.extend([f'{Train_basin_num_list[i]}'])
                i += 1

        print(BC_PCA_pub_data)
            
        ###############
        ### BC_PCA norm
        tick_labels.insert(0, 'BC_og')
        BC_data = [og_eva[f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva'].values]
        BC_data.extend(BC_PCA_pub_data)
        plt.title(f'Box Plot for {benchmark} NORM BC-PCA-lasso_PUB_PC{pc}\n', fontsize=fs)
        box = plt.boxplot(BC_data)  # Ensure the boxplot is created with the correct data
        plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=90)
        plt.xlabel('# Trained Basins', fontsize=fs, labelpad=10)
        plt.ylabel(benchmark, fontsize=fs)
        plt.xticks(fontsize=fs, rotation=90)
        plt.yticks(fontsize=fs)
        
        if benchmark in benchmark_limits_norm:
            min_val = benchmark_limits_norm[benchmark]['min']
            max_val = benchmark_limits_norm[benchmark]['max']
        else:
            min_val = -5
            max_val = 1

        plt.ylim(min_val, max_val)
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/{benchmark}_boxplot_BC-PCA-lasso_PUB{file_tag}_PC{pc}_eva.jpg')  # Save the norm figure
        plt.close()

print(f"Saved to {output_dir}")
print("DONE")