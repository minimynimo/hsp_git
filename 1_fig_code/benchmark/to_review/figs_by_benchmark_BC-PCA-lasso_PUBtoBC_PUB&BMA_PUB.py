import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# compare BC og with BC-PUB based on spatial proximity and BC-PCA-lasso based on PC1-PC5

########
fs = 21

## for RCBC-PUB


#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc = "US"
num_pcs = 2 # max 5
file_tag = ''
scatter = False

reservoir_size = 700
########

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4, 8, 15, 20, 25, 30, 40, 50]
    Train_basin_num_list = [67, 33, 17,9,7,5,5,3,3]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    test_basins_list = [77,84,24,34,40,45,70,11,4] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #40
    #step_val_list = list(range(2, start_val_list[0] + 1))
    step_val_list = [ 2, 4, 8,15,20,25, 30, 40] # 39, 20, 10, 5, 4, 3, 2
    Train_basin_num_list = [39, 20, 10,5,4, 3,3,2]
elif loc == "US":
    file_tot_num = 671
    start_val_list = [336]
    #step_val_list = list(range(2,337))
    step_val_list = [2,5,10,21,42,84,168,337]
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

# BC og, BC pub, BC pca, RCBC og, RCBC pub, RCBC pca
bc_rcbc_list = []
file_tag = ''

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark'

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"] # "logNSE",

# for non pca predictions
BC_og_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv')
og_BMA_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/results/BMA_results_eva.csv')
AVE_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/AVE/results/AVE_results_eva.csv')

def load_BMA_PUB_data(start_val, step_val, BMA_PUB_result_data_dir):
    bma_pub_data = pd.read_csv(os.path.join(BMA_PUB_result_data_dir, f'BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'))
    #filtered_bma_pub_data = bma_pub_data[bma_pub_data['file_num'].isin(test_basins_list)]
    #return filtered_bma_pub_data
    return bma_pub_data

def load_BC_PUB_data(start_val, step_val, BC_PUB_result_data_dir):
    return pd.read_csv(os.path.join(BC_PUB_result_data_dir, f'BC_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

def load_BC_PCA_lasso_PUB_data(start_val, step_val, pc_num, BC_PCA_lasso_PUB_result_data_dir):
    pca_pub_data = pd.read_csv(os.path.join(BC_PCA_lasso_PUB_result_data_dir, f'BC-PCA-lasso_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_rev_PC{pc_num}_eva.csv'))
    #filtered_pca_pub_data = pca_pub_data[pca_pub_data['file_num'].isin(test_basins_list)]
    #filtered_pca_pub_data = pd.DataFrame(filtered_pca_pub_data)
    #return filtered_pca_pub_data
    return pca_pub_data

for pc in range(1, num_pcs + 1):
    print(f'PC{pc}')
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
                BMA_pub_data.append(bma_pub_data[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva']) 

            #BC_PUB
            for step_val in step_val_list:
                bc_pub_data = load_BC_PUB_data(start_val, step_val, BC_PUB_result_data_dir)
                BC_pub_data.append(bc_pub_data[f'BC_PUB_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'])

            #BC_PCA_PUB
            i = 0
            for step_val in step_val_list:
                BC_PCA_lasso_PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB/start{start_val}/step{step_val}/test_basin/results'
                pca_pub_data = load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, BC_PCA_lasso_PUB_result_data_dir)
                BC_PCA_pub_data.append(pca_pub_data[f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'])
                box_colors.extend(['#74c476']) # color: green
                tick_labels.extend([f'{Train_basin_num_list[i]}'])
                i += 1

        print(pca_pub_data.shape) #step 8, test 9
        print(BC_PCA_pub_data)


        for box_type in ['norm', 'dif']:#,, 'ratio'
            if box_type != 'norm':
                output_dir_BMA_box = f'{output_dir}/BC-PCA-lasso&BMA_PUB/box/start{start_val}/n{len(step_val_list)}/{box_type}/PC{pc}'
                output_dir_BC_box = f'{output_dir}/box/BC-PCA-lasso&BC_PUB/box/start{start_val}/n{len(step_val_list)}/{box_type}/PC{pc}'
                os.makedirs(output_dir_BMA_box, exist_ok=True)
                os.makedirs(output_dir_BC_box, exist_ok=True)
            else:
                output_dir_norm_box = f'{output_dir}/BC-PCA-lasso/box/start{start_val}/n{len(step_val_list)}/PC{pc}'
                os.makedirs(output_dir_norm_box, exist_ok=True)
                output_dir_BMA_box = output_dir_norm_box  # Ensure output_dir_BMA_box is defined
                output_dir_BC_box = output_dir_norm_box  # Ensure output_dir_BC_box is defined
            
            ###############
            ### BC_PCA norm
            if box_type == 'norm':
                plt.figure(figsize=(9, 7))  # Set figure size to 9 by 7
                tick_labels = ['BC_og'] + tick_labels  # Ensure tick_labels is correctly populated
                BC_data = [og_eva[f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva']]
                BC_data.extend(BC_PCA_pub_data)
                plt.title(f'Box Plot for {benchmark} NORM BC-PCA-lasso_PUB_PC{pc}\n', fontsize=fs)
                box = plt.boxplot(BC_data)  # Ensure the boxplot is created with the correct data
                plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=90)
                plt.xlabel('# Trained Basins', fontsize=fs, labelpad=10)
                plt.ylabel(benchmark, fontsize=fs)
                plt.xticks(fontsize=fs, rotation=90)
                plt.yticks(fontsize=fs)
                
                if benchmark == "NSE":
                    min_val = -5
                    max_val = 1
                elif benchmark == "KGE":
                    min_val = -5
                    max_val = 1
                elif benchmark == "E1":
                    min_val = max(-5, min_val)
                    max_val = 1
                elif benchmark =="RMSE":
                    min_val = max(0, min_val)
                    max_val = 50

                plt.ylim(min_val, max_val)
                plt.grid()
                plt.tight_layout()
                
                plt.savefig(f'{output_dir_norm_box}/{benchmark}_boxplot_BC-PCA-lasso_PUB{file_tag}_PC{pc}_eva.jpg')  # Save the norm figure
                plt.close()

            ###############################
            ### BMA_PUB vs BC-PCA-lasso_PUB 
            BMA_data = []
            plt.figure(figsize=(9, 7))  # Set figure size to 9 by 7
            if box_type == 'dif':
                BMA_data.append(og_BC_data - og_BMA_data)
                for i in range(len(BC_PCA_pub_data)):
                    BMA_pub = BMA_pub_data[i]
                    BC_PCA_pub = BC_PCA_pub_data[i]
                    ratio = BC_PCA_pub - BMA_pub
                    BMA_data.append(ratio)
                plt.title(f'Box Plot for {benchmark} DIF BC-PCA-lasso_PUB_PC{pc} - BMA_PUB\n ')
                plt.ylim(-5,2)
            elif box_type == 'ratio':
                BMA_data.append(og_BC_data/ og_BMA_data)
                for i in range(len(BC_PCA_pub_data)):
                    BMA_pub = BMA_pub_data[i]
                    BC_PCA_pub = BC_PCA_pub_data[i]
                    ratio = np.where(BC_PCA_pub >= BMA_pub, BC_PCA_pub / BMA_pub, -1 * BMA_pub / BC_PCA_pub)
                    BMA_data.append(ratio)
                plt.title(f'Box Plot for {benchmark} RATIO BC-PCA-lasso_PUB_PC{pc} to BMA_PUB\n')
            
            if BMA_data:
                max_val = max([BMA_data.max() for BMA_data in BMA_data])
                min_val = min([BMA_data.min() for BMA_data in BMA_data])

            if benchmark == "NSE":
                min_val = max(-10, min_val)  
                max_val = min(max_val,10)
                min_val = -5
                max_val = 2
            elif benchmark == "KGE":
                min_val = max(-5, min_val)
                max_vel = min(max_val, 5)
                min_val = -5
                max_val = 2
            elif benchmark == "E1":
                min_val = max(-10, min_val)
                max_val = min(max_val, 10)
            elif benchmark == "RMSE":
                min_val = -20
                max_val = 50

            if not np.isnan(min_val) and not np.isnan(max_val) and not np.isinf(min_val) and not np.isinf(max_val):
                plt.ylim(min_val, max_val)

            box = plt.boxplot(BMA_data) #, patch_artist=True)
            plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=90)

            #for patch, color in zip(box['boxes'], box_colors):
            #     patch.set_facecolor(color)

            #for median in box['medians']:
            #    median.set_color('black')
            #    median.set_linewidth(2)

            plt.xlabel('# Trained Basins', fontsize=fs, labelpad=10)
            plt.ylabel(benchmark, fontsize=fs)
            plt.xticks(fontsize=fs, rotation=90)
            plt.yticks(fontsize=fs)
            plt.grid()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir_BMA_box, f'{benchmark}_boxplot_BC-PCA-lasso_BMA_PUB_PC{pc}_eva.jpg'))
            plt.close()

            ####################
            ### BC-PCA to BC_PUB
            BC_data = []
            plt.figure(figsize=(9, 7))  # Set figure size to 9 by 7
            if box_type == 'dif':
                for i in range(len(BC_PCA_pub_data)):
                    BC_pub = BC_pub_data[i]
                    BC_PCA_pub = BC_PCA_pub_data[i]
                    ratio = BC_PCA_pub - BC_pub
                    BC_data.append(ratio)
                plt.ylim(-5, 2)
                plt.title(f'Box Plot for {benchmark} DIF BC-PCA-lasso_PUB_PC{pc} - BC_PUB \n')
            elif box_type == 'ratio':
                for i in range(len(BC_PCA_pub_data)):
                    BC_pub = BC_pub_data[i]
                    BC_PCA_pub = BC_PCA_pub_data[i]
                    ratio = np.where(BC_PCA_pub >= BC_pub, BC_PCA_pub / BC_pub, -1 * BC_pub / BC_PCA_pub)
                    BC_data.append(ratio)
                plt.title(f'Box Plot for {benchmark} RATIO BC-PCA-lasso_PUB_PC{pc} to BC_PUB \n')

            if BC_data:
                max_val = max([BC_data.max() for BC_data in BC_data])
                min_val = min([BC_data.min() for BC_data in BC_data])


            if benchmark == "NSE":
                min_val = max(-5, min_val)  
                max_val = min(max_val,5)
            elif benchmark == "E1":
                min_val = max(-10, min_val)
                max_val = min(max_val, 10)  
            elif benchmark == "RMSE":
                min_val = -20
                max_val = 50

            if not np.isnan(min_val) and not np.isnan(max_val) and not np.isinf(min_val) and not np.isinf(max_val):
                plt.ylim(min_val, max_val)

            box = plt.boxplot(BC_data) #, patch_artist=True)
            plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=90)
            
            #for patch, color in zip(box['boxes'], box_colors):
            #     patch.set_facecolor(color)

            #for median in box['medians']:
            #    median.set_color('black')
            #    median.set_linewidth(2)

            plt.xlabel('# Trained Basins', fontsize=fs, labelpad=10)
            plt.ylabel(benchmark, fontsize=fs)
            plt.xticks(fontsize=fs, rotation=90)
            plt.yticks(fontsize=fs)
            plt.grid()
            plt.tight_layout()  # Ensure the layout fits within the figure size
            
            plt.savefig(os.path.join(output_dir_BC_box, f'{benchmark}_boxplot_BC-PCA-lasso_PUB{file_tag}_PC{pc}_eva.jpg'))
            plt.close()

        output_dir_scat = f'{output_dir}/BC-PCA-lasso&BMA_PUB/scatter/PC{pc}'
        os.makedirs(output_dir_scat, exist_ok=True)

        # scatter plot
        if scatter:
            plt.figure(figsize=(9, 7))  # Set figure size to 9 by 7
            plt.figure(figsize=(10, 7))
            for start_val in start_val_list:
                for i, step_val in enumerate(step_val_list):
                    color = plt.cm.viridis(i / len(step_val_list))  # Use a colormap to generate colors
                    plt.scatter(BC_pub_data[i], BC_PCA_pub_data[i], label=f'sr{start_val}_st{step_val}', color=color)
            min_val = min(min([data.min() for data in BC_pub_data]), min([data.min() for data in BC_PCA_pub_data]))
            max_val = max(max([data.max() for data in BC_pub_data]), max([data.max() for data in BC_PCA_pub_data]))
            plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
            if benchmark == "NSE":
                min_val = -10
                plt.xlim(min_val, 1)  # Adjust xlim for NSE
                plt.ylim(min_val, 1)  # Adjust ylim for NSE
            else:
                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('BC_PUB', fontsize=fs)
            plt.ylabel(f'BC-PCA-lasso_PC{pc}', fontsize=fs)
            plt.title(f'Scatter Plot for {benchmark} BC_PUB vs BC-PCA-lasso_PC{pc}\n', fontsize=fs)

            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.grid()
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir_scat, f'{benchmark}_scatterplot_BC-PCA-lasso_PUB{file_tag}_PC{pc}_eva.jpg'))

            plt.close()
print('Done')