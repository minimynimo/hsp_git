import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

########
fs = 12  

## for RCBC-PUB_RCnocal_BMAnocal
start_val_list = [68]
step_val_list = [2,10,20,30,40,50]

#bc_fig, rcbc_fig = True, True
bc_fig, rcbc_fig = False, True
#bc_fig, rcbc_fig = True, False
########
end_val = 136

# BC og, BC pub, BC pca, RCBC og, RCBC pub, RCBC pca
bc_rcbc_list = []
file_tag = ''
if bc_fig:
    bc_rcbc_list.extend(['BC'])
    file_tag += 'BC'
if rcbc_fig:
    bc_rcbc_list.extend(['RCBC-BMA'])
    if bc_fig:
        file_tag += '_'
    file_tag += 'RCBC-BMA'

benchmark_list = ["KGE", "NSE", "logNSE", "E1", "VE", "d", "RMSE", "MAE"]

BC_PCA_result_data_dir = '/data0/funato/0_out/99_out/BC-PCA_nocal/results'
RCBC_PCA_result_data_dir = '/data0/funato/0_out/99_out/RCBC-BMA_PCA_nocal/results'
BC_PUB_result_data_dir = '/data0/funato/0_out/99_out/BC_PUB_RCnocal_BMAnocal/start68/test_basin/results'
RCBC_PUB_result_data_dir = '/data0/funato/0_out/99_out/RCBC-BMA_PUB_RCnocal_BMAnocal/start68/test_basin/results'

# for non pca predictions
BC_og_eva = pd.read_csv(os.path.join(BC_PCA_result_data_dir, 'BC-PCA_results_r700_s1_sr0.4_rr0.001_og_eva.csv'))
RCBC_og_eva = pd.read_csv(os.path.join(RCBC_PCA_result_data_dir, 'RCBC-BMA_PCA_results_r700_s1_sr0.4_rr0.001_og_eva.csv'))

## for pca predictions
def load_BC_pc_data(pc_num):
    return pd.read_csv(os.path.join(BC_PCA_result_data_dir, f'BC-PCA_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_num}_eva.csv'))

def load_RCBC_pc_data(pc_num):
    return pd.read_csv(os.path.join(RCBC_PCA_result_data_dir, f'RCBC-BMA_PCA_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_num}_eva.csv'))

def load_BC_PUB_data(start_val, step_val):
    return pd.read_csv(os.path.join(BC_PUB_result_data_dir, f'BC_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

def load_RCBC_PUB_data(start_val, step_val):
    return pd.read_csv(os.path.join(RCBC_PUB_result_data_dir, f'RCBC-BMA_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

for PC_only in [True, False]:
    if PC_only:
        num_pcs = 5
        output_dir = f'/data0/funato/0_out/0_fig/benchmark/PCA/PCA_{file_tag}_nocal/box{num_pcs}'
    else:
        num_pcs = 3
        output_dir = f'/data0/funato/0_out/0_fig/benchmark/PCA/PCA_{file_tag}_nocal/box{num_pcs}_comp'
    os.makedirs(output_dir, exist_ok=True)


    for benchmark in benchmark_list:
        data = []
        box_colors = []
        tick_labels = []
        for bc_rcbc in bc_rcbc_list:
            og_eva = BC_og_eva if bc_rcbc == 'BC' else RCBC_og_eva
            if bc_rcbc == 'BC':
                data.append(og_eva[f'BC-og_r700_s1_sr0.4_rr0.001_{benchmark}_eva'])
            elif bc_rcbc == 'RCBC-BMA':
                data.append(og_eva[f'RCBC-BMA_PCA_r700_s1_sr0.4_rr0.001_{benchmark}_eva_og'])

            if PC_only == False:    # Add BC_PUB data
                box_colors.extend(['silver'] + ['#a6bddb'] * (len(start_val_list) * len(step_val_list)) + ['#fb8072'] * num_pcs)
                tick_labels.extend([f'{bc_rcbc}_og_eva'] + [f'{bc_rcbc}_PUB_sr{start_val}_st{step_val}' for start_val in start_val_list for step_val in step_val_list] + [f'{bc_rcbc}_PC{pc}' for pc in range(1, num_pcs + 1)])
                for start_val in start_val_list:
                    for step_val in step_val_list:
                        if bc_rcbc == 'BC':
                            pub_data = load_BC_PUB_data(start_val, step_val)
                        elif bc_rcbc == 'RCBC-BMA':
                            pub_data = load_RCBC_PUB_data(start_val, step_val)
                        data.append(pub_data[f'{bc_rcbc}_PUB_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'])

            else: # Add PC data only
                box_colors.extend(['silver'] + ['#fb8072'] * num_pcs)
                tick_labels.extend([f'{bc_rcbc}_og_eva'] + [f'{bc_rcbc}_PC{pc}' for pc in range(1, num_pcs + 1)])

            for pc in range(1, num_pcs + 1):
                if bc_rcbc == 'BC':
                    pc_data = load_BC_pc_data(pc)
                    data.append(pc_data[f'BC-PCA_r700_s1_sr0.4_rr0.001_{benchmark}_eva'])            
                elif bc_rcbc == 'RCBC-BMA':
                    pc_data = load_RCBC_pc_data(pc)
                    data.append(pc_data[f'RCBC-BMA_PCA_r700_s1_sr0.4_rr0.001_{benchmark}_eva'])

        plt.figure(figsize=(7, 7))

        box = plt.boxplot(data, patch_artist=True)
        plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=90)
        
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
        
        plt.savefig(os.path.join(output_dir, f'{benchmark}_boxplot_PCA_{file_tag}_eva.jpg'))
        plt.close()
print('Done')