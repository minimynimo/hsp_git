import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
# compare BC og with BC-PUB based on spatial proximity and BC-PCA-lasso based on PC1-PC5

########
fs = 12  

## for RCBC-PUB_RCnocal_BMAnocal
start_val_list = [68]
step_val_list = [2,10,20,30,40,50]

loc = "JP"
loc = "US"
loc = "0_JP_MARRMoT_44"
########
end_val = 136

# BC og, BC pub, BC pca, RCBC og, RCBC pub, RCBC pca
bc_rcbc_list = []
file_tag = ''

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"] # "logNSE",

# for non pca predictions
BC_og_eva = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC_700_nocal/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv')
BC_PUB_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_RCnocal_BMAnocal/start68/test_basin/results'
BC_PCA_lasso_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_nocal/results'


## for pca predictions
def load_BC_pc_data(pc_num):
    return pd.read_csv(os.path.join(BC_PCA_lasso_result_data_dir, f'BC-PCA-lasso_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_num}_eva.csv'))

def load_BC_PUB_data(start_val, step_val):
    return pd.read_csv(os.path.join(BC_PUB_result_data_dir, f'BC_PUB_results_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_eva.csv'))

def create_line_plot(ax, plot_df, benchmark):
    positions = range(1, len(plot_df.columns) + 1)  # Positions for BC-og, BC-PCA-lasso PC1 and PC2
    lw = 0.4
    if benchmark != 'RMSE' and benchmark != 'MAE': # higher values are better
        for i in range(len(plot_df)):  # Loop through each row (data point)
            """
            if plot_df.iloc[i]['PC1'] > plot_df.iloc[i]['BC-og'] and plot_df.iloc[i]['PC1'] > plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            elif plot_df.iloc[i]['PC1'] < plot_df.iloc[i]['BC-og'] and plot_df.iloc[i]['PC1'] < plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            elif plot_df.iloc[i]['BC-og'] > plot_df.iloc[i]['PC1'] and plot_df.iloc[i]['BC-og'] > plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='yellow')"""
            if plot_df.iloc[i]['PC1'] > plot_df.iloc[i]['BC-og'] or plot_df.iloc[i]['PC2'] > plot_df.iloc[i]['BC-og']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
        # for legends
        ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='crimson', label='Good\nPC1 or PC2 > BC-og')        
        #ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='crimson', label='Good\nPC1 > BC-og & PC2')        
        #ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='crimson', label='Good\nPC2 > BC-og & PC1')
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='royalblue', label='Bad\nBC-og > PC1 & PC2')
            
    else: # lower values are better
        for i in range(len(plot_df)):  # Loop through each row (data point)
            """
            if plot_df.iloc[i]['PC1'] > plot_df.iloc[i]['BC-og'] and plot_df.iloc[i]['PC1'] > plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
            elif plot_df.iloc[i]['BC-og'] < plot_df.iloc[i]['PC2'] and plot_df.iloc[i]['PC1'] < plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='mediumslateblue')
            elif plot_df.iloc[i]['BC-og'] > plot_df.iloc[i]['PC1'] and plot_df.iloc[i]['BC-og'] > plot_df.iloc[i]['PC2']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='yellow')"""
            
            if plot_df.iloc[i]['PC1'] < plot_df.iloc[i]['BC-og'] and plot_df.iloc[i]['PC2'] < plot_df.iloc[i]['BC-og']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
        # for legends
        ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='royalblue', label='Bad\nPC1 or PC2 < BC-og')        
        #ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='crimson', label='Bad\nPC1 > BC-og & PC2')        
        #ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='darkorange', label='Bad\nPC2 > BC-og & PC1')
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='crimson', label='Good\nBC-og > PC1 & PC2')
    ax.set_title(f'Line plot for {benchmark}')
    ax.set_ylabel(benchmark)
    ax.set_xticks(positions)
    ax.set_xticklabels(['BC-og', 'PC1', 'PC2'])
    ax.legend(loc='lower left')

for PC_only in [True, False]:
    if PC_only:
        num_pcs = 5 # max 5
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA-lasso/box{num_pcs}'
        output_dir_line = os.path.join(output_dir, 'line')
        os.makedirs(output_dir_line, exist_ok=True)
    else:
        num_pcs = 3 # max 5
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BC-PCA-lasso/box{num_pcs}_comp'
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmark_list:
        data = []
        box_colors = []
        tick_labels = []
        # add BC_og_eva data
        og_eva = BC_og_eva
        data.append(og_eva[f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva'])

        if PC_only == False:    # Add BC_PUB data
            box_colors.extend(['silver'] + ['#a6bddb'] * (len(start_val_list) * len(step_val_list)) + ['#fb8072'] * num_pcs)
            tick_labels.extend([f'BC_og_eva'] + [f'BC_PUB_sr{start_val}_st{step_val}' for start_val in start_val_list for step_val in step_val_list] + [f'BC_PC{pc}' for pc in range(1, num_pcs + 1)])
            ## add BC_PUB data
            for start_val in start_val_list:
                for step_val in step_val_list:
                    pub_data = load_BC_PUB_data(start_val, step_val)
                    data.append(pub_data[f'BC_PUB_r700_s1_sr0.4_rr0.001_start{start_val}_step{step_val}_{benchmark}_eva'])

        else: # Add PC data only
            box_colors.extend(['silver'] + ['#fb8072'] * num_pcs)
            tick_labels.extend([f'BC_og_eva'] + [f'BC_PC{pc}' for pc in range(1, num_pcs + 1)])

        # add BC-PCA-lasso data
        for pc in range(1, num_pcs + 1):
            pc_data = load_BC_pc_data(pc)
            data.append(pc_data[f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_{benchmark}_eva'])            

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
        
        plt.savefig(os.path.join(output_dir, f'{benchmark}_boxplot_BC-PCA-lasso{file_tag}_eva.jpg'))
        plt.close()

        if PC_only == True:
            fig, ax = plt.subplots(1, 1, figsize=(5, 7))
            plot_df = pd.DataFrame(data, index=tick_labels).T
            plot_df = plot_df.iloc[:, :3]
            plot_df.columns = ['BC-og', 'PC1', 'PC2']
            create_line_plot(ax, plot_df, benchmark)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir_line, f'{benchmark}_lineplot_BC-PCA-lasso{file_tag}_eva.jpg'))
            plt.close()

print('Done')