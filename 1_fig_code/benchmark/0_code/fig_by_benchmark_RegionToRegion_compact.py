import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

n_components = 2
fs = 16

non_arid_files = True
samples = 100

#train_loc = 'US'
#train_loc = 'GB'
#train_loc = 'AUS'
train_loc = 'JP'

#test_loc = 'US'
test_loc = 'GB'
#test_loc = 'AUS'
#test_loc = 'BR'
#test_loc = 'JP'
ver = 2

if test_loc != "US":
    non_arid_files = False

r2r_reservoir_size = 200
ridge_param = 1.0

if test_loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    train_basin_int_list = [30,20,15,10,5,3]
    ridge_param = 1.0
elif test_loc == "US" and ver == 2:
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
elif test_loc == "AUS" and ver == 2:
    file_tot_num = 84
    reservoir_size = 200

    test_basins_list = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82] # 17 basins
    train_basin_int_list = [30,20,15,10,5,3,2]
    ridge_param = 1.0
elif test_loc == "GB" and ver == 2:
    file_tot_num = 396
    reservoir_size = 300

    test_basins_list = list(range(4, file_tot_num, 10))
    train_basin_int_list = [170,130,90,60,40,20,10]
    ridge_param = 1.0
else:
    print("Invalid loc and ver")
    exit()

train_basin_dir = f'/data0/funato/0_out/99_out/{train_loc}to{test_loc}/BC-PCA-lasso_PUB_{r2r_reservoir_size}_{ridge_param}/train_basin/results'
test_basin_dir = f'/data0/funato/0_out/99_out/{train_loc}to{test_loc}/BC-PCA-lasso_PUB_{r2r_reservoir_size}_{ridge_param}/test_basin/results'

output_dir = f'/data0/funato/0_out/0_fig/{train_loc}to{test_loc}/benchmark/BC-PCA-lasso_PUB'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/box_compact', exist_ok=True)
os.makedirs(f'{output_dir}/cdf', exist_ok=True)

if non_arid_files:
    non_arid_buf = '_non_arid'
else:
    non_arid_buf = ''

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

benchmark_limits_norm = {
    'KGE': {'min': -2, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -5, 'max': 1},
    'VE': {'min': -5, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 10}
}

def create_cdf_plot(ax, plot_df, benchmark):
    lw_cdf = 1.3
    markers_list = ['D', 'o', 's', '^', 'v', 'x', '*']
    markers = markers_list[:len(plot_df.columns)]
    
    color_gradients = {
        'toReg': plt.cm.Blues,
        'inReg': plt.cm.Reds,
    }
    reset_interval = len(train_basin_int_list)
    legend_entries = {}
    
    for i, column in enumerate(plot_df.columns):
        sorted_data = np.sort(plot_df[column])
        cdf = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))  # Ensure CDF goes to 1
        
        if '->' in column:
            color_key = 'toReg'
            color = 'royalblue'
            group_name = None
        else:
            color_key = 'inReg'
            gradient = color_gradients[color_key]
            color = gradient(0.2 + 0.8 * (1 - (i % reset_interval) / reset_interval))
            group_name = column.split()[0]
        
        line, = ax.plot(sorted_data, cdf, label=f'{column}', marker=markers[i % len(markers)], color=color, markersize=7, linestyle='-', linewidth=lw_cdf)
        
        if group_name not in legend_entries:
            legend_entries[group_name] = []
        if len(column.split()) > 1:
            legend_entries[group_name].append((line, column.split()[1]))
        elif '->' in column:
            legend_entries[group_name].append((line, column))
        else:
            legend_entries[group_name].append((line, ''))

    ax.set_title(f'CDF plot for {benchmark} test\n', fontsize=fs)
    ax.set_xlabel(f'{benchmark}', fontsize=fs)
    ax.set_ylabel('CDF', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.grid(True)

    handles = []
    labels = []
    named_handles = []
    named_labels = []
    for key, lines in legend_entries.items():
        if key is not None:
            # Add group heading
            named_handles.append(plt.Line2D([0], [0], color='none', label=key, linestyle=''))
            named_labels.append(key)
        for line, label in lines:
            if key is None:
                handles.append(line)
                labels.append(label)
            else:
                named_handles.append(line)
                named_labels.append(label)
    
    handles.extend(named_handles)
    labels.extend(named_labels)
    ax.legend(handles, labels, fontsize=fs * 0.8, loc='upper left')

def load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark):
    BC_PCA_PUB_sum_data = None
    for sample in range(1, samples + 1):
        BC_PCA_PUB_data = f'/data0/funato/0_out/99_out/{test_loc}/BC-PCA-lasso_PUB_random_distributed_{reservoir_size}_{ridge_param}{non_arid_buf}/Train{train_basin_int}/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train{train_basin_int}_sample{sample}_rev_PC{pc}_eva.csv'
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

for benchmark in benchmark_list:
    plot_df_train = pd.DataFrame()
    plot_df_test = pd.DataFrame()
    plot_df_test_cdf = pd.DataFrame()

    train_data = pd.read_csv(f'{train_basin_dir}/BC-PCA-lasso_results_r{r2r_reservoir_size}_s1_sr0.4_rr{ridge_param}_eva.csv', index_col=0)
    train_data = train_data[f'BC_og_r{r2r_reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    plot_df_train = pd.concat([plot_df_train, train_data.rename(f'{train_loc}')], axis=1)
    for pc in range(1,n_components+1):
        if pc == n_components:
            for train_basin_int in train_basin_int_list:
                BC_PCA_lasso_PUB_column_test_data = load_BC_PCA_lasso_PUB_data(pc, train_basin_int, benchmark)
                plot_df_test_cdf = pd.concat([plot_df_test_cdf, pd.DataFrame({
                    f'BcReg {train_basin_int}': BC_PCA_lasso_PUB_column_test_data.values.flatten()
                }, index=test_basins_list)], axis=1)
            test_data = pd.read_csv(f'{test_basin_dir}/BC-PCA-lasso_PUB_results_r{r2r_reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_eva.csv', index_col = 0)
            test_data = test_data[f'BC-PCA-lasso_r{r2r_reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']

            plot_df_test = pd.concat([plot_df_test, test_data.rename(f'{test_loc} PC{pc}')], axis=1)
            color_list = 'gainsboro'

            plot_df_test_cdf = pd.concat([plot_df_test_cdf, test_data.rename(f'{train_loc}->{test_loc}')], axis=1)



    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    # box Plot
    # Plot for train data
    plot_df_train.boxplot(ax=axes[0], boxprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'))
    axes[0].set_title(f'Train Data {train_loc} - {benchmark}', fontsize=fs)
    axes[0].set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
    axes[0].set_ylabel(f'{benchmark}', fontsize=fs)
    axes[0].set_xlabel('Training', fontsize=fs)
    axes[0].tick_params(axis='both', which='major', labelsize=fs)
    axes[0].grid(True)

    # Plot for test data
    plot_df_test.boxplot(ax=axes[1], boxprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'))
    axes[1].set_title(f'Test Data {test_loc} - {benchmark}', fontsize=fs)
    axes[1].set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
    axes[1].set_xlabel('Testing', fontsize=fs)
    axes[1].tick_params(axis='both', which='major', labelsize=fs)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/box_compact/{benchmark}_boxplot.png')
    plt.close()

    # Remove NaN values for each column in plot_df_test_cdf
    plot_df_test_cdf = plot_df_test_cdf.dropna()

    # Create CDF plot
    fig_cdf, ax_cdf = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    create_cdf_plot(ax_cdf, plot_df_test_cdf, benchmark)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
    fig_cdf.tight_layout()
    plt.savefig(f'{output_dir}/cdf/{benchmark}_cdfplot.png')
    plt.close()

print(f'{output_dir}/box_compact/{benchmark}_boxplot.png')
print(f'{output_dir}/cdf/{benchmark}_cdfplot.png')

