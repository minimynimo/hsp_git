import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

n_components = 3
ver = 2
#loc = "JP"
#loc = "US"
loc = "GB"
fs = 22
# loc = "US"

test_y_ticks = True

ver_name = "ver1_1" if ver == 1 else "ver2_0"

samples = 100

non_arid_files = True
if loc != "US":
    non_arid_files = False

step_list = [8,15]

ridge_param = 1.0
if loc == "JP":
    file_tot_num = 87
    reservoir_size = 200
    train_basin_int = 15
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    region_list = ['Hokkaido', 'Tohoku', 'Kanto', 'Tokai-Kinki', 'Shikoku-Kyusyu']
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/pub_region_list_{ver_name}.csv')
elif loc == "US":
    file_tot_num = 667
    reservoir_size = 200
    train_basin_int = 75
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
    region_list = ['EastCoast_N','EastCoast_C', 'EastCoast_S', 'Inland_N', 'Inland_S', 'WestCoast_N', 'WestCoast_S']
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/file_num_region.csv')
elif loc == "GB":
    file_tot_num = 396
    reservoir_size = 300
    train_basin_int = 60
    test_basins_list = list(range(4, file_tot_num, 10))
    region_list = ["Northernmost", "NorthMid", "NorthEdge", "SouthEdge", "SouthMid", "Southernmost"]
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/GB/0_data/river_basin/dataset_GB/file_num_valid_small.csv')

if non_arid_files:
    non_arid_buf = "_non_arid"
else:
    non_arid_buf = ""

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

benchmark_limits = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},    
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

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

def load_BC_PUB_region_data(region, benchmark, train_test):
    BC_PUB_region_data = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC_PUB_{reservoir_size}_{ridge_param}{non_arid_buf}/region/{region}/{train_test}_basin/results/BC_PUB_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_eva.csv')
    BC_PUB_region_data = BC_PUB_region_data.sort_values(by='file_num')
    BC_PUB_region_data = BC_PUB_region_data[f'BC_PUB_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    return BC_PUB_region_data

def load_BC_PCA_lasso_PUB_region_data(pc, region, benchmark, train_test):
    BC_PCA_PUB_region_data = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}{non_arid_buf}/region/{region}/{train_test}_basin/results/BC-PCA-lasso_PUB_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_eva.csv')
    BC_PCA_PUB_region_data = BC_PCA_PUB_region_data.sort_values(by='file_num')
    BC_PCA_PUB_region_data = BC_PCA_PUB_region_data[f'BC-PCA-lasso_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    return BC_PCA_PUB_region_data

for pc_n in range(1, n_components+1):
    output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/PUB_distribute_regional_{reservoir_size}_{ridge_param}{non_arid_buf}/box/PC{pc_n}'
    os.makedirs(output_dir, exist_ok=True)
    for benchmark in benchmark_list:
        plot_df_train = pd.DataFrame()
        plot_df_test = pd.DataFrame()

        train_num = []
        test_num = []
        tick_label_train = []
        tick_label_test = []
        color_list_train = []
        color_list_test = []

        # add BCPUB data for distributed
        BC_PUB_data = load_BC_PUB_data(train_basin_int, benchmark)
        plot_df_test = pd.concat([plot_df_test, BC_PUB_data], axis=1)
        tick_label_test.append(f'Distributed')
        test_num.append(train_basin_int)
        color_list_test.append('#9982ab')
        # add BCPUB data for each region
        for region in region_list:
            BC_PUB_region_data_train = load_BC_PUB_region_data(region, benchmark, 'train')
            plot_df_train = pd.concat([plot_df_train, BC_PUB_region_data_train], axis=1)
            tick_label_train.append(f'{region}')
            train_num.append(len(BC_PUB_region_data_train))
            color_list_train.append('#D0B8E6')

            BC_PUB_region_data_test = load_BC_PUB_region_data(region, benchmark, 'test')
            plot_df_test = pd.concat([plot_df_test, BC_PUB_region_data_test], axis=1)
            tick_label_test.append(f'{region}')
            test_num.append(len(BC_PUB_region_data_train))
            color_list_test.append('#D0B8E6')

        # add BCPCA data for distributed
        BC_PCA_PUB_data = load_BC_PCA_lasso_PUB_data(pc_n, train_basin_int, benchmark)
        plot_df_test = pd.concat([plot_df_test, BC_PCA_PUB_data], axis=1)
        tick_label_test.append(f'Distributed')
        test_num.append(train_basin_int)
        color_list_test.append('#C7625E')
        # add BCPCA data for each region
        for region in region_list:
            BC_PCA_PUB_region_data_train = load_BC_PCA_lasso_PUB_region_data(pc_n, region, benchmark, 'train')
            plot_df_train = pd.concat([plot_df_train, BC_PCA_PUB_region_data_train], axis=1)
            tick_label_train.append(f'{region}')
            train_num.append(len(BC_PCA_PUB_region_data_train))
            color_list_train.append('#E8B4A9')
            BC_PCA_PUB_region_data_test = load_BC_PCA_lasso_PUB_region_data(pc_n, region, benchmark, 'test')
            plot_df_test = pd.concat([plot_df_test, BC_PCA_PUB_region_data_test], axis=1)
            tick_label_test.append(f'{region}')
            test_num.append(len(BC_PCA_PUB_region_data_train))
            color_list_test.append('#E8B4A9')

        ### PLOT
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))

        # Box plot for calibration data
        plot_df_train.columns = tick_label_train
        boxplot_train = plot_df_train.boxplot(ax=axes[0], patch_artist=True, return_type='dict')
        axes[0].set_title(f'Training Basins\n', fontsize=fs)
        #axes[0].set_xlabel('Region', fontsize=fs)
        axes[0].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_train, train_num)], rotation=90, fontsize=fs)
        axes[0].set_ylabel(f'{benchmark}', fontsize=fs)
        axes[0].tick_params(axis='y', labelsize=fs)
        
        # Box plot for evaluation data
        plot_df_test.columns = tick_label_test
        boxplot_test = plot_df_test.boxplot(ax=axes[1], patch_artist=True, return_type='dict')
        axes[1].set_title(f'Test Basins\n', fontsize = fs)
        #axes[1].set_xlabel('Region', fontsize=fs)
        axes[1].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_test, test_num)], rotation=90, fontsize=fs)
        if test_y_ticks:
            axes[1].set_ylabel(f'{benchmark}' , fontsize=fs)
            axes[1].tick_params(axis='y', labelsize=fs)
        else:
            axes[1].tick_params(axis='y', labelsize=0)

        if test_y_ticks:
            plt.subplots_adjust(bottom=0.39, wspace=0.25)  # Increase bottom margin and space between columns
        else:
            plt.subplots_adjust(bottom=0.4, wspace=0.07)  # Increase bottom margin and space between columns

        axes[0].tick_params(axis='x', labelsize=fs, pad=10)  # Adjust label size and padding
        axes[1].tick_params(axis='x', labelsize=fs, pad=10)


        axes[0].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
        axes[1].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

        # Ensure color_list dynamically matches the number of boxes
        for patch, color in zip(boxplot_train['boxes'], color_list_train):
            patch.set_facecolor(color)

        for patch, color in zip(boxplot_test['boxes'], color_list_test):
            patch.set_facecolor(color)

        # Change the median line color to black
        for median in boxplot_train['medians']:
            median.set_color('black')
        for median in boxplot_test['medians']:
            median.set_color('black')

        # Add horizontal brackets
        def add_bracket(ax, text, start, end, height, y_offset):
            ax.annotate('', xy=(start, height), xytext=(end, height), 
                        arrowprops={'arrowstyle': '-', 'color': 'black', 'linewidth': 1.5})
            ax.text((start + end) * 0.5, height + y_offset, text, ha='center', va='bottom', fontsize=fs)

        if benchmark == 'KGE' or benchmark == 'NSE':
            height = -2.6
        elif benchmark == 'E1':
            height = -4.3
        elif benchmark == 'RMSE':
            height = -7.8

        # Add brackets for train plot
        add_bracket(axes[0], 'BcProx', 0, len(region_list) + 1, height, 0.05)
        add_bracket(axes[0], 'BcReg', len(region_list) + 1, len(tick_label_train), height, 0.05)

        # Add brackets for test plot
        add_bracket(axes[1], 'BcProx', 0, len(region_list) + 2, height, 0.05)
        add_bracket(axes[1], 'BcReg', len(region_list) + 2, len(tick_label_test), height, 0.05)


        #plt.tight_layout()
        plt.savefig(f'{output_dir}/{benchmark}_box_regions_PC{pc_n}.jpg')
        plt.close()

    print(f'PC{pc_n} is done')
print("DONE")