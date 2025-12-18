import pandas as pd
import matplotlib.pyplot as plt
import os

n_components = 4
fs = 13

train_loc = 'US'
#train_loc = 'GB'
#train_loc = 'AUS'
#train_loc = 'JP'

#test_loc = 'US'
test_loc = 'GB'
#test_loc = 'AUS'
#test_loc = 'BR'
#test_loc = 'JP'

reservoir_size = 200
ridge_param = 1.0


train_basin_dir = f'/data0/funato/0_out/99_out/{train_loc}to{test_loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/train_basin/results'
test_basin_dir = f'/data0/funato/0_out/99_out/{train_loc}to{test_loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/test_basin/results'

output_dir = f'/data0/funato/0_out/0_fig/{train_loc}to{test_loc}/benchmark/BC-PCA-lasso_PUB/box'
os.makedirs(output_dir, exist_ok=True)


benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

benchmark_limits_norm = {
    'KGE': {'min': -5, 'max': 1},
    'NSE': {'min': -5, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -5, 'max': 1},
    'VE': {'min': -5, 'max': 1},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 10}
}

for benchmark in benchmark_list:
    plot_df_train = pd.DataFrame()
    plot_df_test = pd.DataFrame()

    train_data = pd.read_csv(f'{train_basin_dir}/BC-PCA-lasso_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_eva.csv', index_col=0)
    train_data = train_data[f'BC_og_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    plot_df_train = pd.concat([plot_df_train, train_data.rename(f'{train_loc}')], axis=1)
    for pc in range(1,n_components):
        #train_data = pd.read_csv(f'{train_basin_dir}/BC-PCA-lasso_PUB_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_eva.csv', index_col=0)
        test_data = pd.read_csv(f'{test_basin_dir}/BC-PCA-lasso_PUB_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_eva.csv', index_col = 0)

        #train_data = train_data[f'BC-PCA-lasso_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
        test_data = test_data[f'BC-PCA-lasso_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']

        #plot_df_train = pd.concat([plot_df_train, train_data.rename(f'{train_loc} PC{pc}')], axis=1)
        plot_df_test = pd.concat([plot_df_test, test_data.rename(f'{test_loc} PC{pc}')], axis=1)
        color_list = 'gainsboro'

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    # Plot for train data
    plot_df_train.boxplot(ax=axes[0])
    axes[0].set_title(f'Train Data {train_loc} - {benchmark}', fontsize=fs)
    axes[0].set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
    axes[0].set_ylabel(f'{benchmark}', fontsize=fs)
    axes[0].set_xlabel('Training', fontsize=fs)
    axes[0].tick_params(axis='both', which='major', labelsize=fs)
    axes[0].grid(True)

    # Plot for test data
    plot_df_test.boxplot(ax=axes[1])
    axes[1].set_title(f'Test Data {test_loc} - {benchmark}', fontsize=fs)
    axes[1].set_ylim(benchmark_limits_norm[benchmark]['min'], benchmark_limits_norm[benchmark]['max'])
    axes[1].set_xlabel('Testing', fontsize=fs)
    axes[1].tick_params(axis='both', which='major', labelsize=fs)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{benchmark}_boxplot.png')
    plt.close()

print(f'{output_dir}/{benchmark}_boxplot.png')

