import pandas as pd

samples = 100

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


BC_PUB_sum_data = None
for sample in range(1, samples + 1):
    #BC_PUB_data = f"/data0/funato/0_out/99_out/JP/BC-PCA-lasso_PUB_random_distributed_200_1.0/Train30/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train30_sample{sample}_rev_PC3_eva.csv"
    #BC_PUB_data = f"/data0/funato/0_out/99_out/GB/BC-PCA-lasso_PUB_random_distributed_300_1.0/Train130/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train130_sample{sample}_rev_PC3_eva.csv"
    BC_PUB_data = f"/data0/funato/0_out/99_out/US/BC-PCA-lasso_PUB_random_distributed_200_1.0_non_arid/Train200/test_basin/results/sample{sample}/BC-PCA-lasso_PUB_results_Train200_sample{sample}_rev_PC3_eva.csv"

    BC_PUB_df = pd.read_csv(BC_PUB_data, index_col=0, header=0)
    BC_PUB_df = BC_PUB_df.sort_values(by='file_num')
    BC_PUB_df = pd.DataFrame(BC_PUB_df.values)

    if BC_PUB_sum_data is None:
        BC_PUB_sum_data = BC_PUB_df.copy()
    else:
        BC_PUB_sum_data = BC_PUB_sum_data.add(BC_PUB_df, fill_value=0)

# Print the header columns of BC_PUB_data
print(f"Header columns for path {BC_PUB_data}:")
print(BC_PUB_df.columns)

BC_PUB_mean_data = BC_PUB_sum_data.copy()
BC_PUB_mean_data = BC_PUB_mean_data.divide(samples)

BC_PUB_mean_data.columns = benchmark_list
BC_PUB_mean_data.index.name = 'file_num'

print(BC_PUB_mean_data)

# Calculate the median values for each column
median_values = BC_PUB_mean_data.median()

# Print the median values
print(f"Median values for path {BC_PUB_data}:")
print(median_values)
print("\n")
