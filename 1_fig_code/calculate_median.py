import pandas as pd

# List of paths to the CSV files
csv_paths = [
    "/data0/funato/0_out/99_out/JP/BC_hp/spin1/reservoir200/BC_r200_s1_sr0.4_rr1.0_is0.5_a0.1/results/BC_results_r200_s1_sr0.4_rr1.0_bma_eva.csv",
    "/data0/funato/0_out/99_out/JP/BC_hp/spin1/reservoir500/BC_r500_s1_sr0.4_rr1.0_is0.5_a0.1/results/BC_results_r500_s1_sr0.4_rr1.0_bma_eva.csv",
    "/data0/funato/0_out/99_out/UStoGB/BC-PCA-lasso_PUB_200_1.0/train_basin/results/BC-PCA-lasso_results_r200_s1_sr0.4_rr1.0_eva.csv",
    "/data0/funato/0_out/99_out/GBtoUS/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv",
    "/data0/funato/0_out/99_out/JPtoGB/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv"
]

for path in csv_paths:
    # Load the CSV file
    results_df = pd.read_csv(path)
    
    # Calculate the median for each column
    median_values = results_df.median()
    
    # Print the median values
    print(f"Median values for file {path}:")
    print(median_values)
    print("\n")
