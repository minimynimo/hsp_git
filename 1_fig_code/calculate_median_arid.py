import pandas as pd

# Paths to the CSV files
results_csv_path = "/data0/funato/0_out/99_out/US/BC_700_0.001/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv"
arid_file_list_path = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"

# Load the CSV files
results_df = pd.read_csv(results_csv_path)
arid_file_list = pd.read_csv(arid_file_list_path)['File_num'].tolist()

# Filter out the rows where file_num is in the arid file list
non_arid_results_df = results_df[~results_df['file_num'].isin(arid_file_list)]

# Calculate the median for each column
median_values = non_arid_results_df.median()

# Print the median values
print("Median values for non-arid files:")
print(median_values)
