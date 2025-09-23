# for an input file, read it and for each column, return the median value
import pandas as pd


input_file = "/data0/funato/hsp/0_out/BC/results/BC_results_cal.csv"

input_df = pd.read_csv(input_file)
input_df.set_index('file_num', inplace=True)

for col in input_df.columns:
    median_value = input_df[col].median()
    print(f"Median value for {col}: {median_value}")
