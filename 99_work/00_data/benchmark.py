import numpy as np
import pandas as pd
import os

input_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"

file_tot_num = 4

output_dir = "/data0/funato/hsp/0_out/10_HBV"
os.makedirs(output_dir, exist_ok=True)

for file_num in range(1, file_tot_num + 1):
    print(f"Processing file: {file_num}")
    input_file = f"{input_dir}/varssim{file_num:03d}.csv"
    input_df = pd.read_csv(input_file, index_col=0, header=0)
    input_df = input_df.T

    # Save the DataFrame to a new CSV file
    output_file = f"{output_dir}/varssim{file_num:03d}_all_Q_outputs.csv"
    input_df.to_csv(output_file, index=True)
    
    print(f"Saved processed data to {output_file}")
