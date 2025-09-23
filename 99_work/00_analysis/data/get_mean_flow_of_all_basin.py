# get mean flow of all basin

import os
import pandas as pd
import numpy as np

input_data_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
output_dir = "/data0/funato/hsp/0_out/analysis/data"
os.makedirs(output_dir, exist_ok=True)
filt_tot_num = 424

total_mean_flow = []
for file_num in range(1, filt_tot_num + 1):
    file_name = f"{input_data_dir}/varssim{file_num:03d}.csv"
    if not os.path.exists(file_name):
        continue
    
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate mean flow
    mean_flow = df['Obs flow'].mean()

    total_mean_flow.append(mean_flow)

    print(f"Processed file {file_num}: Mean Flow = {mean_flow}")

# Calculate overall mean flow
overall_mean_flow = np.mean(total_mean_flow)
# get the 10th and 90th percentiles
mean_flow_10th = np.percentile(total_mean_flow, 10)
mean_flow_90th = np.percentile(total_mean_flow, 90)
#standard deviation
std_dev_mean_flow = np.std(total_mean_flow)
print(f"Overall Mean Flow (all basins) = {overall_mean_flow}")
print(f"10th Percentile Mean Flow = {mean_flow_10th}")
print(f"90th Percentile Mean Flow = {mean_flow_90th}")
print(f"Standard Deviation of Mean Flow = {std_dev_mean_flow}")

# stort the results of mean, 10th and 90th percentiles, and standard deviation
output_file = f"{output_dir}/mean_flow_all_basins.csv"
mean_flow_df = pd.DataFrame({
    'Overall Mean Flow': [overall_mean_flow],
    '10th Percentile Mean Flow': [mean_flow_10th],
    '90th Percentile Mean Flow': [mean_flow_90th],
    'Standard Deviation of Mean Flow': [std_dev_mean_flow]
})
mean_flow_df.to_csv(output_file, index=False)   

print(f"Results saved to {output_file}")