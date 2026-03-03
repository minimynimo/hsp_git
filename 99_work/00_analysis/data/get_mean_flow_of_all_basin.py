# get mean flow of all basin

import os
import pandas as pd
import numpy as np

loc, file_tot_num = "US", 614
#loc, file_tot_num = "GB", 424

input_data_dir = f"/data0/funato/hsp2/2_MERV/varssim_{loc}"
output_dir = f"/data0/funato/hsp2/0_out/analysis/{loc}/data"
os.makedirs(output_dir, exist_ok=True)

all_precip = []
all_pet = []
all_temp = []
all_flow = []

for file_num in range(1, file_tot_num + 1):
    file_name = f"{input_data_dir}/varssim{file_num:03d}.csv"
    if not os.path.exists(file_name):
        continue
    
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate mean flow
    mean_flow = df['Obs flow'].mean()

    all_precip.append(df['Precip'])
    all_pet.append(df['PET'])
    all_temp.append(df['Temp'])
    all_flow.append(df['Obs flow'])

    # Calculate mean and standard deviation for each basin
    total_mean_precip = [series.mean() for series in all_precip]

    print(f"Processed file {file_num}: Mean Flow = {mean_flow}")

# Calculate overall mean flow
overall_mean_flow = np.mean([series.mean() for series in all_flow])
# get the 10th and 90th percentiles
mean_flow_10th = np.percentile([series.mean() for series in all_flow], 10)
mean_flow_90th = np.percentile([series.mean() for series in all_flow], 90)
#standard deviation
std_dev_mean_flow = np.std([series.mean() for series in all_flow])
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

# save mean and standard deviation for all basins for precip, pet, temp, flow 
output_file_stats = f"{output_dir}/mean_stdev_precip_pet_temp_flow_all_basins.csv"

stats_df = pd.DataFrame({
    'Mean Precip': [np.mean([series.mean() for series in all_precip])],
    'StdDev Precip': [np.std([series.mean() for series in all_precip])],
    'Mean PET': [np.mean([series.mean() for series in all_pet])],
    'StdDev PET': [np.std([series.mean() for series in all_pet])],
    'Mean Temp': [np.mean([series.mean() for series in all_temp])],
    'StdDev Temp': [np.std([series.mean() for series in all_temp])],
    'Mean Flow': [overall_mean_flow],
    'StdDev Flow': [std_dev_mean_flow]
})
stats_df.to_csv(output_file_stats, index=False)   
print(f"Statistics saved to {output_file_stats}")