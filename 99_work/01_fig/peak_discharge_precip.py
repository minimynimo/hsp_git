import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "HBV"  
input_data = "/data0/funato/hsp/0_out/10_HBV/peak_discharge/peak_discharge_comparison_HBV.csv"

output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/example/"
os.makedirs(output_dir, exist_ok=True)

comparison_df = pd.read_csv(input_data, header=0, index_col=None)

sim_discharge_cols = ['peak1_rate0'
                    'peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                    'peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                    'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

start_date_eva = "2001-01-01"
end_date_eva = "2006-12-31"
# Select a representative file for an example plot
# You might choose a file_num that has a median 'peak_flow_og' or based on visual inspection.
# For demonstration, we'll use file_num = 1.
example_file_num = 1 # Change this if you want to pick a different file

# Filter the comparison_df for the chosen file
example_df = comparison_df[comparison_df['file_num'] == example_file_num].copy()

if not example_df.empty:
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Plotting Peak Flows
    plt.plot(example_df['year'].values, example_df['peak_flow_obs'].values, marker='o', linestyle='-', color='blue', label='Observed Peak Flow')
    plt.plot(example_df['year'].values, example_df['peak_flow_og'].values, marker='x', linestyle='--', color='green', label='Simulated Flow (Original Precip)')
    plt.plot(example_df['year'].values, example_df['peak_flow_peak1_rate10'].values, marker='^', linestyle='-.', color='orange', label='Simulated Flow (Peak1_rate10 Precip)')
    plt.plot(example_df['year'].values, example_df['peak_flow_peak1_rate30'].values, marker='s', linestyle=':', color='red', label='Simulated Flow (Peak1_rate30 Precip)')
    plt.plot(example_df['year'].values, example_df['peak_flow_peak1_rate50'].values, marker='d', linestyle=':', color='purple', label='Simulated Flow (Peak1_rate50 Precip)')

    plt.xlabel('Year')
    plt.ylabel('Annual Peak Flow (m³/s)')
    plt.title(f'Annual Peak Flow Comparison for File {example_file_num} ({model_name} Model) - Evaluation Period')
    plt.legend(title='Scenario')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"example_annual_peak_flow_file_{example_file_num}.png"))
    plt.close()
    print(f"Example annual peak flow plot saved to: {os.path.join(output_dir, f'example_annual_peak_flow_file_{example_file_num}.png')}")

    # Plotting Peak Lags
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    plt.plot(example_df['year'].values, example_df['flow_lag_og'].values, marker='x', linestyle='--', color='green', label='Flow Lag (Original Precip)')
    plt.plot(example_df['year'].values, example_df['flow_lag_peak1_rate10'].values, marker='^', linestyle='-.', color='orange', label='Flow Lag (Peak1_rate10 Precip)')
    plt.plot(example_df['year'].values, example_df['flow_lag_peak1_rate30'].values, marker='s', linestyle=':', color='red', label='Flow Lag (Peak1_rate30 Precip)')
    plt.plot(example_df['year'].values, example_df['flow_lag_peak1_rate50'].values, marker='d', linestyle=':', color='purple', label='Flow Lag (Peak1_rate50 Precip)')
    
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Lag (Original Flow Peak Date)') # Reference line for zero lag
    
    plt.xlabel('Year')
    plt.ylabel('Peak Flow Lag (Days relative to Observed Peak Flow)')
    plt.title(f'Annual Peak Flow Lag Comparison for File {example_file_num} ({model_name} Model) - Evaluation Period')
    plt.legend(title='Scenario')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"example_annual_peak_lag_file_{example_file_num}.png"))
    plt.close()
    print(f"Example annual peak lag plot saved to: {os.path.join(output_dir, f'example_annual_peak_lag_file_{example_file_num}.png')}")

else:
    print(f"No data for example file {example_file_num} in the evaluation period.")