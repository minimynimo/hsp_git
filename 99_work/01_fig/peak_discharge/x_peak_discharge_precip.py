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

sim_discharge_cols = ['peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                    'peak3_rate0','peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                    'peak5_rate0','peak5_rate10', 'peak5_rate30', 'peak5_rate50']

start_date_eva = "2001-01-01"
end_date_eva = "2001-02-10"
# Select a representative file for an example plot
# You might choose a file_num that has a median 'peak_flow_og' or based on visual inspection.
file_tot_num = 424 

example_df = comparison_df
if not example_df.empty:
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Plotting Peak Flows
    plt.scatter(example_df['peak_precip_obs'].values, example_df['peak_flow_obs'].values, marker='o', color='blue', label='Observed Peak Flow')
    plt.scatter(example_df['peak_precip_obs'].values, example_df['sim_peak_flow_peak1_rate0'].values, marker='x', color='green', label='Sim (Original Precip)')
    plt.scatter(example_df['sim_peak_precip_0'].values, example_df['sim_peak_flow_peak1_rate0'].values, marker='+', color='cyan', label='Sim (Peak1_rate0 Precip)')
    plt.scatter(example_df['sim_peak_precip_10'].values, example_df['sim_peak_flow_peak1_rate10'].values, marker='^',  color='orange', label='Sim (Peak1_rate10 Precip)')
    plt.scatter(example_df['sim_peak_precip_30'].values, example_df['sim_peak_flow_peak1_rate30'].values, marker='s', color='red', label='Sim (Peak1_rate30 Precip)')
    plt.scatter(example_df['sim_peak_precip_50'].values, example_df['sim_peak_flow_peak1_rate50'].values, marker='d', color='purple', label='Sim (Peak1_rate50 Precip)')

    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Annual Peak Flow (m³/s)')
    plt.title(f'Annual Peak Flow Comparison for  ({model_name} Model) - Evaluation Period')
    plt.legend(title='Scenario')
    plt.ylim([0, 400])  # Adjust y-limits for better visibility
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"example_annual_peak_flow_file.png"))
    plt.close()
    print(f"Example annual peak flow plot saved to: {os.path.join(output_dir, f'example_annual_peak_flow_file.png')}")

    # Plotting Peak Lags
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Prepare data for boxplot
    lag_data = [
        example_df['sim_peak_lag_peak1_rate0'].values,
        example_df['sim_peak_lag_peak1_rate10'].values,
        example_df['sim_peak_lag_peak1_rate30'].values,
        example_df['sim_peak_lag_peak1_rate50'].values,
        example_df['sim_peak_lag_peak3_rate0'].values,
        example_df['sim_peak_lag_peak3_rate10'].values,
        example_df['sim_peak_lag_peak3_rate30'].values,
        example_df['sim_peak_lag_peak3_rate50'].values,
        example_df['sim_peak_lag_peak5_rate0'].values,
        example_df['sim_peak_lag_peak5_rate10'].values,
        example_df['sim_peak_lag_peak5_rate30'].values,
        example_df['sim_peak_lag_peak5_rate50'].values

    ]
    label_list = [
        'Peak1_rate0', 'Peak1_rate10', 'Peak1_rate30', 'Peak1_rate50',
        'Peak3_rate0', 'Peak3_rate10', 'Peak3_rate30', 'Peak3_rate50',
        'Peak5_rate0', 'Peak5_rate10', 'Peak5_rate30', 'Peak5_rate50'
    ]

    plt.boxplot(lag_data, labels=label_list)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)  # Reference line for zero lag

    plt.ylabel('Peak Flow Lag (Days relative to Observed Peak Flow)')
    plt.title(f'Annual Peak Flow Lag Comparison for  ({model_name} Model) - Evaluation Period')
    # Remove plt.legend(title='Scenario') for boxplot
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"example_annual_peak_lag_file.png"))
    plt.close()
    print(f"Example annual peak lag plot saved to: {os.path.join(output_dir, f'example_annual_peak_lag_file.png')}")

else:
    print(f"No data for  in the evaluation period.")