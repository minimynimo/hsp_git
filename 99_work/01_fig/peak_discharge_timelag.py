#General increase in peak flow with increased precipitation, and how the different rate scenarios shift this relationship
import matplotlib.pyplot as plt
import pandas as pd
import os

model_name = "HBV"

input_data = "/data0/funato/hsp/0_out/10_HBV/peak_discharge/peak_discharge_comparison_HBV.csv"

output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/time_lag/"
os.makedirs(output_dir, exist_ok=True)

comparison_df = pd.read_csv(input_data, header=0, index_col=None)

# Prepare data for lag plot
# We need to gather the lag columns into a 'long' format
lag_cols = {
    'Original': 'flow_lag_peak1_rate0',
    'Peak1_rate10': 'flow_lag_peak1_rate10',
    'Peak3_rate10': 'flow_lag_peak3_rate10',
    'Peak5_rate10': 'flow_lag_peak5_rate10',
    'Peak1_rate30': 'flow_lag_peak1_rate30',
    'Peak3_rate30': 'flow_lag_peak3_rate30',
    'Peak5_rate30': 'flow_lag_peak5_rate30',
    'Peak1_rate50': 'flow_lag_peak1_rate50',
    'Peak3_rate50': 'flow_lag_peak3_rate50',
    'Peak5_rate50': 'flow_lag_peak5_rate50',
}

# Create a melted DataFrame for plotting
lag_data_melted = []
for label, col_name in lag_cols.items():
    if col_name in comparison_df.columns:
        temp_df = comparison_df[['file_num', 'year', col_name]].copy()
        temp_df['Scenario'] = label
        temp_df = temp_df.rename(columns={col_name: 'Peak Flow Lag (Days)'})
        lag_data_melted.append(temp_df)

if lag_data_melted:
    lag_df = pd.concat(lag_data_melted)

    # Prepare data for boxplot: group by scenario
    scenario_order = lag_df['Scenario'].unique()
    data_to_plot = [lag_df.loc[lag_df['Scenario'] == scenario, 'Peak Flow Lag (Days)'].dropna() for scenario in scenario_order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data_to_plot,
        labels=scenario_order,
        notch=True
    )
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Lag') # Add a line at zero lag
    plt.xlabel('Simulation Scenario')
    plt.ylabel('Simulated Peak Flow Lag Relative to Observed Flow (Days)')
    plt.title(f'Distribution of Annual Peak Flow Lag by Scenario ({model_name} Model)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"peak_flow_lag_boxplot_{model_name}.png"))
    plt.show()
    plt.close()
    print(f"Peak flow lag box plot saved to: {os.path.join(output_dir, f'peak_flow_lag_boxplot_{model_name}.png')}")
else:
    print("No lag data available for plotting.")