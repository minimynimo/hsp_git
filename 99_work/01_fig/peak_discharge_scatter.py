#General increase in peak flow with increased precipitation, and how the different rate scenarios shift this relationship
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

model_name = "HBV"

input_data = "/data0/funato/hsp/0_out/10_HBV/peak_discharge/peak_discharge_comparison_HBV.csv"

output_fig_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/scatter/"
os.makedirs(output_fig_dir, exist_ok=True)

input_df = pd.read_csv(input_data, header=0, index_col=0)

# Relationship between original precipitation and peak discharge for different scenarios
flow_columns_to_plot = {
    'Original Simulation': 'peak_flow_og',
    'Peak 10%': 'peak_flow_1_rate10', # This will be the actual column name after replace
    'Peak 30%': 'peak_flow_1_rate30',
    'Peak 50%': 'peak_flow_1_rate50',
}

# Explanation of the figure
# Shows the relationship between the original annual peak precipitation and the simulated annual peak flow for the 424 basins.
# Each point represents a year, and the regression line shows the trend for each scenario.
# The x-axis represents the original annual peak precipitation (mm/day), and the y-axis represents the simulated annual peak flow (m³/s).
# The band of variability is shown by the shaded area around the regression line, indicating the uncertainty in the relationship.

flow_columns_to_plot_actual = {
    'Original Simulation': 'peak_flow_og',
    'Peak1_rate10': 'peak_flow_peak1_rate10',
    'Peak1_rate30': 'peak_flow_peak1_rate30',
    'Peak1_rate50': 'peak_flow_peak1_rate50',
    'Peak3_rate10': 'peak_flow_peak3_rate10',
    'Peak3_rate30': 'peak_flow_peak3_rate30',
    'Peak3_rate50': 'peak_flow_peak3_rate50',
    'Peak5_rate10': 'peak_flow_peak5_rate10',
    'Peak5_rate30': 'peak_flow_peak5_rate30',
    'Peak5_rate50': 'peak_flow_peak5_rate50',
}


plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")

for label, col_name in flow_columns_to_plot_actual.items():
    if col_name in input_df.columns:
        sns.regplot(
            data=input_df,
            x='max_precip_obs', # Use original precipitation as the common x-axis
            y=col_name,
            scatter_kws={'alpha':0.2, 's':10}, # Make points smaller and transparent
            line_kws={'lw': 2},
            label=label,
            ax=plt.gca() # Use current axes
        )

plt.xlabel('Original Annual Peak Precipitation (mm/day)')
plt.ylabel('Simulated Annual Peak Flow (m³/s)')
plt.title(f'Relationship between Peak Precipitation and Simulated Peak Flow ({model_name} Model)')
plt.legend(title='Simulation Scenario')
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, f"precip_vs_peak_flow_all_scenarios_{model_name}.png"))
plt.show()
plt.close()

print(f"Relationship plot saved to: {os.path.join(output_fig_dir, f'precip_vs_peak_flow_all_scenarios_{model_name}.png')}")