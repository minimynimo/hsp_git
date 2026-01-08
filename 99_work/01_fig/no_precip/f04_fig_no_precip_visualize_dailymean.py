import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

fs = 20

# Configuration
model_list = ["HBV", "HYPER","RF", "XGB", "TCN", "LSTM"]  # Can be expanded if needed
input_dir = "/data0/funato/hsp/0_out/analysis/no_precip_discharge"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/no_precip_discharge_dailymean_use"

# Ensure output directory exists
os.makedirs(output_fig_dir, exist_ok=True)

color_list = {
    "HBV": "#7B7B7B",    # Kept as requested (Medium Grey)
    "HYPER": "#628AE8",     # Kept as requested (Steel Blue)
    "RF":   "#FF7F00",   # Red
    "XGB":  "#9C4803",   # Bright Orange
    "TCN":  "#E41A1C",   # Reddish Purple / Magenta
    "LSTM": "#9808B2"    # Pink
}

# Load the data
print("Loading data...")
summary_csv = f"{input_dir}/no_precip_discharge_summary.csv"

# Check if files exist
if not os.path.exists(summary_csv):
    print(f"Error: {summary_csv} not found!")
    print("Please run no_precip_calc.py first to generate the data.")
    exit(1)

# Load all discharge data
all_discharge_df = pd.read_csv(summary_csv)
print(f"Loaded {len(all_discharge_df)} discharge data points")

all_discharge_df['Model'] = all_discharge_df['Model'].replace('BC', 'HYPER')  # Change BC to HYPER

# Load summary statistics
if os.path.exists(summary_csv):
    results_df = pd.read_csv(summary_csv)
    print(f"Loaded summary statistics for {len(results_df)} files")

# Prepare data for plotting
all_discharge_data = {}
for model in model_list:
    model_data = all_discharge_df[all_discharge_df['Model'] == model]['Mean_Discharge'].values
    all_discharge_data[model] = model_data
    print(f"{model}: {len(model_data)} data points")

# Create comprehensive visualizations
print("Creating comprehensive visualizations...")

# Prepare valid models list
valid_models = [model for model in model_list if len(all_discharge_data[model]) > 0]
box_data = [all_discharge_data[model] for model in valid_models]

# print the dimension of each model's data
for model in valid_models:
    print(f"{model}: {len(all_discharge_data[model])} data points")


# 7. Focused view - Box plot with limited y-axis
max_for_focus = np.percentile(np.concatenate(box_data), 95)  # 95th percentile
plt.figure(figsize=(8, 6))
# showmeans=True will plot the mean; style it as a '+'

bp3 = plt.boxplot(
    box_data,
    tick_labels=valid_models,
    patch_artist=True,
    showmeans=True,
    meanprops={
        'marker': 'x',
        'markeredgecolor': 'green',
        'markeredgewidth': 2.5,
        'markersize': 15,
    },
)
for patch, model in zip(bp3['boxes'], valid_models):
    patch.set_facecolor(color_list[model])
# Set median line colors to black
for median in bp3['medians']:
    median.set_color('black')
    median.set_linewidth(2)
#plt.title('Daily Mean Discharge', fontsize=fs)
plt.ylabel('Daily Mean Discharge [m³/s]', fontsize=fs)
plt.xlabel('Model', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.ylim(-0.1, max_for_focus)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_fig_dir}/f04_no_precip_boxplot_focused.png", dpi=300, bbox_inches='tight')
plt.close()

print("saved to", f"{output_fig_dir}/f04_no_precip_boxplot_focused.png")

# 9. Focused CDF
plt.figure(figsize=(9, 8))
for model in valid_models:
    data = np.array(all_discharge_data[model])
    data_sorted = np.sort(data)
    p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    plt.plot(data_sorted, p, label=model, color=color_list[model], linewidth=3)
plt.title('Cumulative Distribution of Daily Mean Discharge', fontsize=fs)
plt.xlabel('Daily Mean Discharge [m³/s]', fontsize=fs)
plt.ylabel('CDF', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.xlim(-0.2, 8)
plt.ylim(0, 1.0) 
plt.legend(fontsize=fs)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_fig_dir}/no_precip_cdf_focused.png", dpi=300, bbox_inches='tight')
plt.close()

print("saved to", f"{output_fig_dir}/no_precip_cdf_focused.png")

# 10. Statistical summary
model_stats = []
for model in valid_models:
    data = np.array(all_discharge_data[model])
    stats_dict = {
        'Model': model,
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Q25': np.percentile(data, 25),
        'Q75': np.percentile(data, 75),
        'Q90': np.percentile(data, 90),
        'Q95': np.percentile(data, 95)
    }
    model_stats.append(stats_dict)

stats_df = pd.DataFrame(model_stats)

# Save statistics
stats_df.to_csv(f"{output_fig_dir}/no_precip_statistics_summary.csv", index=False)

print("All plots saved successfully!")
print("Generated files:")
print("- f04_no_precip_boxplot_focused.png")
print("- no_precip_cdf_focused.png")
print("- no_precip_statistics_summary.csv")

