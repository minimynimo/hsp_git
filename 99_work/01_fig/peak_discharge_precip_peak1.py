import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/example/"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424 

# Dictionary to store all model data
all_model_data = {}

# Precipitation rate labels and corresponding column names
rate_labels = [0, 10, 30, 50]
rate_cols = ['sim_peak_flow_peak1_rate0', 'sim_peak_flow_peak1_rate10', 
             'sim_peak_flow_peak1_rate30', 'sim_peak_flow_peak1_rate50']

# Load data for each model
for model in model_list:
    if model == "HBV":
        input_file = "/data0/funato/hsp/0_out/HBV/peak_discharge/peak_discharge_comparison_HBV.csv"
    else:
        input_file = f"/data0/funato/hsp/0_out/{model}/peak_discharge/peak_discharge_comparison_{model}.csv"
    
    try:
        df = pd.read_csv(input_file, header=0, index_col=None)
        
        # Extract peak discharge data for each rate
        model_data = {}
        for i, col in enumerate(rate_cols):
            if col in df.columns:
                model_data[rate_labels[i]] = df[col].dropna().values
            else:
                print(f"Warning: Column {col} not found in {model} data")
                model_data[rate_labels[i]] = np.array([])
        
        all_model_data[model] = model_data
        print(f"Loaded {model} data: {len(df)} records")
        
    except FileNotFoundError:
        print(f"Warning: Data file for {model} not found: {input_file}")
        continue

# Create the visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Peak Discharge vs Precipitation Rate\nMedian with 10th-90th Percentile Uncertainty', 
             fontsize=16, fontweight='bold')

# Define colors for each model
colors = {
    'HBV': '#1f77b4',
    'RF': '#ff7f0e', 
    'XGB': '#2ca02c',
    'TCN': '#d62728',
    'LSTM': '#9467bd'
}

# Plot each model in a separate subplot
for idx, model in enumerate(model_list):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if model not in all_model_data:
        ax.text(0.5, 0.5, f'{model}\nData not available', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax.set_title(f'{model}', fontsize=14, fontweight='bold')
        continue
    
    # Calculate statistics for each rate
    medians = []
    p10s = []
    p90s = []
    
    for rate in rate_labels:
        data = all_model_data[model][rate]
        if len(data) > 0:
            medians.append(np.median(data))
            p10s.append(np.percentile(data, 10))
            p90s.append(np.percentile(data, 90))
        else:
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)
    
    # Plot median line
    ax.plot(rate_labels, medians, 'o-', color=colors[model], 
            linewidth=2, markersize=8, label='Median', zorder=3)
    
    # Plot uncertainty band (10th-90th percentile)
    ax.fill_between(rate_labels, p10s, p90s, color=colors[model], 
                    alpha=0.3, label='10th-90th percentile', zorder=1)
    
    # Plot percentile lines
    ax.plot(rate_labels, p10s, '--', color=colors[model], 
            alpha=0.7, linewidth=1, zorder=2)
    ax.plot(rate_labels, p90s, '--', color=colors[model], 
            alpha=0.7, linewidth=1, zorder=2)
    
    # Formatting
    ax.set_xlabel('Precipitation Rate (%)', fontsize=12)
    ax.set_ylabel('Peak Discharge (mm/day)', fontsize=12)
    ax.set_title(f'{model}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set x-axis ticks
    ax.set_xticks(rate_labels)
    ax.set_xticklabels(rate_labels)

# Remove the empty subplot
axes[1, 2].remove()

# Adjust layout
plt.tight_layout()

# Save the figure
output_file = os.path.join(output_dir, "peak_discharge_vs_precip_rate_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPeak discharge comparison plot saved to: {output_file}")

# Create a combined comparison plot
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Plot all models on the same plot
for model in model_list:
    if model not in all_model_data:
        continue
        
    # Calculate statistics for each rate
    medians = []
    p10s = []
    p90s = []
    
    for rate in rate_labels:
        data = all_model_data[model][rate]
        if len(data) > 0:
            medians.append(np.median(data))
            p10s.append(np.percentile(data, 10))
            p90s.append(np.percentile(data, 90))
        else:
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)
    
    # Plot median line with offset for visibility
    offset = (list(model_list).index(model) - 2) * 0.5  # Small x-offset
    x_pos = [x + offset for x in rate_labels]
    
    ax2.plot(x_pos, medians, 'o-', color=colors[model], 
             linewidth=2, markersize=6, label=f'{model} (median)', zorder=3)
    
    # Plot uncertainty band
    ax2.fill_between(x_pos, p10s, p90s, color=colors[model], 
                     alpha=0.2, zorder=1)

# Formatting
ax2.set_xlabel('Precipitation Rate (%)', fontsize=14)
ax2.set_ylabel('Peak Discharge (mm/day)', fontsize=14)
ax2.set_title('Peak Discharge vs Precipitation Rate - All Models Comparison\n' +
              'Lines: Median, Shaded areas: 10th-90th percentile', 
              fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, loc='best')

# Set x-axis ticks
ax2.set_xticks(rate_labels)
ax2.set_xticklabels(rate_labels)

plt.tight_layout()

# Save the combined figure
output_file2 = os.path.join(output_dir, "peak_discharge_vs_precip_rate_all_models.png")
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Combined comparison plot saved to: {output_file2}")

# Create a clean median-only comparison plot (without uncertainty bands)
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))

# Plot only median lines for all models
for model in model_list:
    if model not in all_model_data:
        continue
        
    # Calculate median values for each rate
    medians = []
    
    for rate in rate_labels:
        data = all_model_data[model][rate]
        if len(data) > 0:
            medians.append(np.median(data))
        else:
            medians.append(np.nan)
    
    # Plot clean median line
    ax3.plot(rate_labels, medians, 'o-', color=colors[model], 
             linewidth=3, markersize=8, label=f'{model}', zorder=3)

# Formatting
ax3.set_xlabel('Precipitation Rate (%)', fontsize=14)
ax3.set_ylabel('Peak Discharge (mm/day)', fontsize=14)
ax3.set_title('Peak Discharge vs Precipitation Rate - Median Values Only\n' +
              'Clean Comparison Across All Models', 
              fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=12, loc='best')

# Set x-axis ticks
ax3.set_xticks(rate_labels)
ax3.set_xticklabels(rate_labels)

# Add some styling
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save the clean median figure
output_file3 = os.path.join(output_dir, "peak_discharge_vs_precip_rate_median_only.png")
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"Clean median-only comparison plot saved to: {output_file3}")

# Print summary statistics
print("\n" + "="*80)
print("PEAK DISCHARGE vs PRECIPITATION RATE ANALYSIS SUMMARY")
print("="*80)

for model in model_list:
    if model not in all_model_data:
        print(f"\n{model}: Data not available")
        continue
        
    print(f"\n{model} Model:")
    print("-" * 40)
    
    for rate in rate_labels:
        data = all_model_data[model][rate]
        if len(data) > 0:
            median_val = np.median(data)
            p10_val = np.percentile(data, 10)
            p90_val = np.percentile(data, 90)
            print(f"  Rate {rate}%: Median={median_val:.2f}, "
                  f"10th={p10_val:.2f}, 90th={p90_val:.2f} mm/day")
        else:
            print(f"  Rate {rate}%: No data available")

print(f"\nVisualization completed!")
print(f"Individual model plots: {output_file}")
print(f"Combined comparison: {output_file2}")
print(f"Clean median-only comparison: {output_file3}")
print(f"Total basins analyzed: {file_tot_num}")

plt.show()