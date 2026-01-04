import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/example/"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424

# Dictionary to store all model data
all_model_data = {}

# Peak length labels and rates to analyze
peak_labels = [1, 3, 5]
rate_list = [0, 10, 30, 50]  # Different precipitation rates to analyze

# Generate column names for all combinations
rate_peak_cols = {}
for rate in rate_list:
    rate_peak_cols[rate] = [f'sim_peak_flow_peak{peak}_rate{rate}' for peak in peak_labels]

# Load data for each model
for model in model_list:
    if model == "HBV":
        input_file = "/data0/funato/hsp/0_out/HBV/peak_discharge/peak_discharge_comparison_HBV.csv"
    else:
        input_file = f"/data0/funato/hsp/0_out/{model}/peak_discharge/peak_discharge_comparison_{model}.csv"
    
    try:
        df = pd.read_csv(input_file, header=0, index_col=None)
        
        # Extract peak discharge data for each rate and peak length combination
        model_data = {}
        for rate in rate_list:
            model_data[rate] = {}
            for i, col in enumerate(rate_peak_cols[rate]):
                if col in df.columns:
                    model_data[rate][peak_labels[i]] = df[col].dropna().values
                else:
                    print(f"Warning: Column {col} not found in {model} data")
                    model_data[rate][peak_labels[i]] = np.array([])
        
        all_model_data[model] = model_data
        print(f"Loaded {model} data: {len(df)} records")
        
    except FileNotFoundError:
        print(f"Warning: Data file for {model} not found: {input_file}")
        continue

# Create visualizations for each rate
for rate in rate_list:
    print(f"\nCreating visualizations for {rate}% rate...")
    
    # Create the visualization
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Peak Discharge vs Peak Length ({rate}% Rate)\nMedian with 10th-90th Percentile Uncertainty', 
                 fontsize=16, fontweight='bold')

    # Define colors for each model
    colors = {
        'HBV': '#1f77b4',
        'BC': "#FC33E5",
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
        
        # Calculate statistics for each peak length
        medians = []
        p10s = []
        p90s = []
        
        for peak in peak_labels:
            data = all_model_data[model][rate][peak]
            if len(data) > 0:
                medians.append(np.median(data))
                p10s.append(np.percentile(data, 10))
                p90s.append(np.percentile(data, 90))
            else:
                medians.append(np.nan)
                p10s.append(np.nan)
                p90s.append(np.nan)
        
        # Plot median line
        ax.plot(peak_labels, medians, 'o-', color=colors[model], 
                linewidth=2, markersize=8, label='Median', zorder=3)
        
        # Plot uncertainty band (10th-90th percentile)
        ax.fill_between(peak_labels, p10s, p90s, color=colors[model], 
                        alpha=0.3, label='10th-90th percentile', zorder=1)
        
        # Plot percentile lines
        ax.plot(peak_labels, p10s, '--', color=colors[model], 
                alpha=0.7, linewidth=1, zorder=2)
        ax.plot(peak_labels, p90s, '--', color=colors[model], 
                alpha=0.7, linewidth=1, zorder=2)
        
        # Formatting
        ax.set_xlabel('Peak Length (days)', fontsize=12)
        ax.set_ylabel('Peak Discharge (mm/day)', fontsize=12)
        ax.set_title(f'{model}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set x-axis ticks
        ax.set_xticks(peak_labels)
        ax.set_xticklabels(peak_labels)

    # Remove the empty subplot
    axes[1, 2].remove()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"peak_discharge_vs_peak_length_rate{rate}_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Peak discharge comparison plot ({rate}% rate) saved to: {output_file}")

    # Create a combined comparison plot for this rate
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

    # Plot all models on the same plot
    for model in model_list:
        if model not in all_model_data:
            continue
            
        # Calculate statistics for each peak length
        medians = []
        p10s = []
        p90s = []
        
        for peak in peak_labels:
            data = all_model_data[model][rate][peak]
            if len(data) > 0:
                medians.append(np.median(data))
                p10s.append(np.percentile(data, 10))
                p90s.append(np.percentile(data, 90))
            else:
                medians.append(np.nan)
                p10s.append(np.nan)
                p90s.append(np.nan)
        
        # Plot median line with offset for visibility
        offset = (list(model_list).index(model) - 2) * 0.1  # Small x-offset
        x_pos = [x + offset for x in peak_labels]
        
        ax2.plot(x_pos, medians, 'o-', color=colors[model], 
                 linewidth=2, markersize=6, label=f'{model} (median)', zorder=3)
        
        # Plot uncertainty band
        ax2.fill_between(x_pos, p10s, p90s, color=colors[model], 
                         alpha=0.2, zorder=1)

    # Formatting
    ax2.set_xlabel('Peak Length (days)', fontsize=14)
    ax2.set_ylabel('Peak Discharge (mm/day)', fontsize=14)
    ax2.set_title(f'Peak Discharge vs Peak Length ({rate}% Rate) - All Models Comparison\n' +
                  'Lines: Median, Shaded areas: 10th-90th percentile', 
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc='best')

    # Set x-axis ticks
    ax2.set_xticks(peak_labels)
    ax2.set_xticklabels(peak_labels)

    plt.tight_layout()

    # Save the combined figure
    output_file2 = os.path.join(output_dir, f"peak_discharge_vs_peak_length_rate{rate}_all_models.png")
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot ({rate}% rate) saved to: {output_file2}")

    # Create a clean median-only comparison plot (without uncertainty bands)
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))

    # Plot only median lines for all models
    for model in model_list:
        if model not in all_model_data:
            continue
            
        # Calculate median values for each peak length
        medians = []
        
        for peak in peak_labels:
            data = all_model_data[model][rate][peak]
            if len(data) > 0:
                medians.append(np.median(data))
            else:
                medians.append(np.nan)
        
        # Plot clean median line
        ax3.plot(peak_labels, medians, 'o-', color=colors[model], 
                 linewidth=3, markersize=8, label=f'{model}', zorder=3)

    # Formatting
    ax3.set_xlabel('Peak Length (days)', fontsize=14)
    ax3.set_ylabel('Peak Discharge (mm/day)', fontsize=14)
    ax3.set_title(f'Peak Discharge vs Peak Length ({rate}% Rate) - Median Values Only\n' +
                  'Clean Comparison Across All Models', 
                  fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, loc='best')

    # Set x-axis ticks
    ax3.set_xticks(peak_labels)
    ax3.set_xticklabels(peak_labels)

    # Add some styling
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Save the clean median figure
    output_file3 = os.path.join(output_dir, f"peak_discharge_vs_peak_length_rate{rate}_median_only.png")
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"Clean median-only comparison plot ({rate}% rate) saved to: {output_file3}")

    plt.close('all')  # Close all figures to free memory

# Print summary statistics for all rates
print("\n" + "="*80)
print("PEAK DISCHARGE vs PEAK LENGTH ANALYSIS SUMMARY")
print("="*80)

for rate in rate_list:
    print(f"\n{rate}% PRECIPITATION RATE:")
    print("="*50)
    
    for model in model_list:
        if model not in all_model_data:
            print(f"\n{model}: Data not available")
            continue
            
        print(f"\n{model} Model:")
        print("-" * 40)
        
        for peak in peak_labels:
            data = all_model_data[model][rate][peak]
            if len(data) > 0:
                median_val = np.median(data)
                p10_val = np.percentile(data, 10)
                p90_val = np.percentile(data, 90)
                print(f"  Peak {peak} day(s): Median={median_val:.2f}, "
                      f"10th={p10_val:.2f}, 90th={p90_val:.2f} mm/day")
            else:
                print(f"  Peak {peak} day(s): No data available")

print(f"\nVisualization completed!")
print(f"Generated figures for rates: {rate_list}%")
print(f"Total basins analyzed: {file_tot_num}")
print(f"Output directory: {output_dir}")

# Create a summary comparison across all rates
fig_summary, axes_summary = plt.subplots(1, 3, figsize=(18, 6))
fig_summary.suptitle('Peak Discharge vs Peak Length - Rate Comparison\nMedian Values Only', 
                     fontsize=16, fontweight='bold')

colors = {
    'HBV': '#1f77b4',
    'BC': "#FC33E5",
    'RF': '#ff7f0e', 
    'XGB': '#2ca02c',
    'TCN': '#d62728',
    'LSTM': '#9467bd'
}

for rate_idx, rate in enumerate(rate_list):
    ax = axes_summary[rate_idx]
    
    for model in model_list:
        if model not in all_model_data:
            continue
            
        # Calculate median values for each peak length
        medians = []
        
        for peak in peak_labels:
            data = all_model_data[model][rate][peak]
            if len(data) > 0:
                medians.append(np.median(data))
            else:
                medians.append(np.nan)
        
        # Plot clean median line
        ax.plot(peak_labels, medians, 'o-', color=colors[model], 
                linewidth=3, markersize=8, label=f'{model}', zorder=3)

    # Formatting
    ax.set_xlabel('Peak Length (days)', fontsize=12)
    ax.set_ylabel('Peak Discharge (mm/day)', fontsize=12)
    ax.set_title(f'{rate}% Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if rate_idx == 0:  # Only show legend on first subplot
        ax.legend(fontsize=10)
    
    # Set x-axis ticks
    ax.set_xticks(peak_labels)
    ax.set_xticklabels(peak_labels)

plt.tight_layout()

# Save the summary comparison figure
output_summary = os.path.join(output_dir, "peak_discharge_vs_peak_length_rate_comparison_summary.png")
plt.savefig(output_summary, dpi=300, bbox_inches='tight')
print(f"Rate comparison summary plot saved to: {output_summary}")

print("Done")
