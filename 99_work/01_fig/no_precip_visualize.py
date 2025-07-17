import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

# Configuration
model_list = ["HBV", "RF", "XGB", "TCN"]  # Can be expanded if needed
input_dir = "/data0/funato/hsp/0_out/analysis/no_precip_discharge"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/no_precip_discharge"

# Ensure output directory exists
os.makedirs(output_fig_dir, exist_ok=True)

# Load the data
print("Loading data...")
all_discharge_csv = f"{input_dir}/no_precip_all_discharge_data.csv"
summary_csv = f"{input_dir}/no_precip_discharge_summary.csv"
model_summary_csv = f"{input_dir}/no_precip_model_summary.csv"

# Check if files exist
if not os.path.exists(all_discharge_csv):
    print(f"Error: {all_discharge_csv} not found!")
    print("Please run no_precip_calc.py first to generate the data.")
    exit(1)

# Load all discharge data
all_discharge_df = pd.read_csv(all_discharge_csv)
print(f"Loaded {len(all_discharge_df)} discharge data points")

# Load summary statistics
if os.path.exists(summary_csv):
    results_df = pd.read_csv(summary_csv)
    print(f"Loaded summary statistics for {len(results_df)} files")

if os.path.exists(model_summary_csv):
    model_summary = pd.read_csv(model_summary_csv, index_col=0)
    print("Loaded model summary statistics")

# Prepare data for plotting
all_discharge_data = {}
for model in model_list:
    model_data = all_discharge_df[all_discharge_df['Model'] == model]['Discharge'].values
    all_discharge_data[model] = model_data
    print(f"{model}: {len(model_data)} data points")

# Create comprehensive visualizations
print("Creating comprehensive visualizations...")

# 1. Box and Violin plots (separate figure with better spacing)
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots

# Box plot for comparison
box_data = [all_discharge_data[model] for model in model_list if len(all_discharge_data[model]) > 0]
box_labels = [model for model in model_list if len(all_discharge_data[model]) > 0]

# Regular scale box plot
bp1 = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
ax1.set_title('No-Precipitation Discharge Comparison\n(Linear Scale)')
ax1.set_ylabel('Discharge')
ax1.set_xlabel('Model')
ax1.grid(True, alpha=0.3)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
    patch.set_facecolor(color)

# Log scale box plot for better visibility near zero
bp2 = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
ax2.set_title('No-Precipitation Discharge Comparison\n(Log Scale)')
ax2.set_ylabel('Discharge (log scale)')
ax2.set_xlabel('Model')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
    patch.set_facecolor(color)

# Regular scale violin plot
if box_data:
    parts1 = ax3.violinplot(box_data, positions=range(1, len(box_data) + 1))
    ax3.set_xticks(range(1, len(box_labels) + 1))
    ax3.set_xticklabels(box_labels)
    ax3.set_title('Distribution of No-Precipitation Discharge\n(Linear Scale)')
    ax3.set_ylabel('Discharge')
    ax3.set_xlabel('Model')
    ax3.grid(True, alpha=0.3)

# Log scale violin plot
if box_data:
    parts2 = ax4.violinplot(box_data, positions=range(1, len(box_data) + 1))
    ax4.set_xticks(range(1, len(box_labels) + 1))
    ax4.set_xticklabels(box_labels)
    ax4.set_title('Distribution of No-Precipitation Discharge\n(Log Scale)')
    ax4.set_ylabel('Discharge (log scale)')
    ax4.set_xlabel('Model')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots
output_fig1 = f"{output_fig_dir}/no_precip_box_violin_comparison.png"
plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
print(f"Box/Violin plots saved to: {output_fig1}")

# 2. PDF and CDF plots (with better spacing)
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots

# Prepare data for PDF/CDF
valid_models = [model for model in model_list if len(all_discharge_data[model]) > 0]
colors_cycle = ['blue', 'green', 'red', 'orange', 'purple']

# PDF plot - Linear scale
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    # Remove zeros and negative values for log calculations
    data_positive = data[data > 0]
    
    if len(data_positive) > 10:  # Need enough points for KDE
        kde = gaussian_kde(data_positive)
        x_range = np.linspace(data_positive.min(), data_positive.max(), 1000)
        density = kde(x_range)
        ax5.plot(x_range, density, label=model, color=colors_cycle[i], linewidth=2)

ax5.set_title('Probability Density Function (PDF)\n(Linear Scale)')
ax5.set_xlabel('Discharge')
ax5.set_ylabel('Density')
ax5.legend()
ax5.grid(True, alpha=0.3)

# PDF plot - Log scale
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    data_positive = data[data > 0]
    
    if len(data_positive) > 10:
        # Use log-transformed data for PDF
        log_data = np.log10(data_positive)
        kde = gaussian_kde(log_data)
        x_range = np.linspace(log_data.min(), log_data.max(), 1000)
        density = kde(x_range)
        ax6.plot(10**x_range, density, label=model, color=colors_cycle[i], linewidth=2)

ax6.set_title('Probability Density Function (PDF)\n(Log Scale)')
ax6.set_xlabel('Discharge (log scale)')
ax6.set_ylabel('Density')
ax6.set_xscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3)

# CDF plot - Linear scale
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    data_sorted = np.sort(data)
    p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax7.plot(data_sorted, p, label=model, color=colors_cycle[i], linewidth=2)

ax7.set_title('Cumulative Distribution Function (CDF)\n(Linear Scale)')
ax7.set_xlabel('Discharge')
ax7.set_ylabel('Cumulative Probability')
ax7.legend()
ax7.grid(True, alpha=0.3)

# CDF plot - Log scale
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    data_positive = data[data > 0]
    if len(data_positive) > 0:
        data_sorted = np.sort(data_positive)
        p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax8.plot(data_sorted, p, label=model, color=colors_cycle[i], linewidth=2)

ax8.set_title('Cumulative Distribution Function (CDF)\n(Log Scale)')
ax8.set_xlabel('Discharge (log scale)')
ax8.set_ylabel('Cumulative Probability')
ax8.set_xscale('log')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots
output_fig2 = f"{output_fig_dir}/no_precip_pdf_cdf_comparison.png"
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"PDF/CDF plots saved to: {output_fig2}")

# 3. Focused view near zero (with better spacing)
fig3, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots

# Box plot with limited y-axis to focus on lower values
bp3 = ax9.boxplot(box_data, labels=box_labels, patch_artist=True)
ax9.set_title('No-Precipitation Discharge\n(Focused on Lower Values)')
ax9.set_ylabel('Discharge')
ax9.set_xlabel('Model')
# Set y-limit to focus on lower values (adjust based on your data)
max_for_focus = np.percentile(np.concatenate(box_data), 90)  # 90th percentile
ax9.set_ylim(0, max_for_focus)
ax9.grid(True, alpha=0.3)

for patch, color in zip(bp3['boxes'], colors[:len(bp3['boxes'])]):
    patch.set_facecolor(color)

# Histogram comparison with focus on lower values
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    # Focus on lower values
    data_focused = data[data <= max_for_focus]
    ax10.hist(data_focused, bins=50, alpha=0.6, label=model, density=True, 
             color=colors_cycle[i])

ax10.set_title('Distribution Comparison\n(Focused on Lower Values)')
ax10.set_xlabel('Discharge')
ax10.set_ylabel('Density')
ax10.legend()
ax10.grid(True, alpha=0.3)

# CDF with focus on lower percentiles
for i, model in enumerate(valid_models):
    data = np.array(all_discharge_data[model])
    data_sorted = np.sort(data)
    p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax11.plot(data_sorted, p, label=model, color=colors_cycle[i], linewidth=2)

ax11.set_title('CDF - Focus on Lower Percentiles')
ax11.set_xlabel('Discharge')
ax11.set_ylabel('Cumulative Probability')
ax11.set_xlim(0, max_for_focus)
ax11.set_ylim(0, 0.9)  # Focus on lower 90% of data
ax11.legend()
ax11.grid(True, alpha=0.3)

# Statistical summary plot
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
x_pos = np.arange(len(valid_models))

ax12.bar(x_pos - 0.3, stats_df['Mean'], 0.2, label='Mean', alpha=0.8)
ax12.bar(x_pos - 0.1, stats_df['Median'], 0.2, label='Median', alpha=0.8)
ax12.bar(x_pos + 0.1, stats_df['Q75'], 0.2, label='Q75', alpha=0.8)
ax12.bar(x_pos + 0.3, stats_df['Q90'], 0.2, label='Q90', alpha=0.8)

ax12.set_title('Statistical Summary by Model')
ax12.set_ylabel('Discharge')
ax12.set_xlabel('Model')
ax12.set_xticks(x_pos)
ax12.set_xticklabels(valid_models)
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots
output_fig3 = f"{output_fig_dir}/no_precip_focused_analysis.png"
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
print(f"Focused analysis plots saved to: {output_fig3}")

# 4. Additional comprehensive analysis (from in-depth analysis)
print("Creating additional comprehensive analysis plots...")
fig4 = plt.figure(figsize=(20, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots

# 4.1. Box plot comparison with log scale
ax13 = plt.subplot(2, 3, 1)
if box_data:
    bp = ax13.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax13.set_title('No-Precipitation Discharge Distribution\n(All Values - Log Scale)', fontsize=12, fontweight='bold')
    ax13.set_ylabel('Discharge (log scale)')
    ax13.grid(True, alpha=0.3)
    ax13.set_yscale('log')

# 4.2. Violin plot with log scale
ax14 = plt.subplot(2, 3, 2)
if box_data:
    parts = ax14.violinplot(box_data, positions=range(1, len(box_data) + 1))
    ax14.set_xticks(range(1, len(box_labels) + 1))
    ax14.set_xticklabels(box_labels)
    ax14.set_title('Discharge Distribution\n(Violin Plot - Log Scale)', fontsize=12, fontweight='bold')
    ax14.set_ylabel('Discharge (log scale)')
    ax14.grid(True, alpha=0.3)
    ax14.set_yscale('log')

# 4.3. Histogram comparison with log-transformed data
ax15 = plt.subplot(2, 3, 3)
for i, model in enumerate(valid_models):
    if len(all_discharge_data[model]) > 0:
        # Use log-transformed data for better visualization
        data = np.array(all_discharge_data[model])
        log_data = np.log10(data + 1e-10)  # Add small value to avoid log(0)
        ax15.hist(log_data, bins=50, alpha=0.6, label=model, density=True, 
                 color=colors_cycle[i])

ax15.set_title('Distribution Comparison\n(Log-transformed)', fontsize=12, fontweight='bold')
ax15.set_xlabel('Log10(Discharge)')
ax15.set_ylabel('Density')
ax15.legend()
ax15.grid(True, alpha=0.3)

# 4.4. Model performance ranking
ax16 = plt.subplot(2, 3, 4)
model_means = []
model_names = []
for model in valid_models:
    if len(all_discharge_data[model]) > 0:
        mean_val = np.mean(all_discharge_data[model])
        model_means.append(mean_val)
        model_names.append(model)

if model_means:
    # Sort by mean (lower is better for zero precipitation)
    sorted_data = sorted(zip(model_names, model_means), key=lambda x: x[1])
    sorted_names, sorted_means = zip(*sorted_data)
    
    bars = ax16.bar(range(len(sorted_means)), sorted_means, 
                   color=colors[:len(sorted_means)])
    ax16.set_xticks(range(len(sorted_names)))
    ax16.set_xticklabels(sorted_names, rotation=45)
    ax16.set_title('Average Mean Discharge by Model\n(Lower = Better for Zero Precip)', fontsize=12, fontweight='bold')
    ax16.set_ylabel('Average Mean Discharge')
    ax16.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sorted_means):
        ax16.text(i, v + max(sorted_means) * 0.01, f'{v:.4f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4.5. Mean vs Standard deviation scatter plot
ax17 = plt.subplot(2, 3, 5)
for i, model in enumerate(valid_models):
    if len(all_discharge_data[model]) > 0:
        data = np.array(all_discharge_data[model])
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax17.scatter(mean_val, std_val, label=model, alpha=0.8, s=100, 
                    color=colors_cycle[i])

ax17.set_xlabel('Mean Discharge')
ax17.set_ylabel('Standard Deviation')
ax17.set_title('Mean vs Variability\n(Model Comparison)', fontsize=12, fontweight='bold')
ax17.legend()
ax17.grid(True, alpha=0.3)
ax17.set_xscale('log')
ax17.set_yscale('log')

# 4.6. Percentile comparison
ax18 = plt.subplot(2, 3, 6)
percentiles = [50, 75, 90, 95, 99]
x_pos = np.arange(len(valid_models))
width = 0.15

for i, percentile in enumerate(percentiles):
    values = []
    for model in valid_models:
        if len(all_discharge_data[model]) > 0:
            values.append(np.percentile(all_discharge_data[model], percentile))
        else:
            values.append(0)
    
    ax18.bar(x_pos + i * width, values, width, label=f'P{percentile}', alpha=0.8)

ax18.set_xlabel('Model')
ax18.set_ylabel('Discharge')
ax18.set_title('Percentile Comparison\n(Higher percentiles)', fontsize=12, fontweight='bold')
ax18.set_xticks(x_pos + width * 2)
ax18.set_xticklabels(valid_models)
ax18.legend()
ax18.grid(True, alpha=0.3)
ax18.set_yscale('log')

output_fig4 = f"{output_fig_dir}/no_precip_comprehensive_analysis.png"
plt.savefig(output_fig4, dpi=300, bbox_inches='tight')
print(f"Comprehensive analysis plots saved to: {output_fig4}")

# 5. Statistical comparison plots
print("Creating statistical comparison plots...")
fig5, ((ax19, ax20), (ax21, ax22)) = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more space between subplots

# 5.1. Zero-value percentage comparison
ax19_data = []
ax19_labels = []
for model in valid_models:
    if len(all_discharge_data[model]) > 0:
        data = np.array(all_discharge_data[model])
        zero_pct = (data < 0.01).sum() / len(data) * 100
        ax19_data.append(zero_pct)
        ax19_labels.append(model)

if ax19_data:
    bars = ax19.bar(ax19_labels, ax19_data, color=colors[:len(ax19_data)])
    ax19.set_title('Percentage of Near-Zero Values\n(< 0.01)', fontsize=12, fontweight='bold')
    ax19.set_ylabel('Percentage (%)')
    ax19.set_xlabel('Model')
    ax19.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(ax19_data):
        ax19.text(i, v + max(ax19_data) * 0.01, f'{v:.1f}%', 
                 ha='center', va='bottom', fontweight='bold')

# 5.2. Range comparison (max - min)
ax20_data = []
for model in valid_models:
    if len(all_discharge_data[model]) > 0:
        data = np.array(all_discharge_data[model])
        range_val = np.max(data) - np.min(data)
        ax20_data.append(range_val)

if ax20_data:
    bars = ax20.bar(ax19_labels, ax20_data, color=colors[:len(ax20_data)])
    ax20.set_title('Discharge Range\n(Max - Min)', fontsize=12, fontweight='bold')
    ax20.set_ylabel('Range')
    ax20.set_xlabel('Model')
    ax20.set_yscale('log')
    ax20.grid(True, alpha=0.3)

# 5.3. Coefficient of Variation comparison
ax21_data = []
for model in valid_models:
    if len(all_discharge_data[model]) > 0:
        data = np.array(all_discharge_data[model])
        cv = np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
        ax21_data.append(cv)

if ax21_data:
    bars = ax21.bar(ax19_labels, ax21_data, color=colors[:len(ax21_data)])
    ax21.set_title('Coefficient of Variation\n(Std/Mean)', fontsize=12, fontweight='bold')
    ax21.set_ylabel('CV')
    ax21.set_xlabel('Model')
    ax21.grid(True, alpha=0.3)

# 5.4. Quartile range comparison
ax22_data_q25 = []
ax22_data_q75 = []
for model in valid_models:
    if len(all_discharge_data[model]) > 0:
        data = np.array(all_discharge_data[model])
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        ax22_data_q25.append(q25)
        ax22_data_q75.append(q75)

if ax22_data_q25 and ax22_data_q75:
    x_pos = np.arange(len(ax19_labels))
    ax22.bar(x_pos - 0.2, ax22_data_q25, 0.4, label='Q25', alpha=0.8, color='lightblue')
    ax22.bar(x_pos + 0.2, ax22_data_q75, 0.4, label='Q75', alpha=0.8, color='lightcoral')
    ax22.set_xlabel('Model')
    ax22.set_ylabel('Discharge')
    ax22.set_title('Quartile Comparison\n(25th and 75th percentiles)', fontsize=12, fontweight='bold')
    ax22.set_xticks(x_pos)
    ax22.set_xticklabels(ax19_labels)
    ax22.legend()
    ax22.grid(True, alpha=0.3)
    ax22.set_yscale('log')

output_fig5 = f"{output_fig_dir}/no_precip_statistical_comparison.png"
plt.savefig(output_fig5, dpi=300, bbox_inches='tight')
print(f"Statistical comparison plots saved to: {output_fig5}")

# Print detailed statistics
print("\n" + "="*60)
print("DETAILED STATISTICAL SUMMARY")
print("="*60)
for model in valid_models:
    data = np.array(all_discharge_data[model])
    print(f"\n{model}:")
    print(f"  Count: {len(data):,}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Median: {np.median(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Q25: {np.percentile(data, 25):.6f}")
    print(f"  Q75: {np.percentile(data, 75):.6f}")
    print(f"  Q90: {np.percentile(data, 90):.6f}")
    print(f"  Q95: {np.percentile(data, 95):.6f}")
    print(f"  Q99: {np.percentile(data, 99):.6f}")
    print(f"  % near zero (<0.01): {(data < 0.01).sum() / len(data) * 100:.2f}%")
    print(f"  % near zero (<0.1): {(data < 0.1).sum() / len(data) * 100:.2f}%")

print("\nVisualization completed!")
print(f"Generated figures:")
print(f"  - Box/Violin plots: {output_fig1}")
print(f"  - PDF/CDF plots: {output_fig2}")
print(f"  - Focused analysis: {output_fig3}")
print(f"  - Comprehensive analysis: {output_fig4}")
print(f"  - Statistical comparison: {output_fig5}")

# Perform statistical analysis (similar to in-depth analysis)
print("\n" + "="*60)
print("STATISTICAL ANALYSIS SUMMARY")
print("="*60)

for model in valid_models:
    data = np.array(all_discharge_data[model])
    print(f"\n{model}:")
    print(f"  Count: {len(data):,}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Median: {np.median(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  CV: {np.std(data)/np.mean(data):.4f}")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Range: {np.max(data) - np.min(data):.6f}")
    print(f"  Q25: {np.percentile(data, 25):.6f}")
    print(f"  Q75: {np.percentile(data, 75):.6f}")
    print(f"  Q90: {np.percentile(data, 90):.6f}")
    print(f"  Q95: {np.percentile(data, 95):.6f}")
    print(f"  Q99: {np.percentile(data, 99):.6f}")
    print(f"  % near zero (<0.01): {(data < 0.01).sum() / len(data) * 100:.2f}%")
    print(f"  % near zero (<0.1): {(data < 0.1).sum() / len(data) * 100:.2f}%")

# Model ranking by mean discharge (lower is better for zero precipitation)
print(f"\nModel Ranking (by mean discharge - lower is better for zero precipitation):")
print("-" * 60)

model_rankings = []
for model in valid_models:
    data = np.array(all_discharge_data[model])
    mean_val = np.mean(data)
    std_val = np.std(data)
    model_rankings.append((model, mean_val, std_val))

model_rankings.sort(key=lambda x: x[1])  # Sort by mean

for rank, (model, mean_val, std_val) in enumerate(model_rankings, 1):
    print(f"{rank}. {model}: {mean_val:.6f} ± {std_val:.6f}")

print(f"\nBest performing model for zero precipitation: {model_rankings[0][0]}")
print(f"Worst performing model for zero precipitation: {model_rankings[-1][0]}")

print("\nAll analyses completed successfully!")