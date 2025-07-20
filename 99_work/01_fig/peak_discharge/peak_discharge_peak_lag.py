import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/time_lag/"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
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
        all_model_data[model] = df
        print(f"Loaded {model}: {len(df)} records")
    except FileNotFoundError:
        print(f"Warning: File not found for {model}: {input_file}")
        continue

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Creating peak discharge analysis figures...")

# =============================================================================
# Figure 1: Relationship between peak rate and time lag
# =============================================================================

# Prepare data for peak rate vs time lag analysis
peak_rate_data = []

for model, df in all_model_data.items():
    for _, row in df.iterrows():
        # For each precipitation rate scenario
        for rate_label, rate_col in zip(rate_labels, rate_cols):
            if rate_label == 0:  # Skip rate 0 as it's baseline
                continue
                
            # Get the lag column name
            lag_col = rate_col.replace('flow', 'lag')
            
            if rate_col in df.columns and lag_col in df.columns:
                peak_rate_data.append({
                    'Model': model,
                    'File_num': row['file_num'],
                    'Rate': rate_label,
                    'Peak_Flow': row[rate_col],
                    'Time_Lag': row[lag_col]
                })

# Convert to DataFrame
peak_rate_df = pd.DataFrame(peak_rate_data)
peak_rate_df = peak_rate_df.dropna()

# Create Figure 1: Peak Rate vs Time Lag
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, rate in enumerate([10, 30, 50]):
    rate_data = peak_rate_df[peak_rate_df['Rate'] == rate]
    
    # Calculate mean time lag for each model at this rate
    mean_data = rate_data.groupby('Model')['Time_Lag'].mean().reset_index()
    
    # Plot as line with different colors for each model
    for j, model in enumerate(model_list):
        model_mean = mean_data[mean_data['Model'] == model]
        if not model_mean.empty:
            axes[i].plot([rate], model_mean['Time_Lag'].values, 
                        marker='o', linewidth=2, markersize=8, label=model)
    
    axes[i].set_xlabel('Peak Rate (%)', fontsize=12)
    axes[i].set_ylabel('Mean Time Lag (days)', fontsize=12)
    axes[i].set_title(f'Peak Rate {rate}%', fontsize=14, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    if i == 0:  # Only show legend on first subplot
        axes[i].legend()

plt.suptitle('Mean Time Lag by Peak Rate', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_rate_vs_time_lag.png", dpi=300, bbox_inches='tight')
plt.close()

# Alternative Figure 1: All rates on one plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for model in model_list:
    model_data = peak_rate_df[peak_rate_df['Model'] == model]
    if not model_data.empty:
        # Calculate mean time lag for each rate
        mean_by_rate = model_data.groupby('Rate')['Time_Lag'].mean()
        ax.plot(mean_by_rate.index.values, mean_by_rate.values, 
                marker='o', linewidth=2, markersize=6, label=model)

ax.set_xlabel('Peak Rate (%)', fontsize=12)
ax.set_ylabel('Mean Time Lag (days)', fontsize=12)
ax.set_title('Mean Time Lag vs Peak Rate by Model', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xticks([10, 30, 50])

plt.tight_layout()
plt.savefig(f"{output_dir}/peak_rate_vs_time_lag_combined.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Relationship between peak days (1, 3, 5) and peak time
# =============================================================================

# Prepare data for peak days analysis
peak_days_data = []

for model, df in all_model_data.items():
    for _, row in df.iterrows():
        # For each peak duration (1, 3, 5 days) and rate (10, 30, 50%)
        for peak_days in [1, 3, 5]:
            for rate in [10, 30, 50]:
                flow_col = f'sim_peak_flow_peak{peak_days}_rate{rate}'
                lag_col = f'sim_peak_lag_peak{peak_days}_rate{rate}'
                
                if flow_col in df.columns and lag_col in df.columns:
                    peak_days_data.append({
                        'Model': model,
                        'File_num': row['file_num'],
                        'Peak_Days': peak_days,
                        'Rate': rate,
                        'Peak_Flow': row[flow_col],
                        'Time_Lag': row[lag_col]
                    })

# Convert to DataFrame
peak_days_df = pd.DataFrame(peak_days_data)
peak_days_df = peak_days_df.dropna()

# Create Figure 2: Peak Days vs Time Lag
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, rate in enumerate([10, 30, 50]):
    rate_data = peak_days_df[peak_days_df['Rate'] == rate]
    
    for model in model_list:
        model_data = rate_data[rate_data['Model'] == model]
        if not model_data.empty:
            # Calculate mean time lag for each peak duration
            mean_by_days = model_data.groupby('Peak_Days')['Time_Lag'].mean()
            axes[i].plot(mean_by_days.index.values, mean_by_days.values, 
                        marker='o', linewidth=2, markersize=6, label=model)
    
    axes[i].set_xlabel('Peak Duration (days)', fontsize=12)
    axes[i].set_ylabel('Mean Time Lag (days)', fontsize=12)
    axes[i].set_title(f'Rate {rate}%', fontsize=14, fontweight='bold')
    axes[i].set_xticks([1, 3, 5])
    axes[i].grid(True, alpha=0.3)
    if i == 0:  # Only show legend on first subplot
        axes[i].legend()

plt.suptitle('Mean Time Lag vs Peak Duration by Rate', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_days_vs_time_lag.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: Peak Flow vs Peak Days comparison
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, rate in enumerate([10, 30, 50]):
    rate_data = peak_days_df[peak_days_df['Rate'] == rate]
    
    # Calculate mean peak flow for each model and peak days combination
    summary_data = rate_data.groupby(['Model', 'Peak_Days'])['Peak_Flow'].mean().reset_index()
    
    for model in model_list:
        model_data = summary_data[summary_data['Model'] == model]
        if not model_data.empty:
            axes[i].plot(model_data['Peak_Days'].values, model_data['Peak_Flow'].values, 
                        marker='o', linewidth=2, markersize=6, label=model)
    
    axes[i].set_xlabel('Peak Duration (days)', fontsize=12)
    axes[i].set_ylabel('Mean Peak Flow (mm/day)', fontsize=12)
    axes[i].set_title(f'Rate {rate}%', fontsize=14, fontweight='bold')
    axes[i].set_xticks([1, 3, 5])
    axes[i].grid(True, alpha=0.3)
    if i == 0:  # Only show legend on first subplot
        axes[i].legend()

plt.suptitle('Mean Peak Flow vs Peak Duration by Rate', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_flow_vs_peak_days.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Additional Figure: Combined Peak Flow vs Peak Days (all rates in one plot)
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Use rate 30% as representative
rate_data = peak_days_df[peak_days_df['Rate'] == 30]
summary_data = rate_data.groupby(['Model', 'Peak_Days'])['Peak_Flow'].mean().reset_index()

for model in model_list:
    model_data = summary_data[summary_data['Model'] == model]
    if not model_data.empty:
        ax.plot(model_data['Peak_Days'].values, model_data['Peak_Flow'].values, 
               marker='o', linewidth=2, markersize=6, label=model)

ax.set_xlabel('Peak Duration (days)', fontsize=12)
ax.set_ylabel('Mean Peak Flow (mm/day)', fontsize=12)
ax.set_title('Mean Peak Flow vs Peak Duration (Rate 30%)', fontsize=14, fontweight='bold')
ax.set_xticks([1, 3, 5])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/peak_flow_vs_peak_days_combined.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Summary Statistics
# =============================================================================

print("\nSummary Statistics:")
print("="*60)

# Time lag statistics by model
print("Mean Time Lag by Model and Rate:")
print("-"*40)
lag_summary = peak_rate_df.groupby(['Model', 'Rate'])['Time_Lag'].mean().unstack()
print(lag_summary.round(2))

# Peak flow statistics by model and peak days
print("\nMean Peak Flow by Model and Peak Days (Rate 30%):")
print("-"*50)
flow_summary = peak_days_df[peak_days_df['Rate'] == 30].groupby(['Model', 'Peak_Days'])['Peak_Flow'].mean().unstack()
print(flow_summary.round(2))

# Save summary statistics
lag_summary.to_csv(f"{output_dir}/time_lag_summary.csv")
flow_summary.to_csv(f"{output_dir}/peak_flow_summary.csv")

print(f"\nFigures saved to: {output_dir}")
print("Files created:")
print("- peak_rate_vs_time_lag.png")
print("- peak_rate_vs_time_lag_combined.png")
print("- peak_days_vs_time_lag.png")
print("- peak_flow_vs_peak_days.png")
print("- peak_flow_vs_peak_days_combined.png")
print("- time_lag_summary.csv")
print("- peak_flow_summary.csv")
        