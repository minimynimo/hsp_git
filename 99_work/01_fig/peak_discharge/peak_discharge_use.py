import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

fs = 24

# Configuration
output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge_use"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "HYPER", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424 

# Dictionary to store all model data
all_model_data = {}

# Precipitation rate labels and corresponding column names
rate_labels = [0, 10, 30, 50]
rate_cols = ['sim_peak_flow_peak1_rate0', 'sim_peak_flow_peak1_rate10', 
             'sim_peak_flow_peak1_rate30', 'sim_peak_flow_peak1_rate50']
peak_labels = [1, 3, 5]

# Define colors for each model
color_list = {
    "HBV": "#7B7B7B",    # Kept as requested (Medium Grey)
    "HYPER": "#628AE8",     # Kept as requested (Steel Blue)
    "RF":   "#FF7F00",   # Red
    "XGB":  "#9C4803",   # Bright Orange
    "TCN":  "#E41A1C",   # Reddish Purple / Magenta
    "LSTM": "#9808B2"    # Pink
}

def BMK(obs_data,sim_data,benchmark):    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    

# Load data for each model
for model in model_list:
    if model == "HBV":
        input_file = "/data0/funato/hsp/0_out/HBV/peak_discharge/peak_discharge_comparison_HBV.csv"
    elif model == "HYPER":
        input_file = "/data0/funato/hsp/0_out/BC/peak_discharge/peak_discharge_comparison_BC.csv"
    else:
        input_file = f"/data0/funato/hsp/0_out/{model}/peak_discharge/peak_discharge_comparison_{model}.csv"
    
    try:
        df = pd.read_csv(input_file, header=0, index_col=None)
        
        # Store the full dataframe for more flexible data access
        all_model_data[model] = df
        print(f"Loaded {model} data: {len(df)} records")
        
    except FileNotFoundError:
        print(f"Warning: Data file for {model} not found: {input_file}")
        continue

#### FIGURE 1: Peak Discharge vs Precipitation Rate for each Peak Duration####
# Create the visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
# increase space between subplots
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle('Precipitation Rate vs Peak Discharge', fontsize=fs, y=0.95)


# Plot each model in a separate subplot
for idx, model in enumerate(model_list):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if model not in all_model_data:
        ax.text(0.5, 0.5, f'{model}\nData not available', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=fs, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax.set_title(f'{model}', fontsize=fs)
        continue
    
    # Calculate statistics for each rate
    medians = []
    p10s = []
    p90s = []
    
    df = all_model_data[model]
    for i, rate in enumerate(rate_labels):
        col_name = rate_cols[i]
        if col_name in df.columns:
            data = df[col_name].dropna().values
            if len(data) > 0:
                medians.append(np.median(data))
                p10s.append(np.percentile(data, 10))
                p90s.append(np.percentile(data, 90))
            else:
                medians.append(np.nan)
                p10s.append(np.nan)
                p90s.append(np.nan)
        else:
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)
    
    # Plot median line
    ax.plot(rate_labels, medians, 'o-', color=color_list[model], 
            linewidth=2, markersize=8, label='Median', zorder=3)
    
    # Plot uncertainty band (10th-90th percentile)
    ax.fill_between(rate_labels, p10s, p90s, color=color_list[model], 
                    alpha=0.3, label='10th-90th percentile', zorder=1)
    
    # Plot percentile lines
    ax.plot(rate_labels, p10s, '--', color=color_list[model], 
            alpha=0.7, linewidth=1, zorder=2)
    ax.plot(rate_labels, p90s, '--', color=color_list[model], 
            alpha=0.7, linewidth=1, zorder=2)
    
    # Formatting
    ax.set_xlabel('Precipitation Rate [%]', fontsize=fs)
    ax.set_ylabel('Peak Discharge [mm/d]', fontsize=fs)
    #set the ylim the same for all subplots
    ax.set_ylim(0, 300)  # Set a common y-limit for better comparison
    ax.set_title(f'{model}', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs*0.7, loc='upper right')

    # Set x-axis ticks
    ax.set_xticks(rate_labels)
    ax.set_xticklabels(rate_labels)

# Adjust layout manually with subplots_adjust instead of tight_layout
# plt.tight_layout()  # Commented out as it overrides subplots_adjust

# Save the figure
output_file = os.path.join(output_dir, "peak_discharge_vs_precip_rate_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Peak discharge comparison plot saved to: {output_file}")
plt.close()

### FIGURE 5: Peak Days vs Mean Time Lag
# This shows peak durations on x-axis and mean time lag between observed and simulated results
fig, ax = plt.subplots(figsize=(11, 7))
#space between subplots
fig.subplots_adjust(wspace=0.4)

rate = 30
for model in model_list:
    if model not in all_model_data:
        continue
    
    df = all_model_data[model]
    mean_lags = []
    
    # For each peak duration, get the time lag for this rate
    for peak_duration in [1, 3, 5]:
        lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
        if lag_col in df.columns:
            data = df[lag_col].dropna().values
            if len(data) > 0:
                mean_lags.append(np.mean(data))
            else:
                mean_lags.append(np.nan)
        else:
            mean_lags.append(np.nan)

    # compare the lag of the HBV with the others
    if model == "HBV":
        mean_lags_hbv = mean_lags
        ax.plot([1, 3, 5], mean_lags_hbv, marker='o', linewidth=2, markersize=6,
                label=f"{model}", color=color_list[model])
    else:
        kge = BMK(mean_lags_hbv, mean_lags, 'KGE')    
        # Plot the line for this model
        ax.plot([1, 3, 5], mean_lags, marker='o', linewidth=2, markersize=6, 
                    label=f"{model}\n(KGE: {kge:.2f})", color=color_list[model])

ax.set_xlabel('Peak Duration [d]', fontsize=fs)
ax.set_ylabel('Mean Time Lag [d]', fontsize=fs)
ax.tick_params(labelsize=fs)
ax.set_xticks([1, 3, 5])
ax.set_ylim(0,8)
ax.grid(True, alpha=0.3)

# add the legend at the bottom of the figure
plt.legend(fontsize=fs*0.8, loc='upper right', ncol=1, bbox_to_anchor=(1.5, 1))
plt.title('Mean Time Lag by Peak Duration\n(Rate 30%)', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_days_vs_mean_time_lag_rate30.png", dpi=300, bbox_inches='tight')
plt.close()



#### FIGURE 1B: Peak Discharge vs Peak Duration for each Model ####
# Create the visualization similar to precipitation rate but with peak durations
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
# increase space between subplots
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle('Peak Duration vs Peak Discharge', fontsize=fs, y=0.95)

# Plot each model in a separate subplot
for idx, model in enumerate(model_list):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if model not in all_model_data:
        ax.text(0.5, 0.5, f'{model}\nData not available', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=fs, bbox=dict(boxstyle="round", facecolor='lightgray'))
        ax.set_title(f'{model}', fontsize=fs)
        continue
    
    df = all_model_data[model]
    
    # Calculate statistics for each peak duration
    for rate in [30]:  # Skip rate 0 for clarity
        medians = []
        p10s = []
        p90s = []
        
        for peak_duration in [1, 3, 5]:
            col_name = f'sim_peak_flow_peak{peak_duration}_rate{rate}'
            if col_name in df.columns:
                data = df[col_name].dropna().values
                if len(data) > 0:
                    medians.append(np.median(data))
                    p10s.append(np.percentile(data, 10))
                    p90s.append(np.percentile(data, 90))
                else:
                    medians.append(np.nan)
                    p10s.append(np.nan)
                    p90s.append(np.nan)
            else:
                medians.append(np.nan)
                p10s.append(np.nan)
                p90s.append(np.nan)
        
        # Plot median line for this rate
        ax.plot([1, 3, 5], medians, 'o-', linewidth=2, markersize=8, color = "royalblue",
                label=f'Median', zorder=3)
        
        # Plot uncertainty band (10th-90th percentile)
        ax.fill_between([1, 3, 5], p10s, p90s, alpha=0.3, color = "royalblue",
                        label=f'10th-90th percentile', zorder=1)
    
    # Formatting
    ax.set_xlabel('Peak Duration [d]', fontsize=fs)
    ax.set_ylabel('Peak Discharge [mm/d]', fontsize=fs)
    ax.set_ylim(0, 450)  # Set a common y-limit for better comparison
    ax.set_title(f'{model}', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs*0.6, loc='upper left')
    
    # Set x-axis ticks
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels([1, 3, 5])

# Save the figure
output_file_durations = os.path.join(output_dir, "peak_discharge_vs_peak_duration_comparison.png")
plt.savefig(output_file_durations, dpi=300, bbox_inches='tight')
print(f"Peak discharge vs peak duration comparison plot saved to: {output_file_durations}")
plt.close()

### FIGURE 2: Peak Discharge vs Precipitation Rate - Median Values Only 
# Create a clean median-only comparison plot (without uncertainty bands)
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))

# Plot only median lines for all models
# First, calculate HBV medians for comparison
hbv_medians = []
if "HBV" in all_model_data:
    hbv_df = all_model_data["HBV"]
    for i, rate in enumerate(rate_labels):
        col_name = rate_cols[i]
        if col_name in hbv_df.columns:
            data = hbv_df[col_name].dropna().values
            if len(data) > 0:
                hbv_medians.append(np.median(data))
            else:
                hbv_medians.append(np.nan)
        else:
            hbv_medians.append(np.nan)

for model in model_list:
    if model not in all_model_data:
        continue
        
    # Calculate median values for each rate
    medians = []
    df = all_model_data[model]
    
    for i, rate in enumerate(rate_labels):
        col_name = rate_cols[i]
        if col_name in df.columns:
            data = df[col_name].dropna().values
            if len(data) > 0:
                medians.append(np.median(data))
            else:
                medians.append(np.nan)
        else:
            medians.append(np.nan)
    
    # Calculate correlation coefficient and KGE between this model's medians and HBV medians
    if model == "HBV":
        coeff = 1.0
        kge = 1.0
    else:
        # Compare medians arrays directly
        valid_indices = ~(np.isnan(medians) | np.isnan(hbv_medians))
        if np.sum(valid_indices) > 1:  # Need at least 2 points for correlation
            coeff = np.corrcoef(np.array(hbv_medians)[valid_indices], np.array(medians)[valid_indices])[0, 1]
            kge = BMK(np.array(hbv_medians)[valid_indices], np.array(medians)[valid_indices], "KGE")
        else:
            coeff = np.nan
            kge = np.nan

    # Plot clean median line
    ax3.plot(rate_labels, medians, 'o-', color=color_list[model], 
             linewidth=3, markersize=8, label=f'{model} \n(KGE: {kge:.2f})' if model != "HBV" else "HBV", zorder=3)

# Formatting
ax3.set_xlabel('Precipitation Rate [%]', fontsize=fs)
ax3.set_ylabel('Median Peak Discharge [mm/d]', fontsize=fs)
ax3.set_title('Median Peak Discharge vs Precipitation Rate', fontsize=fs)
ax3.tick_params(labelsize=fs)
ax3.grid(True, alpha=0.3)
# set the legend outside the plot to the right
ax3.legend(fontsize=fs*0.85, loc='upper right', bbox_to_anchor=(1.4, 1))

# Set x-axis ticks
ax3.set_xticks(rate_labels)
ax3.set_xticklabels(rate_labels)

# Add some styling
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()

# Save the clean median figure
output_file3 = os.path.join(output_dir, "peak_discharge_vs_precip_rate_median_only.png")
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"Clean median-only comparison plot saved to: {output_file3}")
plt.close()

### FIGURE 3: Peak Discharge vs Peak Duration - Rate Comparison Summary   
# Create a summary comparison across all rates
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Peak Discharge vs Peak Duration', 
                     fontsize=fs)

for rate_idx, rate in enumerate([10, 30, 50]):  # Skip rate 0
    ax = axes[rate_idx]
    
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        # Calculate median values for each peak length (1, 3, 5 days)
        medians = []
        
        for peak in peak_labels:
            col_name = f'sim_peak_flow_peak{peak}_rate{rate}'
            if col_name in df.columns:
                data = df[col_name].dropna().values
                if len(data) > 0:
                    medians.append(np.median(data))
                else:
                    medians.append(np.nan)
            else:
                medians.append(np.nan)
        
        # Plot clean median line
        ax.plot(peak_labels, medians, 'o-', color=color_list[model], 
                linewidth=3, markersize=8, label=f'{model}', zorder=3)

    # Formatting
    ax.set_xlabel('Peak Duration [d]', fontsize=fs)
    ax.set_ylabel('Median Peak Discharge [mm/d]', fontsize=fs)
    ax.set_title(f'{rate}% Rate', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)  
    ax.set_ylim(0,130)  
    # Set x-axis ticks
    ax.set_xticks(peak_labels)
    ax.set_xticklabels(peak_labels)

plt.tight_layout()

fig.legend(model_list, fontsize=fs*0.8, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
# Save the summary comparison figure
output_summary = os.path.join(output_dir, "peak_discharge_vs_peak_duration_rate_comparison_summary.png")
plt.savefig(output_summary, dpi=300, bbox_inches='tight')
print(f"Rate comparison summary plot saved to: {output_summary}")
plt.close()

### FIGURE 4: Peak Flow vs Precipitation Rate by Peak Duration
# This shows precipitation rate on x-axis and mean peak flow per model for each peak length
fig, axes = plt.subplots(1, 3, figsize=(19, 8))
fig.subplots_adjust(wspace=0.4)

for i, peak_duration in enumerate([1, 3, 5]):
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        medians = []
        
        # Define rates based on peak duration (include rate 0 only for peak=1)
        if peak_duration == 1:
            rates = [0, 10, 30, 50]
            x_values = [0, 10, 30, 50]
        else:
            rates = [10, 30, 50]
            x_values = [10, 30, 50]
        
        # For each precipitation rate, get the peak flow for this peak duration
        for rate in rates:
            col_name = f'sim_peak_flow_peak{peak_duration}_rate{rate}'
            if col_name in df.columns:
                data = df[col_name].dropna().values
                if len(data) > 0:
                    medians.append(np.median(data))
                else:
                    medians.append(np.nan)
            else:
                medians.append(np.nan)
        
        # Plot the line for this model
        axes[i].plot(x_values, medians, marker='o', linewidth=2, markersize=6, 
                    label=model, color=color_list[model])
    
    axes[i].set_xlabel('Precipitation Rate [%]', fontsize=fs)
    axes[i].set_ylabel('Median Peak Discharge [mm/d]', fontsize=fs)
    axes[i].set_title(f'{peak_duration} Day Peak Duration', fontsize=fs)
    axes[i].tick_params(labelsize=fs)
    
    # Set x-axis ticks based on peak duration
    if peak_duration == 1:
        axes[i].set_xticks([0, 10, 30, 50])
    else:
        axes[i].set_xticks([10, 30, 50])
    
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(0, 135)  # Set a common y-limit for better comparison

fig.legend(model_list, fontsize=fs*0.9, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))

plt.suptitle('Median Peak Discharge vs Precipitation Rate by Peak Duration', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_flow_vs_precip_rate_by_duration.png", dpi=300, bbox_inches='tight')
plt.close()

### FIGURE 5: Peak Days vs Mean Time Lag
# This shows peak durations on x-axis and mean time lag between observed and simulated results
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
#space between subplots
fig.subplots_adjust(wspace=0.4)

for i, rate in enumerate([10, 30, 50]):
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        mean_lags = []
        
        # For each peak duration, get the time lag for this rate
        for peak_duration in [1, 3, 5]:
            lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
            if lag_col in df.columns:
                data = df[lag_col].dropna().values
                if len(data) > 0:
                    mean_lags.append(np.mean(data))
                else:
                    mean_lags.append(np.nan)
            else:
                mean_lags.append(np.nan)
        
        # Plot the line for this model
        axes[i].plot([1, 3, 5], mean_lags, marker='o', linewidth=2, markersize=6, 
                    label=model, color=color_list[model])
    
    axes[i].set_xlabel('Peak Duration [d]', fontsize=fs)
    axes[i].set_ylabel('Mean Time Lag [d]', fontsize=fs)
    axes[i].set_title(f'Rate {rate}%', fontsize=fs)
    axes[i].tick_params(labelsize=fs)
    axes[i].set_xticks([1, 3, 5])
    axes[i].set_ylim(0,8)
    axes[i].grid(True, alpha=0.3)

# add the legend at the bottom of the figure
fig.legend(model_list, fontsize=fs*0.8, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))

plt.suptitle('Mean Time Lag vs Peak Duration by Rate', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_days_vs_mean_time_lag.png", dpi=300, bbox_inches='tight')
plt.close()

### FIGURE 5: Peak Days vs Median Time Lag
# This shows peak durations on x-axis and median time lag between observed and simulated results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#space between subplots
fig.subplots_adjust(wspace=0.4)

for i, rate in enumerate([10, 30, 50]):
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        median_lags = []
        
        # For each peak duration, get the time lag for this rate
        for peak_duration in [1, 3, 5]:
            lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
            if lag_col in df.columns:
                data = df[lag_col].dropna().values
                if len(data) > 0:
                    median_lags.append(np.median(data))
                else:
                    median_lags.append(np.nan)
            else:
                median_lags.append(np.nan)
        
        # Plot the line for this model with special formatting for HBV
        if model == "HBV":
            # Make HBV more visible with thicker line and dashed style
            axes[i].plot([1, 3, 5], median_lags, marker='s', linewidth=4, markersize=8, 
                        linestyle=':', label=model, color=color_list[model], zorder=5)
        else:
            axes[i].plot([1, 3, 5], median_lags, marker='o', linewidth=2, markersize=6, 
                        label=model, color=color_list[model])
    
    axes[i].set_xlabel('Peak Duration [d]', fontsize=fs)
    axes[i].set_ylabel('Median Time Lag [d]', fontsize=fs)
    axes[i].set_title(f'Rate {rate}%', fontsize=fs)
    axes[i].tick_params(labelsize=fs)
    axes[i].set_xticks([1, 3, 5])
    axes[i].set_ylim(0,6)
    axes[i].grid(True, alpha=0.3)

# add the legend at the bottom of the figure
fig.legend(model_list, fontsize=fs*0.8, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))

plt.suptitle('Median Time Lag vs Peak Duration by Rate', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_days_vs_median_time_lag.png", dpi=300, bbox_inches='tight')
plt.close()

plt.savefig(f"{output_dir}/peak_time_lag_vs_precip_rate_by_duration.png", dpi=300, bbox_inches='tight')

### FIGURE 6: Peak time lag vs Precipitation Rate by Peak Duration
# This shows precipitation rate on x-axis and mean time lag of peak discharge flow per model for each peak length
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.subplots_adjust(wspace=0.4)

for i, peak_duration in enumerate([1, 3, 5]):
    # Define rates and x_values outside the model loop
    if peak_duration == 1:
        rates = [0, 10, 30, 50]
        x_values = [0, 10, 30, 50]
    else:
        rates = [10, 30, 50]
        x_values = [10, 30, 50]
    
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        means = []
        
        # For each precipitation rate, get the peak time lag for this peak duration
        for rate in rates:
            lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
            if lag_col in df.columns:
                data = df[lag_col].dropna().values
                if len(data) > 0:
                    means.append(np.mean(data))
                else:
                    means.append(np.nan)
            else:
                means.append(np.nan)
                # Only print warning for the first model to avoid spam
                if model == model_list[0]:
                    print(f"Warning: Column not found: {lag_col}")

        # Plot the line for this model with enhanced HBV visibility
        if model == "HBV":
            axes[i].plot(x_values, means, marker='s', linewidth=4, markersize=8, 
                        linestyle=':', label=model, color=color_list[model], zorder=5)
        else:
            axes[i].plot(x_values, means, marker='o', linewidth=2, markersize=6, 
                        label=model, color=color_list[model])
        
    axes[i].set_xlabel('Precipitation Rate [%]', fontsize=fs)
    axes[i].set_ylabel('Mean Time Lag [d]', fontsize=fs)
    axes[i].set_title(f'{peak_duration} Day Peak Duration', fontsize=fs)
    axes[i].tick_params(labelsize=fs)
    axes[i].set_xticks(x_values)
    # Remove fixed y-limit to allow data to determine the range
    # axes[i].set_ylim(0,8)  # Commented out to let matplotlib auto-scale
    axes[i].grid(True, alpha=0.3)

fig.legend(model_list, fontsize=fs*0.9, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
plt.suptitle('Mean Time Lag vs Precipitation Rate by Peak Duration', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_time_lag_vs_precip_rate_by_duration.png", dpi=300, bbox_inches='tight')
plt.close()

### FIGURE 6: Peak time lag vs Precipitation Rate by Peak Duration
# This shows precipitation rate on x-axis and mean time lag of peak discharge flow per model for each peak length
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.subplots_adjust(wspace=0.4)

for i, peak_duration in enumerate([1, 3, 5]):
    # Define rates and x_values outside the model loop
    if peak_duration == 1:
        rates = [0, 10, 30, 50]
        x_values = [0, 10, 30, 50]
    else:
        rates = [10, 30, 50]
        x_values = [10, 30, 50]
    
    for model in model_list:
        if model not in all_model_data:
            continue
        
        df = all_model_data[model]
        means = []
        
        # For each precipitation rate, get the peak time lag for this peak duration
        for rate in rates:
            lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
            if lag_col in df.columns:
                data = df[lag_col].dropna().values
                if len(data) > 0:
                    means.append(np.mean(data))
                else:
                    means.append(np.nan)
            else:
                means.append(np.nan)
                # Only print warning for the first model to avoid spam
                if model == model_list[0]:
                    print(f"Warning: Column not found: {lag_col}")

        # Plot the line for this model with enhanced HBV visibility
        if model == "HBV":
            axes[i].plot(x_values, means, marker='s', linewidth=4, markersize=8, 
                        linestyle=':', label=model, color=color_list[model], zorder=5)
        else:
            axes[i].plot(x_values, means, marker='o', linewidth=2, markersize=6, 
                        label=model, color=color_list[model])
        
    axes[i].set_xlabel('Precipitation Rate [%]', fontsize=fs)
    axes[i].set_ylabel('Mean Time Lag [d]', fontsize=fs)
    axes[i].set_title(f'{peak_duration} Day Peak Duration', fontsize=fs)
    axes[i].tick_params(labelsize=fs)
    axes[i].set_xticks(x_values)
    # Remove fixed y-limit to allow data to determine the range
    # axes[i].set_ylim(0,8)  # Commented out to let matplotlib auto-scale
    axes[i].grid(True, alpha=0.3)

fig.legend(model_list, fontsize=fs*0.9, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
plt.suptitle('Mean Time Lag vs Precipitation Rate by Peak Duration', fontsize=fs)
plt.tight_layout()
plt.savefig(f"{output_dir}/peak_time_lag_vs_precip_rate_by_duration.png", dpi=300, bbox_inches='tight')
plt.close()



print("All plots saved successfully!")
print("Generated files:")
print("- peak_discharge_vs_precip_rate_comparison.png")
print("- peak_discharge_vs_peak_duration_comparison.png")
print("- peak_discharge_vs_precip_rate_mean_only.png")
print("- peak_discharge_vs_peak_duration_rate_comparison_summary.png")
print("- peak_flow_vs_precip_rate_by_duration.png") 
print("- peak_days_vs_time_lag.png")