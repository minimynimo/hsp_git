import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns 

# Configuration
fs = 24  # Font size
output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/basin_with_largest_precip"
os.makedirs(output_dir, exist_ok=True)

# Analysis parameters
file_tot_num = 424 
start_date_eva = "2001-01-01"
end_date_eva = "2001-02-10" 
file_list = [48]  # Specific basins to analyze

# Model configuration
model_list = ["HBV", "HYPER", "RF", "XGB", "TCN", "LSTM"]

# Define colors for each model
color_list = {
    "HBV": "#7B7B7B",    # Medium Grey
    "HYPER": "#628AE8",  # Steel Blue
    "RF": "#FF7F00",     # Orange
    "XGB": "#9C4803",    # Dark Orange
    "TCN": "#E41A1C",    # Red
    "LSTM": "#9808B2"    # Purple
}


def BMK(obs_data, sim_data, benchmark):
    obs_data = np.array(obs_data).flatten()
    sim_data = np.array(sim_data).flatten()

    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    if benchmark == "KGE":
        # Handle edge cases
        if np.std(obs_data) == 0 or np.std(sim_data) == 0:
            return np.nan
        if np.mean(obs_data) == 0:
            return np.nan
            
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2) 
    
    elif benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.square(obs_data - obs_ave))
        if denom == 0:
            return np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / denom)
    
    elif benchmark == "logNSE":
        temp_obs_data = np.where((obs_data <= 0) | np.isnan(obs_data), 1e-6, obs_data)
        temp_sim_data = np.where((sim_data <= 0) | np.isnan(sim_data), 1e-6, sim_data)
        obs_ave = np.mean(temp_obs_data)
        numer = np.sum((np.log(temp_sim_data) - np.log(temp_obs_data))**2)
        denom = np.sum((np.log(temp_obs_data) - np.log(obs_ave))**2) + 1e-8
        return 1 - numer / denom
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0:
            return np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)
    
    elif benchmark == "Erel":
        obs_ave = np.mean(obs_data)
        temp_obs_data = np.where(obs_data == 0, 1e-6, obs_data)
        numer = np.sum(np.square((temp_obs_data - sim_data) / temp_obs_data))
        denom = np.sum(np.square((obs_data - obs_ave) / obs_ave)) + 1e-8
        return 1 - numer / denom
    
    elif benchmark == "VE":
        obs_sum = np.sum(obs_data)
        if obs_sum == 0:
            return np.nan
        return 1 - np.sum(np.abs(obs_data - sim_data)) / obs_sum
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave))) + 1e-8
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

# Dictionary to store all model data for specific basins
all_model_data = {}

print(f"Analyzing basins: {file_list}")
print("Starting to load model data...")

# Load data for each model (similar structure to peak_discharge_use.py)
for model in model_list:
    if model == "HBV":
        input_file = "/data0/funato/hsp/0_out/HBV/peak_discharge/peak_discharge_comparison_HBV.csv"
    elif model == "HYPER":
        input_file = "/data0/funato/hsp/0_out/BC/peak_discharge/peak_discharge_comparison_BC.csv"
    else:
        input_file = f"/data0/funato/hsp/0_out/{model}/peak_discharge/peak_discharge_comparison_{model}.csv"
    
    print(f"Trying to load {model} from: {input_file}")
    
    try:
        if os.path.exists(input_file):
            df = pd.read_csv(input_file, header=0, index_col=None)
            print(f"Successfully loaded {model}: {len(df)} total records")
            #print(f"Columns in {model}: {list(df.columns)}")
            
            # Check for file identifier column
            if 'file_num' in df.columns:
                df_filtered = df[df['file_num'].isin(file_list)].copy()
                print(f"Filtered by file_num column: {len(df_filtered)} records found")
            elif 'file_id' in df.columns:
                df_filtered = df[df['file_id'].isin(file_list)].copy()
                print(f"Filtered by file_id column")
            elif 'basin_id' in df.columns:
                df_filtered = df[df['basin_id'].isin(file_list)].copy()
                print(f"Filtered by basin_id column")
            elif df.columns[0] in ['file', 'File', 'basin', 'Basin']:
                df_filtered = df[df.iloc[:, 0].isin(file_list)].copy()
                print(f"Filtered by first column: {df.columns[0]}")
            else:
                # Just use all data if no clear identifier
                df_filtered = df.copy()
                print(f"No clear file identifier found, using all data")
            
            all_model_data[model] = df_filtered
            print(f"Final {model} data: {len(df_filtered)} records for analysis")
        else:
            print(f"File does not exist: {input_file}")
        
    except Exception as e:
        print(f"Error loading {model}: {e}")
        continue

print(f"Loaded data for models: {list(all_model_data.keys())}")

if not all_model_data:
    print("No data found for any models!")
    exit()

# Define precipitation rate and peak duration parameters
rate_labels = [0, 10, 30, 50]
rate_cols = ['sim_peak_flow_peak1_rate0', 'sim_peak_flow_peak1_rate10', 
             'sim_peak_flow_peak1_rate30', 'sim_peak_flow_peak1_rate50']
peak_labels = [1, 3, 5]

#### FIGURE 1: Basin-Specific Peak Discharge vs Precipitation Rate ####
print("Creating precipitation rate analysis...")

plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle('Peak Discharge vs Precipitation Rate\n(Basins: ' + ', '.join(map(str, file_list)) + ')', 
             fontsize=fs, y=0.95)

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
    
    # Calculate statistics for each rate
    medians = []
    p10s = []
    p90s = []
    
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
            linewidth=3, markersize=8, label='Median', zorder=3)
    
    # Plot uncertainty band (10th-90th percentile) if we have multiple basins
    if len(file_list) > 1:
        ax.fill_between(rate_labels, p10s, p90s, color=color_list[model], 
                        alpha=0.3, label='10th-90th percentile', zorder=1)
        ax.plot(rate_labels, p10s, '--', color=color_list[model], 
                alpha=0.7, linewidth=1, zorder=2)
        ax.plot(rate_labels, p90s, '--', color=color_list[model], 
                alpha=0.7, linewidth=1, zorder=2)
    
    # Formatting
    ax.set_xlabel('Precipitation Rate [%]', fontsize=fs)
    ax.set_ylabel('Peak Discharge [mm/d]', fontsize=fs)
    ax.set_title(f'{model}', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs*0.7, loc='upper left')
    ax.set_xticks(rate_labels)
    ax.set_xticklabels(rate_labels)

# Save the figure
output_file1 = os.path.join(output_dir, "basin_specific_peak_discharge_vs_precip_rate_for_testing.png")
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"Basin-specific precipitation rate analysis saved to: {output_file1}")
plt.close()

#### FIGURE 2: Basin-Specific Model Comparison ####
print("Creating model comparison analysis...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate HBV medians for comparison
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

# Plot comparison for all models
for model in model_list:
    if model not in all_model_data:
        continue
        
    df = all_model_data[model]
    medians = []
    
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
    
    # Calculate KGE compared to HBV
    if model == "HBV":
        kge = 1.0
        label = "HBV"
    else:
        valid_indices = ~(np.isnan(medians) | np.isnan(hbv_medians))
        if np.sum(valid_indices) > 1:
            kge = BMK(np.array(hbv_medians)[valid_indices], np.array(medians)[valid_indices], "KGE")
        else:
            kge = np.nan
        label = f'{model} (KGE: {kge:.2f})'
    
    # Plot model line
    ax.plot(rate_labels, medians, 'o-', color=color_list[model], 
            linewidth=3, markersize=8, label=label, zorder=3)

# Formatting
ax.set_xlabel('Precipitation Rate [%]', fontsize=fs)
ax.set_ylabel('Median Peak Discharge [mm/d]', fontsize=fs)
ax.set_title('Model Comparison: Peak Discharge vs Precipitation Rate\n(Basins: ' + 
             ', '.join(map(str, file_list)) + ')', fontsize=fs)
ax.tick_params(labelsize=fs)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=fs*0.8, loc='upper left')
ax.set_xticks(rate_labels)
ax.set_xticklabels(rate_labels)

# Save the figure
output_file2 = os.path.join(output_dir, "basin_specific_model_comparison_for_testing.png")
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Basin-specific model comparison saved to: {output_file2}")
plt.close()

print(f"\nAnalysis complete! Results saved to: {output_dir}")
print(f"Basin files analyzed: {file_list}")
print(f"Models included: {list(all_model_data.keys())}") 
