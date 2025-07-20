import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424

start_date_cal = "1993-01-01"
end_date_cal = "2000-12-31"
start_date_eva = "2001-01-01"
end_date_eva = "2006-12-31"

# Paths
input_dir_base = "/data0/funato/hsp/0_out"
output_dir = "/data0/funato/hsp/0_out/analysis/mass_conservation"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

# Load basin characteristics
#basin_char_dir = "/data0/funato/hsp/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv"
#basin_char_df = pd.read_csv(basin_char_dir, index_col='File_num')

# Initialize storage for water balance data
water_balance_data = []

print("Processing water balance analysis...")
print(f"Evaluation period: {start_date_eva} to {end_date_eva}")

# First, calculate ET using observed data
print("Processing observed data...")
for file in range(1, file_tot_num + 1):
    if file % 50 == 0 or file == 1:
        print(f"  Processing file: {file}")

    
    # Read HBV file for observed data
    input_file = f"{input_dir_base}/HBV/varssim_HBV/varssim{file:03d}.csv"
    
    if not os.path.exists(input_file):
        continue
    
    try:
        # Read the input file
        input_df = pd.read_csv(input_file, parse_dates=['Date'])
        input_df.set_index('Date', inplace=True)
        
        # Filter for evaluation period
        input_df = input_df[start_date_cal:end_date_eva]
        
        # Check if required columns exist
        required_cols = ['Precip', 'Obs flow']
        if not all(col in input_df.columns for col in required_cols):
            continue
        
        # Remove rows with missing values
        input_df = input_df[required_cols].dropna()

        if len(input_df) == 0:
            continue
        
        # Calculate water balance components using observed data
        # P - Q = ΔS + ET (assuming ΔS ≈ 0 for long-term averages)
        # Therefore: ET = P - Q
        daily_mean_precip = input_df['Precip'].mean()  # mm/day
        daily_mean_discharge = input_df['Obs flow'].mean()  # mm/day
        daily_mean_et = daily_mean_precip - daily_mean_discharge  # mm/day

        # Get basin characteristics as separate columns
        #basin_char_dict = basin_char_df.loc[file].to_dict()
        
        # Create entry for observed data
        obs_entry = {
            'File_num': file,
            'Model': 'Observed',
            'Daily_Mean_ET': daily_mean_et,
            'Daily_Mean_Precipitation': daily_mean_precip,
            'Daily_Mean_Discharge': daily_mean_discharge,
            'Data_Points': len(input_df)
        }
        
        # Add basin characteristics as separate columns
        #obs_entry.update(basin_char_dict)
        
        water_balance_data.append(obs_entry)
        
    except Exception as e:
        print(f"    Error processing observed data for file {file}: {e}")

print(f"Processed observed data for {len([d for d in water_balance_data if d['Model'] == 'Observed'])} basins")


# Process each model
for model in model_list:
    print(f"Processing model: {model}")
    
    for file in range(1, file_tot_num + 1):
        if file % 50 == 0 or file == 1:
            print(f"  Processing file: {file}")
        
        
        # Determine input file path based on model
        if model == "HBV":
            input_file = f"{input_dir_base}/{model}/varssim_{model}/varssim{file:03d}.csv"
        elif model == "BC":
            input_file = f"{input_dir_base}/{model}/varssim_{model}/varssim{file:03d}.csv"
        else:
            input_file = f"{input_dir_base}/{model}/GB/shap/regional/varssim_{model}/varssim{file:03d}.csv"
        
        if not os.path.exists(input_file):
            continue
        
        try:
            # Read the input file
            input_df = pd.read_csv(input_file, parse_dates=['Date'])
            input_df.set_index('Date', inplace=True)
            
            # Filter for evaluation period
            input_df_eva = input_df[start_date_eva:end_date_eva]
            
            # Check if required columns exist
            required_cols = ['Precip', 'og']  # Use 'og' for simulated discharge
            if not all(col in input_df_eva.columns for col in required_cols):
                continue
            
            # Remove rows with missing values
            input_df_eva = input_df_eva[required_cols].dropna()
            
            if len(input_df_eva) == 0:
                continue
            
            # Calculate water balance components using model data
            # P - Q_sim = ΔS + ET_sim (assuming ΔS ≈ 0 for long-term averages)
            # Therefore: ET_sim = P - Q_sim
            daily_mean_precip = input_df_eva['Precip'].mean()  # mm/day
            daily_mean_sim_discharge = input_df_eva['og'].mean()  # mm/day (simulated)
            daily_mean_et = daily_mean_precip - daily_mean_sim_discharge  # mm/day

            # Create entry for model data
            model_entry = {
                'File_num': file,
                'Model': model,
                'Daily_Mean_ET': daily_mean_et,
                'Daily_Mean_Precipitation': daily_mean_precip,
                'Daily_Mean_Discharge': daily_mean_sim_discharge,
                'Data_Points': len(input_df_eva)
            }
            
            
            water_balance_data.append(model_entry)
            
        except Exception as e:
            print(f"    Error processing file {file} for model {model}: {e}")
    
    print(f"  Completed {model}: {len([d for d in water_balance_data if d['Model'] == model])} basins processed")

# Convert to DataFrame and save
water_balance_df = pd.DataFrame(water_balance_data)

if water_balance_df.empty:
    print("No data to save!")
    exit()

# Save the complete dataset
output_csv = f"{output_dir}/water_balance_data.csv"
water_balance_df.to_csv(output_csv, index=False)
print(f"\nWater balance dataset saved to: {output_csv}")

# Print summary statistics
print("\n" + "="*60)
print("WATER BALANCE DATASET SUMMARY")
print("="*60)
print(f"Total records: {len(water_balance_df)}")
print(f"Models processed: {water_balance_df['Model'].unique()}")
print(f"Number of basins per model:")

for model in water_balance_df['Model'].unique():
    model_data = water_balance_df[water_balance_df['Model'] == model]
    print(f"  {model}: {len(model_data)} basins")

print(f"\nData columns:")
for col in water_balance_df.columns:
    print(f"  - {col}")

# Print ET statistics by model
print(f"\nDaily Mean ET Statistics by Model:")
print("-" * 40)
for model in water_balance_df['Model'].unique():
    model_data = water_balance_df[water_balance_df['Model'] == model]
    valid_et = model_data['Daily_Mean_ET'].dropna()
    if len(valid_et) > 0:
        print(f"{model}:")
        print(f"  Mean ET: {valid_et.mean():.4f} mm/day")
        print(f"  Median ET: {valid_et.median():.4f} mm/day")
        print(f"  Std ET: {valid_et.std():.4f} mm/day")
        print(f"  Min ET: {valid_et.min():.4f} mm/day")
        print(f"  Max ET: {valid_et.max():.4f} mm/day")
        print()

print(f"\nDataset saved for visualization analysis.")
print("Water balance equation: ET = P - Q (assuming ΔS ≈ 0)")