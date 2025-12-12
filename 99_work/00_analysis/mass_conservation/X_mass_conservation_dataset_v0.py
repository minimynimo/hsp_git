import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Load basin characteristics
basin_char_dir = "/data0/funato/hsp/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv"
basin_char_df = pd.read_csv(basin_char_dir, index_col='File_num')

# Initialize storage for mass conservation data
mass_conservation_data = []

print("Processing mass conservation analysis...")
print(f"Evaluation period: {start_date_eva} to {end_date_eva}")

# Process observed data first (from HBV files)
print("Processing observed data...")
for file in range(1, file_tot_num + 1):
    if file % 50 == 0 or file == 1:
        print(f"  Processing file: {file}")
    
    # Get basin characteristics
    if file not in basin_char_df.index:
        print(f"    Basin characteristics not found for file {file}. Skipping.")
        continue
        
    base_flow_index = basin_char_df.loc[file, 'baseflow_index']
    
    # Read HBV file for observed data
    input_file = f"{input_dir_base}/HBV/varssim_HBV/varssim{file:03d}.csv"
    
    if not os.path.exists(input_file):
        print(f"    File {input_file} does not exist. Skipping.")
        continue
    
    try:
        # Read the input file
        input_df = pd.read_csv(input_file, parse_dates=['Date'])
        input_df.set_index('Date', inplace=True)
        
        # Filter for evaluation period
        input_df_eva = input_df[start_date_eva:end_date_eva]
        
        # Check if required columns exist
        required_cols = ['Precip', 'Obs flow']
        if not all(col in input_df_eva.columns for col in required_cols):
            print(f"    Missing required columns in {input_file}. Skipping.")
            continue
        
        # Remove rows with missing values
        input_df_eva = input_df_eva[required_cols].dropna()
        
        if len(input_df_eva) == 0:
            print(f"    No valid data in {input_file}. Skipping.")
            continue
        
        # Calculate mass conservation components for observed data
        # Storm flow = Total flow * (1 - baseflow_index)
        total_flow_obs = input_df_eva['Obs flow'].sum()  # mm total
        total_precip_obs = input_df_eva['Precip'].sum()  # mm total
        
        daily_mean_flow_obs = input_df_eva['Obs flow'].mean()  # mm/day
        daily_mean_precip_obs = input_df_eva['Precip'].mean()  # mm/day
        
        # Storm flow (quick flow) = total flow - base flow
        daily_mean_storm_flow_obs = daily_mean_flow_obs * (1 - base_flow_index)
        daily_mean_base_flow_obs = daily_mean_flow_obs * base_flow_index
        
        # Store observed data
        obs_entry = {
            'Model': 'Observed',
            'File': file,
            'Basin_ID': f'varssim{file:03d}',
            'Base_Flow_Index': base_flow_index,
            'Daily_Mean_Precip': daily_mean_precip_obs,
            'Daily_Mean_Total_Flow': daily_mean_flow_obs,
            'Daily_Mean_Storm_Flow': daily_mean_storm_flow_obs,
            'Daily_Mean_Base_Flow': daily_mean_base_flow_obs,
            'Total_Precip': total_precip_obs,
            'Total_Flow': total_flow_obs,
            'Mass_Balance_Ratio': daily_mean_storm_flow_obs / daily_mean_precip_obs if daily_mean_precip_obs > 0 else np.nan,
            'Data_Points': len(input_df_eva)
        }
        
        mass_conservation_data.append(obs_entry)
        
    except Exception as e:
        print(f"    Error processing observed data in {input_file}: {e}")
        continue

print(f"Processed observed data for {len([d for d in mass_conservation_data if d['Model'] == 'Observed'])} basins")

# Process each model



for model in model_list:
    print(f"Processing model: {model}")
    
    for file in range(1, file_tot_num + 1):
        if file % 50 == 0 or file == 1:
            print(f"  Processing file: {file}")
        
        # Get basin characteristics
        if file not in basin_char_df.index:
            continue
            
        base_flow_index = basin_char_df.loc[file, 'baseflow_index']
        
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
            required_cols = ['Precip', 'og']  # 'og' is the simulated discharge
            if not all(col in input_df_eva.columns for col in required_cols):
                continue
            
            # Remove rows with missing values
            input_df_eva = input_df_eva[required_cols].dropna()
            
            if len(input_df_eva) == 0:
                continue
            
            # Calculate mass conservation components for simulated data
            total_flow_sim = input_df_eva['og'].sum()  # mm total
            total_precip_sim = input_df_eva['Precip'].sum()  # mm total
            
            daily_mean_flow_sim = input_df_eva['og'].mean()  # mm/day
            daily_mean_precip_sim = input_df_eva['Precip'].mean()  # mm/day
            
            # Storm flow (quick flow) = total flow - base flow
            daily_mean_storm_flow_sim = daily_mean_flow_sim * (1 - base_flow_index)
            daily_mean_base_flow_sim = daily_mean_flow_sim * base_flow_index
            
            # Store simulated data
            sim_entry = {
                'Model': model,
                'File': file,
                'Basin_ID': f'varssim{file:03d}',
                'Base_Flow_Index': base_flow_index,
                'Daily_Mean_Precip': daily_mean_precip_sim,
                'Daily_Mean_Total_Flow': daily_mean_flow_sim,
                'Daily_Mean_Storm_Flow': daily_mean_storm_flow_sim,
                'Daily_Mean_Base_Flow': daily_mean_base_flow_sim,
                'Total_Precip': total_precip_sim,
                'Total_Flow': total_flow_sim,
                'Mass_Balance_Ratio': daily_mean_storm_flow_sim / daily_mean_precip_sim if daily_mean_precip_sim > 0 else np.nan,
                'Data_Points': len(input_df_eva)
            }
            
            mass_conservation_data.append(sim_entry)
            
        except Exception as e:
            print(f"    Error processing {input_file}: {e}")
            continue
    
    print(f"  Completed {model}: {len([d for d in mass_conservation_data if d['Model'] == model])} basins processed")

# Convert to DataFrame and save
mass_conservation_df = pd.DataFrame(mass_conservation_data)

if mass_conservation_df.empty:
    print("No data to save!")
    exit()

# Save the dataset
output_csv = f"{output_dir}/mass_conservation_data.csv"
mass_conservation_df.to_csv(output_csv, index=False)
print(f"\nMass conservation dataset saved to: {output_csv}")

# Print summary statistics
print("\n" + "="*60)
print("MASS CONSERVATION DATASET SUMMARY")
print("="*60)
print(f"Total records: {len(mass_conservation_df)}")
print(f"Models processed: {mass_conservation_df['Model'].unique()}")
print(f"Number of basins per model:")

for model in mass_conservation_df['Model'].unique():
    model_data = mass_conservation_df[mass_conservation_df['Model'] == model]
    print(f"  {model}: {len(model_data)} basins")

print(f"\nData columns:")
for col in mass_conservation_df.columns:
    print(f"  - {col}")

print(f"\nDataset saved for visualization analysis.")
print("Run the visualization script to analyze mass conservation properties.")

