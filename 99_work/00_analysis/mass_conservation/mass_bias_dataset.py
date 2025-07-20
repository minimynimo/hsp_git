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
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

# Load basin characteristics
basin_char_dir = "/data0/funato/hsp/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv"
basin_char_df = pd.read_csv(basin_char_dir, index_col='File_num')

# Initialize storage for mass bias data
mass_bias_data = []

print("Processing mass bias analysis...")
print(f"Evaluation period: {start_date_eva} to {end_date_eva}")

# Process each model
for model in model_list:
    print(f"Processing model: {model}")
    
    for file in range(1, file_tot_num + 1):
        if file % 50 == 0 or file == 1:
            print(f"  Processing file: {file}")
        
        # Get basin characteristics
        if file not in basin_char_df.index:
            continue
        
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
            required_cols = ['og', 'Obs flow']
            if not all(col in input_df_eva.columns for col in required_cols):
                continue
            
            # Remove rows with missing values
            input_df_eva = input_df_eva[required_cols].dropna()
            
            if len(input_df_eva) == 0:
                continue
            
            # Calculate mass bias components
            sum_obs_flow = input_df_eva['Obs flow'].sum()  # Sum of observed flow
            sum_sim_flow = input_df_eva['og'].sum()  # Sum of simulated flow
            
            # Calculate mass bias: abs(sum_obs - sum_sim) / sum_obs
            if sum_obs_flow > 0:
                mass_bias = abs(sum_obs_flow - sum_sim_flow) / sum_obs_flow
            else:
                mass_bias = np.nan
            
            # Store mass bias data
            bias_entry = {
                'Model': model,
                'File': file,
                'Basin_ID': f'varssim{file:03d}',
                'Sum_Obs_Flow': sum_obs_flow,
                'Sum_Sim_Flow': sum_sim_flow,
                'Mass_Bias': mass_bias,
                'Data_Points': len(input_df_eva)
            }
            
            mass_bias_data.append(bias_entry)
            
        except Exception as e:
            print(f"    Error processing {input_file}: {e}")
            continue
    
    print(f"  Completed {model}: {len([d for d in mass_bias_data if d['Model'] == model])} basins processed")

# Convert to DataFrame and save
mass_bias_df = pd.DataFrame(mass_bias_data)

if mass_bias_df.empty:
    print("No data to save!")
    exit()

# Save the dataset
output_csv = f"{output_dir}/mass_bias_data.csv"
mass_bias_df.to_csv(output_csv, index=False)
print(f"\nMass bias dataset saved to: {output_csv}")

# Print summary statistics
print("\n" + "="*60)
print("MASS BIAS DATASET SUMMARY")
print("="*60)
print(f"Total records: {len(mass_bias_df)}")
print(f"Models processed: {mass_bias_df['Model'].unique()}")
print(f"Number of basins per model:")

for model in mass_bias_df['Model'].unique():
    model_data = mass_bias_df[mass_bias_df['Model'] == model]
    print(f"  {model}: {len(model_data)} basins")

print(f"\nData columns:")
for col in mass_bias_df.columns:
    print(f"  - {col}")

# Print mass bias statistics by model
print(f"\nMass Bias Statistics by Model:")
print("-" * 40)
for model in mass_bias_df['Model'].unique():
    model_data = mass_bias_df[mass_bias_df['Model'] == model]
    valid_bias = model_data['Mass_Bias'].dropna()
    if len(valid_bias) > 0:
        print(f"{model}:")
        print(f"  Mean Mass Bias: {valid_bias.mean():.4f}")
        print(f"  Median Mass Bias: {valid_bias.median():.4f}")
        print(f"  Std Mass Bias: {valid_bias.std():.4f}")
        print(f"  Min Mass Bias: {valid_bias.min():.4f}")
        print(f"  Max Mass Bias: {valid_bias.max():.4f}")
        print()

print(f"\nDataset saved for visualization analysis.")
print("Mass bias calculation: abs(sum_obs - sum_sim) / sum_obs")

