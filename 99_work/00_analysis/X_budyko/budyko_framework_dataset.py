import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

model_list = ["HBV", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424
start_date_eva = "2001-01-01"
end_date_eva = "2006-12-31"

output_dir = "/data0/funato/hsp/0_out/analysis/budyko_framework"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/budyko_framework"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

# Initialize storage for Budyko data
budyko_data = []
observed_data = {}  # Store observed values once per basin

def theoretical_budyko_curve(phi, w=2.6):
    """
    Calculate theoretical Budyko curve
    phi: aridity index (PET/P)
    w: parameter (default 2.6 for global average)
    Returns: evapotranspiration ratio (ET/P)
    """
    return (phi * np.tanh(1/phi) * (1 - np.exp(-phi)))**0.5

# First pass: collect observed data from any model (they're the same)
print("Collecting observed data from HBV files...")
for file in range(1, file_tot_num + 1):
    if file % 50 == 0 or file == 1:
        print(f"  Processing file: {file}")
        
    input_file = f"/data0/funato/hsp/0_out/HBV/varssim_HBV/varssim{file:03d}.csv"
    
    if not os.path.exists(input_file):
        continue
    
    try:
        # Read the input file
        df = pd.read_csv(input_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        # Filter for evaluation period
        eval_df = df[start_date_eva:end_date_eva]
        
        # Check if required columns exist
        required_cols = ['Precip', 'PET', 'Obs flow']
        if not all(col in eval_df.columns for col in required_cols):
            continue
        
        # Remove rows with any missing values in required columns
        eval_df = eval_df[required_cols].dropna()
        
        if len(eval_df) == 0:
            continue
        
        # Calculate long-term averages for observed data
        long_term_P = eval_df['Precip'].mean()  # mm/day
        long_term_PET = eval_df['PET'].mean()   # mm/day  
        long_term_Q_obs = eval_df['Obs flow'].mean()  # mm/day (observed)
        
        # Check for valid values
        if long_term_P <= 0 or long_term_PET <= 0:
            continue
        
        # Calculate observed Budyko framework components
        aridity_index = long_term_PET / long_term_P  # PET/P
        runoff_ratio_obs = long_term_Q_obs / long_term_P  # Q/P (observed)
        et_ratio_obs = (long_term_P - long_term_Q_obs) / long_term_P  # ET/P (observed)
        
        # Store observed data
        observed_data[file] = {
            'Basin_ID': f'varssim{file:03d}',
            'Aridity_Index': aridity_index,
            'Runoff_Ratio_Obs': runoff_ratio_obs,
            'ET_Ratio_Obs': et_ratio_obs,
            'Long_term_P': long_term_P,
            'Long_term_PET': long_term_PET,
            'Long_term_Q_Obs': long_term_Q_obs,
            'Long_term_ET_Obs': long_term_P - long_term_Q_obs
        }
        
    except Exception as e:
        print(f"    Error processing {input_file}: {e}")
        continue

print(f"Collected observed data for {len(observed_data)} basins")

# Process each model
for model_name in model_list:
    print(f"Processing model: {model_name}")
    
    model_budyko_data = []
    
    for file in range(1, file_tot_num + 1):
        if file % 50 == 0 or file == 1:
            print(f"  Processing file: {file}")
            
        # Determine input file path based on model
        if model_name != "HBV":
            input_file = f"/data0/funato/hsp/0_out/{model_name}/GB/shap/regional/varssim_{model_name}/varssim{file:03d}.csv"
        else:
            input_file = f"/data0/funato/hsp/0_out/HBV/varssim_HBV/varssim{file:03d}.csv"
        
        if not os.path.exists(input_file):
            print(f"    File {input_file} does not exist. Skipping.")
            continue
        
        try:
            # Read the input file
            df = pd.read_csv(input_file, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Filter for evaluation period
            eval_df = df[start_date_eva:end_date_eva]
            
            # Check if required columns exist
            required_cols = ['Precip', 'PET', 'og']
            if not all(col in eval_df.columns for col in required_cols):
                print(f"    Missing required columns in {input_file}. Skipping.")
                continue
            
            # Remove rows with any missing values in required columns
            eval_df = eval_df[required_cols].dropna()
            
            if len(eval_df) == 0:
                print(f"    No valid data in {input_file}. Skipping.")
                continue
            
            # Check if we have observed data for this basin
            if file not in observed_data:
                print(f"    No observed data for file {file}. Skipping.")
                continue
            
            # Calculate long-term averages for simulated data
            long_term_P = eval_df['Precip'].mean()  # mm/day
            long_term_PET = eval_df['PET'].mean()   # mm/day  
            long_term_Q = eval_df['og'].mean()      # mm/day (simulated)
            
            # Check for valid values
            if long_term_P <= 0 or long_term_PET <= 0:
                print(f"    Invalid P or PET values in {input_file}. Skipping.")
                continue
            
            # Calculate simulated Budyko framework components
            aridity_index = long_term_PET / long_term_P  # PET/P (x-axis)
            runoff_ratio = long_term_Q / long_term_P     # Q/P (y-axis) - simulated
            et_ratio = (long_term_P - long_term_Q) / long_term_P  # ET/P = (P-Q)/P - simulated
            
            # Get observed data for this basin
            obs_data = observed_data[file]
            
            # Store combined data (simulated + observed)
            budyko_entry = {
                'Model': model_name,
                'File': file,
                'Basin_ID': obs_data['Basin_ID'],
                'Aridity_Index': obs_data['Aridity_Index'],  # Use observed (same for all models)
                'Runoff_Ratio': runoff_ratio,                # Q/P (simulated)
                'Runoff_Ratio_Obs': obs_data['Runoff_Ratio_Obs'],  # Q/P (observed)
                'ET_Ratio': et_ratio,                        # ET/P (simulated)
                'ET_Ratio_Obs': obs_data['ET_Ratio_Obs'],    # ET/P (observed)
                'Long_term_P': obs_data['Long_term_P'],      # Use observed (same for all models)
                'Long_term_PET': obs_data['Long_term_PET'],  # Use observed (same for all models)
                'Long_term_Q': long_term_Q,                  # simulated
                'Long_term_Q_Obs': obs_data['Long_term_Q_Obs'],     # observed
                'Long_term_ET': long_term_P - long_term_Q,          # simulated
                'Long_term_ET_Obs': obs_data['Long_term_ET_Obs']    # observed
            }
            
            budyko_data.append(budyko_entry)
            model_budyko_data.append(budyko_entry)
            
        except Exception as e:
            print(f"    Error processing {input_file}: {e}")
            continue
    
    print(f"  Completed {model_name}: {len(model_budyko_data)} basins processed")

# Convert to DataFrame
budyko_df = pd.DataFrame(budyko_data)

if budyko_df.empty:
    print("No data to analyze!")
    exit()

# Save the data
budyko_csv = f"{output_dir}/budyko_framework_data.csv"
budyko_df.to_csv(budyko_csv, index=False)
print(f"\nBudyko framework data saved to: {budyko_csv}")

print("\nData processing completed!")
print(f"Evaluation period: {start_date_eva} to {end_date_eva}")
print(f"Total basins analyzed: {len(budyko_df)}")
print(f"Models processed: {budyko_df['Model'].unique()}")
print("\nGenerated files:")
print(f"  - Budyko framework data: {budyko_csv}")
print("\nTo create visualizations, run the separate visualization script.")

