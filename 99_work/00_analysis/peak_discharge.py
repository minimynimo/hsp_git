import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns # Import seaborn for plotting

# Configuration for model and directories
model_name = "HBV"
if model_name == "HBV":
    input_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    output_dir = "/data0/funato/hsp/0_out/10_HBV/peak_discharge/"

os.makedirs(output_dir, exist_ok=True)

file_tot_num = 424 # Total number of files to process

# Evaluation period definition
# IMPORTANT: Current end_date_eva is very short. This means annual peaks
# will be maxes within this short period, not true annual maxes over full years.
start_date_eva = "2001-01-01"
end_date_eva = "2001-02-10" # This short period will limit 'annual' analysis to year 2001's subset

# Columns representing different simulated discharge scenarios
# 'up_peak1_rate0' is now the baseline simulation, analogous to 'og' before.
sim_discharge_cols = ['peak1_rate0',
                    'peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                    'peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                    'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

all_comparison_results = [] # List to store yearly results for all files

for file_num in range(1, 2):#file_tot_num + 1):
    # Print progress for every 50 files or the first one
    if file_num % 50 == 0 or file_num == 1:
        print(f"Processing file number: {file_num}")
    
    yearly_results = []  # List to collect results for each year of the current file

    input_file = f"{input_dir}/varssim{file_num:03d}.csv"
    
    # Skip file if it doesn't exist
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}. Skipping.")
        continue

    input_df = pd.read_csv(input_file, parse_dates=['Date'])

    # Filter data to the evaluation period
    eval_period_df = input_df[(input_df['Date'] >= pd.to_datetime(start_date_eva)) &
                            (input_df['Date'] <= pd.to_datetime(end_date_eva))].copy()

    if eval_period_df.empty:
        print(f"No data for evaluation period in file {file_num}. Skipping.")
        continue
    
    # Get unique years in the evaluation period for looping
    # This will typically be just [2001] given your current start/end dates
    years_in_eval_period = sorted(eval_period_df['Date'].dt.year.unique())

    # --- Pre-calculate annual peaks for observed flow and ORIGINAL precipitation (Precip) ---
    # OBSERVED FLOW peaks are from 'Obs flow' column
    obs_flow_annual_peaks_df = eval_period_df.loc[
        eval_period_df.groupby(eval_period_df['Date'].dt.year)['Obs flow'].idxmax()
    ].copy() # Use .copy() to prevent SettingWithCopyWarning

    # ORIGINAL PRECIPITATION peaks are from 'Precip' column
    original_precip_annual_peaks_df = eval_period_df.loc[
        eval_period_df.groupby(eval_period_df['Date'].dt.year)['Precip'].idxmax()
    ].copy() # Use .copy()

    # --- Loop through each year identified in the evaluation period ---
    for year in years_in_eval_period:
        # --- Extract Observed Precipitation and Flow Data for the current year ---
        original_precip_peak_year_data = original_precip_annual_peaks_df[original_precip_annual_peaks_df['Date'].dt.year == year]
        if not original_precip_peak_year_data.empty:
            peak_precip_obs = original_precip_peak_year_data['Precip'].values[0] # Corrected column name: 'Precip'
            original_precip_peak_date = original_precip_peak_year_data['Date'].values[0]
        else:
            peak_precip_obs = np.nan
            original_precip_peak_date = pd.NaT

        obs_flow_peak_year_data = obs_flow_annual_peaks_df[obs_flow_annual_peaks_df['Date'].dt.year == year]
        if not obs_flow_peak_year_data.empty:
            obs_flow_peak = obs_flow_peak_year_data['Obs flow'].values[0] # Corrected column name: 'Obs flow'
            obs_flow_peak_date = obs_flow_peak_year_data['Date'].values[0]
        else:
            obs_flow_peak = np.nan
            obs_flow_peak_date = pd.NaT

        # Initialize results for the current year
        year_result = {
            'file_num': file_num,
            'year': year,
            'peak_precip_obs': peak_precip_obs, # Original peak precip value
            'peak_flow_obs': obs_flow_peak, # Observed peak flow value
            'peak_precip_obs_date': original_precip_peak_date, # Date of original precip peak
            'peak_flow_obs_date': obs_flow_peak_date, # Date of observed flow peak
        }

        # Add scaled peak precipitation values (these are just derived values for comparison)
        # These reflect the *input* peak precip values for the synthetic scenarios
        year_result['sim_peak_precip_10'] = peak_precip_obs * 1.1 if pd.notnull(peak_precip_obs) else np.nan
        year_result['sim_peak_precip_30'] = peak_precip_obs * 1.3 if pd.notnull(peak_precip_obs) else np.nan
        year_result['sim_peak_precip_50'] = peak_precip_obs * 1.5 if pd.notnull(peak_precip_obs) else np.nan

        # Calculate observed flow lag relative to ORIGINAL precipitation peak
        if pd.notnull(obs_flow_peak_date) and pd.notnull(original_precip_peak_date):
            year_result['obs_flow_lag'] = (pd.to_datetime(obs_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
        else:
            year_result['obs_flow_lag'] = np.nan

        # --- Process 'peak1_rate0' as the baseline simulation ---
        baseline_sim_col = 'peak1_rate0'
        baseline_sim_annual_peaks_df = eval_period_df.loc[
            eval_period_df.groupby(eval_period_df['Date'].dt.year)[baseline_sim_col].idxmax()
        ].copy()

        baseline_sim_flow_peak_year_data = baseline_sim_annual_peaks_df[baseline_sim_annual_peaks_df['Date'].dt.year == year]
        if not baseline_sim_flow_peak_year_data.empty:
            baseline_sim_flow_peak_value = baseline_sim_flow_peak_year_data[baseline_sim_col].values[0]
            baseline_sim_flow_peak_date = baseline_sim_flow_peak_year_data['Date'].values[0]
        else:
            baseline_sim_flow_peak_value = np.nan
            baseline_sim_flow_peak_date = pd.NaT

        year_result[f'sim_peak_flow_{baseline_sim_col}'] = baseline_sim_flow_peak_value
        year_result[f'sim_peak_date_{baseline_sim_col}'] = baseline_sim_flow_peak_date

        # Calculate baseline simulated flow lag relative to ORIGINAL precipitation peak
        if pd.notnull(baseline_sim_flow_peak_date) and pd.notnull(original_precip_peak_date):
            year_result[f'sim_peak_lag_{baseline_sim_col}'] = (pd.to_datetime(baseline_sim_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
        else:
            year_result[f'sim_peak_lag_{baseline_sim_col}'] = np.nan


        # --- Loop through other synthetic scenarios (excluding baseline) ---
        for sim_col in sim_discharge_cols:
            if sim_col == baseline_sim_col:
                continue # Skip baseline as it's already processed
        

            sim_annual_peaks_df = eval_period_df.loc[
                eval_period_df.groupby(eval_period_df['Date'].dt.year)[sim_col].idxmax()
            ].copy()

            sim_flow_peak_year_data = sim_annual_peaks_df[sim_annual_peaks_df['Date'].dt.year == year]
            if not sim_flow_peak_year_data.empty:
                sim_peak_value = sim_flow_peak_year_data[sim_col].values[0]
                sim_flow_peak_date = sim_flow_peak_year_data['Date'].values[0]
            else:
                sim_peak_value = np.nan
                sim_flow_peak_date = pd.NaT


            # Store the simulated peak flow value
            year_result[f'sim_peak_flow_{sim_col}'] = sim_peak_value
            year_result[f'sim_peak_date_{sim_col}'] = sim_flow_peak_date

            # Calculate simulated flow lag relative to ORIGINAL precipitation peak
            if pd.notnull(sim_flow_peak_date) and pd.notnull(original_precip_peak_date):
                lag = (pd.to_datetime(sim_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
            else:
                lag = np.nan
            year_result[f'sim_peak_lag_{sim_col}'] = lag

            # Calculate percentage difference in peak flow from 'up_peak1_rate0' (baseline) simulated peak flow
            if pd.notnull(sim_peak_value) and pd.notnull(baseline_sim_flow_peak_value) and baseline_sim_flow_peak_value != 0:
                peak_flow_rate = ((sim_peak_value - baseline_sim_flow_peak_value) / baseline_sim_flow_peak_value)
            else:
                peak_flow_rate = np.nan
            year_result[f'peak_flow_ratio_base_{sim_col}'] = peak_flow_rate

        yearly_results.append(year_result)

    # Convert yearly_results to DataFrame and append
    if yearly_results: # Ensure yearly_results is not empty before creating DataFrame
        results_df = pd.DataFrame(yearly_results)
        all_comparison_results.append(results_df)

# Concatenate all results into a single DataFrame
output_file = f"{output_dir}/peak_discharge_comparison_{model_name}.csv"
if all_comparison_results: # Only concatenate if there are results
    comparison_df = pd.concat(all_comparison_results, ignore_index=True)
    comparison_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
else:
    print(f"No valid data to save for {model_name}.")
    comparison_df = pd.DataFrame() # Create an empty DataFrame if no results
