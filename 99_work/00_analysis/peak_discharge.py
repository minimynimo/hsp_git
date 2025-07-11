import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration for model and directories
model_name = "HBV"
if model_name == "HBV":
    input_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    output_dir = "/data0/funato/hsp/0_out/10_HBV/peak_discharge/"

os.makedirs(output_dir, exist_ok=True)

file_tot_num = 424 # Total number of files to process

# Evaluation period definition
start_date_eva = "2001-01-01"
end_date_eva = "2006-12-31"

# Columns representing different simulated discharge scenarios
sim_discharge_cols = ['og',
                    'up_peak1_rate10', 'up_peak1_rate30', 'up_peak1_rate50',
                    'up_peak3_rate10', 'up_peak3_rate30', 'up_peak3_rate50',
                    'up_peak5_rate10', 'up_peak5_rate30', 'up_peak5_rate50']

all_comparison_results = [] # List to store yearly results for all files

for file_num in range(1, file_tot_num + 1):
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

    # --- Pre-calculate annual peaks for observed flow and original precipitation ---
    # These are extracted once per file for the entire evaluation period
    obs_flow_annual_peaks = eval_period_df.loc[
        eval_period_df.groupby(eval_period_df['Date'].dt.year)['Obs flow'].idxmax()
    ]
    original_precip_annual_peaks = eval_period_df.loc[
        eval_period_df.groupby(eval_period_df['Date'].dt.year)['Precip'].idxmax()
    ]

    # --- Loop through each year in the evaluation period ---
    for year in range(int(start_date_eva[:4]), int(end_date_eva[:4]) + 1):
        # Extract the peak original precipitation value and date for the current year
        original_precip_peak_year_data = original_precip_annual_peaks[original_precip_annual_peaks['Date'].dt.year == year]
        if not original_precip_peak_year_data.empty:
            original_precip_peak_value = original_precip_peak_year_data['Precip'].values[0]
            original_precip_peak_date = original_precip_peak_year_data['Date'].values[0]
        else:
            original_precip_peak_value = np.nan
            original_precip_peak_date = pd.NaT # Not a Time, use NaT for missing dates

        # Extract the peak observed flow value and date for the current year
        obs_flow_peak_year_data = obs_flow_annual_peaks[obs_flow_annual_peaks['Date'].dt.year == year]
        if not obs_flow_peak_year_data.empty:
            obs_flow_peak_value = obs_flow_peak_year_data['Obs flow'].values[0]
            obs_flow_peak_date = obs_flow_peak_year_data['Date'].values[0]
        else:
            obs_flow_peak_value = np.nan
            obs_flow_peak_date = pd.NaT

        # Initialize results for the current year
        year_result = {
            'file_num': file_num,
            'year': year,
            'og_peak_precip': original_precip_peak_value, # Original peak precip value
            'obs_peak_flow': obs_flow_peak_value, # Observed peak flow value
            'og_peak_precip_date': original_precip_peak_date, # Date of original precip peak
            'obs_flow_peak_date': obs_flow_peak_date, # Date of observed flow peak
        }

        # Add scaled peak precipitation values (these are just derived values for comparison)
        year_result['10_peak_precip'] = original_precip_peak_value * 1.1 if pd.notnull(original_precip_peak_value) else np.nan
        year_result['30_peak_precip'] = original_precip_peak_value * 1.3 if pd.notnull(original_precip_peak_value) else np.nan
        year_result['50_peak_precip'] = original_precip_peak_value * 1.5 if pd.notnull(original_precip_peak_value) else np.nan

        # Calculate observed flow lag relative to original precipitation peak
        if pd.notnull(obs_flow_peak_date) and pd.notnull(original_precip_peak_date):
            year_result['obs_flow_lag_to_og_precip'] = (pd.to_datetime(obs_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
        else:
            year_result['obs_flow_lag_to_og_precip'] = np.nan

        # Process each simulated discharge column
        # Extract 'og' simulated flow peak first to use as a baseline for flow difference percentage
        og_sim_flow_peak_value = np.nan
        og_sim_flow_peak_date = pd.NaT

        og_sim_annual_peaks = eval_period_df.loc[
            eval_period_df.groupby(eval_period_df['Date'].dt.year)['og'].idxmax()
        ]
        og_sim_flow_peak_year_data = og_sim_annual_peaks[og_sim_annual_peaks['Date'].dt.year == year]
        if not og_sim_flow_peak_year_data.empty:
            og_sim_flow_peak_value = og_sim_flow_peak_year_data['og'].values[0]
            og_sim_flow_peak_date = og_sim_flow_peak_year_data['Date'].values[0]

        year_result[f'sim_og_peak_flow'] = og_sim_flow_peak_value

        # Calculate 'og' simulated flow lag relative to original precipitation peak
        if pd.notnull(og_sim_flow_peak_date) and pd.notnull(original_precip_peak_date):
            year_result[f'sim_og_peak_lag'] = (pd.to_datetime(og_sim_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
            year_result[f'sim_og_peak_date'] = og_sim_flow_peak_date
        else:
            year_result[f'sim_og_peak_lag'] = np.nan
            year_result[f'sim_og_peak_date'] = pd.NaT

        # Loop through other synthetic scenarios (excluding 'og' as it's handled)
        for sim_col in sim_discharge_cols:
            if sim_col == 'og':
                continue # Skip 'og' as it's the baseline and already processed

            sim_annual_peaks = eval_period_df.loc[
                eval_period_df.groupby(eval_period_df['Date'].dt.year)[sim_col].idxmax()
            ]

            sim_flow_peak_year_data = sim_annual_peaks[sim_annual_peaks['Date'].dt.year == year]
            if not sim_flow_peak_year_data.empty:
                sim_peak_value = sim_flow_peak_year_data[sim_col].values[0]
                sim_flow_peak_date = sim_flow_peak_year_data['Date'].values[0]
            else:
                sim_peak_value = np.nan
                sim_flow_peak_date = pd.NaT

            output_sim_col_name = sim_col.replace('up_', '') # e.g., 'peak1_rate10'

            # Store the simulated peak flow value
            year_result[f'sim_{output_sim_col_name}_peak_flow'] = sim_peak_value

            # Calculate simulated flow lag relative to original precipitation peak
            if pd.notnull(sim_flow_peak_date) and pd.notnull(original_precip_peak_date):
                precip_flow_lag = (pd.to_datetime(sim_flow_peak_date) - pd.to_datetime(original_precip_peak_date)).days
            else:
                precip_flow_lag = np.nan
            year_result[f'sim_{output_sim_col_name}_peak_lag'] = precip_flow_lag

            # Calculate percentage difference in peak flow from 'og' simulated peak flow
            if pd.notnull(sim_peak_value) and pd.notnull(og_sim_flow_peak_value) and og_sim_flow_peak_value != 0:
                peak_flow_diff_percent = ((sim_peak_value - og_sim_flow_peak_value) / og_sim_flow_peak_value) * 100
            else:
                peak_flow_diff_percent = np.nan
            year_result[f'diff_from_og_perc_flow_{output_sim_col_name}'] = peak_flow_diff_percent

        yearly_results.append(year_result)

    results_df = pd.DataFrame(yearly_results)
    all_comparison_results.append(results_df)

# Concatenate all results into a single DataFrame
output_file = f"{output_dir}/peak_discharge_comparison_{model_name}.csv"
comparison_df = pd.concat(all_comparison_results, ignore_index=True)
comparison_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")