## reshape BC prediction results to match the HBV format
## Fixed version to handle transposed BC prediction files

import pandas as pd
import numpy as np
import os

#loc = "GB"
loc = "US"

# 'camels' (default): raw CAMELS-US driver data, gauge-id-keyed files under
# /2_MERV/{dataset}/varssim_{loc}. 'caravan': Caravan-derived extraction (unchanged legacy
# 2001-01-01 eval start). See create_dataset_HYPER.py's/BMA.py's --dataset docstrings for the
# same switch.
dataset = "camels"  # "caravan"

# NOTE (2026-09-16): this script reshapes an existing "BC" (bias-correction/ESN) model's
# predictions (0_out/BC/{loc}/predict/BC_predict_*.csv) into the HBV varssim format. No script
# in this repo currently produces those BC_predict_*.csv files for either dataset -- same class
# of gap as the missing HYPER 47-model ensemble input flagged in BMA.py's/create_dataset_HYPER.py's
# docstrings. This script is adapted for --dataset consistency but is not runnable until that
# upstream BC model run is (re)generated.

eval_start_date_str = "2001-01-01"  # same for camels and caravan

pred_discharge_cols = ['og', 'no_precip',
                       'peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                       'peak3_rate0','peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                       'peak5_rate0', 'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

# Parameters for synthetic precipitation generation
peak_length_list = [1, 3, 5]
rate_list = [0, 10, 30, 50]

base_input_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_{loc}"
input_dir = f"/data0/funato/hsp2/0_out/BC/{loc}/{dataset}/predict"
output_dir = f"/data0/funato/hsp2/0_out/BC/{loc}/{dataset}/varssim_BC"
os.makedirs(output_dir, exist_ok=True)

basin_list_csv = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/{loc.lower()}_{dataset}_file_num.csv"
basin_df = pd.read_csv(basin_list_csv, dtype={'gauge_id': str})
gauge_ids = [g.split('_')[-1] for g in basin_df['gauge_id']]
file_tot_num = len(gauge_ids)
print(f"[reshape_HYPER_predict] loc={loc} dataset={dataset!r} basins={file_tot_num} "
      f"eval_start={eval_start_date_str}")

def load_transposed_hyper_data(sim_col, gauge_id):
    """Load BC prediction data from transposed format files for a specific scenario and basin"""

    if sim_col == 'og':
        # Handle 'og' scenario with transposed files
        cal_file = f"{input_dir}/BC_predict_cal.csv"
        eva_file = f"{input_dir}/BC_predict_eva.csv"

        if not os.path.exists(cal_file) or not os.path.exists(eva_file):
            print(f"Warning: BC files for og do not exist.")
            return None

        # Load the transposed data
        cal_data = pd.read_csv(cal_file)
        eva_data = pd.read_csv(eva_file)

        # Find the row for this specific basin
        file_identifier_cal = f"gauge_{gauge_id}_cal"
        file_identifier_eva = f"gauge_{gauge_id}_eva"

        cal_row = None
        eva_row = None

        # Find the matching rows
        for _, row in cal_data.iterrows():
            if str(row.iloc[0]) == file_identifier_cal:
                cal_row = row
                break

        for _, row in eva_data.iterrows():
            if str(row.iloc[0]) == file_identifier_eva:
                eva_row = row
                break

        if cal_row is None or eva_row is None:
            print(f"Warning: Could not find data for gauge {gauge_id} in og")
            return None

        # Extract dates and values
        cal_dates = cal_data.columns[1:].tolist()  # Skip first column (file identifier)
        eva_dates = eva_data.columns[1:].tolist()  # Skip first column (file identifier)

        cal_values = cal_row.iloc[1:].values  # Skip first column
        eva_values = eva_row.iloc[1:].values  # Skip first column

        # Create time series
        cal_series = pd.Series(cal_values, index=pd.to_datetime(cal_dates))
        eva_series = pd.Series(eva_values, index=pd.to_datetime(eva_dates))

        # Combine calibration and evaluation data
        combined_data = pd.concat([cal_series, eva_series], axis=0)
        combined_data = combined_data.sort_index()

        return combined_data

    else:
        # Handle other scenarios with transposed files
        cal_file = os.path.join(input_dir,'synthetic', f"BC_predict_{sim_col}_cal.csv")
        eva_file = os.path.join(input_dir,'synthetic', f"BC_predict_{sim_col}_eva.csv")

        if not os.path.exists(cal_file) or not os.path.exists(eva_file):
            print(f"Warning: BC files for {sim_col} do not exist.")
            return None

        # Load the transposed data
        cal_data = pd.read_csv(cal_file)
        eva_data = pd.read_csv(eva_file)

        # Find the row for this specific basin
        file_identifier_cal = f"gauge_{gauge_id}_cal"
        file_identifier_eva = f"gauge_{gauge_id}_eva"

        cal_row = None
        eva_row = None

        # Find the matching rows
        for _, row in cal_data.iterrows():
            if str(row.iloc[0]) == file_identifier_cal:
                cal_row = row
                break

        for _, row in eva_data.iterrows():
            if str(row.iloc[0]) == file_identifier_eva:
                eva_row = row
                break

        if cal_row is None or eva_row is None:
            print(f"Warning: Could not find data for gauge {gauge_id} in {sim_col}")
            return None

        # Extract dates and values
        cal_dates = cal_data.columns[1:].tolist()  # Skip first column (file identifier)
        eva_dates = eva_data.columns[1:].tolist()  # Skip first column (file identifier)

        cal_values = cal_row.iloc[1:].values  # Skip first column
        eva_values = eva_row.iloc[1:].values  # Skip first column

        # Create time series
        cal_series = pd.Series(cal_values, index=pd.to_datetime(cal_dates))
        eva_series = pd.Series(eva_values, index=pd.to_datetime(eva_dates))

        # Combine calibration and evaluation data
        combined_data = pd.concat([cal_series, eva_series], axis=0)
        combined_data = combined_data.sort_index()

        return combined_data

# Process each basin
for gauge_id in gauge_ids:
    print(f"Processing gauge {gauge_id} for model BC")
    base_file = f"{base_input_dir}/varssim_{gauge_id}.csv"

    if not os.path.exists(base_file):
        print(f"Warning: Base file {base_file} does not exist. Skipping...")
        continue

    base_df = pd.read_csv(base_file, parse_dates=['Date'])

    # Copy base columns from the original file
    base_df = base_df[['Date', 'Obs flow', 'Precip', 'Temp', 'PET',
                'syn_no_precip', 'Temp_at_peak',
                'PET_at_peak', 'Obs_flow_at_peak',
                'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                'syn_peak3_rate0','syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                'syn_peak5_rate0','syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']].copy()

    # Add synthetic columns for BC predictions
    for sim_col in pred_discharge_cols:
        base_df[sim_col] = np.nan

    base_df.set_index('Date', inplace=True)
    base_df.sort_index(inplace=True)

    # Generate synthetic precipitation columns (keep existing logic)
    precip_col = 'Precip'
    eval_start_date = pd.Timestamp(eval_start_date_str)
    eval_mask = base_df.index >= eval_start_date

    # Find the peak precipitation value and its location in the evaluation period
    eval_precip = base_df.loc[eval_mask, precip_col]
    if len(eval_precip) > 0 and not eval_precip.isna().all():
        peak_idx = eval_precip.idxmax()
        peak_value = eval_precip.max()

        # Create 41-day window around the peak (20 days before, peak day, 20 days after)
        peak_start = peak_idx - pd.Timedelta(days=20)
        peak_end = peak_idx + pd.Timedelta(days=20)

        # Ensure we stay within the evaluation period and data bounds
        peak_start = max(peak_start, eval_start_date, base_df.index[0])
        peak_end = min(peak_end, base_df.index[-1])

        # Create the source precipitation values (41-day window)
        source_mask = (base_df.index >= peak_start) & (base_df.index <= peak_end)
        source_precip_values = base_df.loc[source_mask, precip_col].values

        # Destination mask for placing the modified event at start of evaluation period
        eval_dates = base_df.index[eval_mask]
        num_dest_rows = min(len(source_precip_values), len(eval_dates))
        dest_mask = eval_dates[:num_dest_rows]

        # Generate synthetic precipitation columns
        new_precip_cols = []
        for peak_length in peak_length_list:
            for rate in rate_list:
                column_name = f"syn_peak{peak_length}_rate{rate}"
                new_precip_cols.append(column_name)

                # STEP 1: Copy original precip to fill the calibration period.
                base_df[column_name] = base_df[precip_col]

                # STEP 2: Set the entire evaluation period to NaN.
                base_df.loc[eval_mask, column_name] = np.nan

                # STEP 3: Create the modified 41-day event block and place it at the start of the evaluation period.
                modified_precip_event = source_precip_values.copy()

                # Calculate the new peak value
                syn_peak_value = peak_value * (1 + rate / 100.0)

                # The peak day is the 11th day in our window (index 10).
                peak_start_index = 10
                peak_end_index = peak_start_index + peak_length
                if peak_end_index <= len(modified_precip_event):
                        modified_precip_event[peak_start_index:peak_end_index] = syn_peak_value

                # Place the entire modified event block into the destination
                if len(dest_mask) > 0 and len(modified_precip_event) > 0:
                    base_df.loc[dest_mask, column_name] = modified_precip_event[:num_dest_rows]
    else:
        print(f"Warning: No valid precipitation data found in evaluation period for gauge {gauge_id}")
        # Initialize synthetic columns with original precipitation values as fallback
        for peak_length in peak_length_list:
            for rate in rate_list:
                if rate == 0 and peak_length != 1:
                    continue
                column_name = f"syn_peak{peak_length}_rate{rate}"
                base_df[column_name] = base_df[precip_col]

    # Process BC predictions for each scenario
    for sim_col in pred_discharge_cols:
        print(f"  Processing BC predictions for scenario: {sim_col}")

        # Load BC prediction data for this scenario and basin
        hyper_predictions = load_transposed_hyper_data(sim_col, gauge_id)

        if hyper_predictions is not None:
            # Align the BC predictions with the base dataframe index
            hyper_predictions = hyper_predictions.reindex(base_df.index)

            # Fill the prediction column
            base_df[sim_col] = hyper_predictions.values
            print(f"    Successfully loaded BC predictions for {sim_col}")
        else:
            print(f"    Warning: Could not load BC predictions for {sim_col}, filling with NaN")
            base_df[sim_col] = np.nan

    # Save the reshaped DataFrame to a new CSV file
    output_file = f"{output_dir}/varssim_{gauge_id}.csv"
    base_df.to_csv(output_file, index=True, header=True)
    print(f"Saved reshaped data to {output_file}")

print("All processing complete!")

###RUN USING####
# python3 /data0/funato/hsp2/99_work/00_analysis/data/reshape_HYPER_predict.py
