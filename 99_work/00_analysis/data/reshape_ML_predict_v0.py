## reshape ML prediction results to match ths HBV format
import pandas as pd
import numpy as np  
import os   

model_list = ["XGB"] #["RF", "XGB", "TCN", "LSTM"]
model_list = ["RF"]
model_list = ["TCN"]
model_list = ["LSTM"]

file_tot_num = 424

pred_discharge_cols = ['og', 'no_precip',
                       'peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                       'peak3_rate0','peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                       'peak5_rate0','peak5_rate10', 'peak5_rate30', 'peak5_rate50']

# Parameters for synthetic precipitation generation
peak_length_list = [1, 3, 5]
rate_list = [0, 10, 30, 50]

base_input_dir = "/data0/funato/hsp2/2_MERV/varssim_HBV"
for model_name in model_list:
    input_dir = f"/data0/funato/hsp2/0_out/{model_name}/GB/regional/predict/synthetic"
    output_dir = f"/data0/funato/hsp2/0_out/{model_name}/GB/regional/varssim_{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for file_num in range(1, file_tot_num + 1):
        print(f"Processing file {file_num} for model {model_name}")
        base_file = f"{base_input_dir}/varssim{file_num:03d}.csv"
        base_df = pd.read_csv(base_file, parse_dates=['Date'])

        #from base_df, copy the following columns to the new dataframe
        # Date,Obs flow,Precip,Temp,PET,syn_no_precip,syn_p_pet,Temp_at_peak,PET_at_peak,Obs_flow_at_peak,syn_peak1_rate0,syn_peak1_rate10,syn_peak1_rate30,syn_peak1_rate50,syn_peak3_rate10,syn_peak3_rate30,syn_peak3_rate50,syn_peak5_rate10,syn_peak5_rate30,syn_peak5_rate50
        # for the following columns, copy the values from the input file
        # og,no_precip,p_pet,peak1_rate0,peak1_rate10,peak1_rate30,peak1_rate50,peak3_rate10,peak3_rate30,peak3_rate50,peak5_rate10,peak5_rate30,peak5_rate50
        # select only the first column if there are duplicate columns
        base_df = base_df[['Date', 'Obs flow', 'Precip', 'Temp', 'PET', 
                 'syn_no_precip', 'syn_p_pet', 'Temp_at_peak', 
                 'PET_at_peak', 'Obs_flow_at_peak', 'syn_peak1_rate0',
                 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']].copy()
        # Add synthetic columns
        for sim_col in pred_discharge_cols:
            base_df[sim_col] = np.nan
        base_df.set_index('Date', inplace=True) 
        base_df.sort_index(inplace=True)
        
        # Generate synthetic precipitation columns
        precip_col = 'Precip'
        
        # Define evaluation period mask 
        # You may need to adjust this logic based on your specific data structure
        # Common approach is to split data chronologically (e.g., first 70% for calibration, last 30% for evaluation)
        eval_start_date = pd.Timestamp("2001-01-01")  # Convert to Timestamp for proper comparison
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
                    
                    ### --- UNIFIED DURATION LOGIC --- ###
                    # STEP 1: Copy original precip to fill the calibration period.
                    base_df[column_name] = base_df[precip_col]
                    
                    # STEP 2: Set the entire evaluation period to NaN.
                    base_df.loc[eval_mask, column_name] = np.nan
                    
                    # STEP 3: Create the modified 41-day event block and place it at the start of the evaluation period.
                    
                    # Make a copy of the original 41-day precipitation event
                    modified_precip_event = source_precip_values.copy()
                    
                    # Calculate the new peak value
                    syn_peak_value = peak_value * (1 + rate / 100.0)
                    
                    # The peak day is the 11th day in our window (index 10).
                    # Replace the values for the specified peak length.
                    peak_start_index = 10 
                    peak_end_index = peak_start_index + peak_length
                    if peak_end_index <= len(modified_precip_event):
                         modified_precip_event[peak_start_index:peak_end_index] = syn_peak_value
                    
                    # Place the entire modified event block into the destination
                    if len(dest_mask) > 0 and len(modified_precip_event) > 0:
                        base_df.loc[dest_mask, column_name] = modified_precip_event[:num_dest_rows]
        else:
            print(f"Warning: No valid precipitation data found in evaluation period for file {file_num}")
            # Initialize synthetic columns with original precipitation values as fallback
            for peak_length in peak_length_list:
                for rate in rate_list:
                    column_name = f"syn_peak{peak_length}_rate{rate}"
                    base_df[column_name] = base_df[precip_col]
        # Loop through synthetic scenarios
        for sim_col in pred_discharge_cols:
            print(f"Processing synthetic column: {sim_col}")
            if sim_col == 'og':
                input_file_cal = f"/data0/funato/hsp2/0_out/{model_name}/GB/regional/predict/og/{model_name}_predict_cal_basin_{file_num}.csv"
                input_file_eva = f"/data0/funato/hsp2/0_out/{model_name}/GB/regional/predict/og/{model_name}_predict_eva_basin_{file_num}.csv"
                input_df_cal = pd.read_csv(input_file_cal, parse_dates=['Date'])
                input_df_cal.set_index('Date', inplace=True)
                input_df_cal.sort_index(inplace=True)
                input_df_eva = pd.read_csv(input_file_eva, parse_dates=['Date'])
                input_df_eva.set_index('Date', inplace=True)
                input_df_eva.sort_index(inplace=True) 
                print(f"Input files for original data: {input_file_cal}, {input_file_eva} with sizes {input_df_cal.shape} and {input_df_eva.shape}")
                
                # Combine cal and eva predictions into a single series using concat with axis=0 (row-wise)
                if model_name == "TCN" or model_name == "LSTM":
                    col_name = "Predicted"
                else:
                    col_name = "Prediction"
                
                # Concatenate the prediction columns vertically (row-wise) to create one continuous time series
                input_df = pd.concat([input_df_cal[col_name], input_df_eva[col_name]], axis=0)
                input_df = input_df.sort_index()  # Ensure proper chronological order
                
                print(f"Combined input DataFrame shape: {input_df.shape}")
                print(f"Combined input df head:\n{input_df.head()}")
                print(f"Combined input df tail:\n{input_df.tail()}")
                
                # Convert to DataFrame with the same column name for consistency with the rest of the code
                input_df = pd.DataFrame({col_name: input_df})
                
            else:
                input_file = f"{input_dir}/{model_name}_predict_{sim_col}.csv"
                col_name = f"file_{file_num}"
                input_df = pd.read_csv(input_file, parse_dates=['Date'])
                input_df.set_index('Date', inplace=True)
                input_df.sort_index(inplace=True)

            # Ensure the index matches the base_df
            input_df = input_df.reindex(base_df.index)

            # Copy the values to the base_df
            base_df[sim_col] = input_df[col_name].values
            print(f"Processed {sim_col} for file {file_num}, shape: {base_df[sim_col].shape} from {input_df[col_name].shape}")

        # Save the reshaped DataFrame to a new CSV file
        output_file = f"{output_dir}/varssim{file_num:03d}.csv"
        base_df.to_csv(output_file, index=True, header=True)
        print(f"Saved reshaped data to {output_file}")




