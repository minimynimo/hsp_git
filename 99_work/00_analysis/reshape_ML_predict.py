## reshape ML prediction results to match ths HBV format
import pandas as pd
import numpy as np  
import os   

model_list = ["XGB"] #["RF", "XGB", "TCN", "LSTM"]

file_tot_num = 424

pred_discharge_cols = ['og', 'no_precip','p_pet','peak1_rate0',
                    'peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                    'peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                    'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

base_input_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
for model_name in model_list:
    input_dir = f"/data0/funato/hsp/0_out/{model_name}/GB/shap/regional/predict/synthetic"
    output_dir = f"/data0/funato/hsp/0_out/{model_name}/GB/shap/regional/varssim_{model_name}"
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
                 'PET_at_peak', 'Obs_flow_at_peak']].copy()
        # Add synthetic columns
        for sim_col in pred_discharge_cols:
            base_df[sim_col] = np.nan
        base_df.set_index('Date', inplace=True) 
        base_df.sort_index(inplace=True)
        # Loop through synthetic scenarios
        for sim_col in pred_discharge_cols:
            print(f"Processing synthetic column: {sim_col}")
            if sim_col == 'og':
                input_file = f"/data0/funato/hsp/0_out/{model_name}/GB/shap/regional/predict/og/{model_name}_predict_cal_basin_{file_num}.csv"
                col_name = "Prediction"
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

        # Save the reshaped DataFrame to a new CSV file
        output_file = f"{output_dir}/varssim{file_num:03d}.csv"
        base_df.to_csv(output_file, index=True, header=True)
        print(f"Saved reshaped data to {output_file}")




