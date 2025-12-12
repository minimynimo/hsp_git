import os
import numpy as np
import pandas as pd
####
PROCESS=15
GLOBAL_SETTINGS = {
    'loc': 'GB',
    'ver': 2,
    'seq_length': 270,
    'train_start': pd.to_datetime('1993-01-01', format='%Y-%m-%d'),
    'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    'val_start': pd.to_datetime('2001-01-01', format='%Y-%m-%d'),
    'val_end': pd.to_datetime('2006-12-31', format='%Y-%m-%d')
}
####

benchmarks = ["NSE", "KGE", "E1", "VE", "RMSE", "MAE"]

syn_precip_cols = ['syn_no_precip', 
                   'syn_peak1_rate0', 'syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                   'syn_peak3_rate0', 'syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                   'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']


input_predict_dir = f"/data0/funato/hsp/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted/predict_p"
input_results_dir = f"/data0/funato/hsp/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted/results_p"

output_predict_dir = f"/data0/funato/hsp/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted/predict"
output_results_dir = f"/data0/funato/hsp/0_out/TCN/{GLOBAL_SETTINGS['loc']}/adapted/results"

os.makedirs(output_predict_dir, exist_ok=True)
os.makedirs(output_results_dir, exist_ok=True)
os.makedirs(f"{output_predict_dir}/synthetic", exist_ok=True)

if GLOBAL_SETTINGS['loc'] == "JP": file_tot_num = 87
else: file_tot_num = 424
file_num_list = list(map(str, range(1, file_tot_num + 1)))

def combine_prediction_files(input_dir, output_dir, file_prefix, output_filename):
    """
    Combine prediction files from multiple processes into a single CSV file.
    Format: columns are dates, index are basins.
    """
    combined_data = {}
    
    for p in range(PROCESS):
        file_path = os.path.join(input_dir, f"{file_prefix}_p{p}.csv")
        if os.path.exists(file_path):
            print(f"Processing {file_path}")
            try:
                # First try to read normally
                df = pd.read_csv(file_path, index_col=0)
            except pd.errors.ParserError:
                try:
                    # If that fails, try reading with error handling
                    print(f"Warning: Parser error in {file_path}, trying to fix...")
                    df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
                except:
                    try:
                        # Last resort: read line by line and fix manually
                        print(f"Warning: Reading {file_path} line by line due to formatting issues...")
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Get header
                        header = lines[0].strip().split(',')
                        header_cols = len(header) - 1  # Subtract 1 for index column
                        
                        # Process data lines
                        data_rows = []
                        for line in lines[1:]:
                            parts = line.strip().split(',')
                            if len(parts) >= header_cols + 1:  # At least basin + expected columns
                                # Take only the expected number of columns
                                row_data = parts[:header_cols + 1]
                                data_rows.append(row_data)
                            elif len(parts) > 1:  # At least basin + some data
                                # Pad with NaN if needed
                                while len(parts) < header_cols + 1:
                                    parts.append('NaN')
                                data_rows.append(parts)
                        
                        # Create DataFrame
                        if data_rows:
                            df_data = pd.DataFrame(data_rows, columns=header)
                            df = df_data.set_index(df_data.columns[0])
                            # Convert to numeric
                            df = df.apply(pd.to_numeric, errors='coerce')
                        else:
                            print(f"No valid data found in {file_path}")
                            continue
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
            
            # Add data from this process to combined_data
            for basin in df.index:
                if basin not in combined_data:
                    combined_data[basin] = df.loc[basin]
                else:
                    # If basin already exists, we might need to handle duplicates
                    # For now, we'll keep the first occurrence
                    pass
    
    if combined_data:
        # Create combined DataFrame
        combined_df = pd.DataFrame(combined_data).T
        combined_df.index.name = 'basin'
        
        # Generate appropriate date headers based on file type
        if 'cal' in file_prefix.lower():
            # Calibration period: train_start + seq_length to train_end
            start_date = GLOBAL_SETTINGS['train_start'] + pd.Timedelta(days=GLOBAL_SETTINGS['seq_length'])
            end_date = GLOBAL_SETTINGS['train_end']
        elif 'eva' in file_prefix.lower():
            # Check if this is a synthetic file
            if 'syn_' in file_prefix.lower():
                # Synthetic data starts from 2001-01-02
                start_date = pd.to_datetime('2001-01-02', format='%Y-%m-%d')
                # For synthetic files, use the actual data length to determine end date
                end_date = start_date + pd.Timedelta(days=combined_df.shape[1]-1)
            else:
                # Regular evaluation period: val_start + seq_length to val_end  
                start_date = GLOBAL_SETTINGS['val_start'] + pd.Timedelta(days=GLOBAL_SETTINGS['seq_length'])
                end_date = GLOBAL_SETTINGS['val_end']
        else:
            # Default to evaluation period for other files
            start_date = GLOBAL_SETTINGS['val_start'] + pd.Timedelta(days=GLOBAL_SETTINGS['seq_length'])
            end_date = GLOBAL_SETTINGS['val_end']
        
        # Create date range based on actual data length
        if 'syn_' in file_prefix.lower():
            # For synthetic files, create exactly as many dates as there are data columns
            date_range = pd.date_range(start=start_date, periods=combined_df.shape[1], freq='D')
        else:
            # For other files, create date range and adjust to match data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Ensure we have the right number of columns
            if len(date_range) != combined_df.shape[1]:
                # Adjust to match the actual number of columns in the data
                date_range = date_range[:combined_df.shape[1]]
                if len(date_range) < combined_df.shape[1]:
                    # If we need more dates, extend the range
                    additional_days = combined_df.shape[1] - len(date_range)
                    extra_dates = pd.date_range(start=date_range[-1] + pd.Timedelta(days=1), 
                                              periods=additional_days, freq='D')
                    date_range = date_range.append(extra_dates)
        
        # Set the date headers
        combined_df.columns = [date.strftime('%Y-%m-%d') for date in date_range]
        
        # Save combined file
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path)
        print(f"Saved combined predictions to {output_path}")
        print(f"Combined data shape: {combined_df.shape}")
    else:
        print(f"No data found for {file_prefix}")

def combine_result_files(input_dir, output_dir, file_prefix, output_filename):
    """
    Combine result files from multiple processes into a single CSV file.
    Format: columns are basin,NSE,KGE,E1,VE,RMSE,MAE
    """
    all_results = []
    
    for p in range(PROCESS):
        file_path = os.path.join(input_dir, f"{file_prefix}_p{p}.csv")
        if os.path.exists(file_path):
            print(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            
            # Ensure the basin column is properly named
            if 'basin' not in df.columns and len(df.columns) > 0:
                df.rename(columns={df.columns[0]: 'basin'}, inplace=True)
            
            all_results.append(df)
    
    if all_results:
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Ensure proper column order
        expected_cols = ['basin', 'NSE', 'KGE', 'E1', 'VE', 'RMSE', 'MAE']
        # Reorder columns if they exist
        available_cols = [col for col in expected_cols if col in combined_df.columns]
        combined_df = combined_df[available_cols]
        
        # Save combined file
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined results to {output_path}")
        print(f"Combined data shape: {combined_df.shape}")
    else:
        print(f"No data found for {file_prefix}")

# Process synthetic precipitation columns
print("Processing synthetic precipitation columns...")

for syn_col in syn_precip_cols:
    print(f"\nProcessing {syn_col}...")
    
    # Combine prediction files for this synthetic column
    combine_prediction_files(
        f"{input_predict_dir}/synthetic",
        f"{output_predict_dir}/synthetic", 
        f"TCN_predict_{syn_col}_eva", 
        f"TCN_predict_{syn_col}_eva.csv"
    )

# Process TCN_predict_cal and eva prediction files
print("\nProcessing additional prediction files...")
additional_predict_files = ['TCN_predict_cal', 'TCN_predict_eva']
for file_prefix in additional_predict_files:
    print(f"\nProcessing {file_prefix}...")
    combine_prediction_files(
        f"{input_predict_dir}",
        f"{output_predict_dir}", 
        file_prefix,
        f"{file_prefix}.csv"
    )



# Process cal results (TCN_results_cal_p0 to TCN_results_cal_p14)
print("\nProcessing cal results...")
combine_result_files(
    input_results_dir,
    output_results_dir,
    "TCN_results_cal",
    "TCN_results_cal_combined.csv"
)

# Process eva results (TCN_results_eva_p0 to TCN_results_eva_p14)
print("\nProcessing eva results...")
combine_result_files(
    input_results_dir,
    output_results_dir,
    "TCN_results_eva",
    "TCN_results_eva_combined.csv"
)

print("\nAll processing completed!")
print(f"Output prediction files saved to: {output_predict_dir}")
print(f"Output result files saved to: {output_results_dir}")



