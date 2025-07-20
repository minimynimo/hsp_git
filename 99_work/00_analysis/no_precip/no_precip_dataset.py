import os
import pandas as pd
import numpy as np

model_list = ["HBV","BC", "RF", "XGB", "TCN","LSTM"]

file_tot_num = 424

start_date_cal = "1993-01-01"
end_date_cal = "2000-12-31"
start_date_eva = "2001-01-01"
end_date_eva = "2006-12-31"

start_date_zero = "2002-01-01"

# Initialize dictionary to store results
results = []
all_discharge_data = {model: [] for model in model_list}

output_dir = "/data0/funato/hsp/0_out/analysis/no_precip_discharge"
os.makedirs(output_dir, exist_ok=True)
    
for model in model_list:
    model_discharge_values = []
    
    for file in range(1, file_tot_num + 1):
        if file % 50 == 0 or file == 1:
            print(f"Processing model: {model}, file number: {file}")
        if model not in ["HBV", "BC"]:
            input_file = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/varssim_{model}/varssim{file:03d}.csv"
        else:
            input_file = f"/data0/funato/hsp/0_out/{model}/varssim_{model}/varssim{file:03d}.csv"
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist. Skipping.")
            continue
        
        # Read the input file
        input_df = pd.read_csv(input_file, parse_dates=['Date'])
        input_df.set_index('Date', inplace=True)
        
        # Filter for the evaluation period (start_date_zero to end_date_eva)
        eval_period_df = input_df[start_date_zero:end_date_eva]
        
        # Check if 'no_precip' column exists
        if 'no_precip' not in eval_period_df.columns:
            print(f"'no_precip' column not found in {input_file}. Skipping.")
            continue
        
        # Extract discharge values for the evaluation period
        discharge_values = eval_period_df['no_precip'].dropna()
        
        if len(discharge_values) == 0:
            print(f"No valid discharge data for {model}, file {file}. Skipping.")
            continue
        
        # Calculate mean discharge for this file
        mean_discharge = discharge_values.mean()
        
        # Store individual discharge values for plotting
        all_discharge_data[model].extend(discharge_values.tolist())
        model_discharge_values.extend(discharge_values.tolist())
        
        # Store results
        results.append({
            'Model': model,
            'File': file,
            'Mean_Discharge': mean_discharge,
            'Std_Discharge': discharge_values.std(),
            'Min_Discharge': discharge_values.min(),
            'Max_Discharge': discharge_values.max(),
            'Count': len(discharge_values)
        })
        
        #print(f"  Mean discharge: {mean_discharge:.4f}")

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Save results to CSV
output_csv = f"{output_dir}/no_precip_discharge_summary.csv"
results_df.to_csv(output_csv, index=False)
print(f"Results saved to: {output_csv}")

# Create summary statistics by model
model_summary = results_df.groupby('Model').agg({
    'Mean_Discharge': ['mean', 'std', 'min', 'max'],
    'File': 'count'
}).round(4)

model_summary.columns = ['Mean_Mean_Discharge', 'Std_Mean_Discharge', 'Min_Mean_Discharge', 'Max_Mean_Discharge', 'File_Count']
print("\nModel Summary Statistics:")
print(model_summary)

# Save model summary
summary_csv = f"{output_dir}/no_precip_model_summary.csv"
model_summary.to_csv(summary_csv)
print(f"Model summary saved to: {summary_csv}")

# Save all discharge data for visualization script
all_discharge_csv = f"{output_dir}/no_precip_all_discharge_data.csv"

# Convert all_discharge_data to a DataFrame for easier handling
all_discharge_list = []
for model in model_list:
    for discharge_value in all_discharge_data[model]:
        all_discharge_list.append({
            'Model': model,
            'Discharge': discharge_value
        })

all_discharge_df = pd.DataFrame(all_discharge_list)
all_discharge_df.to_csv(all_discharge_csv, index=False)
print(f"All discharge data saved to: {all_discharge_csv}")

print("\nData processing completed!")
print(f"Evaluation period: {start_date_zero} to {end_date_eva}")
print(f"Total files processed: {len(results_df)}")
print(f"Models analyzed: {results_df['Model'].unique()}")
print("\nGenerated files:")
print(f"  - Summary statistics: {output_csv}")
print(f"  - Model summary: {summary_csv}")
print(f"  - All discharge data: {all_discharge_csv}")
print("\nTo create visualizations, run the separate visualization script.")


