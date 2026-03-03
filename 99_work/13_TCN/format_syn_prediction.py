import os
import pandas as pd

input_dir = "/data0/funato/hsp2/0_out/TCN/GB/regional/predict/synthetic_wrongformat"
output_dir = "/data0/funato/hsp2/0_out/TCN/GB/regional/predict/synthetic"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if not file_name.endswith('.csv'):
        continue
        
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    
    print(f"Processing {file_name}...")
    
    # Read the CSV with dates as index
    df = pd.read_csv(input_path, index_col=0)
    
    # Transpose: dates as rows -> basins as rows, dates as columns
    df_transposed = df.T
    
    # Clean up the index: remove 'file_' prefix from basin names
    df_transposed.index = df_transposed.index.str.replace('file_', '')
    df_transposed.index.name = 'basin'
    
    # Convert column names (dates) to string and extract just the date part
    df_transposed.columns = df_transposed.columns.astype(str).str.split(' ').str[0]
    
    # Save the transposed data
    df_transposed.to_csv(output_path)
    print(f"  Saved to {output_path}")

print("\nAll files have been reformatted!")