"""
Script to combine existing results from all processes (including incomplete process 2)

This script will:
1. Combine all existing process results into final prediction and results files
2. Handle process 2's partial results
3. Convert numeric column indices to actual dates if needed
"""

import sys
from pathlib import Path
import pandas as pd
import glob
import re
import numpy as np

def generate_date_columns(period, num_cols):
    """Generate date column names for a given period."""
    if period == 'cal':
        start_date = pd.to_datetime('1993-01-01')
        end_date = pd.to_datetime('2000-12-31')
    elif period == 'eva':
        start_date = pd.to_datetime('2001-01-01')
        end_date = pd.to_datetime('2006-12-31')
    else:
        # Default to creating a date range
        start_date = pd.to_datetime('1993-01-01')
        end_date = start_date + pd.Timedelta(days=num_cols-1)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Return only the number of columns we need
    return date_range[:num_cols]

def has_numeric_columns(df):
    """Check if dataframe has numeric column names (0, 1, 2, ...)."""
    try:
        # Check if all columns can be converted to integers
        cols_as_int = [int(col) for col in df.columns]
        # Check if they form a sequence starting from 0
        return cols_as_int == list(range(len(df.columns)))
    except (ValueError, TypeError):
        return False

def combine_results():
    """Combine results from all processes into final files"""
    print("\n" + "="*60)
    print("Combining results from all processes...")
    print("="*60)
    
    base_dir = Path("/data0/funato/hsp2/0_out/TCN/GB/adapted")
    
    # Check which directory structure is being used
    if (base_dir / "predict_p").exists():
        predict_dir = "predict_p"
        results_dir = "results_p"
        print("  Using directory structure: predict_p/ and results_p/")
    else:
        predict_dir = "predict"
        results_dir = "results"
        print("  Using directory structure: predict/ and results/")
    
    # Define file types to combine
    file_types = [
        (predict_dir, "TCN_predict_cal", "csv"),
        (predict_dir, "TCN_predict_eva", "csv"),
        (results_dir, "TCN_results_cal", "csv"),
        (results_dir, "TCN_results_eva", "csv"),
    ]
    
    
  
    # Combine each file type
    for subdir, base_name, ext in file_types:
        output_dir = base_dir / subdir
        pattern = str(output_dir / f"{base_name}_p*.{ext}")
        files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'_p(\d+)\.', x).group(1)))
        
        if not files:
            print(f"\n  - No files found for {base_name}")
            continue
        
        print(f"\n  Combining {len(files)} files for {base_name}...")
        
        dfs = []
        for file in files:
            process_id = re.search(r'_p(\d+)\.', file).group(1)
            try:
                # Try normal reading first
                try:
                    df = pd.read_csv(file, index_col='basin')
                    print(f"    ✓ Process {process_id}: {len(df)} basins")
                    dfs.append(df)
                except pd.errors.ParserError:
                    # If CSV has inconsistent columns, read line by line
                    print(f"    ⚠ Process {process_id}: Inconsistent columns, reading carefully...")
                    with open(file, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse header
                    header = lines[0].strip().split(',')
                    
                    # Parse each data line
                    basins_data = []
                    for i, line in enumerate(lines[1:], start=2):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:  # At least basin ID and some data
                            basin_id = parts[0]
                            # Take only as many data columns as header specifies
                            data_values = parts[1:len(header)]
                            # Pad with NaN if needed
                            while len(data_values) < len(header) - 1:
                                data_values.append('')
                            basins_data.append([basin_id] + data_values[:len(header)-1])
                    
                    # Create DataFrame
                    df = pd.DataFrame(basins_data, columns=header)
                    df = df.set_index('basin')
                    df.index = df.index.astype(int)
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"    ✓ Process {process_id}: {len(df)} basins (repaired)")
                    dfs.append(df)
                    
            except Exception as e:
                print(f"    ✗ Process {process_id}: Error reading file - {e}")
                continue
        
        if dfs:
            # Combine all dataframes with outer join to handle different columns
            combined_df = pd.concat(dfs, axis=0, join='outer')
            
            # Remove duplicates (keep last occurrence in case of re-runs)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            # Sort by basin index
            combined_df = combined_df.sort_index()
            
            # Convert numeric columns to dates if needed (for prediction files)
            if 'predict' in base_name and has_numeric_columns(combined_df):
                print(f"    → Converting numeric columns to dates...")
                # Extract period from filename (cal or eva)
                period_match = re.search(r'_(cal|eva)$', base_name)
                period = period_match.group(1) if period_match else 'cal'
                
                # Generate date columns
                date_cols = generate_date_columns(period, len(combined_df.columns))
                
                # Rename columns to dates
                if len(date_cols) == len(combined_df.columns):
                    combined_df.columns = date_cols
                    print(f"    → Converted {len(combined_df.columns)} columns to dates")
                else:
                    print(f"    ⚠ Warning: Expected {len(combined_df.columns)} dates but got {len(date_cols)}")
            
            # Save combined file - determine output path
            if subdir.startswith("predict_p"):
                output_subdir = subdir.replace("predict_p", "predict")
            elif subdir.startswith("results_p"):
                output_subdir = subdir.replace("results_p", "results")
            else:
                output_subdir = subdir
            
            output_path = base_dir / output_subdir
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{base_name}.{ext}"
            combined_df.to_csv(output_file)
            
            print(f"    → Combined file: {output_file}")
            print(f"    → Total basins: {len(combined_df)}")
        else:
            print(f"    ✗ No valid dataframes to combine")

def verify_results():
    """Verify that all basins have predictions"""
    print("\n" + "="*60)
    print("Verifying combined results...")
    print("="*60)
    
    base_dir = Path("/data0/funato/hsp2/0_out/TCN/GB/adapted")
    
    # Check main prediction files
    files_to_check = [
        base_dir / "predict" / "TCN_predict_cal.csv",
        base_dir / "predict" / "TCN_predict_eva.csv",
    ]
    
    total_basins = 424  # For GB dataset
    
    for file in files_to_check:
        if file.exists():
            df = pd.read_csv(file, index_col='basin')
            print(f"\n  {file.name}:")
            print(f"    Basins found: {len(df)} / {total_basins}")
            if len(df) < total_basins:
                missing_basins = set(range(1, total_basins + 1)) - set(df.index)
                print(f"    ⚠ Missing {total_basins - len(df)} basins")
                print(f"    Missing basin IDs: {sorted(list(missing_basins))[:10]}...")  # Show first 10
            else:
                print(f"    ✓ All basins present")
        else:
            print(f"\n  {file.name}: File not found")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("TCN Results Combination (Existing Data)")
    print("="*60)
    
    # Combine results from all processes
    combine_results()
    
    # Verify combined results
    verify_results()
    
    print("\n" + "="*60)
    print("Process completed!")
    print("="*60)

if __name__ == "__main__":
    main()
