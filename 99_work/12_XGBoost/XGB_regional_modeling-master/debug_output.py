"""
Hydrograph Visualization Tool for XGBoost Model Results

This script reads the output prediction files from the evaluation script
and generates plots comparing observed vs. predicted streamflow for a
random sample of basins.

Usage:
    python /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/debug_output.py --run_dir /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/runs_xgb/xgb_run_1015_0335_seed300
"""
# plot the calibration and evaluation period discharge in the same figure with different colors. also create additional sub plots for the input data that shows the original and the rescaled data to check that the rescaling was done correctly.
"""
Hydrograph Visualization and Input Data Debugging Tool

This script reads the output prediction files and the scaler file from a run 
directory to generate detailed diagnostic plots. Each plot includes:
1. A unified hydrograph comparing observed vs. predicted flow for both 
   calibration and evaluation periods.
2. A plot of the original, unscaled input data (Precip, PET, Temp).
3. A plot of the scaled input data to verify normalization.

Usage:
    python visualize_results.py --run_dir /path/to/your/xgb_run_directory
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import json
import numpy as np

def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize XGBoost hydrograph predictions and debug inputs."
    )
    parser.add_argument(
        '--run_dir', 
        type=str, 
        required=True, 
        help="Path to the specific XGBoost run directory (e.g., 'runs_xgb/xgb_run_...')."
    )
    parser.add_argument(
        '--num_basins', 
        type=int, 
        default=5, 
        help="Number of random basins to plot."
    )
    return parser.parse_args()

def load_data_for_basin(file_num: int) -> pd.DataFrame:
    """Load original timeseries data for a specific basin."""
    varssim_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"
    filename = f"varssim{str(file_num).zfill(3)}.csv"
    filepath = Path(varssim_dir) / filename
    
    if not filepath.is_file():
        print(f"Warning: Data file not found for basin {file_num} at {filepath}")
        return None
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def main():
    """Main function to load results and generate plots."""
    args = get_args()
    run_dir = Path(args.run_dir)
    
    if not run_dir.is_dir():
        print(f"Error: Run directory not found at '{run_dir}'")
        return

    # --- Load Scalers (CRITICAL for debugging inputs) ---
    scaler_path = run_dir / 'scalers.json'
    if not scaler_path.is_file():
        print(f"Error: 'scalers.json' not found in {run_dir}. Cannot verify scaling.")
        return
    with open(scaler_path, 'r') as fp:
        scalers = json.load(fp)
        # Convert to numpy arrays for calculations
        for key in scalers:
            if isinstance(scalers[key], list):
                scalers[key] = np.array(scalers[key])
    print("Loaded scalers for input data verification.")

    # Define paths to prediction files
    output_base = Path("/data0/funato/hsp/0_out/XGB/GB/regional")
    cal_pred_path = output_base / "predict" / "XGB_predict_cal.csv"
    eva_pred_path = output_base / "predict" / "XGB_predict_eva.csv"

    if not cal_pred_path.is_file() or not eva_pred_path.is_file():
        print(f"Error: Prediction files not found.")
        print(f"Looked for: {cal_pred_path}")
        print(f"And: {eva_pred_path}")
        return
        
    print("Loading prediction files...")
    df_cal_pred = pd.read_csv(cal_pred_path, index_col='date', parse_dates=True)
    df_eva_pred = pd.read_csv(eva_pred_path, index_col='date', parse_dates=True)
    
    available_basins = df_cal_pred.columns.tolist()
    
    if len(available_basins) < args.num_basins:
        print(f"Warning: Fewer basins available ({len(available_basins)}) than requested ({args.num_basins}). Plotting all available.")
        basins_to_plot = available_basins
    else:
        basins_to_plot = random.sample(available_basins, args.num_basins)
        
    print(f"Selected basins to plot: {basins_to_plot}")
    
    plot_dir = run_dir / "../debug_hydrograph_plots"
    plot_dir.mkdir(exist_ok=True)
    print(f"Saving plots to: {plot_dir}")

    for basin_id in basins_to_plot:
        print(f"  Plotting basin {basin_id}...")
        
        df_obs_full = load_data_for_basin(int(basin_id))
        if df_obs_full is None:
            continue
            
        # --- Create Figure with 3 subplots ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
        fig.suptitle(f'Debug Analysis for Basin {basin_id}', fontsize=18)

        # --- 1. Hydrograph Comparison Plot ---
        ax1.set_title('Hydrograph Comparison (Observed vs. Predicted)', fontsize=14)
        
        # Calibration Data
        cal_start, cal_end = df_cal_pred.index.min(), df_cal_pred.index.max()
        obs_cal = df_obs_full.loc[cal_start:cal_end, 'Obs flow']
        pred_cal = df_cal_pred[basin_id]
        ax1.plot(obs_cal.index, obs_cal.values, label='Observed (All)', color='black', linewidth=1.5)
        ax1.plot(pred_cal.index, pred_cal.values, label='Predicted (Calibration)', color='red', linestyle='--', alpha=0.8)

        # Evaluation Data
        eva_start, eva_end = df_eva_pred.index.min(), df_eva_pred.index.max()
        obs_eva = df_obs_full.loc[eva_start:eva_end, 'Obs flow']
        pred_eva = df_eva_pred[basin_id]
        ax1.plot(obs_eva.index, obs_eva.values, label='Observed (All)', color='black', linewidth=1.5)
        ax1.plot(pred_eva.index, pred_eva.values, label='Predicted (Evaluation)', color='blue', linestyle='--', alpha=0.8)
        
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- 2. Original Input Data Plot ---
        ax2.set_title('Original Input Data', fontsize=14)
        input_cols = ['Precip', 'PET', 'Temp']
        combined_inputs = df_obs_full.loc[cal_start:eva_end, input_cols]
        
        ax2.plot(combined_inputs.index, combined_inputs['PET'], label='PET', color='orange')
        ax2.plot(combined_inputs.index, combined_inputs['Temp'], label='Temp', color='green')
        ax2.set_ylabel('PET & Temp')
        ax2.legend(loc='upper left')
        
        # Use a secondary y-axis for precipitation
        ax2_twin = ax2.twinx()
        ax2_twin.bar(combined_inputs.index, combined_inputs['Precip'], label='Precipitation', color='lightblue', width=1.0)
        ax2_twin.set_ylabel('Precipitation (mm)')
        ax2_twin.legend(loc='upper right')
        ax2_twin.invert_yaxis() # Common practice for precip plots
        
        # --- 3. Scaled Input Data Plot ---
        ax3.set_title('Scaled (Normalized) Input Data', fontsize=14)
        scaled_inputs = (combined_inputs - scalers['dynamic_means']) / scalers['dynamic_stds']
        
        ax3.plot(scaled_inputs.index, scaled_inputs['Precip'], label='Precip (scaled)', color='lightblue')
        ax3.plot(scaled_inputs.index, scaled_inputs['PET'], label='PET (scaled)', color='orange')
        ax3.plot(scaled_inputs.index, scaled_inputs['Temp'], label='Temp (scaled)', color='green')
        ax3.set_ylabel('Standardized Value')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = plot_dir / f"basin_{basin_id}_debug_analysis.png"
        plt.savefig(save_path)
        plt.close(fig)

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()

