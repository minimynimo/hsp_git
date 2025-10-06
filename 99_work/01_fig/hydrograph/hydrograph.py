import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import glob


def get_prediction(model, train_type, basin_id, ce):
    """Get prediction data for a specific model, train type, basin, and ce"""
    base_path = '/data0/funato/hsp/0_out'
    
    # Try different file patterns for different train types
    if train_type == 'individual':
        if model == "LSTM":
            # Individual files have different naming pattern and structure
            file_pattern = f"{base_path}/{model}/GB/{train_type}/predict/{model}_predict_*_{ce}.csv"
        else:
            # Regional and adapted files
            file_pattern = f"{base_path}/{model}/GB/{train_type}/predict/{model}_predict_{ce}.csv"
    elif model == 'TCN' and train_type == "regional":
        file_pattern = f"{base_path}/TCN/GB/shap/regional/predict/og/TCN_predict_{ce}_basin_{basin_id}.csv"
    else:
        # Regional and adapted files
        file_pattern = f"{base_path}/{model}/GB/{train_type}/predict/og/{model}_predict_{ce}_basin_{basin_id}.csv"

    
    try:
        files = glob.glob(file_pattern)
        if not files:
            print(f"No file found for pattern: {file_pattern}")
            return None
            
        data = pd.read_csv(files[0])
        
        if train_type == 'individual':
            # Individual files have dates as columns and basin data as rows
            # Find the row for this basin
            basin_row_id = f"file_{basin_id}_{ce}"
            
            # Check if basin exists in the data
            if basin_row_id not in data.iloc[:,0].values:
                print(f"Basin {basin_id} not found in individual file {files[0]}")
                return None
            
            # Find the correct row
            row_idx = data[data.iloc[:,0] == basin_row_id].index[0]
            
            # Extract dates (columns 1 onwards) and values for this basin
            dates = data.columns[1:]  # Skip first column which contains row identifiers
            values = data.iloc[row_idx, 1:].values
            
            result = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Predicted': values
            })
            return result
            
        else:
            # Regional/adapted files have standard format
            if 'Date' not in data.columns:
                print(f"Date column not found in {files[0]}")
                return None
                
            # Standardize column names
            if 'Predicted' in data.columns:
                predicted_col = 'Predicted'
            elif 'Prediction' in data.columns:
                predicted_col = 'Prediction'
            else:
                print(f"No prediction column found in {files[0]}")
                return None
                
            # Create standardized dataframe
            result = pd.DataFrame({
                'Date': pd.to_datetime(data['Date']),
                'Predicted': data[predicted_col]
            })
            
            # Add observed data if available
            if 'Observed' in data.columns:
                result['Observed'] = data['Observed']
                
            return result
        
    except Exception as e:
        print(f"Error reading {file_pattern}: {e}")
        return None

def check_available_train_types(model, basin_id):
    """Check which train types are available for a given model and basin"""
    available_types = []
    base_types = ['regional', 'adapted', 'individual']
    
    for train_type in base_types:
        # Check if files exist for calibration ce
        cal_data = get_prediction(model, train_type, basin_id, 'cal')
        if cal_data is not None:
            available_types.append(train_type)
    
    return available_types


model_list = ['LSTM', "XGB", "TCN", "RF"]
basin_id_list = list(range(1,11))

output_dir = "/data0/funato/hsp/0_out/1_fig/hydrograph/ML_comparison"
os.makedirs(output_dir, exist_ok=True)


# Color scheme for different training types
color_scheme = {
    "regional": "red",
    "individual": "blue", 
    "adapted": "green"
}

# create a hydrograph with both the calibration and evaluation periods for each of the model_list with the train_types shown in different line styles

# New: One file per model, all train types and basins overlayed
for model in model_list:
    print(f"=== Processing Model {model} ===")
    
    # Check which train types are available for the first basin as reference
    available_train_types = check_available_train_types(model, basin_id_list[0])
    print(f"Available train types for {model}: {available_train_types}")
    
    if not available_train_types:
        print(f"No train types available for {model}, skipping...")
        continue
    
    plt.figure(figsize=(15, 7))
    
    for basin_id in basin_id_list:
        print(f"  Processing Basin {basin_id}")
        has_observed = False
        
        for train_type in available_train_types:
            # Get calibration and evaluation data
            cal_data = get_prediction(model, train_type, basin_id, 'cal')
            eva_data = get_prediction(model, train_type, basin_id, 'eva')
            
            if cal_data is not None and eva_data is not None:
                # Convert to lists for plotting
                dates_cal = cal_data['Date'].tolist()
                predicted_cal = cal_data['Predicted'].tolist()
                dates_eva = eva_data['Date'].tolist()
                predicted_eva = eva_data['Predicted'].tolist()
                
                # Get color for this train type
                train_color = color_scheme[train_type]
                
                # Plot predictions with same color and style for cal/eva
                plt.plot(dates_cal, predicted_cal, 
                        label=f'{model} {train_type} (Basin {basin_id})', 
                        color=train_color, linestyle='-', alpha=0.7)
                plt.plot(dates_eva, predicted_eva, 
                        color=train_color, linestyle='-', alpha=0.7)  # No label for eva to avoid duplicate legend entries
                
                # Add vertical line between calibration and evaluation periods
                if dates_cal and dates_eva:
                    # Find the boundary between cal and eva periods
                    cal_end = max(dates_cal)
                    eva_start = min(dates_eva)
                    boundary_date = cal_end  # Use end of calibration as boundary
                    plt.axvline(x=boundary_date, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                
                # Only plot observed once per basin (from first available dataset)
                if not has_observed and 'Observed' in cal_data.columns:
                    observed_cal = cal_data['Observed'].tolist()
                    observed_eva = eva_data['Observed'].tolist() if 'Observed' in eva_data.columns else None
                    
                    plt.plot(dates_cal, observed_cal, 
                            label=f'Observed (Basin {basin_id})', 
                            color='black', linestyle='-', alpha=0.8, linewidth=2)
                    if observed_eva is not None:
                        plt.plot(dates_eva, observed_eva, 
                                color='black', linestyle='-', alpha=0.8, linewidth=2)  # No label to avoid duplicate
                    has_observed = True
            else:
                print(f"    Warning: Could not load data for {train_type} basin {basin_id}")
    
        plt.title(f'Hydrograph Comparison for {model}')
        plt.xlabel('Date')
        plt.ylabel('Discharge [mm/day]')
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hydrograph_basin_{basin_id}_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.join(output_dir, f'hydrograph_basin_{basin_id}_{model}.png')}")