import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from tcn import TCN
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import shap
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TCN_regional_modeling-master"))
from papercode.utils import get_basin_list

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


#loc, ver = "JP", 2
loc, ver = "GB", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

attribute_dir = f'/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

# The 27-attribute CAMELS-US catchment-attribute set from
# /data0/funato/hsp2/CATCHMENT_ATTRIBUTE.md (see papercode/datautils.py's VALID_ATTR in both
# the LSTM and TCN regional-modeling dirs for the canonical copy / full naming-typo notes).
VALID_ATTR = [
    'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq',
    'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2',
    'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'soil_depth_pelletier',
    'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content',
    'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac', 'geol_permeability'
]

if loc == "JP" and ver == 2:
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio', 'File_num']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col='gauge_id', header=0, dtype={'gauge_id': str})
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8', index_col='gauge_id', header=0, dtype={'gauge_id': str})
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_GB"
elif loc == "US" and ver == 2:
    # Use the 27-attribute CAMELS-original combined CSV (built from the raw
    # camels_clim/topo/vege/soil/geol.txt files), not the old HydroATLAS-derived
    # camels_attribute_combined.csv, which does not contain any of VALID_ATTR's names.
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_original_attribute_combined.csv', encoding= 'UTF-8', index_col='gauge_id', header=0, dtype={'gauge_id': str})
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_US"

if loc in ("US", "GB"):
    # Restrict to VALID_ATTR (intersected with whatever is actually present -- GB's
    # CAMELS-GB-native attribute CSV only shares 11 of the 27 names outright; see
    # CATCHMENT_ATTRIBUTE.md / datautils.py's VALID_ATTR notes for the full list).
    available_attr = [c for c in VALID_ATTR if c in attribute_values.columns]
    missing_attr = [c for c in VALID_ATTR if c not in attribute_values.columns]
    if missing_attr:
        print(f"WARNING: {len(missing_attr)}/{len(VALID_ATTR)} VALID_ATTR columns are missing "
              f"for loc={loc} and will be skipped: {missing_attr}")
    attribute_values = attribute_values[available_attr]
    columns_drop = []

gauge_id_list = get_basin_list(loc)

input_dim = 3 + len(attribute_values.columns) - len(columns_drop)  # 3 meteorological + N attributes

timesteps, input_dim = 365, input_dim
epochs = 50 # Changed from 5 to 50 for better training potential
batch_size = 64
nb_filters = 128
kernel_size = 4
nb_stacks = 1


start_date_cal = '1990-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp2/0_out/TCN/{loc}/shap"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True) # New directory for hydrographs

output_dir_shap = f"0_out/1_fig/shap/TCN/regional"
os.makedirs(output_dir_shap, exist_ok=True)


# clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns]) # 87,193

# Convert all columns except 'File_num' to numeric, coercing errors to NaN
cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped
attribute_mean = attribute_values_numeric.mean()
attribute_std = attribute_values_numeric.std()
attribute_values_normalized = attribute_values_dropped.copy()
attribute_values_normalized.loc[:, attribute_values_numeric.columns] = (attribute_values_numeric - attribute_mean) / attribute_std


early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)


def load_data(gauge_id):
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")

    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df


def BMK(obs_data,sim_data,benchmark):    
    # Ensure data are flat for calculations if they come in (N, 1) shape
    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

    # Handle cases where obs_data might be all zero or very close to zero
    # This prevents division by zero or extremely large values in some metrics
    if np.sum(obs_data) == 0 and benchmark not in ["RMSE", "MAE"]:
        return np.nan # Or some other indicator for undefined metric

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        # Avoid division by zero if obs_data is constant
        if np.sum(np.square(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan # Perfect match for constant series
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        
        # Guard against zero standard deviation for constant series
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        # Guard against zero mean for very small values
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
        
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        if np.sum(np.abs(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "VE":
        if np.sum(obs_data) == 0:
             return 1.0 if np.all(obs_data == sim_data) else np.nan # If observed is zero, perfect if sim is also zero
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return 1.0 if numer == 0 else np.nan
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))
    
def create_sequences(data, targets, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

if os.path.exists(output_dir + f'/TCN_log.txt'):
    open(output_dir + f'/TCN_log.txt', 'w').close()
log_file = open(output_dir + f'/TCN_log.txt', 'a')


# results_cal and results_eva will store dicts for each gauge_id
results_cal = []
results_eva = []

dynamic_features = ["Precip", "Temp", "PET"]
static_features = attribute_values_normalized.columns.tolist()
feature_names = dynamic_features + static_features

# Initialize lists to store data for all basins
all_input_cal_list = []
all_input_eva_list = []

all_target_cal_list = []
all_target_eva_list = []

start_time_overall = datetime.now() # Renamed to avoid conflict inside loop later
start_time_overall_st = start_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall start: {start_time_overall_st}\n")
log_file.flush()

# --- Collecting Data for all Basins (for combined training) ---
# Also store individual basin data for later plotting and individual prediction processing
individual_basin_data = {}

for gauge_id in gauge_id_list:
    start_time_basin_load = datetime.now() 
    start_time_st_basin_load = start_time_basin_load.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id} - Data Loading\n")
    log_file.write(f"start: {start_time_st_basin_load}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()  
    df_eva = df[start_date_eva:end_date_eva].copy()

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal_dyn = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)])
    input_eva_dyn = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)])
    
    # Ensure masks consider both dynamic inputs and target
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    input_cal_dyn = input_cal_dyn[mask_cal]
    input_eva_dyn = input_eva_dyn[mask_eva]

    # Load static features once for this basin
    static_features_for_basin = attribute_values_normalized.loc[gauge_id]
    # Repeat static features for every timestep
    static_array_cal = np.tile(static_features_for_basin.values, (input_cal_dyn.shape[0], 1))
    static_array_eva = np.tile(static_features_for_basin.values, (input_eva_dyn.shape[0], 1))

    # Combine dynamic + static features
    input_cal_full = np.hstack([input_cal_dyn, static_array_cal])
    input_eva_full = np.hstack([input_eva_dyn, static_array_eva])

    # Append to global lists for combined training
    all_input_cal_list.append(input_cal_full)
    all_input_eva_list.append(input_eva_full)

    # Targets
    target_cal = df_cal['Obs flow'].values
    target_cal = target_cal[mask_cal]  # Apply mask to targets
    all_target_cal_list.append(target_cal)

    target_eva = df_eva['Obs flow'].values
    target_eva = target_eva[mask_eva]  # Apply mask to targets
    all_target_eva_list.append(target_eva)

    # Store original dates (DatetimeIndex) for individual plotting (before sequence creation)
    dates_cal_basin_original_full = df_cal.index[mask_cal] 
    dates_eva_basin_original_full = df_eva.index[mask_eva] 

    individual_basin_data[gauge_id] = {
        'input_cal_full_pre_norm': input_cal_full.copy(), # Store before combined norm
        'target_cal': target_cal,
        'dates_cal_original_full': dates_cal_basin_original_full,
        'input_eva_full_pre_norm': input_eva_full.copy(), # Store before combined norm
        'target_eva': target_eva,
        'dates_eva_original_full': dates_eva_basin_original_full,
    }
    end_time_basin_load = datetime.now()
    log_file.write(f"elapsed: {end_time_basin_load - start_time_basin_load}\n")
    log_file.flush()

# Stack all basins together for combined training
all_input_cal_combined = np.vstack(all_input_cal_list)
all_target_cal_combined = np.hstack(all_target_cal_list)

all_input_eva_combined = np.vstack(all_input_eva_list)
all_target_eva_combined = np.hstack(all_target_eva_list)

all_input_cal_combined = np.array(all_input_cal_combined, dtype=float)

# Compute mean/std for normalization of dynamic features from the *combined* calibration data
mean_dyn_combined = np.mean(all_input_cal_combined[:, :3], axis=0)
std_dyn_combined = np.std(all_input_cal_combined[:, :3], axis=0)

# Normalize dynamic features for combined training
all_input_cal_combined[:, :3] = (all_input_cal_combined[:, :3] - mean_dyn_combined) / std_dyn_combined
all_input_eva_combined[:, :3] = (all_input_eva_combined[:, :3] - mean_dyn_combined) / std_dyn_combined


# Create sequences for combined data for model training
input_cal_seq_combined, target_cal_seq_combined = create_sequences(all_input_cal_combined, all_target_cal_combined, timesteps)
input_eva_seq_combined, target_eva_seq_combined = create_sequences(all_input_eva_combined, all_target_eva_combined, timesteps)


log_file.write(f"Model Training Starting\n")
log_file.flush()

model_in = Input(shape=(timesteps, all_input_cal_combined.shape[1])) # (365, 196)

# TCN block 
model_out = TCN(
    nb_filters=nb_filters,
    kernel_size=kernel_size,
    nb_stacks=nb_stacks,
    return_sequences=False
)(model_in)

# Output
model_out = Dense(1, activation='linear')(model_out)

# Build and compile
TCN_model = Model(inputs=model_in, outputs=model_out)
TCN_model.compile(optimizer='adam', loss='mse')


# Fit the model
TCN_model.fit(input_cal_seq_combined, target_cal_seq_combined, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

log_file.write(f"Model Training Finished\n")
log_file.flush()


### SHAP Analysis (only on a subset of the combined calibration data) ###
# Background data (e.g., 50 samples)
background = input_cal_seq_combined[np.random.choice(input_cal_seq_combined.shape[0], 50, replace=False)]

# Samples to explain (e.g., 10-100)
X_sample = input_cal_seq_combined[:100]

explainer = shap.GradientExplainer((TCN_model.input, TCN_model.output), background)
shap_values = explainer.shap_values(X_sample)

shap_array = np.stack(shap_values, axis=0)  # Shape: (100, 365, 196, 1)
shap_vals_mean = shap_array.mean(axis=1)  # Shape: (100, 196, 1)
shap_vals_mean = shap_vals_mean.squeeze(axis=-1)  # Remove the last dimension, now shape: (100, 196)

# Average input over time for visualization
X_sample_mean = X_sample.mean(axis=1)  # Shape: (samples, features)

X_sample_df = pd.DataFrame(X_sample_mean, columns=feature_names)

# SHAP summary plot (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_mean, X_sample_df, show=False)
plt.tight_layout()
plt.savefig(f"{output_dir_shap}/TCN_shap_summary_beeswarm.png")
plt.close()

# bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_mean, X_sample_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{output_dir_shap}/TCN_shap_summary_bar.png")
plt.close()


# SHAP dependence plots (top 3 features based on summary plot)
for i in range(min(3, len(feature_names))): # Plot for up to top 3 features
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(i, shap_vals_mean, X_sample_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/TCN_shap_dependence_{feature_names[i].replace('/', '_')}.png") 
    plt.close()

base_values = TCN_model.predict(X_sample).flatten() # Ensure base_values is 1D

# Force plots (top 3 examples)
for i in range(min(3, X_sample.shape[0])): # Plot for up to top 3 samples
    shap.save_html(
        f"{output_dir_shap}/force_plot_sample_{i}.html",
        shap.force_plot(
            base_value=base_values[i],
            shap_values=shap_vals_mean[i],
            features=X_sample_df.iloc[i],
            feature_names=feature_names
        )
    )

log_file.write(f"SHAP Analysis Finished\n")
log_file.flush()

# --- Prepare CSV files for individual basin predictions (write headers once) ---
# Date ranges for combined predictions (needed for CSV headers)
all_dates_cal_orig_flat = np.hstack([d.to_numpy() for d in [v['dates_cal_original_full'] for k, v in individual_basin_data.items()]])
all_dates_eva_orig_flat = np.hstack([d.to_numpy() for d in [v['dates_eva_original_full'] for k, v in individual_basin_data.items()]])

# Dates for the actual sequences used in prediction (shifted by timesteps)
predict_cal_dates_combined_for_csv = all_dates_cal_orig_flat[timesteps:]
predict_eva_dates_combined_for_csv = all_dates_eva_orig_flat[timesteps:]

date_row_cal_csv = ['Date'] + [str(pd.to_datetime(date).date()) for date in predict_cal_dates_combined_for_csv]
date_row_eva_csv = ['Date'] + [str(pd.to_datetime(date).date()) for date in predict_eva_dates_combined_for_csv]

# Write header rows to CSV files (mode='w' to overwrite if they exist from previous runs)
with open(output_dir + f"/predict/TCN_predict_cal.csv", 'w') as file:
    pd.DataFrame([date_row_cal_csv]).to_csv(file, header=False, index=False)
with open(output_dir + f"/predict/TCN_predict_eva.csv", 'w') as file:
    pd.DataFrame([date_row_eva_csv]).to_csv(file, header=False, index=False)


# --- Loop to process, plot, and evaluate EACH individual file ---
for gauge_id in gauge_id_list:
    start_time_basin_eval = datetime.now()
    start_time_st_basin_eval = start_time_basin_eval.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id} - Evaluation and Plotting\n")
    log_file.write(f"start: {start_time_st_basin_eval}\n")
    log_file.flush()

    basin_data = individual_basin_data[gauge_id]

    # --- Calibration Data for current basin ---
    # Use the pre-normalized input data from `individual_basin_data`
    input_cal_full_basin = basin_data['input_cal_full_pre_norm'].copy() # Need copy to modify for normalization
    target_cal_basin = basin_data['target_cal']
    dates_cal_basin_original_full = basin_data['dates_cal_original_full'] 

    # Normalize dynamic features using the *combined* mean/std from combined training data
    input_cal_full_basin[:, :3] = (input_cal_full_basin[:, :3] - mean_dyn_combined) / std_dyn_combined
    
    input_cal_seq_basin, target_cal_seq_basin = create_sequences(input_cal_full_basin, target_cal_basin, timesteps)
    predict_cal_basin = TCN_model.predict(input_cal_seq_basin).flatten()
    predict_cal_basin[predict_cal_basin < 0] = 0

    # Adjust dates for the predictions based on timesteps and convert to numpy
    dates_cal_plot = dates_cal_basin_original_full[timesteps:].to_numpy()

    # Plot for individual basin Calibration
    plt.figure(figsize=(15, 6))
    plt.plot(dates_cal_plot, target_cal_seq_basin, label='Observed', color='blue', alpha=0.7)
    plt.plot(dates_cal_plot, predict_cal_basin, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Hydrograph (Calibration) - Basin {gauge_id}')
    plt.xlabel('Date')
    plt.ylabel('Observed Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hydrographs", f"hydrograph_cal_basin_{gauge_id}.png"))
    plt.close()

    # --- Evaluation Data for current basin ---
    input_eva_full_basin = basin_data['input_eva_full_pre_norm'].copy() # Need copy to modify for normalization
    target_eva_basin = basin_data['target_eva']
    dates_eva_basin_original_full = basin_data['dates_eva_original_full'] 

    # Normalize dynamic features using the *combined* mean/std
    input_eva_full_basin[:, :3] = (input_eva_full_basin[:, :3] - mean_dyn_combined) / std_dyn_combined

    input_eva_seq_basin, target_eva_seq_basin = create_sequences(input_eva_full_basin, target_eva_basin, timesteps)
    predict_eva_basin = TCN_model.predict(input_eva_seq_basin).flatten()
    predict_eva_basin[predict_eva_basin < 0] = 0

    # Adjust dates for the predictions based on timesteps and convert to numpy
    dates_eva_plot = dates_eva_basin_original_full[timesteps:].to_numpy()

    # Plot for individual basin Evaluation
    plt.figure(figsize=(15, 6))
    plt.plot(dates_eva_plot, target_eva_seq_basin, label='Observed', color='blue', alpha=0.7)
    plt.plot(dates_eva_plot, predict_eva_basin, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Hydrograph (Evaluation) - Basin {gauge_id}')
    plt.xlabel('Date')
    plt.ylabel('Observed Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hydrographs", f"hydrograph_eva_basin_{gauge_id}.png"))
    plt.close()

    # --- Prepare data for CSV saving (individual file) ---
    # Ensure predict_cal_basin and target_cal_seq_basin match lengths with dates_cal_plot
    # (which they should due to create_sequences)
    file_row_cal = [f'basin_{gauge_id}_cal'] + list(predict_cal_basin) # Use individual basin predictions
    file_row_eva = [f'basin_{gauge_id}_eva'] + list(predict_eva_basin) # Use individual basin predictions

    # Append individual file's prediction data to the pre-initialized CSVs
    with open(output_dir + f"/predict/TCN_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/TCN_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)


    # --- Calculate and store benchmark results for individual basins ---
    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

    for benchmark in benchmark_list:
        file_results_cal.update({
            f'TCN_{benchmark}_cal': BMK(target_cal_seq_basin, predict_cal_basin, benchmark)
        })
        file_results_eva.update({
            f'TCN_{benchmark}_eva': BMK(target_eva_seq_basin, predict_eva_basin, benchmark)
        })  

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results to CSV after each file
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)

    print(f"gauge_id: {gauge_id}, NSE_cal: {BMK(target_cal_seq_basin, predict_cal_basin, 'NSE'):.4f}, NSE_eva: {BMK(target_eva_seq_basin, predict_eva_basin, 'NSE'):.4f}")

    end_time_basin_eval = datetime.now()
    log_file.write(f"end: {end_time_basin_eval}\n")
    log_file.write(f"elapsed: {end_time_basin_eval - start_time_basin_eval}\n")
    log_file.write(f"Results saved for basin {gauge_id}\n")
    print(f"Results saved for basin {gauge_id}")
    log_file.flush()


end_time_overall = datetime.now()
end_time_overall_st = end_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall end: {end_time_overall_st}\n")
log_file.write(f"Overall elapsed: {end_time_overall - start_time_overall}\n")
log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)


print("DONE")