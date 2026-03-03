import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors

from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from tcn import TCN
from datetime import datetime

import shap
import sys

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 2
#loc, ver = "GB", 2
loc, ver = "US", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"

attribute_dir = f'/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

# Define synthetic precipitation columns
syn_precip_cols = ['syn_no_precip',
                    'syn_peak1_rate0','syn_peak1_rate10', 'syn_peak1_rate30', 'syn_peak1_rate50',
                    'syn_peak3_rate0','syn_peak3_rate10', 'syn_peak3_rate30', 'syn_peak3_rate50',
                    'syn_peak5_rate0', 'syn_peak5_rate10', 'syn_peak5_rate30', 'syn_peak5_rate50']

# Define the 18 valid basin attributes
VALID_ATTR = [
    'p_mean', 'pet_mean', 'aridity', 'seasonality', 'frac_snow', 'high_prec_freq',
    'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'ele_mt_smn', 'slp_dg_sav', 'area',
    'for_pc_sse', 'snd_pc_sav', 'slt_pc_sav', 'cly_pc_sav','soc_th_sav', 'swc_pc_syr']

if loc == "JP" and ver == 2:
    file_tot_num = 87
    file_list = list(range(1, file_tot_num + 1))
    columns_drop = ['grdc_no','river','station','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','lat_org','long_org', 'land_GolfCourse','land_GolfCourse_Ratio']
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    file_tot_num = 424
    file_list = list(range(1, file_tot_num + 1))
    columns_drop = ['gauge_id', 'gauge_name', 'country']
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_GB"
elif loc == "US" and ver == 2:
    # Load small basins list (area < 1000 km²) for US
    small_basins_csv = "/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/small_basins_1000.csv"
    small_basins_df = pd.read_csv(small_basins_csv)
    file_list = small_basins_df['basin_id'].tolist()
    columns_drop = ['gauge_id','gauge_name','country']
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8', index_col=0, header=0)
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_US"
    file_tot_num = len(file_list)

# TCN parameters (matching regional code)
timesteps = 270  # seq_length from regional code
epochs = 50
batch_size = 32  # Matching regional code
nb_filters = 64  # Matching regional code (num_filters)
kernel_size = 4
nb_stacks = 3
dropout = 0.4  # Matching regional code

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f"/data0/funato/hsp2/0_out/TCN/{loc}/shap/regional"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict/og", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True)
os.makedirs(output_dir + "/predict/synthetic", exist_ok=True)

output_dir_shap = f"0_out/1_fig/shap/TCN/{loc}/regional"
os.makedirs(output_dir_shap, exist_ok=True)


# clean attribute values and normalize
attribute_values_dropped = attribute_values.drop(columns=[col for col in columns_drop if col in attribute_values.columns])

# Keep only valid attributes (File_num if it exists as column, plus VALID_ATTR)
keep_cols = [col for col in VALID_ATTR if col in attribute_values_dropped.columns]
if 'File_num' in attribute_values_dropped.columns:
    keep_cols = ['File_num'] + keep_cols
attribute_values_dropped = attribute_values_dropped[keep_cols]

cols_to_numeric = [col for col in attribute_values_dropped.columns if col != 'File_num']
attribute_values_dropped[cols_to_numeric] = attribute_values_dropped[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
attribute_values_numeric = attribute_values_dropped.drop(columns=['File_num']) if 'File_num' in attribute_values_dropped.columns else attribute_values_dropped
attribute_mean = attribute_values_numeric.mean()
attribute_std = attribute_values_numeric.std()
attribute_values_normalized = attribute_values_dropped.copy()
attribute_values_normalized.loc[:, attribute_values_numeric.columns] = (attribute_values_numeric - attribute_mean) / attribute_std

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    df = df.reindex(full_date_range)
    return df

def BMK(obs_data,sim_data,benchmark):
    if len(obs_data) == 0 or len(sim_data) == 0: return np.nan
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
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

dynamic_features = ["Precip", "Temp", "PET"]
static_feature_names = [col for col in attribute_values_normalized.columns if col != 'File_num']
feature_names = dynamic_features + static_feature_names

all_input_cal_list, all_target_cal_list = [], []
all_input_eva_list, all_target_eva_list = [], []
individual_basin_data = {}

start_time = datetime.now()
log_file.write(f"start: {start_time.strftime('%a %b %d %I:%M:%S %p JST %Y')}\n")
log_file.flush()

print("--- Collecting Data for all Basins ---")
log_file.write("Collecting data for all basins...\n")
for file_num in file_list:
    log_file.write(f"Processing basin {file_num}/{file_tot_num}\n")
    log_file.flush()
    
    df = load_data(file_num)
    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    input_cal_dyn = df_cal[dynamic_features].values
    input_eva_dyn = df_eva[dynamic_features].values
    
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    static_features_for_basin = attribute_values_normalized.loc[file_num].drop('File_num', errors='ignore')
    static_array_cal = np.tile(static_features_for_basin.values, (np.sum(mask_cal), 1))
    static_array_eva = np.tile(static_features_for_basin.values, (np.sum(mask_eva), 1))

    input_cal_full = np.hstack([input_cal_dyn[mask_cal], static_array_cal])
    input_eva_full = np.hstack([input_eva_dyn[mask_eva], static_array_eva])

    all_input_cal_list.append(input_cal_full)
    all_input_eva_list.append(input_eva_full)

    target_cal = df_cal['Obs flow'].values[mask_cal]
    target_eva = df_eva['Obs flow'].values[mask_eva]
    all_target_cal_list.append(target_cal)
    all_target_eva_list.append(target_eva)

    individual_basin_data[file_num] = {
        'input_cal_full': input_cal_full, 'target_cal': target_cal, 'dates_cal': df_cal.index[mask_cal],
        'input_eva_full': input_eva_full, 'target_eva': target_eva, 'dates_eva': df_eva.index[mask_eva],
    }

all_input_cal = np.vstack(all_input_cal_list)
all_target_cal = np.hstack(all_target_cal_list)
all_input_eva = np.vstack(all_input_eva_list)
all_target_eva = np.hstack(all_target_eva_list)

print("--- Normalizing Data and Creating Sequences ---")
log_file.write("Normalizing data and creating sequences...\n")
log_file.flush()
mean_dyn = np.mean(all_input_cal[:, :3], axis=0)
std_dyn = np.std(all_input_cal[:, :3], axis=0)
std_dyn = np.where(std_dyn == 0, 1e-6, std_dyn)

all_input_cal[:, :3] = (all_input_cal[:, :3] - mean_dyn) / std_dyn
all_input_eva[:, :3] = (all_input_eva[:, :3] - mean_dyn) / std_dyn

input_cal_seq, target_cal_seq = create_sequences(all_input_cal, all_target_cal, timesteps)
input_eva_seq, target_eva_seq = create_sequences(all_input_eva, all_target_eva, timesteps)

print(f"Shape of training sequences: {input_cal_seq.shape}")
print(f"Shape of training targets: {target_cal_seq.shape}")

print("--- Building and Training TCN Model ---")
model_in = Input(shape=(timesteps, all_input_cal.shape[1]))
model_out = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, return_sequences=False)(model_in)
model_out = Dense(1, activation='linear')(model_out)
TCN_model = Model(inputs=model_in, outputs=model_out)
TCN_model.compile(optimizer='adam', loss='mse')

TCN_model.fit(input_cal_seq, target_cal_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
model_save_path = os.path.join(output_dir, "TCN_model.h5")
TCN_model.save(model_save_path)
log_file.write(f"Model saved to: {model_save_path}\n")
log_file.flush()
print(f"Model saved to: {model_save_path}")

print("--- Performing SHAP Analysis using KernelExplainer (this may be slow) ---")
# Use a subset of the training data as background for the explainer
# KernelExplainer is slow, so we use a very small background set.
background_size = 50
if input_cal_seq.shape[0] > background_size:
    background_indices = np.random.choice(input_cal_seq.shape[0], background_size, replace=False)
    background = input_cal_seq[background_indices]
else:
    background = input_cal_seq

# Use a small subset of the evaluation data to explain
explain_size = 1000
if input_eva_seq.shape[0] > explain_size:
    explain_indices = np.random.choice(input_eva_seq.shape[0], explain_size, replace=False)
    X_sample = input_eva_seq[explain_indices]
else:
    X_sample = input_eva_seq

if X_sample.shape[0] > 0:
    # We need a function that can take a 2D array and predict, as KernelExplainer works with 2D data.
    def predict_fn(x):
        # x has shape (n_samples, n_features)
        # We need to expand it to (n_samples, timesteps, n_features)
        x_3d = np.tile(x[:, np.newaxis, :], (1, timesteps, 1))
        return TCN_model.predict(x_3d, batch_size=batch_size)

    # We summarize the background data by taking the mean over the time axis
    background_summary = np.mean(background, axis=1)
    
    print("Initializing SHAP KernelExplainer (this may be slow)...")
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    
    # We explain the time-series-averaged samples
    X_sample_summary = np.mean(X_sample, axis=1)
    
    print("Calculating SHAP values with KernelExplainer...")
    shap_values = explainer.shap_values(X_sample_summary)
    
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(axis=2)
    
    X_sample_df = pd.DataFrame(X_sample_summary, columns=feature_names)
    
    # Create the Explanation object for summary plots
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample_summary,
        feature_names=feature_names
    )

    # --- Save SHAP results for later analysis ---
    print("Saving SHAP objects for later analysis...")
    shap_results_dir = os.path.join(output_dir, "shap_objects")
    os.makedirs(shap_results_dir, exist_ok=True)
    
    np.save(os.path.join(shap_results_dir, "shap_values.npy"), shap_explanation.values)
    np.save(os.path.join(shap_results_dir, "expected_value.npy"), shap_explanation.base_values)
    pd.DataFrame(shap_explanation.data, columns=shap_explanation.feature_names).to_csv(os.path.join(shap_results_dir, "X_sample.csv"))
    
    print("SHAP objects saved.")

    # --- Generate summary plots using the Explanation object ---
    print("Generating SHAP summary plots...")
    shap.summary_plot(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/TCN_shap_summary_beeswarm.png")
    plt.close()

    shap.summary_plot(shap_explanation, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir_shap}/TCN_shap_summary_bar.png")
    plt.close()
    
    # --- FIX: Generate force plots by passing components explicitly ---
    print("Generating SHAP force plots...")
    for i in range(min(5, X_sample_summary.shape[0])):
        # Pass the base value, the i-th row of shap_values, and the i-th row of features
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[i,:], 
            X_sample_df.iloc[i,:]
        )
        shap.save_html(f"{output_dir_shap}/TCN_force_plot_sample_{i}.html", force_plot)

log_file.write("SHAP analysis completed.\n")
log_file.flush()
log_file.close()
print("--- SHAP Analysis Complete ---")
print("DONE")
