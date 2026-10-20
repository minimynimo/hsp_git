# Description: Bayesian Model Averaging for hydrological model ensembles
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd


#######
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

loc = "US"

# dataset: 'camels' (default) -- built directly from raw CAMELS-US -- or 'caravan', the older
# Caravan-derived extraction. Only affects loc='US'. Both now use the same cal/eva split as
# main_HYPER.py / hparam_main_HYPER.py and the other model families (main_random_forest_indiv.py,
# main_xgboost_indiv.py, main_TCN_indiv.py, main_LSTM_indiv.py). See also
# ealstm_regional_modeling/main_lstm_regional.py's --dataset for the same split.
#
# NOTE: this only changes which per-basin gauge list / cal-eval dates BMA.py uses -- it does
# NOT by itself make /2_MERV/{dataset}/HYPER/{loc}/{scenario}/{ver_name}/varssim_{gauge_id}.csv
# exist. That directory (a merged {m01..m47} ensemble-prediction matrix per basin/scenario) is
# built by 2_MERV/create_dataset_HYPER.py from the upstream 47-model ensemble output.
dataset = "camels"  # "caravan"

#ver = 1
ver = 2
############

ver_name = 'ver1_1' if ver == 1 else 'ver2_0' if ver == 2 else 'ver3_0'

varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/HYPER/{loc}"

# Basin list, keyed by gauge_id: /2_MERV/{dataset}/HYPER/{loc}/{scenario}/{ver_name}/ stores its
# per-basin CSVs as varssim_{gauge_id}.csv (not the old sequential varssim{NNN}.csv), matching
# what main_HYPER.py / hparam_main_HYPER.py already load from. This script's own output
# (BMA_weights.csv, BMA_predict_*.csv, BMA_results_*.csv) is now keyed by gauge_id too --
# main_HYPER.py/hparam_main_HYPER.py look up a basin's 'Sim flow' by gauge_id, not file_num.
basin_list_csv = f"/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_{dataset}_file_num.csv"
basin_df = pd.read_csv(basin_list_csv, dtype={'gauge_id': str})
basins = [g.split('_')[-1] for g in basin_df['gauge_id']]

precip_cols = ['no_precip',
               'peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
               'peak3_rate0','peak3_rate10', 'peak3_rate30', 'peak3_rate50',
               'peak5_rate0', 'peak5_rate10', 'peak5_rate30', 'peak5_rate50']

# Define the calibration and evaluation periods
if dataset == "camels":
    start_date_cal = '1987-01-01'
    end_date_cal = '2000-12-31'
    start_date_eva = '2001-01-01'
    end_date_eva = '2006-12-31'
else:  # caravan
    start_date_cal = '1990-01-01'
    end_date_cal = '2000-12-31'
    start_date_eva = '2001-01-01'
    end_date_eva = '2006-12-31'

print(f"[BMA] loc={loc} dataset={dataset!r} "
      f"cal={start_date_cal}..{end_date_cal} eva={start_date_eva}..{end_date_eva}")

# List of models
model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
            "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
            "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
            "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39", "m40",
            "m41", "m42", "m43", "m44", "m45","m46", "m47"]

output_dir = f'/data0/funato/hsp2/0_out/BMA/{loc}/{dataset}'

os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict/synthetic', exist_ok=True)
os.makedirs(output_dir + '/weights', exist_ok=True)

#model_list = ["m34"]
#buf = "m34"
buf = ""

def BMK(obs_data,sim_data,benchmark):
    """
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)
    """
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]


    if benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2) 
    
    elif benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
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

# Function to load and preprocess data
def load_data(data_name, gauge_id):
    df = pd.read_csv(f"{varssim_dir}/{data_name}/{ver_name}/varssim_{gauge_id}.csv")

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

# Log-likelihood function
def log_likelihood(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]
    obs = np.where(obs == 0, 1e-6, obs)
    sim = np.where(sim == 0, 1e-6, sim)
    residuals = obs - sim
    sigma2 = np.var(residuals)
    if sigma2 == 0:  # Avoid division by zero
        sigma2 = 10e-6
    return -0.5 * (residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))


# Bayesian Model Averaging
# Initialize arrays to store log-likelihoods and posterior probabilities
log_likelihoods_cal = np.zeros(len(model_list))
log_likelihoods_eva = np.zeros(len(model_list))
priors = np.ones(len(model_list)) / len(model_list)  # Uniform priors

results_cal = []
results_eva = []

# Run BMA for each dataset
for gauge_id in basins:
    # Load the data
    df = load_data("varssim_nocal_og", gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Observations
    obs_cal = df_cal['Obs flow'].values
    obs_eva = df_eva['Obs flow'].values

    # CALIBRATION
    for i, model in enumerate(model_list):
        model_cal = df_cal[model].values
        log_likelihoods_cal[i] = np.sum(log_likelihood(obs_cal, model_cal))

    # Update priors based on log-likelihoods
    max_log_likelihood_cal = np.max(log_likelihoods_cal)  # For numerical stability
    weights_cal = np.exp(log_likelihoods_cal - max_log_likelihood_cal)
    posterior_cal = weights_cal * priors
    posterior_cal /= np.sum(posterior_cal) ##turn the weight sum into 1

    if gauge_id == basins[0]:
        model_row = np.concatenate((["model"], model_list))
        weight_df = pd.DataFrame([model_row])
        weight_df.to_csv(output_dir + f"/weights/BMA_weights.csv", mode='w', index=False, header=False)

    weight_row = np.concatenate(([f'gauge_{gauge_id}'], posterior_cal))
    with open(output_dir + f"/weights/BMA_weights.csv", 'a') as file:
        pd.DataFrame([weight_row]).to_csv(file, header=False, index=False)

    # EVALUATION
    model_matrix_eva = df_eva[model_list].values  # shape: (n_dates, n_models)
    predict_eva = np.dot(np.nan_to_num(model_matrix_eva, nan=0.0), posterior_cal)

    # Calibration testing for analysis
    model_matrix_cal = df_cal[model_list].values  # shape: (n_dates, n_models)
    predict_cal = np.dot(np.nan_to_num(model_matrix_cal, nan=0.0), posterior_cal)

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    file_row_cal = [f'gauge_{gauge_id}'] + list(predict_cal.flatten())
    file_row_eva = [f'gauge_{gauge_id}'] + list(predict_eva.flatten())

    if gauge_id == basins[0]:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/BMA_predict_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/BMA_predict_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/BMA_predict_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/BMA_predict_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            benchmark: BMK(obs_cal, predict_cal, benchmark)
        })
        file_results_eva.update({
            benchmark: BMK(obs_eva, predict_eva, benchmark)
        })  
    
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # using the same weights, calculate the BMA for the synthetic data
    for precip_col in precip_cols:
        if precip_col == 'no_precip':
            syn_df = load_data("varssim_nocal_no_precip", gauge_id)
        else:
            syn_df = load_data(f"varssim_nocal_{precip_col}", gauge_id)
        syn_cal = syn_df[start_date_cal:end_date_cal].copy()
        syn_eva = syn_df[start_date_eva:end_date_eva].copy()    

        # EVALUATION
        syn_model_matrix_eva = syn_eva[model_list].values
        syn_predict_eva = np.dot(np.nan_to_num(syn_model_matrix_eva, nan=0.0), posterior_cal)

        # Calibration testing for analysis
        syn_model_matrix_cal = syn_cal[model_list].values
        syn_predict_cal = np.dot(np.nan_to_num(syn_model_matrix_cal, nan=0.0), posterior_cal)

        syn_predict_cal[syn_predict_cal < 0] = 0
        syn_predict_eva[syn_predict_eva < 0] = 0

        file_row_cal = [f'gauge_{gauge_id}_cal'] + list(syn_predict_cal.flatten())
        file_row_eva = [f'gauge_{gauge_id}_eva'] + list(syn_predict_eva.flatten())

        if gauge_id == basins[0]:
            predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

            date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

            predict_cal_df = pd.DataFrame([date_row_cal])
            predict_eva_df = pd.DataFrame([date_row_eva])

            predict_cal_df.to_csv(output_dir + f"/predict/synthetic/BMA_predict_{precip_col}_cal.csv", mode='w', index=False, header=False)
            predict_eva_df.to_csv(output_dir + f"/predict/synthetic/BMA_predict_{precip_col}_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/predict/synthetic/BMA_predict_{precip_col}_cal.csv", 'a') as file:
            pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/predict/synthetic/BMA_predict_{precip_col}_eva.csv", 'a') as file:
            pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)



df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/BMA_results{buf}_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/BMA_results{buf}_eva.csv', index=False, header=True)


print(f"BMA")
print("DONE")