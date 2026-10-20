import numpy as np
import pandas as pd
import os

#loc = "GB"
loc = "US"

dataset = "camels" # "caravan"

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]


input_dir = f"/data0/funato/hsp2/0_out/HBV/{loc}/{dataset}/varssim_HBV"

output_dir = f"/data0/funato/hsp2/0_out/HBV/{loc}/{dataset}/results"
os.makedirs(output_dir, exist_ok=True)

# 'camels' (default): matches create_dataset_HBV.py's camels cal/eval window. 'caravan':
# matches create_dataset_HBV.py's legacy caravan window (unchanged).
if dataset == "camels":
    start_date_cal = "1987-01-01"
    end_date_cal = "2000-12-31"
    start_date_eva = "2001-01-01"
    end_date_eva = "2006-12-31"
else:  # caravan
    start_date_cal = "1993-01-01"
    end_date_cal = "2000-12-31"
    start_date_eva = "2001-01-01"
    end_date_eva = "2006-12-31"


def BMK(obs_data, sim_data, benchmark):
    """
    Calculate benchmark metrics for observed vs simulated data
    """
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)

    # Additional check for empty arrays
    if len(obs_data) == 0 or len(sim_data) == 0:
        return np.nan

    # Check for all NaN values
    if np.all(np.isnan(obs_data)) or np.all(np.isnan(sim_data)):
        return np.nan

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        if np.sum(np.square(obs_data - obs_ave)) == 0:
            return np.nan
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))

    elif benchmark == "KGE":
        if len(obs_data) < 2:  # Need at least 2 points for correlation
            return np.nan
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        if np.isnan(r):
            r = 0
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        if obs_std == 0:
            obs_std = 1e-6
        if obs_ave == 0:
            obs_ave = 1e-6
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)

    elif benchmark == "logNSE":
        temp_obs_data = np.where((obs_data <= 0) | np.isnan(obs_data), 1e-6, obs_data)
        temp_sim_data = np.where((sim_data <= 0) | np.isnan(sim_data), 1e-6, sim_data)
        obs_ave = np.mean(temp_obs_data)
        numer = np.sum((np.log(temp_sim_data) - np.log(temp_obs_data))**2)
        denom = np.sum((np.log(temp_obs_data) - np.log(obs_ave))**2)
        if denom == 0:
            return np.nan
        return 1 - numer / denom

    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        denom = np.sum(np.abs(obs_data - obs_ave))
        if denom == 0:
            return np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / denom)

    elif benchmark == "Erel":
        obs_ave = np.mean(obs_data)
        temp_obs_data = np.where(obs_data == 0, 1e-6, obs_data)
        numer = np.sum(np.square((temp_obs_data - sim_data) / temp_obs_data))
        denom = np.sum(np.square((obs_data - obs_ave) / obs_ave))
        if denom == 0:
            return np.nan
        return 1 - numer / denom

    elif benchmark == "VE":
        total_obs = np.sum(obs_data)
        if total_obs == 0:
            return np.nan
        return 1 - np.sum(np.abs(obs_data - sim_data)) / total_obs

    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return np.nan
        return 1 - numer / denom

    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))

    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

    else:
        return np.nan


# compare Obs flow to no_precip,p_pet,up_peak1_rate10,up_peak1_rate30,up_peak1_rate50,up_peak3_rate10,up_peak3_rate30,up_peak3_rate50,up_peak5_rate10,up_peak5_rate30,up_peak5_rate50

basin_list_csv = f"/data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/{loc.lower()}_{dataset}_file_num.csv"
gauge_id_file_num_df = pd.read_csv(basin_list_csv, header=0, index_col=None)
file_tot_num = len(gauge_id_file_num_df)
print(f"[HBV_benchmark] loc={loc} dataset={dataset!r} basins={file_tot_num} "
      f"cal={start_date_cal}..{end_date_cal} eva={start_date_eva}..{end_date_eva}")

results_cal = []
results_eva = []
gauge_ids_short = []

for _, row in gauge_id_file_num_df.iterrows():
    gauge_id_short = row['gauge_id'].split('_')[-1]
    gauge_ids_short.append(gauge_id_short)
    input_file = f"{input_dir}/varssim_{gauge_id_short}.csv"

    try:
        input_df = pd.read_csv(input_file)
        input_df_cal = input_df[(input_df['Date'] >= start_date_cal) & (input_df['Date'] <= end_date_cal)]
        input_df_eva = input_df[(input_df['Date'] >= start_date_eva) & (input_df['Date'] <= end_date_eva)]

        obs_cal = input_df_cal["Obs flow"].values
        obs_eva = input_df_eva["Obs flow"].values

        sim_cal = input_df_cal["og"].values
        sim_eva = input_df_eva["og"].values

        # Remove NaN values for calculations
        mask_cal = ~np.isnan(obs_cal) & ~np.isnan(sim_cal)
        mask_eva = ~np.isnan(obs_eva) & ~np.isnan(sim_eva)

        obs_cal_clean = obs_cal[mask_cal]
        sim_cal_clean = sim_cal[mask_cal]
        obs_eva_clean = obs_eva[mask_eva]
        sim_eva_clean = sim_eva[mask_eva]

        row_cal = []
        row_eva = []

        for benchmark in benchmark_list:
            if len(obs_cal_clean) > 0 and len(sim_cal_clean) > 0:
                result_cal = BMK(obs_cal_clean, sim_cal_clean, benchmark)
            else:
                result_cal = np.nan

            if len(obs_eva_clean) > 0 and len(sim_eva_clean) > 0:
                result_eva = BMK(obs_eva_clean, sim_eva_clean, benchmark)
            else:
                result_eva = np.nan

            row_cal.append(result_cal)
            row_eva.append(result_eva)

        results_cal.append(row_cal)
        results_eva.append(row_eva)

        print(f"Processed gauge {gauge_id_short}: {os.path.basename(input_file)}")

    except Exception as e:
        print(f"Error processing gauge {gauge_id_short}: {e}")
        # Add NaN values for this file
        results_cal.append([np.nan] * len(benchmark_list))
        results_eva.append([np.nan] * len(benchmark_list))

# Create DataFrames for calibration and evaluation results with HBV prefix
cal_columns = [f"HBV_{benchmark}_cal" for benchmark in benchmark_list]
eva_columns = [f"HBV_{benchmark}_eva" for benchmark in benchmark_list]

results_cal_df = pd.DataFrame(results_cal, columns=cal_columns, index=gauge_ids_short)
results_cal_df.index.name = "gauge_id"

results_eva_df = pd.DataFrame(results_eva, columns=eva_columns, index=gauge_ids_short)
results_eva_df.index.name = "gauge_id"

# Save results to CSV files
cal_output_file = f"{output_dir}/HBV_results_cal.csv"
eva_output_file = f"{output_dir}/HBV_results_eva.csv"

results_cal_df.to_csv(cal_output_file)
results_eva_df.to_csv(eva_output_file)

print(f"Saved calibration results to: {cal_output_file}")
print(f"Saved evaluation results to: {eva_output_file}")
print(f"Processing complete! Processed {len(results_cal)} files.")

###RUN USING####
# python3 /data0/funato/hsp2/99_work/00_analysis/data/HBV_benchmark.py
