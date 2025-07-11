import numpy as np
import pandas as pd
import os

input_dir = "/data0/funato/hsp/2_MERV/varssim_HBV"

file_tot_num = 424

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

output_dir = "/data0/funato/hsp/0_out/10_HBV/results/"
os.makedirs(output_dir, exist_ok=True)

sim_cols = ["og"]

def BMK(obs_data,sim_data,benchmark):
    """
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)
    
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]
    """
    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
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
        return 1 - numer / denom
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "Erel":
        obs_ave = np.mean(obs_data)
        temp_obs_data = np.where(obs_data == 0, 1e-6, obs_data)
        numer = np.sum(np.square((temp_obs_data - sim_data) / temp_obs_data))
        denom = np.sum(np.square((obs_data - obs_ave) / obs_ave))
        return 1 - numer / denom
    
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


# compare Obs flow to no_precip,p_pet,up_peak1_rate10,up_peak1_rate30,up_peak1_rate50,up_peak3_rate10,up_peak3_rate30,up_peak3_rate50,up_peak5_rate10,up_peak5_rate30,up_peak5_rate50

for sim_col in sim_cols:
    results = []
    for file_num in range(1, file_tot_num + 1):
        input_file = f"{input_dir}/varssim{file_num:03d}.csv"
        input_df = pd.read_csv(input_file)
        obs = input_df["Obs flow"].values
        sim = input_df[sim_col].values
        row = []
        for benchmark in benchmark_list:
            try:
                val = BMK(obs, sim, benchmark)
            except Exception as e:
                val = np.nan
            row.append(val)
        results.append(row)
    # Save results for this sim_col
    results_df = pd.DataFrame(results, columns=benchmark_list, index=range(1, file_tot_num + 1))
    results_df.index.name = "file_num"
    results_df.to_csv(f"{output_dir}/HBV_results_{sim_col}.csv")
    print(f"Saved results for {sim_col} to {output_dir}/HBV_results_{sim_col}.csv")