# Using nofix files, zerofix files are created.
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from progressbar import ProgressBar

#########
file_tag = 'r700_s1_sr0.4_rr0.001' 
#out_type = 'modelbased'
#out_type = 'biascor'
out_type = 'withBMAinput'
#########

if out_type == 'modelbased':
    input_dir = '/data0/funato/0_out/bias_correction/model_based/predict'
    predict_output_dir = '/data0/funato/0_out/bias_correction/model_based/predict'
    results_output_dir = '/data0/funato/0_out/bias_correction/model_based/results'
    buf = f'_modelbased'
elif out_type == 'biascor' or out_type == 'withBMAinput':
    input_dir = '/data0/funato/0_out/bias_correction/predict'
    predict_output_dir = '/data0/funato/0_out/bias_correction/predict'
    results_output_dir = '/data0/funato/0_out/bias_correction/results'
    buf = f''


def convert_negatives_to_zero_df(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].applymap(lambda x: max(x, 0))
    return df

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"/data0/funato/2_MERV-Jp/varssim/ver1_1/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def BMK(obs_data,sim_data,benchmark):    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
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


# Define the calibration and evaluation periods
start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2003-12-31'
    
cal_eva = ['cal', 'eva']

if out_type == 'modelbased':
    #model_list = ["bma"]
    model_list = ["bma","m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m41", "m42", "m43", "m44", "m46"]
elif out_type == 'biascor' or out_type == 'withBMAinput':
    model_list = ["temp"]

benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]

fix =['nofix', 'zerofix']

i = 0
p = ProgressBar(0,len(model_list)*2*135)
for model_name in model_list:
    if out_type == 'modelbased':
        out_model_name = f'_{model_name}'
    elif out_type == 'biascor':
        out_model_name = ''
    elif out_type == 'withBMAinput':
        out_model_name = f'_withBMAinput'

    for ce in cal_eva:
        results_nofix = []
        results_zerofix = []
        original_file = os.path.join(input_dir, f'BiasCor_predict_{file_tag}_nofix{out_model_name}_{ce}.csv')

        og_df = pd.read_csv(original_file)

        nofix = og_df.copy()
        zerofix = convert_negatives_to_zero_df(og_df)

        nofix.to_csv(os.path.join(predict_output_dir, f'BiasCor_predict_{file_tag}_nofix{out_model_name}_{ce}.csv'), index=False)
        zerofix.to_csv(os.path.join(predict_output_dir, f'BiasCor_predict_{file_tag}_zerofix{out_model_name}_{ce}.csv'), index=False)

        for file_num in range(1,136):
            i += 1
            p.update(i)
            predict_nofix = nofix[f'file_{file_num}_{ce}'].values
            predict_zerofix = zerofix[f'file_{file_num}_{ce}'].values

            ### INPUT DATA ###
            df = load_data(file_num)

            df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
            df_eva = df[start_date_eva:end_date_eva].copy()

            if ce == 'cal':
                target = np.concatenate((df_cal['Obs flow'].values[1:], df_eva['Obs flow'].values[0:1])) #2898

            elif ce == 'eva':
                target = np.concatenate((df_eva['Obs flow'].values[1:],)) #1089

            file_results_nofix = {'file_num': file_num}
            file_results_zerofix = {'file_num': file_num}

            # Loop over each benchmark and add results to the current file's dictionary
            for benchmark in benchmark_list:
                file_results_nofix.update({
                    f'BiasCor_{file_tag}_nofix{out_model_name}_{benchmark}_{ce}': BMK(target, predict_nofix, benchmark)
                })
                file_results_zerofix.update({
                    f'BiasCor_{file_tag}_zerofix{out_model_name}_{benchmark}_{ce}': BMK(target, predict_zerofix, benchmark)
                })  


            # Append the completed dictionary to the results list
            results_nofix.append(file_results_nofix)
            results_zerofix.append(file_results_zerofix)


        df_results_nofix = pd.DataFrame(results_nofix)
        df_results_nofix.to_csv(os.path.join(results_output_dir, f'BiasCor_results_{file_tag}_nofix{out_model_name}_{ce}.csv'), index=False)
        df_results_zerofix = pd.DataFrame(results_zerofix)
        df_results_zerofix.to_csv(os.path.join(results_output_dir, f'BiasCor_results_{file_tag}_zerofix{out_model_name}_{ce}.csv'), index=False)
p.finish()

print("Done!")
