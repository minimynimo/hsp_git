# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
import matplotlib
matplotlib.use('Agg')
from BC.esn_BC import ESN
from lorenz96data import Lorenz96, readBinary
import struct, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.ma import masked_array

#### should not include flow error in input
#### revised : main_BiasCor.py
#####################
#benchmark = "KGE"
#benchmark = "NSE"
benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]

spin = 1

nexttime = True

fix = ""
#fix = "_nofix" ##
#fix = "_zerofix"##

buf = f"{fix}"

input_size = 4
output_size = 1
reservoir_size = 700
spectral_radius = 0.4
washout = 0
ridge_param = 0.001 

file_tag_prev = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
#####################
print(file_tag_prev)

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2003-12-31'

## only when testing BMA
model_list = ["bma"]


# when testing individual models 
"""
model_list = [ "bma","m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m41", "m42", "m43", "m44", "m46"]
"""

# Helper function to format file names
def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"/data0/funato/2_MERV-Jp/varssim/ver1_1/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
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


for model_name in model_list:
    results_cal = []
    results_eva = []
    predict_cal_all = {}
    predict_eva_all = {}
    reservoir_rows = []
    file_tag = file_tag_prev + f"_{model_name}"
    for file_num in range(1,136):
        print(file_num)
        ### INPUT DATA ###
        df = load_data(file_num)

        df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva = df[start_date_eva:end_date_eva].copy()

        # loading the sim flow data for the BMA and 44 MARRMoT models
        if model_name == "bma":
            # Load BMA predictions
            bma_df_cal = pd.read_csv('/data0/funato/0_out/BMA/predict/BMA_predict_cal.csv')
            bma_df_eva = pd.read_csv('/data0/funato/0_out/BMA/predict/BMA_predict_eva.csv')

            df_cal['Sim flow'] = bma_df_cal[f'file_{file_num}'].values
            df_eva['Sim flow'] = bma_df_eva[f'file_{file_num}'].values

        else:
            df_cal["Sim flow"] = df_cal[model_name].values
            df_eva["Sim flow"] = df_eva[model_name].values


        # Calculate Flow Error
        df_cal['Flow Error'] = df_cal['Obs flow'] - df_cal['Sim flow']
        df_eva['Flow Error'] = df_eva['Obs flow'] - df_eva['Sim flow']

        """
        # Save df_cal and df_eva for testing
        df_cal.to_csv('/data0/funato/0_out/bias_correction/df_cal.csv')
        df_eva.to_csv('/data0/funato/0_out/bias_correction/df_eva.csv')
        """
        # Observations
        #obs_cal = masked_array(df_cal['Obs flow'].values, mask=(df_cal['Obs flow'].values == -999))
        #obs_eva = masked_array(df_eva['Obs flow'].values, mask=(df_eva['Obs flow'].values == -999))
        # Remove NaNs from observations and corresponding simulations
        #obs_cal = obs_cal[~np.isnan(obs_cal)]
        #df_cal = df_cal.dropna(subset=['Obs flow'])
        #obs_eva = obs_eva[~np.isnan(obs_eva)]
        #df_eva = df_eva.dropna(subset=['Obs flow'])

        # For each timestep, Precip,Temp,PET,Obs flow
        input_cal = np.hstack([
            df_cal['Precip'].values.reshape(-1, 1),
            df_cal['Temp'].values.reshape(-1, 1),
            df_cal['PET'].values.reshape(-1, 1),
            df_cal['Flow Error'].values.reshape(-1, 1)]).T # Observed - Simulation
        input_eva = np.hstack([
            df_eva['Precip'].values.reshape(-1, 1),
            df_eva['Temp'].values.reshape(-1, 1),
            df_eva['PET'].values.reshape(-1, 1),
            df_eva['Flow Error'].values.reshape(-1, 1)]).T # Observed - Simulation

        #target_cal = np.concatenate((input_cal[3][1:], input_eva[3][:input_cal.shape[1] - 1])).tolist()

        # target error
        error_target_cal = np.concatenate((input_cal[3][1:], input_eva[3][0:1])) #2898
        error_target_eva = np.concatenate((input_eva[3][1:],))  #1089

        # Target flow
        target_cal = np.concatenate((df_cal['Obs flow'].values[1:], df_eva['Obs flow'].values[0:1])) #2898
        target_eva = np.concatenate((df_eva['Obs flow'].values[1:],)) #1089

        input_eva = input_eva[:, :-1]

        ##########!!!!
        #data = readBinary(path, precision=4, nsteps=total_length*time_thinning_step, npoints=output_size*space_thinning_step)
        #data = data.getData(step=time_thinning_step).T[::space_thinning_step, :]

        # CALIBRATION
        # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        model = ESN(input_size=input_size,
                    output_size=output_size,
                    reservoir_size=reservoir_size,
                    adjacency_density=0.0006,
                    spectral_radius=spectral_radius,
                    input_scale=0.5)

        W_out, reservoir = model.train(input_cal, target_data=error_target_cal, washout=washout, ridge_param=ridge_param, spinup = spin)

        reservoir_row = np.concatenate((np.array([f'file_{file_num}']), reservoir.flatten()))
        reservoir_rows.append(reservoir_row)

        # W_out (1,R)
        # reservoir (R, )

        #W_out_df = pd.DataFrame(W_out)
        #W_out_df.to_csv(f"/data0/funato/0_out/RC/W_out_file_{file_num}.csv", index=False)
        
        #predict = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)


        # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
        ### PREDICTIONS OF ERRORS ###
        error_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
        # predict = error_cal + BMAsim
        # predict のtimestep = t+1
        predict_cal = error_cal + np.concatenate([df_cal['Sim flow'].values[1:], df_eva['Sim flow'].values[0:1]])
        predict_eva = error_eva + df_eva['Sim flow'].values[1:]

        print(predict_cal)
        print(predict_eva)
        print(BMK(target_cal,predict_cal,"KGE"), BMK(target_eva, predict_eva,"KGE"))

        predict_cal_all[f'file_{file_num}_cal']  = predict_cal.flatten()
        predict_eva_all[f'file_{file_num}_eva']  = predict_eva.flatten()


        file_results_cal = {'file_num': file_num}
        file_results_eva = {'file_num': file_num}

        # Loop over each benchmark and add results to the current file's dictionary
        for benchmark in benchmark_list:
            file_results_cal.update({
                f'BiasCor{file_tag}_{benchmark}_cal': BMK(target_cal, predict_cal, benchmark)
            })
            file_results_eva.update({
                f'BiasCor{file_tag}_{benchmark}_eva': BMK(target_eva, predict_eva, benchmark)
            })  


        # Append the completed dictionary to the results list
        results_cal.append(file_results_cal)
        results_eva.append(file_results_eva)

        del W_out, reservoir, predict_eva, predict_cal


    reservoir_df = pd.DataFrame(reservoir_rows)
    reservoir_df.to_csv(f"/data0/funato/0_out/bias_correction/reservoir{file_tag}.csv", index=False, header=False)


    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(f'/data0/funato/0_out/bias_correction/model_based/results/BiasCor_results{file_tag}_cal.csv', index=False)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(f'/data0/funato/0_out/bias_correction/model_based/results/BiasCor_results{file_tag}_eva.csv', index=False)

    predict_cal_df = pd.DataFrame.from_dict(predict_cal_all, orient='index').T
    predict_cal_df['Date'] = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal_df))
    predict_cal_df.insert(0, 'Date', predict_cal_df.pop('Date'))
    predict_cal_df.to_csv(f"/data0/funato/0_out/bias_correction/model_based/predict/BiasCor_predict{file_tag}_cal.csv", index=False, header = "predict_cal")

    predict_eva_df = pd.DataFrame.from_dict(predict_eva_all, orient='index').T
    predict_eva_df['Date'] = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva_df))
    predict_eva_df.insert(0, 'Date', predict_eva_df.pop('Date'))
    predict_eva_df.to_csv(f"/data0/funato/0_out/bias_correction/model_based/predict/BiasCor_predict{file_tag}_eva.csv", index=False, header = "predict_eva")

print(f"BiasCor{file_tag}")
print("DONE")