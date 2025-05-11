# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
import matplotlib
matplotlib.use('Agg')
from esn_BC import ESN
import os
import pandas as pd
import numpy as np
from datetime import datetime

#####################
benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

spin = 1

nexttime = True
#arid_files = True
arid_files = False

fix = ""
#fix = "_nofix" ##
#fix = "_zerofix"##

buf = f"{fix}"

input_size = 3
output_size = 1
reservoir_size = 700
#reservoir_size = 200
spectral_radius = 0.4
washout = 0
#ridge_param = 1.0 
ridge_param = 0.001

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc, ver = "US", 1
#loc, ver = "US", 2
#loc, ver = "AUS", 2
#loc, ver = "GB", 2

if loc != "US":
    arid_files = False

file_tag_prev = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
#####################
print(file_tag_prev)

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"
elif loc == "US" and ver == 1:
    file_tot_num = 669
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver1_1"
elif loc == "US" and ver == 2:
    file_tot_num = 667
    arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/ver2_0"
elif loc == "AUS" and ver == 2:
    file_tot_num = 84
elif loc == "GB" and ver == 2:
    file_tot_num = 396
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

if arid_files:
    arid_file_list = pd.read_csv(arid_file_list)
    file_list = arid_file_list['File_num'].tolist()
    file_list.sort()
else:
    file_list = list(range(1, file_tot_num+1))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

## only when testing BMA
model_list = ["bma"]
#model_list = ["m44", "m46"]

if arid_files:
    output_dir = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}_arid'
else:
    output_dir = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}_testtime'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/Win_A', exist_ok=True)
os.makedirs(output_dir + '/Wout', exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/reservoir', exist_ok=True)

param_file_path = os.path.join(output_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write(f"spin = {spin}\n")
    param_file.write("\n===PCA===\n")

# when testing individual models 
"""
model_list = [ "bma","m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]
"""

# Helper function to format file names
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
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
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

model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

W_in_vals = model.W_in_val()
W_in_df = pd.DataFrame(W_in_vals)
W_in_df.to_csv(output_dir + f"/Win_A/BC_Win{file_tag_prev}.csv", mode='a', index=False, header=False)

A_vals = model.A_val()
A_val_df = pd.DataFrame(A_vals)
A_val_df.to_csv(output_dir + f"/Win_A/BC_A_val{file_tag_prev}.csv", mode = 'a', index=False, header=False)

for model_name in model_list:
    file_tag = file_tag_prev + f"_{model_name}"

    if os.path.exists(output_dir + f'/BC{file_tag}_log.txt'):
        open(output_dir + f'/BC{file_tag}_log.txt', 'w').close()
    log_file = open(output_dir + f'/BC{file_tag}_log.txt', 'a')

    results_cal = []
    results_eva = []
    for file_num in file_list:
        print(file_num)
        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"FILE_{file_num}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        ### INPUT DATA ###
        df = load_data(file_num)

        df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva = df[start_date_eva:end_date_eva].copy()

        # loading the sim flow data for the BMA and 44 MARRMoT models
        if model_name == "bma":
            # Load BMA predictions
            bma_df_cal = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_cal.csv', index_col=0)
            bma_df_eva = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_eva.csv', index_col=0)
            
            df_cal['Sim flow'] = bma_df_cal.loc[f'file_{file_num}_cal'].values
            df_eva['Sim flow'] = bma_df_eva.loc[f'file_{file_num}_eva'].values

        else:
            df_cal["Sim flow"] = df_cal[model_name].values
            df_eva["Sim flow"] = df_eva[model_name].values


        # Calculate Flow Error
        df_cal['Flow Error'] = df_cal['Obs flow'] - df_cal['Sim flow']
        df_eva['Flow Error'] = df_eva['Obs flow'] - df_eva['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_cal = np.hstack([
            df_cal['Precip'].values.reshape(-1, 1),
            df_cal['Temp'].values.reshape(-1, 1),
            df_cal['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
        input_eva = np.hstack([
            df_eva['Precip'].values.reshape(-1, 1),
            df_eva['Temp'].values.reshape(-1, 1),
            df_eva['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        #target_cal = np.concatenate((input_cal[3][1:], input_eva[3][:input_cal.shape[1] - 1])).tolist()

        obs_error_cal = df_cal['Flow Error'].values.reshape(-1, 1)
        obs_error_eva = df_eva['Flow Error'].values.reshape(-1, 1)

        # target error
        error_target_cal = np.concatenate([obs_error_cal[1:], obs_error_eva[0:1]]) #2898
        error_target_eva = obs_error_eva[1:]  #1089

        # Target flow
        target_cal = np.concatenate((df_cal['Obs flow'].values[1:], df_eva['Obs flow'].values[0:1])) #2898
        target_eva = np.concatenate((df_eva['Obs flow'].values[1:],)) #1089

        input_eva = input_eva[:, :-1]

        ##########!!!!
        # CALIBRATION
        # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW


        Wout, reservoir = model.train(input_cal, target_data=error_target_cal, washout=washout, ridge_param=ridge_param, spinup = spin)


        reservoir_row = np.concatenate(([f'file_{file_num}'], reservoir.flatten()))
        reservoir_df = pd.DataFrame([reservoir_row])
        if file_num == 1:
            reservoir_df.to_csv(output_dir + f"/reservoir/BC_reservoir{file_tag}.csv", mode='w', index=False, header=False)
        else:
            reservoir_df.to_csv(output_dir + f"/reservoir/BC_reservoir{file_tag}.csv", mode='a', index=False, header=False)

        # Wout (1,R)
        # reservoir (R, )

        Wout_row = np.concatenate(([f'file_{file_num}'], Wout.flatten()))
        Wout_df = pd.DataFrame([Wout_row])
        if file_num == 1:
            Wout_df.to_csv(output_dir + f"/Wout/BC_Wout{file_tag}.csv", mode='w', index=False, header=False)
        else:
            Wout_df.to_csv(output_dir + f"/Wout/BC_Wout{file_tag}.csv", mode='a', index=False, header=False)

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")

        start_time_pred = datetime.now()
        start_time_pred_st = start_time_pred.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"pred start: {start_time_pred_st}\n")

        # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
        ### PREDICTIONS OF ERRORS ###
        error_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
        # predict = error_cal + BMAsim
        # predict のtimestep = t+1
        predict_cal = error_cal + np.concatenate([df_cal['Sim flow'].values[1:], df_eva['Sim flow'].values[0:1]])
        predict_eva = error_eva + df_eva['Sim flow'].values[1:]

        predict_cal[predict_cal < 0] = 0
        predict_eva[predict_eva < 0] = 0

        end_time_pred = datetime.now()
        end_time_pred_st = end_time_pred.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"pred end: {end_time_pred_st}\n")
        log_file.write(f"pred elapsed: {end_time_pred - start_time_pred}\n")

        #print(predict_cal)
        #print(predict_eva)
        print(BMK(target_cal.T,predict_cal,"KGE"), BMK(target_eva.T, predict_eva,"KGE"))

        file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
        file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

        if file_num == 1:
            predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal[0]))
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva[0]))

            date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

            predict_cal_df = pd.DataFrame([date_row_cal])
            predict_eva_df = pd.DataFrame([date_row_eva])

            predict_cal_df.to_csv(output_dir + f"/predict/BC_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
            predict_eva_df.to_csv(output_dir + f"/predict/BC_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/predict/BC_predict{file_tag}_cal.csv", 'a') as file:
            pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/predict/BC_predict{file_tag}_eva.csv", 'a') as file:
            pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

        file_results_cal = {'file_num': file_num}
        file_results_eva = {'file_num': file_num}

        # Loop over each benchmark and add results to the current file's dictionary
        for benchmark in benchmark_list:
            file_results_cal.update({
                f'BC{file_tag}_{benchmark}_cal': BMK(target_cal, predict_cal, benchmark)
            })
            file_results_eva.update({
                f'BC{file_tag}_{benchmark}_eva': BMK(target_eva, predict_eva, benchmark)
            })  
        
        results_cal.append(file_results_cal)
        results_eva.append(file_results_eva)


        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")

        del Wout, reservoir, predict_eva, predict_cal

    log_file.write(f"DONE\n")
    log_file.close()


    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/BC_results{file_tag}_cal.csv', index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/BC_results{file_tag}_eva.csv', index=False, header=True)

print(f"BC{file_tag} reservoir size {reservoir_size} done")
print(f"BC{file_tag}")
print("DONE")