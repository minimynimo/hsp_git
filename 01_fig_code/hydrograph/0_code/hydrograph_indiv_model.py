# hydrograph of observed and simulated flow
import matplotlib
matplotlib.use('Agg')
import struct, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.ma import masked_array
import os


####
#start_plot_date = '2001-01-01'
start_plot_date = '2002-01-01'
end_plot_date = '2005-12-31'
####

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

#########
nocal = True
plot_input_data = True  # Add this variable to control input data plotting

cal_list = ["eva"] #["cal", "eva"]
fs = 15

#loc,ver = "JP",2
loc,ver = "US",2
#loc,ver = "AUS",2

reservoir_size = 200
ridge_param = 1.0#0.001
pc = 1

main_lw = 1.0
#########

file_tag = f'r{reservoir_size}_s1_sr0.4_rr{ridge_param}'

if nocal:
    nocal_tag = "_nocal"
else:
    nocal_tag = ""
    
#calc_mode_list = ["BMA", "RC", "RCHBMA", "BC", "BCPCA", "BC_PUB", "BC_PCA_lasso_PUB"] 
calc_mode_list = ["BC_arid", "BCPCA_arid","BCPCA", "BMA", "RC",  "BC"] #"RCHBMA",

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP":
    file_tot_num = 87
    start_val = 36
    step_val = 2
elif loc == "US":
    file_tot_num = 667
    start_val = 301
    step_val = 2
elif loc == "AUS":
    file_tot_num = 84

varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

subdir_name = "hydrograph"

output_base_dir = os.path.join(f'/data0/funato/0_out/0_fig/{loc}', subdir_name)
os.makedirs(output_base_dir, exist_ok=True)

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
    full_date_range = pd.date_range(start=start_plot_date, end=end_plot_date)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    df = df[(df.index >= start_plot_date) & (df.index <= end_plot_date)]  # Filter the date range
    return df

def load_BMA_data(file_num, loc):
    BMA_PUB_column_data = []
    for ce in cal_list:
        BMA_data = f'/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_{ce}.csv'
        BMA_df = pd.read_csv(BMA_data, index_col=0).T  # Transpose to have dates as index
        BMA_df.index = pd.to_datetime(BMA_df.index)
        BMA_df = BMA_df[(BMA_df.index >= start_plot_date) & (BMA_df.index <= end_plot_date)]  # Filter the date range
        BMA_PUB_column_data.append(np.array(BMA_df[f'file_{file_num}_{ce}']).tolist())
    return BMA_PUB_column_data

def load_RC_data(file_num):
    RC_row_data = []
    for ce in cal_list:   
        RC_data = f'/data0/funato/0_out/99_out/{loc}/RC_{reservoir_size}_{ridge_param}/predict/RC_predict_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{ce}.csv'
        RC_df = pd.read_csv(RC_data, index_col=0).T
        RC_df.index = pd.to_datetime(RC_df.index)
        RC_df = RC_df[(RC_df.index >= start_plot_date) & (RC_df.index <= end_plot_date)]  # Filter the date range
        RC_row_data.append(np.array(RC_df[f'file_{file_num}_{ce}'].tolist()))
    return RC_row_data

def load_RCH_data(file_num):
    RCH_row_data = []
    for ce in cal_list:
        RCH_data = f'/data0/funato/0_out/99_out/{loc}/RCHBMA_{reservoir_size}_{ridge_param}/predict/RCHBMA_predict_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{ce}.csv'
        RCH_df = pd.read_csv(RCH_data, index_col=0).T  
        RCH_df.index = pd.to_datetime(RCH_df.index)
        RCH_df = RCH_df[(RCH_df.index >= start_plot_date) & (RCH_df.index <= end_plot_date)]  # Filter the date range
        RCH_row_data.append(np.array(RCH_df[f'file_{file_num}_{ce}'].tolist()))
    return RCH_row_data

def load_BC_data(file_num):
    BC_row_data = []
    for ce in cal_list:
        BC_data = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}/predict/BC_predict_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_bma_{ce}.csv'
        BC_df = pd.read_csv(BC_data, index_col=0).T
        BC_df.index = pd.to_datetime(BC_df.index)
        BC_df = BC_df[(BC_df.index >= start_plot_date) & (BC_df.index <= end_plot_date)]  # Filter the date range
        BC_row_data.append(np.array(BC_df[f'file_{file_num}_{ce}'].tolist()))
    return BC_row_data

def load_BCPCA_data(file_num):
    BCPCA_row_data = []
    for ce in cal_list:
        BCPCA_data = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}_{ridge_param}/predict/BC-PCA-lasso_predict_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_{ce}.csv'
        BCPCA_df = pd.read_csv(BCPCA_data, index_col=0).T
        BCPCA_df.index = pd.to_datetime(BCPCA_df.index)
        BCPCA_df = BCPCA_df[(BCPCA_df.index >= start_plot_date) & (BCPCA_df.index <= end_plot_date)]  # Filter the date range
        if f'file_{file_num}_{ce}' in BCPCA_df.columns:
            BCPCA_row_data.append(np.array(BCPCA_df[f'file_{file_num}_{ce}'].tolist()))
    return BCPCA_row_data
    
def load_BC_PUB_data(file_num, train_test, start_val, step_val, pc):
    BC_PUB_row_data =[]
    for ce in cal_list:
        BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_{reservoir_size}_{ridge_param}/start{start_val}/step{step_val}/{train_test}_basin/predict/BC_PUB_predict_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_start{start_val}_step{step_val}_rev_PC{pc}_{ce}.csv'
        BC_PUB_df = pd.read_csv(BC_PUB_data, index_col=0).T
        BC_PUB_df.index = pd.to_datetime(BC_PUB_df.index)
        BC_PUB_df = BC_PUB_df[(BC_PUB_df.index >= start_plot_date) & (BC_PUB_df.index <= end_plot_date)]  # Filter the date range
        BC_PUB_row_data.append(np.array(BC_PUB_df[f'file_{file_num}_{ce}'].tolist()))
    return BC_PUB_row_data

def load_BC_PCA_lasso_PUB_data(file_num, train_test, start_val, step_val, pc):
    BC_PCA_lasso_row_data = []
    for ce in cal_list:
        BC_PCA_lasso_result_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/start{start_val}/step{step_val}/{train_test}_basin/predict/BC-PCA-lasso_PUB_predict_{file_tag}_start{start_val}_step{step_val}_rev_PC{pc}_{ce}.csv', index_col=0).T
        BC_PCA_lasso_result_df.index = pd.to_datetime(BC_PCA_lasso_result_df.index)
        BC_PCA_lasso_result_df = BC_PCA_lasso_result_df[(BC_PCA_lasso_result_df.index >= start_plot_date) & (BC_PCA_lasso_result_df.index <= end_plot_date)]  # Filter the date range
        BC_PCA_lasso_row_data.append(np.array(BC_PCA_lasso_result_df[f'file_{file_num}_{ce}'].tolist()))
    return BC_PCA_lasso_row_data

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


for basin_num in range(133, file_tot_num + 1):
    df = load_data(basin_num)
    df_plot = df[start_plot_date:end_plot_date].copy()
    target = df_plot['Obs flow'].values 

    for calc_mode in calc_mode_list:
        fig, ax1 = plt.subplots(figsize=(13, 4))
        if plot_input_data:
            output_dir = f'{output_base_dir}/{calc_mode}_in'
        else:
            output_dir = f'{output_base_dir}/{calc_mode}'
        os.makedirs(output_dir, exist_ok=True)
        
        for model in range(len(model_list)):
            model_num = model_list[model]
            data_s = df_plot[f'{model_num}'].values
            data_s = data_s[~np.isnan(data_s)].flatten()
            ax1.plot(df_plot.index, data_s,  linewidth = 0.5, color = 'silver') 
        
        ax1.plot([], [], label=f"MARRMoT (n=43)", linewidth=0.4, color='gray') 
        ax1.plot(df_plot.index, target, label="Observed Runoff", linewidth=main_lw*1.1, color='black')

        if calc_mode == "BMA":
            predict_row = load_BMA_data(basin_num, loc)
        elif calc_mode == "RC":
            predict_row = load_RC_data(basin_num)
        elif calc_mode == "RCHBMA":
            predict_row = load_RCH_data(basin_num)
        elif calc_mode == "BC":
            predict_row = load_BC_data(basin_num)
        elif calc_mode == "BCPCA":
            predict_row = load_BCPCA_data(basin_num)
        elif calc_mode == "BC_PUB":
            predict_row = load_BC_PUB_data(basin_num, "eva", 1, 1, pc)
        elif calc_mode == "BC_PCA_lasso_PUB":
            predict_row = load_BC_PCA_lasso_PUB_data(basin_num, "eva", 1, 1, pc)

        if len(predict_row) > 1:
            ax1.plot(df_plot.index, predict_row[1], label=f"{calc_mode}", linewidth=main_lw, color='crimson')

            nse = BMK(target, predict_row[1], "NSE")
            kge = BMK(target, predict_row[1], "KGE")
            ax1.text(0.03, 0.95, f'NSE: {nse:.3f}\nKGE: {kge:.3f}', transform=ax1.transAxes, fontsize=fs, verticalalignment='top')

        ax1.legend(fontsize=fs, loc='upper right', framealpha=1)
        ax1.set_xlabel('Day', fontsize=fs)
        ax1.set_ylabel('Flow[mm/day]', fontsize=fs)
        ax1.tick_params(axis='x', labelsize=fs)
        ax1.tick_params(axis='y', labelsize=fs)
        ax1.set_ylim(0, max(target) * 1.1)
        ax1.set_title(f'Basin {basin_num}\n', fontsize=fs)

        if start_plot_date < '2001-01-01':
            ax1.axvline(pd.to_datetime(start_date_eva), color='dimgray', linestyle='-', linewidth=2)

        if plot_input_data:
            ax2 = ax1.twinx()
            ax2.plot(df_plot.index, df_plot['Temp'], label='Temperature', color='blue', alpha=0.6)
            ax2.plot(df_plot.index, df_plot['Precip'], label='Precipitation', color='green', alpha=0.6)
            ax2.plot(df_plot.index, df_plot['PET'], label='PET', color='orange', alpha=0.6)
            ax2.set_ylabel('Input Data', fontsize=fs)
            ax2.legend(fontsize=fs, loc='upper left', framealpha=1)

        fig.tight_layout()
        plt.savefig(f'{output_dir}/{calc_mode}_basin{basin_num}_hg_nocal.jpg')
        plt.close()

print("Done")
