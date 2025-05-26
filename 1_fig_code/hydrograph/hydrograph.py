# hydrograph of observed and simulated flow
import matplotlib
matplotlib.use('Agg')
import struct, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from progressbar import ProgressBar
import os

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2003-12-31'
#########
benchmark = "KGE"
file = "RCH_BMA_predict_r700_s1_sr0.4_rr0.001_zerofix_star" #no .csv
file2 = False
#file2 = "RCH_BMA_predict_r700_s1_sr0.4_zerofix_aft"
#########

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m41", "m42", "m43", "m44", "m46"]

path = "/data0/funato/0_out/0_fig/bybasin/indiv_model"  # no / at end
#path = "/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/RC/out"  # no / at end


def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"/data0/funato/2_MERV-Jp/varssim/ver1_1/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    return df

def BMK(obs_data,sim_data):
    """
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)
    
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]
    """

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum((obs_data - sim_data) ** 2) / np.sum((obs_data - obs_ave) ** 2))
    elif benchmark == "KGE":
        # KGE = 1-{(r-1)^2 + (sigma_sim/sigma_obs -1)^2 + (myu_sim/myu_obs -1)^2 )}^ 0.5
        r = np.corrcoef(obs_data,sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)

        return 1 - ((r-1)**2 + (sim_std/obs_std -1)**2 + (sim_ave/obs_ave -1)**2 )** 0.5
    
p = ProgressBar(0, 136)

for basin_num in range(1,136):
    p.update(basin_num)
    df = load_data(basin_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    df_cal.replace(-999, np.nan, inplace=True)
    df_cal.dropna(inplace=True)
    df_eva.replace(-999, np.nan, inplace=True)
    df_eva.dropna(inplace=True)

    data_eva = df_eva.loc[start_date_eva:end_date_eva] #2922

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.vstack([
        df_cal['Precip'].values,
        df_cal['Temp'].values,
        df_cal['PET'].values,
        df_cal['Obs flow'].values
    ])
    input_eva = np.vstack([
        df_eva['Precip'].values,
        df_eva['Temp'].values,
        df_eva['PET'].values,
        df_eva['Obs flow'].values
    ])


    # Prepare target data
    target_cal = np.concatenate((input_cal[3][1:], input_eva[3][0:1])) #2898
    target_eva = np.concatenate((input_eva[3][1:],))  #1089

    data_eva = data_eva[1:]
    
    ##For simulated in RC & RC hybrid
    sim_data = pd.read_csv(f"{path}/{file}.csv")
    sim_data = sim_data[f'file_{basin_num}'].values
    sim_data = sim_data[~np.isnan(sim_data)].flatten()

    if file2 != False:
        sim_data2 = pd.read_csv(f"{path}/{file2}.csv")
        sim_data2 = sim_data2[f'file_{basin_num}'].values
        sim_data2 = sim_data2[~np.isnan(sim_data2)].flatten()

    #For BMA_ave
    """
    BMA_ave = data_eva[model_list].mean(axis=1).to_list()


    # Remove NaN values from target_eva and BMA_ave
    non_nan_indices = ~np.isnan(target_eva) & ~np.isnan(BMA_ave)
    target_eva = target_eva[non_nan_indices]
    data_eva = data_eva[non_nan_indices]
    BMA_ave = np.array(BMA_ave)[non_nan_indices]
    """
    plt.figure(figsize=(12,6))

    ## plot the 44 modeled flow
    """###
    for model in range(44):
        model_num = model_list[model]
        plt.plot(data_eva.index,data_eva[f'{model_num}'],  linewidth = 0.5,color = 'darkgray') 
    """

    ## plot the ansemble flow
    plt.plot(data_eva.index,target_eva,label= "Observed Runoff",linewidth = 1.5, color = 'crimson')
    if file2 != False:
        plt.plot(data_eva.index, sim_data2,label = "Simulated_2", linewidth = 1.5, color = 'limegreen') ###FOR RC AND RC-HYBRID
    plt.plot(data_eva.index, sim_data,label = "Simulated", linewidth = 1.5, color = 'royalblue') ###FOR RC AND RC-HYBRID

    #plt.plot(data_eva.index, BMA_ave,label = "Model Average", linewidth = 1.5, color = 'royalblue') ###FOR BMA 

    #for label only
    plt.plot( [],[] ,label = f"MARRMoT (n=44)", linewidth = 0.5,color = 'gray') 

    plt.axhline(np.average(target_eva), linestyle = '--')
    plt.axhline(np.average(target_eva)+11*np.std(target_eva), linestyle = ':')

    ### fig setting and output
    plt.legend(fontsize = 14)
    plt.xlabel('Day',fontsize = 14)
    plt.ylabel('Flow[mm/day]',fontsize = 14)
    if file2 != False:
        plt.title(f'Basin {basin_num}\nKGE: {round(BMK(target_eva,sim_data),3)} sim2KGE: {round(BMK(target_eva, sim_data2),3)}\n')
    else:
        plt.title(f'Basin {basin_num}\nKGE: {round(BMK(target_eva,sim_data),3)}\n')
    #plt.title(f'Basin {basin_num}\nKGE: {round(BMK(target_eva,BMA_ave),3)}\n')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)

    hg_path = f'/data0/funato/0_out/fig/bybasin/RCH-BMA/{file}'
    #hg_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/fig/bybasin/BMA/KGE'

    if not os.path.exists(hg_path):
        os.mkdir(hg_path)
    plt.savefig(f'{hg_path}/{file}_{basin_num}_hg.png')
    plt.close()
