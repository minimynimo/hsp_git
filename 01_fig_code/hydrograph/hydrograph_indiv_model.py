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


####
#start_plot_date = '2001-01-01'
start_plot_date = '2002-01-01'
end_plot_date = '2006-12-31'
####

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

#########
nocal = True

cal_eva = "eva"
fs = 27

reservoir_size = 200

main_lw = 2.0
#########

file_tag = f'r{reservoir_size}_s1_sr0.4_rr0.001'

if nocal:
    nocal_tag = "_nocal"
else:
    nocal_tag = ""

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

ave_path = f'/data0/funato/0_out//AVE{nocal_tag}/predict'
bma_path = f"/data0/funato/0_out/BMA{nocal_tag}/predict"
BC_path = f'/data0/funato/0_out/BC{nocal_tag}/predict'
path = "/data0/funato/0_out/0_fig/bybasin/indiv_model"  # no / at end

#path = "/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/RC/out"  # no / at end

subdir_name = "hydrograph"

hg_path = os.path.join(f'/data0/funato/0_out/0_fig/bybasin/nocal_hydro_model', subdir_name)
if not os.path.exists(hg_path):
    os.makedirs(hg_path)

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"/data0/funato/2_MERV-Jp/varssim{nocal_tag}/ver1_1/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    return df

    
p = ProgressBar(0, 136)

for basin_num in range(1,136):
    p.update(basin_num)
    df = load_data(basin_num)

    df_plot = df[start_plot_date:end_plot_date].copy()

    df_cal = df_plot[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df_plot[start_date_eva:end_date_eva].copy()

    data_cal = df_cal.loc[start_date_cal:end_date_cal] #2922
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

    target_s = df_plot['Obs flow'].values
    

    data_eva = data_eva[1:]
    
    plt.figure(figsize=(20, 5))


    if file2 != False:
        sim_data2 = pd.read_csv(f"{path}/{file2}.csv")
        sim_data2 = sim_data2[f'file_{basin_num}'].values
        sim_data2 = sim_data2[~np.isnan(sim_data2)].flatten()


    for model in range(len(model_list)):
        model_num = model_list[model]
        data_s = df_plot[f'{model_num}'].values
        data_s = data_s[~np.isnan(data_s)].flatten()
        plt.plot(df_plot.index, data_s,  linewidth = 0.5,color = 'silver') 

    #plt.plot(data_eva.index, BMA_ave,label = "Model Average", linewidth = 1.5, color = 'royalblue') ###FOR BMA 

    #for label only
    plt.plot( [],[] ,label = f"MARRMoT (n=44)", linewidth = 0.4,color = 'gray') 

    ## plot the ansemble flow
    plt.plot(df_plot.index,target_s,label= "Observed Runoff",linewidth = main_lw, color = 'black')

    if ave == True:
        bma_data_cal = pd.read_csv(f"{ave_path}/AVE_predict_cal.csv")
        bma_data_eva = pd.read_csv(f"{ave_path}/AVE_predict_eva.csv")
        bma_data_cal_s = bma_data_cal[f'file_{basin_num}'].values
        bma_data_eva_s = bma_data_eva[f'file_{basin_num}'].values
        bma_data = np.concatenate((bma_data_cal_s,bma_data_eva_s))
        bma_dates = pd.date_range(start=start_date_cal, end=end_date_eva, freq='D')  # Adjust frequency as needed
        bma_data = pd.Series(bma_data, index=bma_dates).dropna()
        bma_data = bma_data[(bma_data.index >= start_plot_date) & (bma_data.index <= end_plot_date)]
        plt.plot(df_plot.index,bma_data,label = "AVE", linewidth = main_lw, color = 'darkorange') #orange

    if bma == True:
        bma_data_cal = pd.read_csv(f"{bma_path}/BMA_predict_cal.csv")
        bma_data_eva = pd.read_csv(f"{bma_path}/BMA_predict_eva.csv")
        bma_data_cal_s = bma_data_cal[f'file_{basin_num}'].values
        bma_data_eva_s = bma_data_eva[f'file_{basin_num}'].values
        bma_data = np.concatenate((bma_data_cal_s,bma_data_eva_s))
        bma_dates = pd.date_range(start=start_date_cal, end=end_date_eva, freq='D')  # Adjust frequency as needed
        bma_data = pd.Series(bma_data, index=bma_dates).dropna()
        bma_data = bma_data[(bma_data.index >= start_plot_date) & (bma_data.index <= end_plot_date)]
        plt.plot(df_plot.index,bma_data,label = "BMA", linewidth = main_lw, color = 'royalblue')

    if BiasCor == True:
        biascor_data_cal = pd.read_csv(f"{BC_path}/BC_predict_{file_tag}_bma_cal.csv")
        biascor_data_eva = pd.read_csv(f"{BC_path}/BC_predict_{file_tag}_bma_eva.csv")
        biascor_data_cal_s = biascor_data_cal[f'file_{basin_num}_cal'].values
        biascor_data_eva_s = biascor_data_eva[f'file_{basin_num}_eva'].values
        biascor_data = np.concatenate((biascor_data_cal_s,biascor_data_eva_s))
        biascor_data = np.insert(biascor_data, 0, 0)
        biascor_dates = pd.date_range(start=start_date_cal, end=end_date_eva, freq='D')  # Adjust frequency as needed
        biascor_data = pd.Series(biascor_data, index=biascor_dates).dropna()
        biascor_data = biascor_data[(biascor_data.index >= start_plot_date) & (biascor_data.index <= end_plot_date)]
        plt.plot(df_plot.index,biascor_data,label = "BiasCor", linewidth = main_lw, color = 'crimson')




    #plt.axhline(np.average(target_eva), linestyle = '--')
    #plt.axhline(np.average(target_eva)+11*np.std(target_eva), linestyle = ':')

    ### fig setting and output
    plt.legend(fontsize = fs, loc = 'upper right')
    plt.xlabel('Day',fontsize = fs)
    plt.ylabel('Flow[mm/day]',fontsize = fs)
    #plt.title(f'Basin {basin_num}\nKGE: {round(BMK(target_eva,BMA_ave),3)}\n')
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.ylim(0,max(target_eva)*1.4)

    if start_plot_date < '2001-01-01':
        plt.axvline(pd.to_datetime(start_date_eva), color='black', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.savefig(f'{hg_path}/basin{basin_num}_hg_nocal.jpg')
    plt.close()

p.finish()
print("Done")
