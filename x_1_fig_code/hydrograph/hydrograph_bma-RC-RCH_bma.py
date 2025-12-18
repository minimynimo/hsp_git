# creates hydrograph for BMA, RC, and RCH-bma
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
from progressbar import ProgressBar
import numpy as np

##########
file_tag = 'r700_s1_sr0.4_rr0.001_zerofix'
cal_or_eva = ['cal', 'eva'][1]  # 0: cal, 1: eva

fig_start_date = '2001-01-01'
fig_end_date = '2003-12-31'

zoom_start_date = '2001-01-01'
zoom_end_date = '2002-12-31'

zoom = True # Set to True to zoom in on base flow of the hydrograph 
##########

if 'zerofix' in file_tag:
    fix = 'zerofix'
elif 'nofix' in file_tag:
    fix = 'nofix'
else:
    raise ValueError(f'Invalid fix: set zerofix or nofix in file_tag') 

if zoom:
    fig_start_date = zoom_start_date
    fig_end_date = zoom_end_date
    buf = '_zoom'
    output_dir = f'/data0/funato/0_out/0_fig/bybasin/bma-RC-RCH_bma/{fix}/zoom'
else:
    buf = '_normal'
    output_dir = f'/data0/funato/0_out/0_fig/bybasin/bma-RC-RCH_bma/{fix}/normal'

bma_file_path = f'/data0/funato/0_out/BMA/predict/BMA_predict_{cal_or_eva}.csv'
rc_file_path = f'/data0/funato/0_out/RC/predict/RC_predict_{file_tag}_{cal_or_eva}.csv'
rch_bma_file_path = f'/data0/funato/0_out/RC_hybrid_bma/predict/RCH_BMA_predict_{file_tag}_{cal_or_eva}.csv'

os.makedirs(output_dir, exist_ok=True)

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2003-12-31'

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"/data0/funato/2_MERV-Jp/varssim/ver1_1/varssim{file_name(file_num, 3)}.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    return df

bma_df = pd.read_csv(bma_file_path)
rc_df = pd.read_csv(rc_file_path)
rch_bma_df = pd.read_csv(rch_bma_file_path)

p = ProgressBar(0, 136)

# Loop through each file (column) in the CSV files
for file_num in range(1, 136):
    p.update(file_num)
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Remove the first row from df_eva and df_cal
    df_cal = df_cal.iloc[1:]
    df_eva = df_eva.iloc[1:]
    bma_data = bma_df[f'file_{file_num}'].iloc[1:]


    # Extract data for the current file
    rc_data = rc_df[f'file_{file_num}_{cal_or_eva}']
    rch_bma_data = rch_bma_df[f'file_{file_num}_{cal_or_eva}']

    if cal_or_eva == 'cal':
        plot_df = pd.DataFrame({
            'Date': df_cal.index,
            'Obs flow': df_cal['Obs flow'],
            'BMA': bma_data.values,
            'RC': rc_data.values,
            'RCH-bma': rch_bma_data.values
        })
    elif cal_or_eva == 'eva':
        plot_df = pd.DataFrame({
            'Date': df_eva.index,
            'Obs flow': df_eva['Obs flow'],
            'BMA': bma_data.values,
            'RC': rc_data.values,
            'RCH-bma': rch_bma_data.values
        })

    plot_df = plot_df[(plot_df['Date'] >= fig_start_date) & (plot_df['Date'] <= fig_end_date)]

    lw = 0.9
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df['Date'], plot_df['Obs flow'], label='Obs flow', color='black', linewidth=lw*1.4)
    plt.plot(plot_df['Date'], plot_df['BMA'], label='BMA', color='#80b1d3',linewidth=lw)
    plt.plot(plot_df['Date'], plot_df['RC'], label='RC', color='#7fc97f', linewidth=lw)
    plt.plot(plot_df['Date'], plot_df['RCH-bma'], label='RCH-bma', color='#fb8072', linewidth=lw)
    plt.xlabel('Date')
    plt.ylabel('Flow')
    if zoom:
        plt.title(f'Zoomed in Hydrograph for File {file_num}')
    else:
        plt.title(f'Hydrograph for File {file_num}')
    plt.legend()
    plt.tight_layout()

    if zoom:
        plt.ylim(top = 10)

    # Save the figure
    output_path = os.path.join(output_dir, f'{file_num}_hg{buf}.jpg')
    plt.savefig(output_path)
    plt.close()

p.finish()

print("DONE!")