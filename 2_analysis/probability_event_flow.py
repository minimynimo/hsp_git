# get the discharge data for a event flow probability 
# once in 50 years, once in 100 years, once in 300 years, once in 500 years, and once in 1000 years
# might not use becuase some observed are over the return period flows.

import pandas as pd
import numpy as np
import scipy.io
import os
import glob
import matplotlib.pyplot as plt


gauge_id_file = "/data0/funato/hsp/camels_GB/river_basin/gb_file_num.csv"
gauge_id_df = pd.read_csv(gauge_id_file)

output_dir = f"/data0/funato/hsp/0_out/2_analysis/probability_event"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/fig", exist_ok=True)

file_tot_num = 424

input_dir = "/data0/funato/hsp/camels_GB/data/timeseries"

exceed_prob_50 = 1.0 / 50.0
exceed_prob_100 = 1.0 / 100.0
exceed_prob_300 = 1.0 / 300.0
exceed_prob_500 = 1.0 / 500.0
exceed_prob_1000 = 1.0 / 1000.0

precip_obs_max =[]
precip_exceed_50 = []
precip_exceed_100 = []
precip_exceed_300 = []
precip_exceed_500 = []
precip_exceed_1000 = []

for file in range(1, file_tot_num + 1):
    gauge_id = gauge_id_df.loc[gauge_id_df['File_num'] == file, 'gauge_id'].values[0]

    input_file = glob.glob(f"{input_dir}/CAMELS_GB_hydromet_timeseries_{gauge_id}_*.csv")
    input_df = pd.read_csv(input_file[0])
    input_df['date'] = pd.to_datetime(input_df['date'])

    precip_data = input_df['precipitation'].values

    #sort the precipitation data 
    sorted_precip = np.sort(precip_data)[::-1]
    #calculate the exceedance probability
    exceed_prob = np.arange(1, len(sorted_precip) + 1) / len(sorted_precip) 

    if file>= 1 and file <= 10:
        #plot the probability of exceedance of the precipitation data. 
        plt.figure(figsize=(10, 10))
        plt.plot(sorted_precip, exceed_prob, label='Exceedance Probability')
        #plot as dots the individual precipitation values as dots
        plt.scatter(sorted_precip, exceed_prob, color='red', s=10, label='Individual Precipitation Values')
        #plot the threshold lines for 50, 100, and 300
        plt.axhline(y=exceed_prob_50, color='orange', linestyle='--', label='50-year Threshold')
        plt.axhline(y=exceed_prob_100, color='green', linestyle='--', label='100-year Threshold')
        plt.axhline(y=exceed_prob_300, color='blue', linestyle='--', label='300-year Threshold')
        plt.axhline(y=exceed_prob_500, color='purple', linestyle='--', label='500-year Threshold')
        plt.axhline(y=exceed_prob_1000, color='brown', linestyle='--', label='1000-year Threshold')
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Exceedance Probability')
        plt.title('Exceedance Probability of Precipitation')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_fig/exceedance_probability_precipitation_{file}.png")
        plt.close()

        #plot pdf of the precipitation data
        plt.figure(figsize=(10, 10))
        plt.hist(precip_data, bins=50, density=True, alpha=0.5, color='blue', label='Precipitation PDF')
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Probability Density')
        plt.title('Probability Density Function of Precipitation')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_fig/probability_density_function_precipitation_{file}.png")
        plt.close()

    # get the maximum precip from dates 1993 to 2000
    start_date_cal = '2001-01-01'
    end_date_cal = '2006-12-31'
    cal_data = input_df[(input_df['date'] >= start_date_cal) & (input_df['date'] <= end_date_cal)].copy()   
    cal_precip = cal_data['precipitation'].values
    precip_obs_max.append(round(np.max(cal_precip), 2))

    #find the precipitation data that exceeds the threshold
    exceed_50 = sorted_precip[exceed_prob <= exceed_prob_50]
    exceed_100 = sorted_precip[exceed_prob <= exceed_prob_100]
    exceed_300 = sorted_precip[exceed_prob <= exceed_prob_300]
    exceed_500 = sorted_precip[exceed_prob <= exceed_prob_500]
    exceed_1000 = sorted_precip[exceed_prob <= exceed_prob_1000]

    # Calculate the exceedance probabilities based on the regression line
    if len(exceed_50) > 0:
        precip_exceed_50.append(round(np.mean(exceed_50), 2))
    else:
        precip_exceed_50.append(np.NaN)
    if len(exceed_100) > 0:
        precip_exceed_100.append(round(np.mean(exceed_100), 2))
    else:
        precip_exceed_100.append(np.NaN)
    if len(exceed_300) > 0:
        precip_exceed_300.append(round(np.mean(exceed_300), 2))
    else:
        precip_exceed_300.append(np.NaN)
    if len(exceed_500) > 0:
        precip_exceed_500.append(round(np.mean(exceed_500), 2))
    else:
        precip_exceed_500.append(np.NaN)
    if len(exceed_1000) > 0:
        precip_exceed_1000.append(round(np.mean(exceed_1000), 2))
    else:
        precip_exceed_1000.append(np.NaN)


# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'file_num': range(1, file_tot_num + 1),
    'precip_obs_max': precip_obs_max,
    'precip_exceed_50': precip_exceed_50,
    'precip_exceed_100': precip_exceed_100,
    'precip_exceed_300': precip_exceed_300,
    'precip_exceed_500': precip_exceed_500,
    'precip_exceed_1000': precip_exceed_1000
})

# Save the results to a CSV file
results_df.to_csv(f"{output_dir}/precipitation_exceedance_probabilities.csv", index=False)

print("Exceedance probabilities for precipitation have been calculated and saved to CSV.")