# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, Erel, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

#########
all_mode = False # include all models
fs = 22 ## 22 or over for presentation, 13 for paper
alpha = 0.8

loc, ver = "JP", 2
#loc, ver = "US", 2
#loc, ver = "AUS", 2
#loc, ver = "GB", 2

reservoir_size = 200
#reservoir_size = 700

non_arid_files = False


benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

def load_data(loc, ce, model, benchmark):
    file_tag = f'r{reservoir_size}_s1_sr0.4_rr{ridge_param}'
    if model != 'BC':
        file_path = f'/data0/funato/0_out/99_out/{loc}/{model}_{reservoir_size}_{ridge_param}/results/{model}_results_{file_tag}_{ce}.csv'
        column = f'{model}_{file_tag}_{benchmark}_{ce}'
    else:
        file_path = f'/data0/funato/0_out/99_out/{loc}/{model}_{reservoir_size}_{ridge_param}/results/{model}_results_{file_tag}_bma_{ce}.csv'
        column = f'{model}_{file_tag}_bma_{benchmark}_{ce}'
    df = pd.read_csv(file_path)
    data = df[column]
    return data

def create_scatter_plots(benchmark, characteristic, model, data_dict, attribute_values):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    fig.suptitle(f'{benchmark} - {model}', fontsize=fs)

    # Calibration
    ax = axes[0]
    for file_num, data in data_dict['cal'].items():
        x_values = attribute_values.loc[attribute_values['File_num'] == file_num, characteristic]
        if len(x_values) == len(data):
            ax.scatter(x_values, data, label=f'{file_num} Cal', alpha=alpha)
    ax.set_xlabel(model, fontsize=fs)
    ax.set_ylabel(benchmark, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    ax.set_title('Calibration', fontsize=fs)

    # Evaluation
    ax = axes[1]
    for file_num, data in data_dict['eva'].items():
        x_values = attribute_values.loc[attribute_values['File_num'] == file_num, characteristic]
        if len(x_values) == len(data):
            ax.scatter(x_values, data, label=f'{file_num} Eva', alpha=alpha)
    ax.set_xlabel(model, fontsize=fs)
    ax.set_ylabel(benchmark, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    ax.set_title('Evaluation', fontsize=fs)

    plt.tight_layout()
    os.makedirs(f'{output_base_dir}/{model}', exist_ok=True)
    plt.savefig(f'{output_base_dir}/{model}/{characteristic}_{benchmark}.png')
    plt.close()

def create_combined_figure(benchmark, data_dict, attribute_values,model):
    num_characteristics = len(characteristic_list)
    num_cols = 5
    num_rows = (num_characteristics + num_cols - 1) // num_cols  # Calculate the number of rows needed
    
    # Calibration figure
    fig_cal, axes_cal = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(25, 5 * num_rows))
    fig_cal.suptitle(f'{benchmark} - All Characteristics (Calibration)', fontsize=fs)

    output_dir_combined = f'{output_base_dir}/0_combined/{model}'
    os.makedirs(output_dir_combined, exist_ok=True)

    for i, (characteristic, char_name) in enumerate(zip(characteristic_list, char_list_name)):
        row = i // num_cols
        col = i % num_cols

        ax = axes_cal[row, col]
        for file_num, data in data_dict['cal'].items():
            x_values = attribute_values.loc[attribute_values['File_num'] == file_num, characteristic]
            if len(x_values) == len(data[i]):
                ax.scatter(x_values, data[i], label=f'{file_num} Cal', alpha=alpha)
        ax.set_xlabel(char_name, fontsize=fs)
        ax.set_ylabel(benchmark, fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.legend()
        ax.grid(True)
        ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
        ax.set_title(f'{char_name}', fontsize=fs)

    plt.tight_layout()
    plt.savefig(f'{output_dir_combined}/{benchmark}_all_characteristics_cal.png')
    plt.close()

    # Evaluation figure
    fig_eva, axes_eva = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(25, 5 * num_rows))
    fig_eva.suptitle(f'{benchmark} - All Characteristics (Evaluation)', fontsize=fs)

    for i, (characteristic, char_name) in enumerate(zip(characteristic_list, char_list_name)):
        row = i // num_cols
        col = i % num_cols

        ax = axes_eva[row, col]
        for file_num, data in data_dict['eva'].items():
            x_values = attribute_values.loc[attribute_values['File_num'] == file_num, characteristic]
            if len(x_values) == len(data[i]):
                ax.scatter(x_values, data[i], label=f'{file_num} Eva', alpha=alpha)
        ax.set_xlabel(char_name, fontsize=fs)
        ax.set_ylabel(benchmark, fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.legend()
        ax.grid(True)
        ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
        ax.set_title(f'{char_name}', fontsize=fs)

    plt.tight_layout()
    plt.savefig(f'{output_dir_combined}/{benchmark}_all_characteristics_eva.png')
    plt.close()

# Load attribute values for each location
attribute_values = {
    "US": pd.read_csv(f'/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/camels_attribute_combined.csv', encoding='UTF-8'),
    "JP": pd.read_csv(f'/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/basin_data_JP_cleaned_limited.csv', encoding='UTF-8'),
    "GB": pd.read_csv(f'/data0/funato/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv', encoding='UTF-8'),
    "AUS": pd.read_csv(f'/data0/funato/3_gis_data/AUS/0_data/river_basin/dataset_AUS/camelsaus_attribute_combined.csv', encoding='UTF-8')
}

for model in ['RC', 'RCHBMA', 'BC']:
    for benchmark in benchmark_list:
        data_dict = {'cal': {}, 'eva': {}}
        
        for loc in ["US", "JP", "GB", "AUS"]:

            if loc == "JP" or loc == "AUS":
                ridge_param = 0.001
            else:
                ridge_param = 1.0

            ver_name = "ver1_1" if ver == 1 else "ver2_0"

            #nocal_tag = "_nocal"
            nocal_tag = ""

            output_base_dir = f'/data0/funato/0_out/0_fig/scatter/benchmark_to_characteristics_all_regions/'
            os.makedirs(output_base_dir, exist_ok=True)

            if loc != "US":
                non_arid_files = False

            attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

            char_list_name = ['Basin Area', 'Latitude', 'Longitude', 'Mean Elevation', 'Max Elevation', 'Min Elevation', 'Mean Slope', 'Aridity Index', 'Annual Mean Precipitation', 'Daily Precipitation', 'Annual Mean Temperature', 'Annual Max Temperature', 'Annual Min Temperature', 'Daily Mean PET', 'Annual Mean PET']

            if loc == "JP" and ver == 1:
                file_tot_num  = 135
                start_val_list = [68]
                #step_val_list = list(range(2,69))
                step_val_list = [2,4,8,15,20,25,30,40,50,68]
                attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_JP_cleaned_limited.csv', encoding= 'UTF-8') # File_num: 1~135
                characteristic_list = ['area','lat_org','long_org','lat_pp','long_pp','ElevMean','ElevMax','ElevMin','SlopeMean','AridityIndex','met_An_precip','met_Daily_precip','met_An_meanT','met_An_maxT','met_An_minT','DailyMeanPET','AnnualMeanPET']
                columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
            elif loc == "JP" and ver == 2:
                file_tot_num = 87
                #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
                test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
                train_basin_int_list = [30,20,15,10,5,3,2]
                train_basin_int_list = [30]
                #train_basin_int_list = [2]
                basin_data_df = pd.read_csv("/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
                attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_JP_cleaned_limited.csv', encoding= 'UTF-8') # File_num: 1~135
                characteristic_list = ['area','lat_org','long_org','ElevMean','ElevMax','ElevMin','SlopeMean','AridityIndex','met_An_precip','met_Daily_precip','met_An_meanT','met_An_maxT','met_An_minT','DailyMeanPET','AnnualMeanPET'] #'dist_km',
                columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
            elif loc == "US" and ver == 1:
                file_tot_num = 671
                attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
                characteristic_list = ['area','gauge_lat','gauge_lon','ele_mt_sav','ele_mt_smx','ele_mt_smn','slp_dg_sav','aridity','pre_mm_syr','p_mean','tmp_dc_syr','tmp_dc_smx','tmp_dc_smn','pet_mean','pet_mm_syr']
                columns_drop = ['File_num','gauge_id','gauge_name','country']
            elif loc == "US" and ver == 2:
                file_tot_num = 667
                if non_arid_files:
                    arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
                    arid_file_list = pd.read_csv(arid_file_list)
                    arid_file_list = arid_file_list['File_num'].tolist()
                    arid_file_list.sort()
                    non_arid_file_list = [i for i in range(1,file_tot_num+1) if i not in arid_file_list]
                    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
                    test_basins_list = [basin for basin in test_basins_list if basin in non_arid_file_list]
                    train_basin_int_list = [200,150,100,50,25,10,5,3]
                    train_basin_int_list = [3]
                else:
                    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
                    train_basin_int_list = [300,200,150,100,50,25,10,5,3]
                    #train_basin_int_list = [200]
                characteristic_list = ['area','gauge_lat','gauge_lon','ele_mt_sav','ele_mt_smx','ele_mt_smn','slp_dg_sav','aridity','pre_mm_syr','p_mean','tmp_dc_syr','tmp_dc_smx','tmp_dc_smn','pet_mean','pet_mm_syr']
                basin_data_df = pd.read_csv("/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/pub_region_list_ver2_0.csv")
                attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
                columns_drop = ['File_num','gauge_id','gauge_name','country']
            elif loc == "AUS" and ver == 2:
                file_tot_num = 84
                test_basins_list = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82] # 17 basins
                train_basin_int_list = [30,20,15,10,5,3,2]
                #train_basin_int_list = [30,20,15] #,10,5,3,2
                train_basin_int_list = [10,5,3,2]

                basin_data_df = pd.read_csv("/data0/funato/3_gis_data/AUS/0_data/river_basin/dataset_AUS/file_num_valid_small.csv")
                attribute_values = pd.read_csv(f'{attribute_dir}/camelsaus_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
                characteristic_list = ['area','gauge_lat','gauge_lon','ele_mt_sav','ele_mt_smx','ele_mt_smn','slp_dg_sav','aridity','pre_mm_syr','p_mean','tmp_dc_syr','tmp_dc_smx','tmp_dc_smn','pet_mean','pet_mm_syr']
                columns_drop = ['File_num','gauge_id','gauge_name','country']
            elif loc == "GB" and ver == 2:
                file_tot_num = 396
                test_basins_list = list(range(4, file_tot_num, 10))
                train_basin_int_list = [150,100,50,25,10,5,3]
                train_basin_int_list = [150,100,50]
                #train_basin_int_list = [25,10,5,3]
                basin_data_df = pd.read_csv(f"/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/file_num_valid_small.csv")
                attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
                characteristic_list = ['area','gauge_lat','gauge_lon','ele_mt_sav','ele_mt_smx','ele_mt_smn','slp_dg_sav','aridity','pre_mm_syr','p_mean','tmp_dc_syr','tmp_dc_smx','tmp_dc_smn','pet_mean','pet_mm_syr']
                columns_drop = ['File_num','gauge_id','gauge_name','country']

            #########
            file_tag = f'r{reservoir_size}_s1_sr0.4_rr{ridge_param}'

            if loc == "JP":
                benchmark_limits = {
                    'KGE': {'min': 0, 'max': 1},
                    'NSE': {'min': -1, 'max': 1},
                    'logNSE': {'min': -150, 'max': 1},
                    'E1': {'min': -1, 'max': 1},
                    'VE': {'min': -2, 'max': 1},
                    'd': {'min': -2, 'max': 1},
                    'RMSE': {'min': 0, 'max': 10},
                    'MAE': {'min': 0, 'max': 10}
                }
            else:
                benchmark_limits = {
                    'KGE': {'min': -1, 'max': 1},
                    'NSE': {'min': -1, 'max': 1},
                    'E1': {'min': -1, 'max': 1},
                    'VE': {'min': -2, 'max': 1},
                    'd': {'min': -2, 'max': 1},
                    'RMSE': {'min': 0, 'max': 10},
                    'MAE': {'min': 0, 'max': 10}
                }

            data_dict['cal'][loc] = []
            data_dict['eva'][loc] = []
            
            for characteristic in characteristic_list:
                if model == 'RC':
                    RC_data_cal = load_data(loc, 'cal', 'RC', benchmark)
                    RC_data_eva = load_data(loc, 'eva', 'RC', benchmark)
                    data_dict['cal'][loc].append(RC_data_cal)
                    data_dict['eva'][loc].append(RC_data_eva)
                elif model == 'RCHBMA':
                    RCH_data_cal = load_data(loc, 'cal', 'RCHBMA', benchmark)
                    RCH_data_eva = load_data(loc, 'eva', 'RCHBMA', benchmark)
                    data_dict['cal'][loc].append(RCH_data_cal)
                    data_dict['eva'][loc].append(RCH_data_eva)
                elif model == 'BC':
                    BC_data_cal = load_data(loc, 'cal', 'BC', benchmark)
                    BC_data_eva = load_data(loc, 'eva', 'BC', benchmark)
                    data_dict['cal'][loc].append(BC_data_cal)
                    data_dict['eva'][loc].append(BC_data_eva)
            
            create_scatter_plots(benchmark, characteristic, model, data_dict, attribute_values)
        
        create_combined_figure(benchmark, data_dict, attribute_values, model)

print("Done")