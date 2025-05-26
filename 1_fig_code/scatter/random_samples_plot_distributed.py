import random
import pandas as pd
import matplotlib.pyplot as plt
import os

#loc, ver = "JP", 1
loc, ver = "JP", 2
# loc, ver = "US", 1
#loc, ver = "US", 2
#loc, ver = "AUS", 2
#loc, ver = "GB", 2

random_samples = 100
random_seed = 42 

arid_files = True
if loc != "US":
    arid_files = False

ver_name = "ver1_1" if ver == 1 else "ver2_0" 
attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    train_basin_int_list = [30,20,10,5,4,3,2]
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    distance_matrix = pd.read_csv('/data0/funato/3_gis_data/JP/0_data/distance_matrix_v2_sorted.csv', index_col=0, header=0)
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
elif loc == "US" and ver == 2:
    file_tot_num = 667
    if arid_files:
        arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
        arid_file_list = pd.read_csv(arid_file_list)
        arid_file_list = arid_file_list['File_num'].tolist()
        arid_file_list.sort()
        non_arid_file_list = [i for i in range(1,file_tot_num+1) if i not in arid_file_list]
        print(len(non_arid_file_list))
        test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
        test_basins_list = [basin for basin in test_basins_list if basin in non_arid_file_list]
        print(len(test_basins_list))
        train_basin_int_list = [200,150,100,75,50,25,10]
    else:
        test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
        train_basin_int_list = [300,200,150,100,50,25,10,5,3]
        #train_basin_int_list = [200]
        ####
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/pub_region_list_ver2_0.csv")
    distance_matrix = pd.read_csv('/data0/funato/3_gis_data/US/0_data/distance_matrix_sorted.csv', index_col=0, header=0)
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
elif loc == "AUS" and ver == 2:
    file_tot_num = 84
    test_basins_list = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82]
    train_basin_int_list = [30,20,15,10,5,3,2]
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/AUS/0_data/river_basin/dataset_AUS/file_num_valid_small.csv")
    distance_matrix = pd.read_csv('/data0/funato/3_gis_data/AUS/0_data/distance_matrix_sorted.csv', index_col=0, header=0)
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsaus_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
elif loc == "GB" and ver == 2:
    file_tot_num = 396
    test_basins_list = list(range(4, file_tot_num, 10))
    print(len(test_basins_list))
    train_basin_int_list = [170,130,90,60,40,20,10]
    basin_data_df = pd.read_csv(f"/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/file_num_valid_small.csv")
    distance_matrix = pd.read_csv(f'/data0/funato/3_gis_data/{loc}/0_data/distance_matrix_sorted.csv', index_col=0, header=0)
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']

if arid_files:
    file_list = non_arid_file_list
else:
    file_list = list(range(1, file_tot_num+1))

def split(a_list, n):
    # devide a_list into roughly n equal parts
    k, m = divmod(len(a_list), n)
    return list(a_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

output_dir = f'/data0/funato/0_out/0_fig/randomness_distributed/{loc}'
os.makedirs(output_dir,exist_ok=True)

random.seed(random_seed)

# possible training basins to select the training basins from
PosTrainBasin = [i for i in file_list if i not in test_basins_list]
if loc == "JP":
    PosTrainBasin = sorted(PosTrainBasin, key=lambda x: basin_data_df.loc[basin_data_df['File_num'] == x, 'long_org'].values[0])[::-1]
else:
    PosTrainBasin = sorted(PosTrainBasin, key=lambda x: basin_data_df.loc[basin_data_df['File_num'] == x, 'lon'].values[0])[::-1]


for train_basin_int in train_basin_int_list:
    if loc == 'JP' or loc == 'AUS':
        fig, ax = plt.subplots(figsize=(10, 6))
    elif loc == 'US' or loc == 'GB':
        fig, ax = plt.subplots(figsize=(10, 8))

    SplitPosTrainBasin = split(PosTrainBasin, train_basin_int)
    print(SplitPosTrainBasin)

    for sample in range(1, random_samples + 1):
        train_basins = []
        for sPosTrainBasin in SplitPosTrainBasin:
            train_basins.append(random.sample(sPosTrainBasin, 1)[0])
        train_basins.sort()

        #print(train_basins)
        ax.scatter([sample] * len(train_basins), train_basins, alpha=0.6)

    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Train Basin IDs')
    ax.set_title(f'Dispersal of Train Basin IDs for Different Samples Train{train_basin_int}')
    plt.savefig(f'{output_dir}/randomness_{loc}_{ver}_Train{train_basin_int}.png')
    plt.clf()

