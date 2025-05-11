import random
import pandas as pd
import matplotlib.pyplot as plt


#loc, ver = "JP", 1
loc, ver = "JP", 2
# loc, ver = "US", 1
#loc, ver = "US", 2

random_samples = 100
random_seed = 42 


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
    train_basin_int_list = [40,20,10,5,4,3,2]
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    distance_matrix = pd.read_csv('/data0/funato/3_gis_data/JP/0_data/distance_matrix_v2_sorted.csv', index_col=0, header=0)
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']

train_basin_int_list = [40,20,10,5,4,3,2]


random.seed(random_seed)

# possible training basins to select the training basins from
PosTrainBasin = [i for i in range(1,file_tot_num) if i not in test_basins_list]
if loc == "JP":
    PosTrainBasin = sorted(PosTrainBasin, key=lambda x: basin_data_df.loc[basin_data_df['File_num'] == x, 'long_org'].values[0])[::-1]
elif loc == "US":
    PosTrainBasin = sorted(PosTrainBasin, key=lambda x: basin_data_df.loc[basin_data_df['File_num'] == x, 'lon'].values[0])[::-1]

for train_basin_int in train_basin_int_list:
    fig, ax = plt.subplots(figsize=(10, 6))

    for sample in range(1, random_samples + 1):
        train_basins = random.sample(PosTrainBasin, train_basin_int)
        ax.scatter([sample] * len(train_basins), train_basins, alpha=0.6)

    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Train Basin IDs')
    ax.set_title(f'Dispersal of Train Basin IDs for Different Samples Train{train_basin_int}')
    plt.savefig(f'/data0/funato/0_out/0_fig/randomness/randomness_{loc}_{ver}_Train{train_basin_int}.png')
    plt.clf()

