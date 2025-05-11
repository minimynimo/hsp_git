# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
import warnings
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colors
import math

warnings.filterwarnings('ignore')

#########
all_mode = True # include all models

#loc, ver = "JP", 2
#loc, ver = "US", 2
#loc, ver = "AUS", 2
loc, ver = "GB", 2

#nocal_tag = "_nocal"
nocal_tag = ""

non_arid = True
if loc != "US":
    non_arid = False

n_components = 2

#reservoir_size = 200
reservoir_size = 700
#ridge_param = 1.0
ridge_param = 0.001

fs = 24 ## 22 or over for presentation, 13 for paper
#########
file_tag = f'r{reservoir_size}_s1_sr0.4_rr{ridge_param}'


if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
    lat_min, lat_max = 30.0, 46.0
    lon_min, lon_max = 127.0,149.0
    crs = 6667 #4947
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)]
    #step_val_list = list(range(2, start_val_list[0] + 1))
    step_val_list = [2,4,8,15,20,25,30,40] # 39, 20, 10, 5, 4, 3, 2
    lat_min, lat_max = 30.0, 46.0
    lon_min, lon_max = 127.0,149.0
    crs = 6667 #4947
elif loc == "US" and ver == 1:
    file_tot_num = 671
    start_val_list = [336]
    #step_val_list = list(range(2,337))
    step_val_list = [2,5,10,21,42,84,168,337]
    lat_min, lat_max = 23.0, 52.0
    lon_min, lon_max = -127.0, -65.0
    crs = 4326
elif loc == "US" and ver == 2:
    file_tot_num = 667
    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #301
    step_val_list = [2,4,6,12,24,47,84,125,149,168,277] #### Train basin num: 299,150,100,50,25,13,7,5,4,3,3
    lat_min, lat_max = 23.0, 52.0
    lon_min, lon_max = -127.0, -65.0
    crs = 4326
    arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
    arid_file_list = pd.read_csv(arid_file_list)
    arid_file_list = arid_file_list['File_num'].tolist()
elif loc == "AUS" and ver == 2:
    file_tot_num = 84
    lat_min, lat_max = -46.0, -8.0
    lon_min, lon_max = 110.0, 156.0
    crs = 4326
elif loc == "GB" and ver == 2:
    file_tot_num = 396
    lat_min, lat_max = 49.0, 61.0
    lon_min, lon_max = -8.0, 2.0
    crs = 4326
data_folder = f'/data0/funato/3_gis_data/{loc}/0_data/shape_file/'

all_basins = set(range(1, file_tot_num+1))

font_prop = FontProperties(size=20)

##base: maps where the trained and test besins are
##result: maps the benchmark values for the trained basins
##dif: maps difference of calibrated BMA benchmark values with PUB values
map_type_list = ['result'] 

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]
#benchmark_list = ["RMSE","MAE"]

# Read the shapefile
### country MAP & RIVER BASINS MAP ###
country_path = data_folder + f'{loc}_shape.shp'
country_gdf = gpd.read_file(country_path)
basins_path = data_folder + f'river_basin_v{ver}.shp'
basins_gdf = gpd.read_file(basins_path)

points = gpd.GeoSeries(
    [Point(lon_max, (lat_min+lat_max)/2.0), Point(lon_min, (lat_min+lat_max)/2.0)], crs=crs
)  # Geographic EPSG 6667
utm_zone_number = int((lon_min + 180) / 6) + 1  # Calculate UTM zone
utm_epsg_code = 32600 + utm_zone_number
points_utm = points.to_crs(utm_epsg_code)

x_km = points_utm.x / 1000.0
y_km = points_utm.y / 1000.0

points_km = gpd.GeoSeries([Point(x, y) for x, y in zip(x_km, y_km)], crs=f'EPSG:{utm_epsg_code}')

def get_basin_result_color(file_num):
    index = result_df[result_df['file_num'] == file_num].index    
    if not index.empty: ## test basin
        value = result_column[index[0]]        
        return cmap(norm(value))
    else:
        return 'dimgray' ##For the train basins 

def plot_basin_markers(ax, basins_gdf):
    offset = 0.3  # Offset to prevent markers from overlapping too much
    for i, row in basins_gdf.iterrows():
        if non_arid and row['File_num'] in arid_file_list:
            continue
        basin_center = row['geometry'].centroid
        color = get_basin_result_color(row['File_num'])
        x_offset = (i % 2) * offset - offset / 2
        y_offset = ((i // 2) % 2) * offset - offset / 2
        ax.plot(basin_center.x + x_offset, basin_center.y + y_offset, marker='o', markersize=10, color=color, markeredgecolor='black')

def setup_figure(width, lat_min, lat_max, lon_min, lon_max):
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    height = width * aspect_ratio *1.7
    plt.figure(figsize=(width, height))
    ax = plt.gca()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')
    return ax

def apply_labels(ax):
    ax.set_xlabel('Longitude', fontsize=fs)
    ax.set_ylabel('Latitude', fontsize=fs)
    lon_labels = [f"{int(label)}°E" for label in ax.get_xticks()]
    lat_labels = [f"{int(label)}°N" for label in ax.get_yticks()]
    ax.set_xticklabels(lon_labels, fontsize=fs)
    ax.set_yticklabels(lat_labels, fontsize=fs)
    lon_positions = ax.get_xticks()
    lat_positions = ax.get_yticks()
    ax.xaxis.set_major_locator(FixedLocator(lon_positions))
    ax.xaxis.set_major_formatter(FixedFormatter(lon_labels))
    ax.yaxis.set_major_locator(FixedLocator(lat_positions))
    ax.yaxis.set_major_formatter(FixedFormatter(lat_labels))
    ax.tick_params(axis='both', which='both', labelsize=fs)
    plt.grid(True)

def add_scalebar(ax):
    scalebar = ScaleBar(100, location='lower right', units='km', scale_loc='bottom',
                        pad=0.1, length_fraction=1, height_fraction=0.01, 
                        fixed_value=100, font_properties={"size": 20})
    ax.add_artist(scalebar)

def add_north_arrow(ax):
    north_arrow_symbol = '\u25B2'  # Upward-pointing triangle
    plt.text(0.930, 0.15, north_arrow_symbol, transform=ax.transAxes, fontsize=50, fontweight='bold', color='black', rotation=0)
    plt.text(0.937, 0.10, 'N', transform=ax.transAxes, fontsize=30, fontweight='bold', color='black')

##LOAD DATA
def load_AVE_data():
    AVE_file_path = f'/data0/funato/0_out/99_out/{loc}/AVE{nocal_tag}/results/AVE_results_eva.csv'
    AVE_df = pd.read_csv(AVE_file_path)
    return AVE_df

def load_BMA_data():
    BMA_file_path = f'/data0/funato/0_out/99_out/{loc}/BMA{nocal_tag}/results/BMA_results_eva.csv'
    BMA_df = pd.read_csv(BMA_file_path)
    return BMA_df

def load_RC_data():
    RC_file_path = f'/data0/funato/0_out/99_out/{loc}/RC_{reservoir_size}_{ridge_param}/results/RC_results_{file_tag}_eva.csv'
    RC_df = pd.read_csv(RC_file_path)
    return RC_df

def load_RCH_data():
    RCH_file_path = f'/data0/funato/0_out/99_out/{loc}/RCHBMA_{reservoir_size}_{ridge_param}/results/RCHBMA_results_{file_tag}_eva.csv'
    RCH_df = pd.read_csv(RCH_file_path)
    return RCH_df

def load_BC_data():    
    BC_file_path = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}/results/BC_results_{file_tag}_bma_eva.csv'
    BC_df = pd.read_csv(BC_file_path)
    return BC_df
    
def load_RCBC_data():  
    RCBC_file_path = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_{reservoir_size}_{ridge_param}/results/RCBC-BMA_results_{file_tag}_eva.csv'
    RCBC_df = pd.read_csv(RCBC_file_path)
    return RCBC_df

def load_BCPCA_data(pc):
    BCPCA_file_path = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_{reservoir_size}_{ridge_param}/results/BC-PCA-lasso_results_{file_tag}_rev_PC{pc}_eva.csv'
    BCPCA_df = pd.read_csv(BCPCA_file_path)
    return BCPCA_df

def load_BC_PUB_data(start_val, step_val, train_test):
    BC_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BC_PUB_{reservoir_size}_{ridge_param}/start{start_val}/{train_test}_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
    BC_PUB_df = pd.read_csv(BC_PUB_data)
    BC_PUB_column_data = np.array(BC_PUB_df[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        BC_PUB_column_data = BC_PUB_column_data[BC_PUB_df['file_num'].isin(test_basins_list)]
    return BC_PUB_column_data

def load_BMA_PUB_data(start_val, step_val, train_test):
    BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/BMA_PUB/start{start_val}/{train_test}_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
    BMA_PUB_df = pd.read_csv(BMA_PUB_data)
    BMA_PUB_column_data = np.array(BMA_PUB_df[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        BMA_PUB_column_data = BMA_PUB_column_data[BMA_PUB_df['file_num'].isin(test_basins_list)]
    return BMA_PUB_column_data
    
def load_RCBC_BMA_PUB_data(start_val, step_val, train_test):
    RCBC_BMA_PUB_data = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_PUB_{reservoir_size}_{ridge_param}/start{start_val}/{train_test}_basin/results/RCBC-BMA_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
    RCBC_BMA_PUB_df = pd.read_csv(RCBC_BMA_PUB_data)
    RCBC_BMA_PUB_column_data =  np.array(RCBC_BMA_PUB_df[f'RCBC-BMA_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist())
    if train_test == 'test':
        RCBC_BMA_PUB_column_data = RCBC_BMA_PUB_column_data[RCBC_BMA_PUB_df['file_num'].isin(test_basins_list)]
    return RCBC_BMA_PUB_column_data

def load_BC_PCA_lasso_PUB_data(start_val, step_val, pc, train_test):
    BC_PCA_lasso_result_data_dir = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/start{start_val}/step{step_val}/{train_test}_basin/results/BC-PCA-lasso_PUB_results_{file_tag}_start{start_val}_step{step_val}_rev_PC{pc}_eva.csv')
    BC_PCA_lasso_PUB_column_data = BC_PCA_lasso_result_data_dir[f'BC-PCA-lasso_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva']
    if train_test == 'test':
        BC_PCA_lasso_PUB_column_data = BC_PCA_lasso_PUB_column_data[BC_PCA_lasso_result_data_dir['file_num'].isin(test_basins_list)]
    return BC_PCA_lasso_PUB_column_data

def load_GBtoUS_data():
    GBtoUS_data = f'/data0/funato/0_out/99_out/GBtoUS/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv'
    GBtoUS_df = pd.read_csv(GBtoUS_data)
    return GBtoUS_df

def load_UStoGB_data(): 
    UStoGB_data = f'/data0/funato/0_out/99_out/UStoGB/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv'
    UStoGB_df = pd.read_csv(UStoGB_data)
    return UStoGB_df

def load_GBtoJP_data():
    GBtoJP_data = f'/data0/funato/0_out/99_out/GBtoJP/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv'
    GBtoJP_df = pd.read_csv(GBtoJP_data)
    return GBtoJP_df

def load_JPtoGB_data():
    JPtoGB_data = f'/data0/funato/0_out/99_out/JPtoGB/BC-PCA-lasso_PUB_200_1.0/test_basin/results/BC-PCA-lasso_PUB_results_r200_s1_sr0.4_rr1.0_rev_PC2_eva.csv'
    JPtoGB_df = pd.read_csv(JPtoGB_data)
    return JPtoGB_df

calc_mode_to_name = {
    #'BMA_norm': 'BMA',
    #'BC_norm': 'BC',
    #'BCPCA_norm': 'BCPCA',
    #'BC_BMA_dif': 'dif_BC & BMA',
    #'BC_BCPCA_dif': 'dif_BC & BCPCA',

    #'RCBC_norm': 'RCBC-BMA_PUB',
    #'RCBC_BMA_dif': 'dif_RCBC-BMA & BMA',
    #'RCBC_BC_dif': 'dif_RCBC-BMA & BC',

    'GBtoUS_norm': 'GBtoUS',    
    'UStoGB_norm': 'UStoGB',
    'GBtoJP_norm': 'GBtoJP',
    'JPtoGB_norm': 'JPtoGB',
}

if loc == "US":
    calc_mode_to_name.pop('UStoGB_norm', None)
    calc_mode_to_name.pop('GBtoJP_norm', None)
    calc_mode_to_name.pop('JPtoGB_norm', None)
elif loc == "GB":
    calc_mode_to_name.pop('GBtoUS_norm', None)
    calc_mode_to_name.pop('GBtoJP_norm', None)
elif loc == "JP":
    calc_mode_to_name.pop('UStoGB_norm', None)
    calc_mode_to_name.pop('JPtoGB_norm', None)
    calc_mode_to_name.pop('GBtoUS_norm', None)


benchmark_limits_norm = {
    'KGE': {'min': -3, 'max': 1},
    'NSE': {'min': -3, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 20}
}
benchmark_limits_dif = {
    'KGE': {'min': -3, 'max': 3},
    'NSE': {'min': -3, 'max': 3},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -3, 'max': 3},
    'VE': {'min': -2, 'max': 2},
    'd': {'min': -1, 'max': 1},
    'RMSE': {'min': -10, 'max': 10},
    'MAE': {'min': -3, 'max': 3}
}

for calc_mode in calc_mode_to_name.keys():
    calc_name = calc_mode_to_name[calc_mode]

    if 'to' not in calc_mode:
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/map/'
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f'/data0/funato/0_out/0_fig/{calc_name}/map/'
        os.makedirs(output_dir, exist_ok=True)

    if 'to' in calc_mode:
        rrr_buf = ''
    else:
        rrr_buf = f'_{reservoir_size}_{ridge_param}'
 
    if non_arid:
        if 'dif' not in calc_name: # results
            output_result_dir = f'{output_dir}/{calc_name}{rrr_buf}_nonarid_marker/'
        elif 'dif' in calc_name: # difference
            output_result_dir = f'{output_dir}/{calc_name}{rrr_buf}_nonarid_marker/'    
    else:
        if 'dif' not in calc_name: # results
            output_result_dir = f'{output_dir}/{calc_name}{rrr_buf}_marker/'
        elif 'dif' in calc_name: # difference
            output_result_dir = f'{output_dir}/{calc_name}{rrr_buf}_marker/'

    os.makedirs(output_result_dir, exist_ok=True)
    
    for pc in range(1, n_components + 1):
        if 'PCA' in calc_name:
            BCPCA_result_df = load_BCPCA_data(pc)
        
        if pc > 1 and calc_mode not in ['BCPCA_norm', 'BC_BCPCA_dif']:
            continue

        for benchmark in benchmark_list:
            AVE_result_df = load_AVE_data()
            BMA_result_df = load_BMA_data()
            RC_result_df = load_RC_data()
            RCH_result_df = load_RCH_data()
            BC_result_df = load_BC_data()
            GBtoUS_result_df = load_GBtoUS_data()   
            UStoGB_result_df = load_UStoGB_data()
            GBtoJP_result_df = load_GBtoJP_data()
            JPtoGB_result_df = load_JPtoGB_data()

            AVE_column = f'AVE_{benchmark}_eva'
            BMA_column = f'BMA_{benchmark}_eva'
            RC_column = f'RC_{file_tag}_{benchmark}_eva'
            RCH_column = f'RCHBMA_{file_tag}_{benchmark}_eva'
            BC_column = f'BC_{file_tag}_bma_{benchmark}_eva'
            RCBC_column = f'RCBC-BMA_{file_tag}_{benchmark}_eva'
            BCPCA_column = f'BC-PCA-lasso_{file_tag}_{benchmark}_eva'
            GBtoUS_column = f'BC-PCA-lasso_r200_s1_sr0.4_rr1.0_{benchmark}_eva'
            UStoGB_column = f'BC-PCA-lasso_r200_s1_sr0.4_rr1.0_{benchmark}_eva'
            GBtoJP_column = f'BC-PCA-lasso_r200_s1_sr0.4_rr1.0_{benchmark}_eva'
            JPtoGB_column = f'BC-PCA-lasso_r200_s1_sr0.4_rr1.0_{benchmark}_eva'

            AVE_result_column = np.array(AVE_result_df[AVE_column].tolist())
            BMA_result_column = np.array(BMA_result_df[BMA_column].tolist())
            RC_result_column = np.array(RC_result_df[RC_column].tolist())
            RCH_result_column = np.array(RCH_result_df[RCH_column].tolist())
            BC_result_column = np.array(BC_result_df[BC_column].tolist())
            if 'PCA' in calc_name:
                BCPCA_result_column = np.array(BCPCA_result_df[BCPCA_column].tolist())
            GBtoUS_result_column = np.array(GBtoUS_result_df[GBtoUS_column].tolist())
            UStoGB_result_column = np.array(UStoGB_result_df[UStoGB_column].tolist())
            GBtoJP_result_column = np.array(GBtoJP_result_df[GBtoJP_column].tolist())
            JPtoGB_result_column = np.array(JPtoGB_result_df[JPtoGB_column].tolist())

            # result
            if calc_name == 'BMA':
                result_df, result_column  = BMA_result_df, BMA_result_column
            elif calc_name == 'BC':
                result_df, result_column = BC_result_df, BC_result_column
            elif calc_name == 'BCPCA':
                result_df, result_column = BCPCA_result_df, BCPCA_result_column
            elif calc_name == 'RC':
                result_df, result_column = RC_result_df, RC_result_column
            elif calc_name == 'RCH':
                result_df, result_column = RCH_result_df, RCH_result_column
            elif calc_name == 'BC_BMA_dif':
                result_df = BC_result_df
                result_column = BC_result_column - BMA_result_column
            elif calc_name == 'BC_PCA_dif':
                result_df = BC_result_df
                result_column = BC_result_column - BCPCA_result_column
            elif calc_name == 'GBtoUS':
                result_df, result_column = GBtoUS_result_df, GBtoUS_result_column
            elif calc_name == 'UStoGB':
                result_df, result_column = UStoGB_result_df, UStoGB_result_column
            elif calc_name == 'GBtoJP':
                result_df, result_column = GBtoJP_result_df, GBtoJP_result_column
            elif calc_name == 'JPtoGB':
                result_df, result_column = JPtoGB_result_df, JPtoGB_result_column


            width = 13 if loc == 'JP' else 10 if loc == 'GB' else 15  # Adjust width based on location
            ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

            facecolor = 'white'
            ax.set_facecolor(facecolor) ## light yellow
            country_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=1.3)

            # Add a colorbar ### INCLUDE 0: R1, NSE, KGE, VE
            if 'dif' not in calc_name:
                if benchmark in ['RMSE', 'MAE']:
                    norm = plt.Normalize(vmin = benchmark_limits_norm[benchmark]['min'], vmax = benchmark_limits_norm[benchmark]['max'])
                    cmap = plt.cm.inferno_r  # Reverse the colormap
                elif benchmark in ['KGE', 'NSE', 'E1','VE', 'd']:
                    ## divide colors below and above 0 
                    min_value = min(result_column)
                    max_value = 1
                    vmin = math.floor(max(min_value, benchmark_limits_norm[benchmark]['min']))
                    vcenter = 0
                    vmax = benchmark_limits_norm[benchmark]['max']
                    if vmin >= vcenter:
                        vmin = vcenter - 1
                    if vcenter >= vmax:
                        vmax = vcenter + 1
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    cmap = plt.cm.seismic
                elif benchmark in ['logNSE', 'Erel']:
                    norm = plt.Normalize(vmin=min(result_column), vmax=1)
                    cmap = plt.cm.inferno
                else:
                    norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
                    cmap = plt.cm.inferno
            else:
                if benchmark in ['RMSE', 'MAE']:
                    norm = plt.Normalize(vmin = benchmark_limits_dif[benchmark]['min'], vmax = benchmark_limits_dif[benchmark]['max'])
                    cmap = plt.cm.inferno_r
                elif benchmark in ['KGE', 'NSE', 'E1','VE', 'd']:
                    ## divide colors below and above 0 
                    min_value = min(result_column)
                    max_value = 1
                    vmin = benchmark_limits_dif[benchmark]['min']
                    vcenter = 0
                    vmax = benchmark_limits_dif[benchmark]['max']
                    if vmin >= vcenter:
                        vmin = vcenter - 1
                    if vcenter >= vmax:
                        vmax = vcenter + 1
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    cmap = plt.cm.seismic
                elif benchmark in ['logNSE', 'Erel']:
                    norm = plt.Normalize(vmin=min(result_column), vmax=1)
                    cmap = plt.cm.inferno
                else:
                    norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
                    cmap = plt.cm.inferno

            # Apply the color function to each row in the GeoDataFrame
            basins_gdf.loc[:, 'color'] = basins_gdf['File_num'].apply(get_basin_result_color)
            plot_basin_markers(ax, basins_gdf)

            apply_labels(ax)
            add_scalebar(ax)
            add_north_arrow(ax) 

            ### COLOR BAR ###
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(benchmark, fontsize=fs)
            cbar.ax.tick_params(labelsize=fs)

            plt.grid(True)
            if calc_name not in ['BCPCA_norm', 'BC_BCPCA_dif']:
                #plt.title(f'{calc_name} ({benchmark})', fontsize=fs)
                plt.title(f'')
            else:
                plt.title(f'{calc_name} PC{pc} ({benchmark})', fontsize=fs)
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.90, top=0.93, bottom=0.07)  # Adjusted margins

            if calc_name not in ['BCPCA_norm', 'BC_BCPCA_dif']:
                plt.savefig(f'{output_result_dir}/{calc_name}_basin_map_{benchmark}.jpg')
            else:
                plt.savefig(f'{output_result_dir}/{calc_name}_basin_map_{benchmark}_PC{pc}.jpg')
            print(f'results {calc_name} {benchmark} Done!')
            plt.close()     

print("DONE!!!")




