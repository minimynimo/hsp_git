# Description: This script generates maps of the river basins colored by the benchmark values.
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from shapely.geometry.point import Point
from matplotlib_scalebar.scalebar import ScaleBar
import pyproj
from matplotlib.colors import TwoSlopeNorm
import math
from matplotlib import colors
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import Patch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

##

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
#loc, ver = "US", 2
loc, ver = "UStoBR", 2

fs = 35

n_components = 3
reservoir_size = 200

file_tag = f'r{reservoir_size}_s1_sr0.4_rr0.001'
###

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
if loc == "UStoBR" and ver == 2:
    locR = "BR"    
    file_tot_num = 667
    lat_min, lat_max = -35.0, 6.0
    lon_min, lon_max = -75.0, -32.0
    crs = 4326
data_folder = f'/data0/funato/3_gis_data/{locR}/0_data/shape_file/'

all_basins = set(range(1, file_tot_num+1))

font_prop = FontProperties(size=20)

##base: maps where the trained and test besins are
##result: maps the benchmark values for the trained basins
##dif: maps difference of calibrated BMA benchmark values with PUB values
map_type_list = ['result'] 

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
#benchmark_list = ["RMSE","MAE"]

# Read the shapefile
### country MAP & RIVER BASINS MAP ###
country_path = data_folder + f'{locR}_shape.shp'
country_gdf = gpd.read_file(country_path)
basins_path = data_folder + 'river_basin.shp'
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
        return 'white' ##For the train basins 

def setup_figure(width, lat_min, lat_max, lon_min, lon_max):
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    height = width * aspect_ratio
    plt.figure(figsize=(width, height * aspect_ratio))
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

benchmark_limits_norm = {
    'KGE': {'min': -2, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 20},
    'MAE': {'min': 0, 'max': 20}
}

output_folder = f'/data0/funato/0_out/0_fig/{loc}/map/'

for map_type in map_type_list:
    if map_type == 'result': ## result
        output_folder_map_type = f'{output_folder}/result_map/'
    if not os.path.exists(output_folder_map_type):
        os.makedirs(output_folder_map_type, exist_ok=True)

    for pc in range(1, n_components + 1):
        ## BMA PUB results 
        BC_PCA_path = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}/test_basin/results/BC-PCA-lasso_PUB_results_{file_tag}_rev_PC{pc}_eva.csv'

        result_output_folder = f'{output_folder_map_type}/BC_PCA/PC{pc}' 
        if not os.path.exists(result_output_folder):
            os.makedirs(result_output_folder, exist_ok=True)
    
        result_df = pd.read_csv(BC_PCA_path)

        for benchmark in benchmark_list:            
            ### RESULT MAP ###
            width = 20 if loc == 'JP' else 25 if loc == 'UStoBR' else 40  # Adjust width based on location
            ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

            facecolor = 'white'
            ax.set_facecolor(facecolor) ## light yellow
            country_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=1.3)

            result_column = result_df[f'BC-PCA-lasso_{file_tag}_{benchmark}_eva'].tolist()

            # Add a colorbar ### INCLUDE 0: R1, NSE, KGE, VE
            if benchmark in ['RMSE', 'MAE']:
                norm = plt.Normalize(vmin=0, vmax=max(result_column))
                cmap = plt.cm.inferno_r  # Reverse the colormap
            elif benchmark in ['KGE', 'NSE', 'E1','VE', 'd']:
                ## divide colors below and above 0 
                min_value = min(result_column)
                max_value = 1
                norm = TwoSlopeNorm(vmin=math.floor(max(min_value,-3)), vcenter=0, vmax=max_value)
                cmap = plt.cm.seismic
            elif benchmark in ['logNSE', 'Erel']:
                norm = plt.Normalize(vmin=min(result_column), vmax=1)
                cmap = plt.cm.inferno
            else:
                norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
                cmap = plt.cm.inferno

            # Apply the color function to each row in the GeoDataFrame
            basins_gdf.loc[:, 'color'] = basins_gdf['File_num'].apply(get_basin_result_color)

            basins_gdf.plot(ax=ax, facecolor=basins_gdf['color'], edgecolor='black', linewidth=1.0)
            
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

            plt.title(f'BC_PCA Result Map {benchmark}\n', fontsize=fs)

            plt.tight_layout()
            plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.05)  # Adjusted left and right padding

            plt.savefig(f'{result_output_folder}/BC_PCA_basin_{map_type}_map_{benchmark}.jpg')
            print(f'results BC_PCA {benchmark} Done!')
            plt.close()     


print("DONE!!!")