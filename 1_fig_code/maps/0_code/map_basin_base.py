# Description: This script generates maps of the river basins colored by the benchmark values.
# To create the base map of the train and test basins
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
file_tag = 'r700_s1_sr0.4_rr0.001'

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
loc, ver = "US", 2

fs = 25
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

all_basins = set(range(1, file_tot_num+1))
PosTrainBasin = [i for i in range(1,file_tot_num) if i not in test_basins_list]

font_prop = FontProperties(size=20)

data_folder = f'/data0/funato/3_gis_data/{loc}/0_data/shape_file/'

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
#benchmark_list = ["RMSE","MAE"]

# Read the shapefile
### country MAP & RIVER BASINS MAP ###
country_path = data_folder + f'{loc}_shape.shp'
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

def get_basin_base_color(file_num):
    if file_num in train_basins:
        return '#FFCD43' # Light orange
    else:
        return '#87CED5' # Light blue

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

for start_val in start_val_list:
    output_folder = f'/data0/funato/0_out/0_fig/{loc}/map/base/start{start_val}'
    for step_val in step_val_list:
        # Define the list of basins to be colored in light orange
        train_basins = sorted([PosTrainBasin[i] for i in range(start_val, len(PosTrainBasin), step_val)] + 
                              [PosTrainBasin[i] for i in range(start_val - step_val, -1, -step_val)])
        
        test_basins = list(all_basins - set(train_basins))

        train_basins_gdf = basins_gdf.loc[basins_gdf['File_num'].isin(train_basins)]
        test_basins_gdf = basins_gdf.loc[~basins_gdf['File_num'].isin(train_basins)]

        # Plot the shapefile
        width = 20 if loc == 'JP' else 40  # Adjust width based on location
        ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

        country_gdf.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=1.0)
        
        # Apply the color function to each row in the GeoDataFrame
        test_basins_gdf.loc[:, 'color'] = "#87CED5"  # Light blue
        train_basins_gdf.loc[:, 'color'] = '#FFCD43'  # Light orange

        train_basins_gdf.loc[:, 'color'] = train_basins_gdf['color'].fillna('none')
        test_basins_gdf.loc[:, 'color'] = test_basins_gdf['color'].fillna('none')

        test_basins_gdf.plot(ax=ax, facecolor=test_basins_gdf['color'], edgecolor='black', linewidth=1.0)
        train_basins_gdf.plot(ax=ax, facecolor=train_basins_gdf['color'], edgecolor='black', linewidth=1.0)

        # Add legend
        legend_elements = [Line2D([0], [0], marker='s', color='w', label='Train', markersize=20, markerfacecolor='#FFCD43'),
                        Line2D([0], [0], marker='s', color='w', label='Test', markersize=20, markerfacecolor='#87CED5')]
        if loc == 'JP':
            legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=35)
        elif loc == 'US':
            legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=35)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(1)

        apply_labels(ax)
        add_scalebar(ax)
        add_north_arrow(ax)

        plt.title(f'Base Map start{start_val} step{step_val}\n', fontsize=fs)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(f'{output_folder}/basin_base_map_start{start_val}_step{step_val}.png')
        print('base Done!')

        plt.close()

print("DONE!!!")