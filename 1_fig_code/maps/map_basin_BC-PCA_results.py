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

warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")
warnings.simplefilter(action='ignore', category=UserWarning)

###
file_tag = 'r700_s1_sr0.4_rr0.001'
n_components = 3
fs = 25
prefecture_lw = 0.7
basin_lw = 1.2
facecolor = 'white' # '#FFFFF0'
PCA_type = 'BC-PCA-lasso' # PCA using lasso PC weight estimation by reverting PCA weights using lasso weights estimation
#PCA_type = 'BC-PCA' # PCA without PC weight estimation by reverting PCA weights without lasso weights estimation (no use of basin attributes)
###
# end val is always 135
end_val = 136

all_basins = set(range(1, end_val))

##base: maps where the trained and test besins are
##result: maps the benchmark values for the trained basins
##dif: maps difference of calibrated BMA benchmark values with PUB values
#map_type_list = ['result','dif','ratio']
map_type_list = ['ratio']  #

data_folder = '/data0/funato/3_gis_data/0_data/'
PCA_result_data_dir = f'/data0/funato/0_out/99_out/{PCA_type}_{reservoir_size}/results' 

benchmark_list = ["KGE","NSE","logNSE","E1", "VE", "d","RMSE","MAE"]
#benchmark_list = ["RMSE","MAE"]

# Read the shapefile
### PREFECTURES MAP & RIVER BASINS MAP ###
lat_min, lat_max = 30.0, 46.0
lon_min, lon_max = 127.0,149.0
prefectures_path = data_folder + 'shape_file/Japan_shape.shp'
prefectures_gdf = gpd.read_file(prefectures_path)
basins_path = data_folder + 'shape_file/river_basin.shp'
basins_gdf = gpd.read_file(basins_path)


points = gpd.GeoSeries(
    [Point(lon_max, (lat_min+lat_max)/2.0), Point(lon_min, (lat_min+lat_max)/2.0)], crs=4947
)  # Geographic EPSG 6667
utm_zone_number = int((lon_min + 180) / 6) + 1  # Calculate UTM zone
utm_epsg_code = 32600 + utm_zone_number
points_utm = points.to_crs(utm_epsg_code)

utm_to_geo = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg_code}", "EPSG:6667", always_xy=True)


x_km = points_utm.x / 1000.0
y_km = points_utm.y / 1000.0

points_km = gpd.GeoSeries([Point(x, y) for x, y in zip(x_km, y_km)], crs=f'EPSG:{utm_epsg_code}')

def load_pc_data(pc_num):
    return pd.read_csv(os.path.join(PCA_result_data_dir, f'{PCA_type}_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_num}_eva.csv'))

def get_basin_result_color(file_num):
    value = result_column[file_num - 1]  # Adjust index to match file_num
    return cmap(norm(value))
    
def get_basin_dif_color(file_num):
    value = np.array(BC_PCA_result_column[file_num - 1]) - np.array(BMA_result_column[file_num - 1])
    return cmap(norm(value))

def get_basin_ratio_color(file_num):
    value = result_column[file_num - 1]  # Adjust index to match file_num
    return cmap(norm(value))

def setup_figure(width, lat_min, lat_max, lon_min, lon_max):
    aspect_ratio = (lat_max - lat_min) / (lon_min - lon_max)
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

font_prop = FontProperties(size=20)

## BMA PUB results 
BMA_nocal_path = '/data0/funato/0_out/99_out/BMA_nocal/results/BMA_results_eva.csv'
BMA_result_df = pd.read_csv(BMA_nocal_path)

for pc_n in range(1, n_components+1):
    output_folder = f'/data0/funato/0_out/0_fig/map_{PCA_type}/PC{pc_n}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    BC_PCA_result_df = load_pc_data(pc_n)

    for map_type in map_type_list:
        if map_type == 'result': ## result
            output_folder_map_type = f'{output_folder}/result_map/'
        elif map_type == 'dif': ## difference
            output_folder_map_type = f'{output_folder}/dif_map/'
        elif map_type == 'ratio': ## ratio
            output_folder_map_type = f'{output_folder}/ratio_map/'
        if not os.path.exists(output_folder_map_type):
            os.makedirs(output_folder_map_type, exist_ok=True)

        for benchmark in benchmark_list:
            ### if map_type = 'result' then color based on the benchmark by using a colorbar
            BMA_result_column = BMA_result_df[f'BMA_{benchmark}_eva'].tolist()
            BC_PCA_result_column = BC_PCA_result_df[f'{PCA_type}_r700_s1_sr0.4_rr0.001_{benchmark}_eva'].tolist()
            
            ### RESULT MAP ###
            if map_type == 'result':
                result_output_folder = f'{output_folder_map_type}/'
                if not os.path.exists(result_output_folder):
                    os.makedirs(result_output_folder, exist_ok=True)

                result_column = BC_PCA_result_column 

                width = 20  # Adjust this for the desired width of the figure
                ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                # Add a colorbar ### INCLUDE 0: R1, NSE, KGE, VE
                if benchmark in ['RMSE', 'MAE']:
                    facecolor = 'white'
                    norm = plt.Normalize(vmin=0, vmax=max(result_column))
                    cmap = plt.cm.inferno_r  # Reverse the colormap
                elif benchmark in ['KGE', 'NSE', 'E1','VE', 'd']:
                    ## divide colors below and above 0 
                    facecolor = facecolor
                    min_value = min(result_column)
                    max_value = 1
                    norm = TwoSlopeNorm(vmin=math.floor(min(min_value,-0.5)), vcenter=0, vmax=max_value)
                    cmap = plt.cm.seismic
                elif benchmark in ['logNSE']:
                    facecolor = 'white'
                    norm = plt.Normalize(vmin=min(result_column), vmax=1)
                    cmap = plt.cm.inferno
                else:
                    facecolor = 'white'
                    norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
                    cmap = plt.cm.inferno

                ax.set_facecolor(facecolor) ## light yellow
                prefectures_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=prefecture_lw)

                # Apply the color function to each row in the GeoDataFrame
                basins_gdf['color'] = basins_gdf['File_num'].apply(get_basin_result_color)
                basins_gdf.loc[:, 'color'] = basins_gdf['color'].fillna('none')
                basins_gdf.plot(ax=ax, facecolor=basins_gdf['color'].apply(lambda x: x if x != 'none' else (1, 1, 1, 0)), edgecolor='black', linewidth=basin_lw)

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
                plt.title(f'BC_PCA Result Map {benchmark} PC{pc_n}\n', fontsize=fs)

                plt.tight_layout()

                plt.savefig(f'{result_output_folder}/{PCA_type}_basin_{map_type}_map_PC{pc_n}_{benchmark}.jpg')
                print(f'results PC{pc_n} {benchmark} Done!')
                plt.close()     

            ### DIFFERENCE MAP ###
            elif map_type == 'dif':
                width = 20  # Adjust this for the desired width of the figure
                ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                ax.set_facecolor(facecolor) ## light yellow
                prefectures_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=prefecture_lw)

                result_column = np.array(BC_PCA_result_column) - np.array(BMA_result_column)

                norm = TwoSlopeNorm(vmin=math.floor(min(min(result_column),-0.5)), vcenter=0, vmax=math.ceil(max(max(result_column),0.5)))
                if benchmark in ['RMSE', 'MAE']:
                    cmap = plt.cm.seismic_r  
                else:
                    cmap = plt.cm.seismic

                basins_gdf['color'] = basins_gdf['File_num'].apply(get_basin_dif_color)
                basins_gdf.loc[:, 'color'] = basins_gdf['color'].fillna('none')
                basins_gdf.plot(ax=ax, facecolor=basins_gdf['color'].apply(lambda x: x if x != 'none' else (1, 1, 1, 0)), edgecolor='black', linewidth=basin_lw)

                apply_labels(ax)
                add_scalebar(ax)
                add_north_arrow(ax) 

                ### COLOR BAR ###
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(benchmark, fontsize=fs)
                cbar.ax.tick_params(labelsize=fs)

                plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
                plt.title(f'Difference Map ({PCA_type} - BMA){benchmark} PC{pc_n}\n', fontsize=fs)

                plt.tight_layout()
                plt.savefig(f'{output_folder_map_type}/basin_{map_type}_map_PC{pc_n}_{benchmark}.jpg')
                print(f'difference PC{pc_n} {benchmark} Done!')
                plt.close()

            elif map_type == 'ratio':
                width = 20
                ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                ax.set_facecolor(facecolor) ## light yellow
                prefectures_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=prefecture_lw)

                result_column = []
                for i in range(len(BC_PCA_result_column)):
                    if BC_PCA_result_column[i] >= BMA_result_column[i]:
                        result_column.append(BC_PCA_result_column[i] / BMA_result_column[i] * 100.0)
                    else:
                        result_column.append(BMA_result_column[i] / BC_PCA_result_column[i] * -100.0)

                norm = TwoSlopeNorm(vmin=math.floor(min(min(result_column),-0.5)), vcenter=0, vmax=math.ceil(max(max(result_column),0.5)))
                if benchmark in ['RMSE', 'MAE']:
                    cmap = plt.cm.seismic_r  
                else:
                    cmap = plt.cm.seismic

                basins_gdf['color'] = basins_gdf['File_num'].apply(get_basin_ratio_color)
                basins_gdf.loc[:, 'color'] = basins_gdf['color'].fillna('none')
                basins_gdf.plot(ax=ax, facecolor=basins_gdf['color'].apply(lambda x: x if x != 'none' else (1, 1, 1, 0)), edgecolor='black', linewidth=basin_lw)

                apply_labels(ax)
                add_scalebar(ax)
                add_north_arrow(ax) 

                ### COLOR BAR ###
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(benchmark, fontsize=fs)
                cbar.ax.tick_params(labelsize=fs)

                plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
                plt.title(f'Ratio Map ({PCA_type} to BMA){benchmark} PC{pc_n}\n', fontsize=fs)

                plt.tight_layout()
                plt.savefig(f'{output_folder_map_type}/basin_{map_type}_map_PC{pc_n}_{benchmark}.jpg')
                print(f'ratio PC{pc_n} {benchmark} Done!')
                plt.close()

print("DONE!!!")