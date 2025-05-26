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

warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")
warnings.simplefilter(action='ignore', category=UserWarning)

###
n_components = 2

file_tag = 'r700_s1_sr0.4_rr0.001'
PCA_type = 'BC-PCA-lasso_PUB_region'

loc = "JP"
# loc = "US"
facecolor = 'white'
fs = 25

reservoir_size = 700
###

if loc == "JP":
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
elif loc == "US":
    file_tot_num = 671
    start_val_list = [336]
    #step_val_list = list(range(2,337))
    step_val_list = [2,5,10,21,42,84,168,337]


region_list = ["Hokkaido", "N_Tohoku","S_Tohoku", "Kanto", "N_Chubu","S_Chubu", "Kinki", "Chugoku", "Shikoku", "Kyushu"]
region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/JP/0_data/river_basin/dataset/pub_region_list.csv') # might not use

all_basins = set(range(1, file_tot_num + 1))

##base: maps where the trained and test besins are
##result: maps the benchmark values for the trained basins
##dif: maps difference of calibrated BMA benchmark values with PUB values
#map_type_list = ['result', 'dif','base']  ###base map already generated for start1###
map_type_list = ['base','result']  #

data_folder = '/data0/funato/3_gis_data/JP/0_data/'
PCA_result_data_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB/region' 

benchmark_list = ["KGE","NSE","logNSE","E1", "VE", "d","RMSE","MAE"]

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

def get_basin_base_color(file_num):
    if file_num in train_basins:
        return '#FFCD43' # Light orange
    else:
        return '#87CED5' # Light blue

def get_basin_result_color(file_num):
    index = result_df[result_df['file_num'] == file_num].index    
    if not index.empty: ## test basin
        value = result_column[index[0]]        
        return cmap(norm(value))
    else:
        return 'white' ##For the train basins 

def get_hatch(file_num):
    index = result_df[result_df['file_num'] == file_num].index    
    if index.empty: ## test basin
        return '//////' ##For the train basins 
    else:
        return None

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

font_prop = FontProperties(size=20)

for pc_n in range(1, n_components + 1):
    for region in region_list:
        #BMA_nocal_path = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BMA_PUB_nocal/start68/test_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
        #BMA_result_df = pd.read_csv(BMA_nocal_path)
        output_folder = f'/data0/funato/0_out/0_fig/{loc}/map/map_{PCA_type}/'
        os.makedirs(output_folder, exist_ok=True)

        # Define the list of basins to be colored in light orange
        train_basins = sorted(region_list_df[region_list_df['Region'] == region]['File_num'].tolist())
        test_basins = [num for num in range(1, file_tot_num + 1) if num not in train_basins]

        train_basins_gdf = basins_gdf.loc[basins_gdf['File_num'].isin(train_basins)]
        test_basins_gdf = basins_gdf.loc[~basins_gdf['File_num'].isin(train_basins)]

        for map_type in map_type_list:
            if map_type == 'base': ## base
                output_folder_map_type = f'{output_folder}/base/PC{pc_n}/{region}/'
            if map_type == 'result': ## result
                output_folder_map_type = f'{output_folder}/result/PC{pc_n}/{region}/'
            if not os.path.exists(output_folder_map_type):
                os.makedirs(output_folder_map_type, exist_ok=True)

            if map_type == 'base':
                # Plot the shapefile
                width = 20  # Adjust this for the desired width of the figure
                ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                prefectures_gdf.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=1.0)
            
                # Apply the color function to each row in the GeoDataFrame
                test_basins_gdf['color'] = "#87CED5"  # Light blue
                train_basins_gdf['color'] = '#FFCD43'  # Light orange

                train_basins_gdf.loc[:, 'color'] = train_basins_gdf['color'].fillna('none')
                test_basins_gdf.loc[:, 'color'] = test_basins_gdf['color'].fillna('none')

                test_basins_gdf.plot(ax=ax, facecolor = test_basins_gdf['color'], edgecolor='black', linewidth=1.0)
                train_basins_gdf.plot(ax=ax, facecolor = train_basins_gdf['color'], edgecolor='black', linewidth=1.0)

                # Add legend
                legend_elements = [Line2D([0], [0], marker='s', color='w', label='Train', markersize=20, markerfacecolor='#FFCD43'),
                                Line2D([0], [0], marker='s', color='w', label='Test', markersize=20, markerfacecolor='#87CED5')]
                legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=35)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(1)

                apply_labels(ax)
                add_scalebar(ax)
                add_north_arrow(ax)

                plt.title(f'Base Map {region}\n', fontsize=fs)
                plt.tight_layout()
                plt.savefig(f'{output_folder_map_type}/basin_{map_type}_map_{region}_PC{pc_n}.png')
                print('base Done!')

                plt.close()

            if map_type == 'result':
                result_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB/region/{region}/test_basin/results/BC-PCA-lasso_PUB_results_r700_s1_sr0.4_rr0.001_rev_PC{pc_n}_eva.csv')
                for benchmark in benchmark_list:
                    result_column = result_df[f'BC-PCA-lasso_r700_s1_sr0.4_rr0.001_{benchmark}_eva']

                    width = 20  # Adjust this for the desired width of the figure
                    ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                    facecolor = 'white'
                    ax.set_facecolor(facecolor) ## light yellow
                    prefectures_gdf.plot(ax=ax, facecolor=facecolor, edgecolor='black', linewidth=1.3)
                    
                    # Add a colorbar ### INCLUDE 0: R1, NSE, KGE, VE
                    if benchmark in ['RMSE', 'MAE']:
                        norm = plt.Normalize(vmin=0, vmax=max(result_column))
                        cmap = plt.cm.inferno_r  # Reverse the colormap
                    elif benchmark in ['KGE', 'NSE', 'E1','VE', 'd']:
                        ## divide colors below and above 0 
                        min_value = min(result_column)
                        max_value = 1
                        norm = TwoSlopeNorm(vmin=math.floor(min(min_value,-0.5)), vcenter=0, vmax=max_value)
                        cmap = plt.cm.seismic
                    elif benchmark in ['logNSE', 'Erel']:
                        norm = plt.Normalize(vmin=min(result_column), vmax=1)
                        cmap = plt.cm.inferno
                    else:
                        norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
                        cmap = plt.cm.inferno

                    # Apply the color function to each row in the GeoDataFrame
                    test_basins_gdf['color'] = basins_gdf['File_num'].apply(get_basin_result_color)
                    train_basins_gdf['color'] = 'white'  # Light orange
                    train_basins_gdf['hatch'] = train_basins_gdf['File_num'].apply(get_hatch)

                    test_basins_gdf.loc[:, 'color'] = test_basins_gdf['color'].fillna('none')
                    train_basins_gdf.loc[:, 'color'] = train_basins_gdf['color'].fillna('none')

                    test_basins_gdf.plot(ax=ax, facecolor = test_basins_gdf['color'], edgecolor='black', linewidth=1.0)
                    for _, row in train_basins_gdf.iterrows():
                        row_geom = row['geometry']
                        row_color = row['color']
                        row_hatch = row['hatch']
                        
                        # Check if the geometry is a MultiPolygon
                        if row_geom.geom_type == 'MultiPolygon':
                            for polygon in row_geom.geoms:  # Use .geoms instead of direct iteration
                                ax.add_patch(plt.Polygon(list(polygon.exterior.coords), facecolor=row_color, edgecolor='black', linewidth=1.0, hatch=row_hatch))
                        else:  # For simple Polygons
                            ax.add_patch(plt.Polygon(list(row_geom.exterior.coords), facecolor=row_color, edgecolor='black', linewidth=1.0, hatch=row_hatch))

                    apply_labels(ax)
                    add_scalebar(ax)
                    add_north_arrow(ax) 

                    ### COLOR BAR ###
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(benchmark, fontsize=fs)
                    cbar.ax.tick_params(labelsize=fs)

                    # Add legend
                    legend_elements = [Patch(facecolor='white', edgecolor='black', label='Train',hatch='//////', linewidth=1.0)]
                    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=30)
                    legend.get_frame().set_facecolor('white')
                    legend.get_frame().set_alpha(1)

                    plt.grid(True)
                    plt.title(f'BC_PUB Region Result Map {benchmark} {region} PC{pc_n}\n', fontsize=fs)
                    plt.tight_layout()

                    plt.savefig(f'{output_folder_map_type}/BC_PUB_basin_{map_type}_map_{region}_PC{pc_n}_{benchmark}.jpg')
                    print(f'results {region} _PUB {benchmark} Done!')
                    plt.close()     

print("DONE!!!")