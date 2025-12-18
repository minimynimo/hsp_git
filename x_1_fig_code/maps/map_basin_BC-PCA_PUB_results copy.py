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
start_val_list = [68]
n_components = 2
step_val_list = [2,4,10,20,30,40,50] # Train: 67, 27, 13, 7, 5, 3, 3
Train_basin_num_list = [67, 33, 13, 7, 5, 3, 3]
RC_nocal = True
BMA_nocal = True
file_tag = 'r700_s1_sr0.4_rr0.001'
PCA_type = 'BC-PCA-lasso_PUB'

facecolor = 'white'
fs = 25
###
# end val is always 135
end_val = 136

all_basins = set(range(1, end_val))

if RC_nocal:
    nocal_tag = '_RCnocal'
else:
    nocal_tag = ''
if BMA_nocal:
    nocal_tag += '_BMAnocal'


##base: maps where the trained and test besins are
##result: maps the benchmark values for the trained basins
##dif: maps difference of calibrated BMA benchmark values with PUB values
#map_type_list = ['result', 'dif','base']  ###base map already generated for start1###
map_type_list = ['base','dif','result']  #
map_type_list = ['ratio']  #

data_folder = '/data0/funato/3_gis_data/JP/0_data/'
PCA_result_data_dir = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/{PCA_type}_nocal/results' 

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
        return '#FFFFF0' ##For the train basins 
    
def get_basin_dif_color(file_num):
    BC_PUB_index = BC_PUB_result_df[BC_PUB_result_df['file_num'] == file_num].index
    BMA_PUB_index = BMA_PUB_result_df[BMA_PUB_result_df['file_num'] == file_num].index    
    if not BC_PUB_index.empty: ## test basin
        value = BC_PUB_result_column[BC_PUB_index[0]] - BMA_PUB_result_column[BMA_PUB_index[0]]
        return cmap(norm(value))
    else:
        return '#FFFFF0' ##For the train basins 

def get_hatch(file_num):
    index = BC_PUB_result_df[BC_PUB_result_df['file_num'] == file_num].index    
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

for start_val in start_val_list:
    for step_val in step_val_list:
        BMA_nocal_path = f'/data0/funato/0_out/99_out/0_JP_MARRMoT_44/BMA_PUB_nocal/start68/test_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
        BMA_result_df = pd.read_csv(BMA_nocal_path)
        output_folder = f'/data0/funato/0_out/0_fig/map_{PCA_type}/step{step_val}/'

        # Define the list of basins to be colored in light orange
        train_basins_file = f'/data0/funato/3_gis_data/train_list/start{start_val}/training_basin_start{start_val}_step{step_val}.csv'
        train_basins_df = pd.read_csv(train_basins_file)
        train_basins = train_basins_df['Basin'].tolist()
        test_basins = list(all_basins - set(train_basins))

        train_basins_gdf = basins_gdf.loc[basins_gdf['File_num'].isin(train_basins)]
        test_basins_gdf = basins_gdf.loc[~basins_gdf['File_num'].isin(train_basins)]

        for map_type in map_type_list:
            if map_type == 'result': ## result
                output_folder_map_type = f'{output_folder}/result_map/'
            elif map_type == 'dif': ## difference
                output_folder_map_type = f'{output_folder}/dif_map/start{start_val}_step{step_val}'
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

                plt.title(f'Base Map start{start_val} step{step_val}\n', fontsize=fs)
                plt.tight_layout()
                plt.savefig(f'{output_folder_map_type}/basin_{map_type}_map_start{start_val}_step{step_val}.png')
                print('base Done!')

                plt.close()
            
            else: ## map_type = 'result' or 'dif'
                ## BMA PUB results 
                BMA_PUB_path = f'/data0/funato/0_out/BMA_PUB_nocal/start{start_val}/test_basin/results/BMA_PUB_results_start{start_val}_step{step_val}_eva.csv'
                BMA_PUB_result_df = pd.read_csv(BMA_PUB_path)

                BC_PUB_path = f'/data0/funato/0_out/BC_PUB{nocal_tag}/start{start_val}/test_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
                BC_PUB_result_df = pd.read_csv(BC_PUB_path)

                for benchmark in benchmark_list:
                    ### if map_type = 'result' then color based on the benchmark by using a colorbar

                    BMA_PUB_result_column = BMA_PUB_result_df[f'BMA_PUB_start{start_val}_step{step_val}_{benchmark}_eva'].tolist()
                    BC_PUB_result_column = BC_PUB_result_df[f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'].tolist()
                    
                    ### RESULT MAP ###
                    if map_type == 'result':
                        for bma_bc in ['BC', 'BMA']:
                            result_output_folder = f'{output_folder_map_type}/{bma_bc}_PUB/start{start_val}_step{step_val}'
                            if not os.path.exists(result_output_folder):
                                os.makedirs(result_output_folder, exist_ok=True)

                            result_df = BMA_PUB_result_df if bma_bc == 'BMA' else BC_PUB_result_df
                            result_column = BMA_PUB_result_column if bma_bc == 'BMA' else BC_PUB_result_column

                            width = 20  # Adjust this for the desired width of the figure
                            ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                            facecolor = '#FFFFF0'
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
                            train_basins_gdf['color'] = '#FFFFF0'  # Light orange
                            train_basins_gdf['hatch'] = train_basins_gdf['File_num'].apply(get_hatch)

                            test_basins_gdf.loc[:, 'color'] = test_basins_gdf['color'].fillna('none')
                            train_basins_gdf.loc[:, 'color'] = train_basins_gdf['color'].fillna('none')

                            test_basins_gdf.plot(ax=ax, facecolor = test_basins_gdf['color'], edgecolor='black', linewidth=1.0)
                            for _, row in train_basins_gdf.iterrows():
                                row_geom = row['geometry']
                                row_color = row['color']
                                
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
                            legend_elements = [Patch(facecolor='#FFFFF0', edgecolor='black', label='Train',hatch='//////', linewidth=1.0)]
                            legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=30)
                            legend.get_frame().set_facecolor('white')
                            legend.get_frame().set_alpha(1)

                            plt.grid(True)

                            if bma_bc == 'BMA':
                                plt.title(f'BMA_PUB Result Map {benchmark}{nocal_tag} start{start_val} step{step_val}\n', fontsize=fs)
                            else:
                                plt.title(f'BC_PUB Result Map {benchmark}{nocal_tag} start{start_val} step{step_val}\n', fontsize=fs)

                            plt.tight_layout()

                            if bma_bc == 'BMA':
                                plt.savefig(f'{result_output_folder}/BMA_PUB_basin_{map_type}_map_start{start_val}_step{step_val}_{benchmark}.jpg')
                            else:
                                plt.savefig(f'{result_output_folder}/BC_PUB_basin_{map_type}_map_start{start_val}_step{step_val}_{benchmark}.jpg')
                            print(f'results start{start_val} step{step_val} {bma_bc}_PUB {benchmark}{nocal_tag} Done!')
                            plt.close()     

                    ### DIFFERENCE MAP ###
                    elif map_type == 'dif':
                        width = 20  # Adjust this for the desired width of the figure
                        ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

                        ax.set_facecolor('#FFFFF0') ## light yellow
                        prefectures_gdf.plot(ax=ax, facecolor='#FFFFF0', edgecolor='black', linewidth=1.3)

                        result_column = []
                        result_df = BC_PUB_result_df - BMA_PUB_result_df

                        for file_num in test_basins:
                            BC_PUB_index = BC_PUB_result_df[BC_PUB_result_df['file_num'] == file_num].index    
                            BMA_PUB_index = BMA_PUB_result_df[BMA_PUB_result_df['file_num'] == file_num].index
                            BC_PUB_value = BC_PUB_result_column[BC_PUB_index[0]]
                            BMA_PUB_value = BMA_PUB_result_column[BMA_PUB_index[0]]
                            value = BC_PUB_value - BMA_PUB_value
                            result_column.append(value) 

                        norm = TwoSlopeNorm(vmin=math.floor(min(min(result_column),-0.5)), vcenter=0, vmax=math.ceil(max(max(result_column),0.5)))
                        if benchmark in ['RMSE', 'MAE']:
                            cmap = plt.cm.seismic_r  
                        else:
                            cmap = plt.cm.seismic

                        # Apply the color function to each row in the GeoDataFrame
                        test_basins_gdf['color'] = test_basins_gdf['File_num'].apply(get_basin_dif_color)
                        train_basins_gdf['color'] = '#FFFFF0'  # Light yellow
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
                        legend_elements = [Patch(facecolor='#FFFFF0', edgecolor='black', label='Train',
                                                hatch='//////', linewidth=1.5)]
                        legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=30)
                        legend.get_frame().set_facecolor('white')
                        legend.get_frame().set_alpha(1)

                        plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
                        plt.title(f'Difference Map (PUB{nocal_tag} - BMA_nocal){benchmark} start{start_val} step{step_val}\n', fontsize=fs)

                        plt.tight_layout()
                        plt.savefig(f'{output_folder_map_type}/basin_{map_type}_map_start{start_val}_step{step_val}_{benchmark}.png')
                        print(f'differnce start{start_val} step{step_val} {benchmark}{nocal_tag} Done!')
                        plt.close()

print("DONE!!!")