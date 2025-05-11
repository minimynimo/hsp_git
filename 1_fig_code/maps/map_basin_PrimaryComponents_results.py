# Description: This script generates maps of the river basins colored by the Principle components values of the basins.
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
from matplotlib.colors import TwoSlopeNorm, PowerNorm
from warnings import filterwarnings
filterwarnings('ignore')

data_folder = '/data0/funato/3_gis_data/0_data/'
output_dir = '/data0/funato/0_out/0_fig/map/PCA/PrinCompMap'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
PrinComp_path = '/data0/funato/4_PCA/PCA_primary_components_bybasin.csv'
PrinComp_df = pd.read_csv(PrinComp_path)

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
    ax.set_xlabel('Longitude', fontsize=25)
    ax.set_ylabel('Latitude', fontsize=25)
    lon_labels = [f"{int(label)}°E" for label in ax.get_xticks()]
    lat_labels = [f"{int(label)}°N" for label in ax.get_yticks()]
    ax.set_xticklabels(lon_labels, fontsize=25)
    ax.set_yticklabels(lat_labels, fontsize=25)
    ax.tick_params(axis='both', which='both', labelsize=25)
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

for PrinComp in range(13,16):
    result_df = PrinComp_df[['file_num', f'Principal Component {PrinComp}']]

    width = 20  # Adjust this for the desired width of the figure
    ax = setup_figure(width, lat_min, lat_max, lon_min, lon_max)

    facecolor = 'white'
    ax.set_facecolor(facecolor)
    prefectures_gdf.plot(ax=ax, facecolor = facecolor, edgecolor='black', linewidth=1.1)

    ### if map_type = 'result' then color based on the Principle Component values of the basins by using a colorbar        
    result_column = result_df[f'Principal Component {PrinComp}'].tolist()
    norm = TwoSlopeNorm(vmin=min(result_column), vcenter = 0, vmax=max(result_column))

    """
    if PrinComp not in [4, 5, 6, 7, 8, 10, 12, 13,14]:
        norm = plt.Normalize(vmin=min(result_column), vmax=max(result_column))
    else:
        norm = PowerNorm(gamma=0.5, vmin=min(result_column), vmax=max(result_column))
    """
    
    cmap = plt.cm.bwr
    basins_gdf = basins_gdf.merge(result_df, left_on='File_num', right_on='file_num', how='left')
    basins_gdf['color'] = basins_gdf[f'Principal Component {PrinComp}'].apply(lambda x: cmap(norm(x)) if pd.notnull(x) else 'dimgray')

    basins_gdf.plot(ax=ax, facecolor = basins_gdf['color'], edgecolor='black', linewidth=1.8)

    apply_labels(ax)
    add_scalebar(ax)
    add_north_arrow(ax) 

    ### COLOR BAR ###
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'PC{PrinComp}', fontsize=20)
    cbar.ax.tick_params(labelsize=15)

    plt.grid(True)

    fs = 23
    plt.title(f'Principle Component {PrinComp} Map\n', fontsize=fs)

    plt.tight_layout()

    plt.savefig(f'{output_dir}/PC{PrinComp}_map.jpg', bbox_inches='tight')
    plt.close()

    print(f'PC{PrinComp} map is done!')
print("DONE!!!")