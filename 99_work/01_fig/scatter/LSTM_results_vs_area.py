import pandas as pd             
import matplotlib.pyplot as plt
import os

lstm_results = "/data0/funato/hsp2/0_out/LSTM/US/regional/results/LSTM_results_hs256_do0.4_e30_eva.csv"
basin_attr ="/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/camels_attribute_combined.csv"

lstm_results_df = pd.read_csv(lstm_results, index_col=0)
basin_attr_df = pd.read_csv(basin_attr, index_col=0)

output_fig_dir = "/data0/funato/hsp2/0_out/1_fig/US/scatter/"
os.makedirs(output_fig_dir, exist_ok=True)
output_dir = "/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US"
os.makedirs(output_dir, exist_ok=True)

# Merge the two DataFrames on the basin ID (assuming it's the index)
merged_df = lstm_results_df.merge(basin_attr_df, left_index=True, right_index=True)

# Create scatter plots for each attribute against the LSTM benchmark (e.g., KGE)
results = merged_df['KGE']  # Replace 'KGE' with the actual column name of the LSTM results if different
area = merged_df['area']  # Replace 'area' with the actual column name of the basin area if different

plt.figure(figsize=(10, 6))
plt.scatter(area, results, alpha=0.7)
plt.xscale('log')  # Use logarithmic scale for area if it spans several orders
plt.xlabel('Basin Area (log scale)')
plt.ylabel('LSTM KGE')
plt.title('Scatter Plot of LSTM KGE vs Basin Area')
plt.grid(True)
plt.tight_layout()
plt.ylim(-1, 1)  # Set y-axis limits for better visualization if KGE is the metric
plt.savefig(os.path.join(output_fig_dir, 'LSTM_KGE_vs_Basin_Area.png'), dpi=400)
plt.close() 


# create a csv list of basins under the area size of 1000 and 2000 km^2
small_basins_1000 = merged_df[merged_df['area'] < 1000].index.tolist()
small_basins_2000 = merged_df[merged_df['area'] < 2000].index.tolist()
with open(os.path.join(output_dir, 'small_basins_1000.csv'), 'w') as f:
    f.write("basin_id\n")
    for basin in small_basins_1000:
        f.write(f"{basin}\n")
with open(os.path.join(output_dir, 'small_basins_2000.csv'), 'w') as f:
    f.write("basin_id\n")
    for basin in small_basins_2000:
        f.write(f"{basin}\n")

