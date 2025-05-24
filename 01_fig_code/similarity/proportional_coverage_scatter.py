import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import seaborn as sns

non_arid_files = True
#loc_list = ["US", "GB"]
loc_list = ["JP", "GB"]

fs = 19
markersize = 7

output_dir = f"/data0/funato/0_out/0_fig/scatter/propotional_coverage"
os.makedirs(output_dir, exist_ok=True)

def get_attribute_dir(loc):
    return f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

def calculate_proportional_index(attribute_values1, attribute_values2, columns_drop1, columns_drop2):
    common_columns = set(attribute_values1.columns) & set(attribute_values2.columns)
    common_columns -= set(columns_drop1) | set(columns_drop2)
    
    proportional_indices = []
    for col in common_columns:
        range1 = attribute_values1[col].max() - attribute_values1[col].min()
        range2 = attribute_values2[col].max() - attribute_values2[col].min()
        intersection_range = min(attribute_values1[col].max(), attribute_values2[col].max()) - max(attribute_values1[col].min(), attribute_values2[col].min())
        
        if range1 > 0 and range2 > 0 and intersection_range > 0:
            prop_index1 = intersection_range / range1
            prop_index2 = intersection_range / range2
        else:
            prop_index1 = 0
            prop_index2 = 0
        
        proportional_indices.append((col, prop_index1, prop_index2))
    
    return pd.DataFrame(proportional_indices, columns=['Characteristic', f'{loc_list[0]} ⊆ {loc_list[1]}', f'{loc_list[0]} ⊇ {loc_list[1]}'])

def get_colors(attribute_group, proportional_indices_df, color_group):
    colors = [color_group[attribute_group.loc[attribute_group['Attributes'] == char, 'AttributeGroup'].values[0]] 
              for char in proportional_indices_df['Characteristic']]
    return colors

def compute_normalized_pdf(data, bins=30):
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, 1), density=False)
    pdf = counts / counts.sum()  # Normalize the histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get the center of each bin
    return bin_centers, pdf

if "US" in loc_list and "GB"in loc_list:
    attribute_group = pd.read_csv(f'/data0/funato/3_gis_data/catchment_attribute_group_caravan.csv')
    color_group = {
        'Physiograph': 'tab:red',
        'Climate': 'tab:orange',
        'Hydrology': 'tab:blue',
        'Land Cover': 'tab:green',
        'Soils & Geology': 'tab:brown',
        'Anthropogenic': 'tab:purple',
    }
    ## US ##
    loc1 = "US"
    file_tot_num1 = 667
    if non_arid_files:
        arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
        arid_file_list = pd.read_csv(arid_file_list)
        arid_file_list = arid_file_list['File_num'].tolist()
        arid_file_list.sort()
        file_list1 = [i for i in range(1,file_tot_num1+1) if i not in arid_file_list]
    else:
        file_list1 = [i for i in range(1,file_tot_num1+1)]
    basin_data_df1 = pd.read_csv("/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/pub_region_list_ver2_0.csv")
    attribute_values1 = pd.read_csv(f'{get_attribute_dir(loc1)}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop1 = ['File_num','gauge_id','gauge_name','country', 'area_fraction_used_for_aggregation']

    ## GB ##
    loc2 = "GB"
    file_tot_num2 = 396
    file_list2 = [i for i in range(1,file_tot_num2+1)]
    basin_data_df2 = pd.read_csv(f"/data0/funato/3_gis_data/{loc2}/0_data/river_basin/dataset_{loc2}/file_num_valid_small.csv")
    attribute_values2 = pd.read_csv(f'{get_attribute_dir(loc2)}/camelsgb_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop2 = ['File_num','gauge_id','gauge_name','country', 'area_fraction_used_for_aggregation']

elif "GB" in loc_list and "JP" in loc_list:
    attribute_group = pd.read_csv(f'/data0/funato/3_gis_data/catchment_attribute_group_JP.csv')
    color_group = {
        'Physiograph': 'tab:red',
        'Climate': 'tab:orange',
        'Landuse': 'tab:brown'
    }
    ## JP ##
    loc1 = "JP"
    file_tot_num1 = 87
    basin_data_df1 = pd.read_csv("/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    attribute_values1 = pd.read_csv(f'{get_attribute_dir(loc1)}/basin_data_JP_cleaned_limited_R2R.csv', encoding= 'UTF-8')
    columns_drop1 = ['File_num','grdc_no','river','station','lat_org','long_org']

    ## GB ##
    loc2 = "GB"
    file_tot_num2 = 396
    file_list2 = [i for i in range(1,file_tot_num2+1)]
    basin_data_df2 = pd.read_csv(f"/data0/funato/3_gis_data/{loc2}/0_data/river_basin/dataset_{loc2}/file_num_valid_small.csv")
    attribute_values2 = pd.read_csv(f'{get_attribute_dir(loc2)}/camelsgb_attribute_combined_R2R.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop2 = ['File_num','gauge_id','gauge_name','country']

else:
    print("Invalid loc_list")
    exit()

# Calculate proportional indices
proportional_indices_df = calculate_proportional_index(attribute_values1, attribute_values2, columns_drop1, columns_drop2)

# Get colors for each point based on AttributeGroup
colors = get_colors(attribute_group, proportional_indices_df, color_group)


# Plotting
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6], wspace=0.05, hspace=0.05)
ax_scatter = fig.add_subplot(gs[1, 0])  # Main scatter plot
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)  # Top PDF
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)  # Right PDF

# Extract the data
x_us_subset_gb = proportional_indices_df[f'{loc_list[0]} ⊆ {loc_list[1]}']
x_us_superset_gb = proportional_indices_df[f'{loc_list[0]} ⊇ {loc_list[1]}']

# Compute normalized PDF for the top plot (US ⊆ GB)
bin_centers_top, pdf_top = compute_normalized_pdf(x_us_subset_gb)

# Compute normalized PDF for the right plot (US ⊇ GB)
bin_centers_right, pdf_right = compute_normalized_pdf(x_us_superset_gb)

# Plot Top PDF (US ⊆ GB)
ax_top.bar(bin_centers_top, pdf_top, width=bin_centers_top[1] - bin_centers_top[0], color='gray', alpha=0.5, align='center')
ax_top.set_ylim(0, None)  # Let seaborn handle the correct scaling
ax_top.tick_params(axis='y', labelsize=fs*0.8)  # Set y-ticks for the top plot
ax_top.set_ylabel('PDF', fontsize=fs)
ax_top.xaxis.set_visible(False)  # Hide x-axis completely


# Plot Right PDF (US ⊇ GB)
ax_right.barh(bin_centers_right, pdf_right, height=bin_centers_right[1] - bin_centers_right[0], color='gray', alpha=0.5, align='center')
ax_right.set_xlim(0, None)
ax_right.tick_params(axis='x', labelsize=fs*0.8)  # Set x-ticks for the right plot
ax_right.set_xlabel('PDF', fontsize=fs)
ax_right.yaxis.set_visible(False)  # Hide x-axis completely


# Main scatter plot
sc = ax_scatter.scatter(
    proportional_indices_df[f'{loc_list[0]} ⊆ {loc_list[1]}'],
    proportional_indices_df[f'{loc_list[0]} ⊇ {loc_list[1]}'],
    c=colors, s=markersize**2, alpha=0.8
)
ax_scatter.set_xlabel(f'{loc_list[0]} ⊆ {loc_list[1]}', fontsize=fs)
ax_scatter.set_ylabel(f'{loc_list[0]} ⊇ {loc_list[1]}', fontsize=fs)
ax_scatter.set_ylim(-0.05, 1.05)
ax_scatter.set_xlim(-0.05, 1.05)

# Ensure the x and y ticks are visible
ax_scatter.set_xticks(np.arange(0, 1.1, 0.2))  # Set x-ticks
ax_scatter.set_yticks(np.arange(0, 1.1, 0.2))  # Set y-ticks

# Make sure both x and y ticks are enabled
ax_scatter.tick_params(axis='x', which='both', bottom=True, top=True, labelsize=fs*0.8)  # Show both bottom and top ticks
ax_scatter.tick_params(axis='y', which='both', left=True, right=True, labelsize=fs*0.8)  # Show both left and right ticks

ax_scatter.set_aspect('equal', adjustable='box')

fig.suptitle('Proportional Index Coverage Scatter Plot\n', fontsize=fs)

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=markersize*1.5) for color in color_group.values()]
labels = color_group.keys()
legend = ax_scatter.legend(handles, labels, title='Attribute Group', fontsize=fs*0.7, title_fontsize=fs*0.8, loc='lower center')
legend.get_frame().set_edgecolor('silver')
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_alpha(0.7)

#plt.tight_layout()
plt.savefig(f"{output_dir}/proportional_index_scatter_{'_'.join(loc_list)}.png")

# Save proportional indices
proportional_indices_df.to_csv(f"{output_dir}/proportional_indices_{'_'.join(loc_list)}.csv", index=False)
plt.close()

print("Done")