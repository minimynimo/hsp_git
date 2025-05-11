import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

loc = "US"

results_path = '/data0/funato/0_out/99_out/US/RC_700_0.001_bin_flow/predict/RC_predict_r700_s1_sr0.4_rr0.001_eva.csv'

output_dir = f'/data0/funato/0_out/0_fig/{loc}/bin_flow'

# Read the CSV file with low_memory=False to suppress DtypeWarning
data = pd.read_csv(results_path, header=0, low_memory=False)

# Ensure only numeric data is used for histograms
data = data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

# Load the arid file numbers
arid_file_path = '/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv'
arid_files = pd.read_csv(arid_file_path)["File_num"].tolist()

# Create grouped figures for histograms (50 per PNG, 10 columns x 5 rows)
os.makedirs(output_dir, exist_ok=True)
group_size = 50
num_rows = data.shape[0]

for group_start in range(0, num_rows, group_size):
    group_end = min(group_start + group_size, num_rows)
    fig, axes = plt.subplots(5, 10, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, row in enumerate(data.iloc[group_start:group_end].iterrows()):
        ax = axes[i]
        file_num = group_start + i + 1
        values = row[1].dropna()
        bins = 20
        hist, bin_edges = np.histogram(values, bins=bins, range=(-1, 2))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Separate colors for values < 0 and >= 0
        colors = ['red' if center < 0 else 'blue' for center in bin_centers]
        ax.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), color=colors, alpha=0.7)
        
        # Change background to cream if file_num is in arid_files
        if file_num in arid_files:
            ax.set_facecolor('cornsilk')
        
        ax.set_title(f'Timeseries {file_num}', fontsize=10)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

    # Hide unused subplots
    for j in range(group_end - group_start, len(axes)):
        axes[j].axis('off')

    output_file = os.path.join(output_dir, f'histograms_{group_start + 1}_to_{group_end}.png')
    plt.savefig(output_file)
    plt.close()

    print(f"Histograms saved to {output_file}")

