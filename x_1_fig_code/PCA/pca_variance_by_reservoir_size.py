import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
loc, ver = "US", 2
#loc, ver = "GB",2

non_arid_files = True
if loc != "US":
    non_arid_files = False
if non_arid_files:
    non_arid_dir = "non_arid/"

model_type = "BC-PCA-lasso"
fs = 20
n_components = 5

input_scale = 0.5
alpha = 0.1
spectral_radius = 0.4
spin = 1
ridge_param = 1.0

#reservoir_size_list = [5,10,50,100,200,250,300,500,700]
#reservoir_size_list = [10,100,200,300,500,700,1000]
reservoir_size_list = [10,100,200,300,500,700] #6 ,1000


output_dir = f"/data0/funato/0_out/0_fig/{loc}/ReservoirSize/{model_type}_s{spin}_sr{spectral_radius}_rr{ridge_param}/variance/"
os.makedirs(output_dir, exist_ok=True)

plot_df = pd.DataFrame(index=[f'PC{i}' for i in range(1, n_components + 1)])
for reservoir_size in reservoir_size_list:
    csv_path = f'/data0/funato/0_out/99_out/{loc}/{model_type}_hp/{non_arid_dir}PCA/explained_variance_ratio_PC5.csv'
    df = pd.read_csv(csv_path, index_col=0, header=0)
    row = df.loc[f'_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_is{input_scale}_a{alpha}']
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]  # Select the first row if there are multiple rows
    values = row.values.astype(float)  # Convert values to float
    if len(values) == n_components:
        plot_df[f'R{reservoir_size}'] = values
    else:
        print(f"Skipping reservoir size {reservoir_size} due to mismatched length")

print(plot_df)

fig, ax1 = plt.subplots(figsize=(10, 8))

bar_width = 0.95
num_reservoirs = len(reservoir_size_list)

colors = plt.get_cmap('turbo', num_reservoirs)
for idx, (reservoir_size, pc_values) in enumerate(plot_df.items()):
    pc_values = pc_values.values * 100
    ax1.bar(np.arange(1, n_components+1) - bar_width / 2 + idx * bar_width / num_reservoirs, pc_values, width=bar_width / num_reservoirs, label=reservoir_size, color=colors(idx))

ax1.set_xlabel('Principal Components', fontsize=fs)
ax1.set_ylabel('Explained Variance Ratio [%]', fontsize=fs)
ax1.set_title('Scree Plot by Reservoir Size\n', fontsize=fs)

# Set the legend with a white background and alpha = 1
legend = ax1.legend(loc='upper left', fontsize=fs*0.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1)

ax1.tick_params(axis='both', which='major', labelsize=fs)
ax2 = ax1.twinx()
ax2.tick_params(axis='both', which='major', labelsize=fs)

ax1.set_xticks(np.arange(1, n_components +1))
ax1.set_xticklabels([f'PC{i}' for i in range(1, n_components +1)], fontsize=fs)

for idx, (reservoir_size, pc_values) in enumerate(plot_df.items()):
    cumulative_variance = np.cumsum(pc_values.values) * 100
    ax2.plot(np.arange(1, n_components +1), cumulative_variance, marker='o', label=reservoir_size, color=colors(idx), markersize=4) 

ax2.set_ylabel('Cumulative Explained Variance [%]', fontsize=fs)

ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)

plt.tight_layout()

# Make the grid lines lighter
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)

plt.savefig(os.path.join(output_dir, 'scree_plot_reservoir_size_comp.png'))
print(f"saved at {os.path.join(output_dir, 'scree_plot_reservoir_size_comp.png')}")

print("DONE")