import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

start_val_list = [68]
step_val_list = list(range(2, 34))
nums = len(step_val_list)
end_val = 136

plot_df = pd.DataFrame(
    np.full((end_val, len(step_val_list)), np.nan, dtype=object), 
    columns=[f'start{start_val}_step{step_val}' for start_val in start_val_list for step_val in step_val_list]
)

for start_val in start_val_list:
    for step_val in step_val_list:
        file_path = f'/data0/funato/3_gis_data/closest/start{start_val}/closest_basins_start{start_val}_step{step_val}.csv'
        
        df = pd.read_csv(file_path)
        filtered_df = df[df['Distance'] != 0]
        distances = filtered_df['Distance'].values

        distances = list(distances)
        if len(distances) < end_val:
            distances += [np.nan] * (end_val - len(distances))  # Extend with NaN values if shorter
        
        distances = distances[:end_val]
        
        plot_df[f'Str{start_val}_Stp{step_val}'] = distances

plot_df = plot_df / 1000

fs = 13   
# Create a box plot
fig, ax = plt.subplots(figsize=(19, 6))
plot_df.boxplot(ax=ax)
plt.xticks(rotation=90, fontsize=fs)
plt.yticks(fontsize=fs)
plt.title(f'Distance from test basin to the closest train basin start{start_val}\n', fontsize=fs)
plt.ylabel('Distance [km]', fontsize=fs)
plt.xlabel('start_val_step_val', fontsize=fs)
plt.ylim(0, )

plt.tight_layout()
plt.savefig(f'/data0/funato/0_out/0_fig/shortest_length/shortest_length_start{start_val}_{nums}_boxplot.png')

print('Done!')