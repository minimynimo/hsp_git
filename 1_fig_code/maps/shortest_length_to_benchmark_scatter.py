import matplotlib 
matplotlib.use('Agg')
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Settings
start_val_list = [1]
step_val_list = list(range(2, 136))
nums = len(step_val_list)

####
nocal = True
file_tag = 'r700_s1_sr0.4_rr0.001'
####

if nocal:
    nocal_tag = '_nocal'
else:
    nocal_tag = ''

cal_eva = ['train', 'test']
max_median = ['Max', 'Median']  ## FOR MAX OR MEDIAN OF THE DISTANCE BETWEEN THE BASINS
# USE MEDIAN FOR BENCHMARK VALUES
output_dir = f'/data0/funato/0_out/0_fig/benchmark/BC_PUB{nocal_tag}/scatter'
distance_dir = '/data0/funato/3_gis_data/closest'

os.makedirs(output_dir, exist_ok=True)

# Benchmarks to plot
benchmark_list = ["KGE", "NSE", "logNSE", "E1", "Erel", "VE", "d", "RMSE", "MAE"]

# Create lists to store the data
median_shortest_distance = []
max_shortest_distance = [ ]
benchmark_medians = {benchmark: {'train': [], 'test': []} for benchmark in benchmark_list}

for start_val in start_val_list:
    for step_val in step_val_list:
        distance_file_path = f'{distance_dir}/closest_basins_start{start_val}_step{step_val}.csv'
        if os.path.exists(distance_file_path):
            df_distance = pd.read_csv(distance_file_path)
            filtered_df = df_distance[df_distance['Distance'] != 0]
            for max_med in max_median:
                if max_med == 'Max':
                    max_distance = filtered_df['Distance'].max() / 1000
                else:
                    median_distance = filtered_df['Distance'].median() / 1000  # Convert to km
            
        valid_data = False
        for ce in cal_eva:
            for benchmark in benchmark_list:
                benchmark_file_path = f'/data0/funato/0_out/BC_PUB{nocal_tag}/{ce}_basin/results/BC_PUB_results_{file_tag}_start{start_val}_step{step_val}_eva.csv'
                if os.path.exists(benchmark_file_path):
                    df_benchmark = pd.read_csv(benchmark_file_path)
                    col_name = f'BC_PUB_{file_tag}_start{start_val}_step{step_val}_{benchmark}_eva'
                    if col_name in df_benchmark.columns:
                        median_benchmark_value = df_benchmark[col_name].median()
                        benchmark_medians[benchmark][ce].append(median_benchmark_value)
                        valid_data = True
        if valid_data:
            median_shortest_distance.append(median_distance)
            max_shortest_distance.append(max_distance)

for benchmark in benchmark_list:
    for ce in cal_eva:
        for max_med in max_median:
            plt.figure(figsize=(8, 8))
            if max_med == 'Max':
                x_values = max_shortest_distance
                y_values = benchmark_medians[benchmark][ce]
            else:
                x_values = median_shortest_distance
                y_values = benchmark_medians[benchmark][ce]
            plt.scatter(max_shortest_distance, benchmark_medians[benchmark][ce], c='royalblue', label=ce)

            plt.title(f'Scatter plot of {max_med} Basin Distance vs {benchmark}{nocal_tag} ({ce})', fontsize=15)
            plt.xlabel(f'{max_med} Basin Distance [km]', fontsize=12)
            plt.ylabel(f'Median {benchmark} {ce}', fontsize=12)

            #min_val = min(min(x_values), min(y_values))
            #max_val = max(max(x_values), max(y_values))           
            
            plt.xlim(0, )
            if benchmark in ['KGE', 'NSE', 'E1', 'Erel', 'VE', 'd']:
                plt.ylim(min(y_values),1)
            elif benchmark in ['RMSE', 'MAE']:
                plt.ylim(0, )
            #plt.ylim(min_val, max_val)
            
            # Add y=x line
            #plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            cur_output_dir = os.path.join(output_dir, f'{ce}/{max_med}')
            os.makedirs(cur_output_dir, exist_ok=True)
            output_file = os.path.join(cur_output_dir, f'scatter_{max_med}_{benchmark}{nocal_tag}_{ce}_{nums}.jpg')
            plt.savefig(output_file)
            plt.close()

print("Done!")
