import os
import pandas as pd
import matplotlib.pyplot as plt


cal_eva = ['cal', 'eva']
benchmark_list = ['KGE', 'NSE', 'E1', 'VE','d', 'RMSE', 'MAE']

#model = 'RandomForest'  # 'RandomForest', 'XGBoost', 'TCN'
#model = 'XGBoost'  # 'RandomForest', 'XGBoost', 'TCN'
model = 'TCN'  # 'RandomForest', 'XGBoost', 'TCN'

output_dir = f'/data0/funato/hsp/0_out/1_fig/hyperparam/{model}'
os.makedirs(output_dir, exist_ok=True)

if model == 'RandomForest':
    # Random forest
    #max_depth_list = [1,2, 5,10, 15, 20,25]
    #max_features_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = '/data0/funato/hsp/0_out/11_RandomForest/JP/hyperparam/results'
elif model == 'XGBoost':
    #XGBoost
    #n_estimators_list = [200,500,1000,2000,5000,10000]
    #learning_rate_list = [0.001,0.005,0.01,0.05,0.1,0.5,0.7,0.9,1.0]
    results_dir = '/data0/funato/hsp/0_out/12_XGBoost/JP/hyperparam/results'
elif model == 'TCN':
    #TCN
    #filter_size_list = [32, 64, 128]
    #kernel_size_list = [4,6, 8]
    #nb_stacks_list = [1,2,3]
    results_dir = '/data0/funato/hsp/0_out/13_TCN/JP/hyperparam/results'

# Collect results
plot_df_cal = []
plot_df_eva = []
file_names_cal = []
file_names_eva = []

benchmark_limits = {
    'KGE': {'min': 0., 'max': 1},
    'NSE': {'min': -0.5, 'max': 1},
    'E1': {'min': -0.5, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0., 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

for fname in os.listdir(results_dir):
    fpath = os.path.join(results_dir, fname)
    if not os.path.isfile(fpath):
        continue
    if fname.endswith('cal.csv'):
        df = pd.read_csv(fpath)
        plot_df_cal.append(df)
        fname = fname.replace('_cal.csv', '')
        if 'results_' in fname:
            fname = fname.split('results_')[-1]
        file_names_cal.append(fname)
    elif fname.endswith('eva.csv'):
        df = pd.read_csv(fpath)
        plot_df_eva.append(df)
        fname = fname.replace('_eva.csv', '')
        if 'results_' in fname:
            fname = fname.split('results_')[-1]
        file_names_eva.append(fname)

print(f'plot_df_cal: {len(plot_df_cal)} files')
print(f'plot_df_eva: {len(plot_df_eva)} files')

# For each benchmark, plot boxplots for cal and eva
for benchmark in benchmark_list:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    # CAL
    data_cal = []
    for df in plot_df_cal:
        col = [c for c in df.columns if benchmark in c]
        if col:
            data_cal.append(df[col[0]].dropna())
        else:
            data_cal.append(pd.Series(dtype=float))
    # Only plot if data and labels match
    axes[0].boxplot(data_cal, labels=file_names_cal, vert=True)
    axes[0].set_title(f'cal - {benchmark}')
    axes[0].set_ylabel(benchmark)
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    axes[0].grid(True)

    # EVA
    data_eva = []
    for df in plot_df_eva:
        col = [c for c in df.columns if benchmark in c]
        if col:
            data_eva.append(df[col[0]].dropna())
        else:
            data_eva.append(pd.Series(dtype=float))
    axes[1].boxplot(data_eva, labels=file_names_eva, vert=True)
    axes[1].set_title(f'eva - {benchmark}')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'{output_dir}/{model}_{benchmark}.png')
    plt.close()
