import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

model_list = ["RF", "XGB", "TCN", "LSTM"]
train_types = ["regional", "adapted","individual"]

benchmark_list = ["KGE", "NSE", "E1", "VE", "RMSE", "MAE"]

fs = 15

##LSTM_individual parameter
num_epochs = 100 # Number of training epochs
hidden_size = 16 # Number of LSTM cells
learning_rate = 0.005 # Learning rate used to update the weights
window_size = 365 # Length of the meteorological record provided to the network
batch_size = 256 # Number of samples in each batch
dropout_rate = 0.4 # Dropout rate of the final fully connected Layer [0.0, 1.0]

color_list = {
    "RF":   "#FF7F00",   # Orange
    "XGB":  "#9C4803",   # Dark Orange
    "TCN":  "#E41A1C",   # Red
    "LSTM": "#9808B2"    # Purple
}

# Markers for different training types
marker_list = {
    "RF":  'o',      # Circle
    "XGB":  's',    # Square
    "TCN":  '^',    # Triangle
    "LSTM": 'd'     # Diamond
}

# Line styles for different training types
linestyle_list = {
    "regional": "-",     # Solid
    "adapted": "--",       # Dotted
    "individual": ":",  # Dashed
}


benchmark_limits = {
    'KGE': {'min': 0, 'max': 1},
    'NSE': {'min': 0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': 0, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

benchmark_limits_cdf = {
    'KGE': {'min': 0, 'max': 1},
    'NSE': {'min': 0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': 0, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 5},
    'MAE': {'min': 0, 'max': 5}
}

output_dir_box = "/data0/funato/hsp/0_out/1_fig/benchmark/ML_training_comparison/box"
output_dir_cdf = "/data0/funato/hsp/0_out/1_fig/benchmark/ML_training_comparison/cdf"
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_cdf, exist_ok=True)

def get_file_path(model, train_type):
    if model == "TCN" and train_type == "regional":
        return f"/data0/funato/hsp/0_out/TCN/GB/regional/results/{model}_results_eva.csv"
    elif model == "LSTM" and train_type == "individual":
        return f"/data0/funato/hsp/0_out/LSTM/GB/individual/results/LSTM_results_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}_eva.csv"
    else:
        if train_type == "adapted":
            return f"/data0/funato/hsp/0_out/{model}/GB/adapted_396/results/{model}_results_eva.csv"
        else:
            return f"/data0/funato/hsp/0_out/{model}/GB/{train_type}/results/{model}_results_eva.csv"

for benchmark in benchmark_list:
    print(f"\n=== Processing {benchmark} ===")
    
    # === DEBUGGING: Check file existence ===
    print("File existence check:")
    for model in model_list:
        for train_type in train_types:
            file_path = get_file_path(model, train_type)
            exists = os.path.exists(file_path)
            df = pd.read_csv(file_path)
            has_benchmark = benchmark in df.columns
            data_count = len(df[benchmark].dropna())
            print(f"  {model} - {train_type}: Exists: {exists}, Has '{benchmark}': {has_benchmark}, Data count: {data_count}")
            if not exists or not has_benchmark or data_count == 0:
                print(f"    Warning: Missing or incomplete data for {model} - {train_type}")
    
    # === BOXPLOT FIGURE ===
    plt.figure(figsize=(12, 6))
    
    all_data = []
    all_positions = []
    all_labels = []
    box_colors = []
    
    position_counter = 0
    
    for model_idx, model in enumerate(model_list):
        for train_idx, train_type in enumerate(train_types):
            file_path = get_file_path(model, train_type)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if benchmark in df.columns:
                    data = df[benchmark].dropna()
                    if len(data) > 0:
                        all_data.append(data)
                        all_positions.append(position_counter)
                        all_labels.append(f"{model}_{train_type}")
                        box_colors.append(color_list[model])
                        position_counter += 1
    
    # Create boxplots only if we have data
    if all_data:
        bp = plt.boxplot(all_data, positions=all_positions, widths=0.6, patch_artist=True,
                        medianprops=dict(color='black'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set custom tick labels
        plt.xticks(all_positions, all_labels, rotation=45, ha='right')
    
    plt.ylabel(benchmark)
    plt.title(f'Box Plot: {benchmark}')
    plt.grid(axis='y', alpha=0.3)
    
    # Apply benchmark limits to box plot if available
    if benchmark in benchmark_limits:
        plt.ylim(benchmark_limits[benchmark]['min'], 
                 benchmark_limits[benchmark]['max'])
        print(f"Applied box plot limits: {benchmark_limits[benchmark]}")
    
    plt.tight_layout()
    
    # Save boxplot
    output_path_box = os.path.join(output_dir_box, f'fig_by_benchmark_{benchmark}_boxplot.png')
    plt.savefig(output_path_box, dpi=300, bbox_inches='tight')
    plt.close()
    
    # === CDF FIGURE with subplots per model ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Collect data for CDF plotting
    cdf_data = {}
    for model in model_list:
        cdf_data[model] = {}
        for train_type in train_types:
            file_path = get_file_path(model, train_type)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if benchmark in df.columns:
                    data = df[benchmark].dropna()
                    if len(data) > 0:
                        cdf_data[model][train_type] = data
    
    # Plot CDFs - one subplot per model
    for model_idx, model in enumerate(model_list):
        ax = axes[model_idx]
        
        for train_type in train_types:
            if train_type in cdf_data[model]:
                data = cdf_data[model][train_type]
                sorted_data = np.sort(data)
                cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                ax.plot(sorted_data, cdf_values, 
                        linestyle=linestyle_list[train_type],
                        label=train_type,
                        linewidth=2.5,
                        alpha=0.8,
                        color=color_list[model])
        
        ax.set_xlabel(benchmark, fontsize=fs*0.9)
        ax.set_ylabel('CDF', fontsize=fs*0.9)
        ax.tick_params(axis='both', which='major', labelsize=fs*0.9)
        ax.set_title(f'{model}', fontsize=fs, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=fs*0.8)
        ax.set_ylim(0, 1)
        
        # Apply benchmark limits if available
        if benchmark in benchmark_limits_cdf:
            ax.set_xlim(benchmark_limits_cdf[benchmark]['min'], 
                       benchmark_limits_cdf[benchmark]['max'])

    
    fig.suptitle(f'CDF Comparison: {benchmark}', fontsize=fs, y=0.995)
    plt.tight_layout()
    
    # Save CDF plot
    output_path_cdf = os.path.join(output_dir_cdf, f'fig_by_benchmark_{benchmark}_cdf.png')
    plt.savefig(output_path_cdf, dpi=300, bbox_inches='tight')
    plt.close()

print("All benchmark plots have been generated successfully!")
print(f"Box plots saved to: {output_dir_box}")
print(f"CDF plots saved to: {output_dir_cdf}")