import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

model_list = ["RF", "XGB", "TCN", "LSTM"]
train_types = ["regional", "individual", "adapted"]

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

color_list = {
    "RF":   "#FF7F00",   # Orange
    "XGB":  "#9C4803",   # Dark Orange
    "TCN":  "#E41A1C",   # Red
    "LSTM": "#9808B2"    # Purple
}

# Markers for different training types
marker_list = {
    "regional": "",     # Circle
    "individual": "",   # Square
    "adapted": ""       # Triangle
}

# Line styles for different training types
linestyle_list = {
    "regional": "-",     # Solid
    "individual": "--",  # Dashed
    "adapted": ":"       # Dotted
}


benchmark_limits = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

benchmark_limits_cdf = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

output_dir_box = "/data0/funato/hsp/0_out/1_fig/benchmark/ML_training_comparison/box"
output_dir_cdf = "/data0/funato/hsp/0_out/1_fig/benchmark/ML_training_comparison/cdf"
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_cdf, exist_ok=True)

def get_file_path(model, train_type):
    if model == "TCN" and train_type == "regional":
        return f"/data0/funato/hsp/0_out/TCN/GB/shap/regional/results/{model}_results_eva.csv"
    elif model == "LSTM" and train_type == "individual":
        return f"/data0/funato/hsp/0_out/LSTM/GB/individual/results/LSTM_results_h20_lr0.005_e100_w365_b512_d0.1_eva.csv"
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
    
    # === CDF FIGURE ===
    plt.figure(figsize=(10, 6))
    
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
    
    # Plot CDFs
    for model in model_list:
        for train_type in train_types:
            if train_type in cdf_data[model]:
                data = cdf_data[model][train_type]
                sorted_data = np.sort(data)
                cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                plt.plot(sorted_data, cdf_values, 
                        color=color_list[model], 
                        linestyle=linestyle_list[train_type],
                        marker=marker_list[train_type],
                        markersize=4,
                        markevery=max(1, len(sorted_data)//20),  # Show markers sparingly
                        label=f"{model}_{train_type}",
                        linewidth=2,
                        alpha=0.8)
    
    plt.xlabel(benchmark)
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF: {benchmark}')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.ylim(0, 1)
    
    # Apply benchmark limits if available (use the same limits as box plots)
    if benchmark in benchmark_limits:
        plt.xlim(benchmark_limits[benchmark]['min'], 
                 benchmark_limits[benchmark]['max'])
        print(f"Applied CDF limits: {benchmark_limits[benchmark]}")
    elif benchmark in benchmark_limits_cdf:
        plt.xlim(benchmark_limits_cdf[benchmark]['min'], 
                 benchmark_limits_cdf[benchmark]['max'])
        print(f"Applied CDF-specific limits: {benchmark_limits_cdf[benchmark]}")
    
    plt.tight_layout()
    
    # Save CDF plot
    output_path_cdf = os.path.join(output_dir_cdf, f'fig_by_benchmark_{benchmark}_cdf.png')
    plt.savefig(output_path_cdf, dpi=300, bbox_inches='tight')
    plt.close()

print("All benchmark plots have been generated successfully!")
print(f"Box plots saved to: {output_dir_box}")
print(f"CDF plots saved to: {output_dir_cdf}")