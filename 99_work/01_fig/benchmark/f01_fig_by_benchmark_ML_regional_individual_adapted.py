import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

loc = "US"

model_list = ["RF", "XGB", "TCN", "LSTM"]
train_types = ["regional", "adapted","individual"]

benchmark_list = ["KGE", "NSE", "E1", "VE", "RMSE", "MAE"]

fs = 15

# us_camels_file_num.csv is filtered (by us_basin_filter.py) to only the basins with complete
# Obs flow/Precip/Temp/PET data over the full calibration+evaluation period.
basin_ids = pd.read_csv("/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_camels_file_num.csv")["gauge_id"].str.replace("camels_", "", regex=False).astype(int).tolist()


"""
MEMO INDIVIDUAL (CAMELS)
/data0/funato/hsp2/0_out/RF/{loc}/camels/individual/results/RF_results_n300_d10_l20_fsqrt_eva.csv
/data0/funato/hsp2/0_out/XGB/{loc}/camels/individual/results/XGB_results_n500_d2_lr0.01_ss0.6_cbt1.0_mcw10_eva.csv
/data0/funato/hsp2/0_out/TCN/{loc}/camels/individual/f16_k5_d0.4_w90/results/TCN_results_f16_k5_d0.4_w90_eva.csv
/data0/funato/hsp2/0_out/LSTM/{loc}/camels/individual/results/LSTM_results_h32_lr0.001_e100_w270_b64_d0.5_lossnse_eva.csv

MEMO ADAPTED (CAMELS)
/data0/funato/hsp2/0_out/TCN/{loc}/camels/adapted/results/TCN_results_n64_k5_d0.2_nsFalse_sl180_eva.csv
/data0/funato/hsp2/0_out/LSTM/{loc}/camels/adapted/results/LSTM_results_nsFalse_csFalse_hs64_do0.4_e30_sl365_ft5_ftlr0.0001_eva.csv

MEMO REGIONAL (CAMELS)
/data0/funato/hsp2/0_out/RF/{loc}/camels/regional/results/RF_results_n300_d30_s5_l20_mfsqrt_ms0.8_w270_nsFalse_eva.csv
/data0/funato/hsp2/0_out/XGB/{loc}/camels/regional/results/XGB_results_ns0_n1000_d7_lr0.1_sub0.7_col0.7_sl270_eva.csv
/data0/funato/hsp2/0_out/TCN/{loc}/camels/regional/results/TCN_results_n64_k5_d0.2_nsFalse_sl180_eva.csv
/data0/funato/hsp2/0_out/LSTM/{loc}/camels/regional/results/LSTM_results_nsFalse_csFalse_hs64_do0.4_e30_sl365_eva.csv

NOTE: under camels/, RF/XGB/TCN/LSTM regional and adapted results live directly under
regional/ and adapted/ (no _ealstm / _614 suffix -- those only exist under the legacy
caravan/ tree). TCN individual is the only one written under a hyperparameter-named
subdirectory (individual/{tag}/results), not a flat individual/results/ dir.
"""

# Tags match the selected hyperparameters in SELECTED_HYPERPARAMETERS_camels.md (Individual),
# verified against the actual filenames present on disk under 0_out/{model}/{loc}/camels/individual/.
param_idv_tag_dict = {
    "RF": "_n300_d10_l20_fsqrt",
    "XGB": "_n500_d2_lr0.01_ss0.6_cbt1.0_mcw10",
    "TCN": "_f16_k5_d0.4_w90",
    "LSTM": "_h32_lr0.001_e100_w270_b64_d0.5_lossnse"
}

# Tags match the selected hyperparameters in SELECTED_HYPERPARAMETERS_camels.md (Regional),
# verified against the actual filenames present on disk under 0_out/{model}/{loc}/camels/regional/results.
# NOTE: TCN regional's selected NB_FILTERS is 32 (see SELECTED_HYPERPARAMETERS_camels.md), but
# only the n64 combo has a production regional run on disk -- using n64 until n32 is (re)trained.
param_reg_tag_dict = {
    "RF": "_n300_d30_s5_l20_mfsqrt_ms0.8_w270_nsFalse",
    "XGB": "_ns0_n1000_d7_lr0.1_sub0.7_col0.7_sl270",
    "TCN": "_n64_k5_d0.2_nsFalse_sl180",
    "LSTM": "_nsFalse_csFalse_hs64_do0.4_e30_sl365"
}

# Adapted tags match the selected Regional hyperparameters above plus the fine-tune params
# (TCN adapted reuses the regional checkpoint's tag; LSTM adapted appends ft{epochs}_ftlr{lr}),
# verified against the actual filenames on disk under 0_out/{model}/{loc}/camels/adapted/results.
param_adp_tag_dict = {
    "TCN": "_n64_k5_d0.2_nsFalse_sl180",
    "LSTM": "_nsFalse_csFalse_hs64_do0.4_e30_sl365_ft5_ftlr0.0001"
}

###RF


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

output_dir_box = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/ML_training_comparison/box"
output_dir_cdf = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/ML_training_comparison/cdf"
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_cdf, exist_ok=True)

def get_file_path(model, train_type):
    # Under camels/, regional/adapted/individual results for all four ML models live
    # directly under those names (no _ealstm / _614 suffix -- that split only exists under
    # the legacy caravan/ tree).
    if model == "TCN" and train_type == "individual":
        # TCN individual runs are written under a hyperparameter-named subdirectory
        # (e.g. individual/f16_k5_d0.4_w90/results), not a flat individual/results/ dir.
        dir_name = f"individual/{param_idv_tag_dict['TCN'].lstrip('_')}"
    else:
        dir_name = train_type
    base_path = f"/data0/funato/hsp2/0_out/{model}/{loc}/camels/{dir_name}/results"
    if train_type == "individual":
        param_tag = param_idv_tag_dict.get(model, "")
    elif train_type == "adapted":
        param_tag = param_adp_tag_dict.get(model, "")
    else:  # regional
        param_tag = param_reg_tag_dict.get(model, "")
    
    return os.path.join(base_path, f"{model}_results{param_tag}_eva.csv")


for benchmark in benchmark_list:
    print(f"\n=== Processing {benchmark} ===")
    
    # === DEBUGGING: Check file existence ===
    print("File existence check:")
    for model in model_list:
        for train_type in train_types:
            if model in ["RF", "XGB"] and train_type == "adapted":
                # Skip adapted for RF and XGB as they don't have adapted training
                continue
            file_path = get_file_path(model, train_type)
            exists = os.path.exists(file_path)
            if exists:
                df = pd.read_csv(file_path)
                # Column name format: try {MODEL}_{BENCHMARK}_eva first, then {BENCHMARK}
                col_name = f"{model}_{benchmark}_eva"
                if col_name not in df.columns:
                    col_name = benchmark
                if col_name not in df.columns:
                    print(f"    Warning: Missing or incomplete data for {model} - {train_type}")
            else:
                print(f"  {model} - {train_type}: File does not exist: {file_path}")
    
    # === BOXPLOT FIGURE ===
    plt.figure(figsize=(12, 6))
    
    all_data = []
    all_positions = []
    all_labels = []
    box_colors = []
    
    position_counter = 0
    
    for model_idx, model in enumerate(model_list):
        for train_idx, train_type in enumerate(train_types):
            if model in ["RF", "XGB"] and train_type == "adapted":
                # Skip adapted for RF and XGB as they don't have adapted training
                continue
            file_path = get_file_path(model, train_type)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Filter to only include basins in basin_ids list
                # LSTM's results use 'gauge_id' instead of 'basin' for the same id
                basin_col = 'basin' if 'basin' in df.columns else 'gauge_id' if 'gauge_id' in df.columns else None
                if basin_col is not None:
                    df = df[df[basin_col].isin(basin_ids)]
                # Column name format: try {MODEL}_{BENCHMARK}_eva first, then {BENCHMARK}
                col_name = f"{model}_{benchmark}_eva"
                if col_name not in df.columns:
                    col_name = benchmark
                if col_name in df.columns:
                    data = df[col_name].dropna()
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
    plt.savefig(output_path_box, dpi=400, bbox_inches='tight')
    plt.close()
    
    # === CDF FIGURE with subplots per model ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Collect data for CDF plotting
    cdf_data = {}
    for model in model_list:
        cdf_data[model] = {}
        for train_type in train_types:
            if model in ["RF", "XGB"] and train_type == "adapted":
                # Skip adapted for RF and XGB as they don't have adapted training
                continue
            file_path = get_file_path(model, train_type)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Filter to only include basins in basin_ids list
                # LSTM's results use 'gauge_id' instead of 'basin' for the same id
                basin_col = 'basin' if 'basin' in df.columns else 'gauge_id' if 'gauge_id' in df.columns else None
                if basin_col is not None:
                    df = df[df[basin_col].isin(basin_ids)]
                # Column name format: try {MODEL}_{BENCHMARK}_eva first, then {BENCHMARK}
                col_name = f"{model}_{benchmark}_eva"
                if col_name not in df.columns:
                    col_name = benchmark
                if col_name in df.columns:
                    data = df[col_name].dropna()
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

    
    #fig.suptitle(f'CDF Comparison: {benchmark}', fontsize=fs, y=0.995)
    plt.tight_layout()
    
    # Save CDF plot
    output_path_cdf = os.path.join(output_dir_cdf, f'fig_by_benchmark_{benchmark}_cdf.png')
    plt.savefig(output_path_cdf, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved box plot to {output_path_box}")

print("All benchmark plots have been generated successfully!")
print(f"Box plots saved to: {output_dir_box}")
print(f"CDF plots saved to: {output_dir_cdf}")