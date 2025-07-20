import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a png file for each benchmark metric. each png should include two figures next to each other, each representing calibration and evaluation results. 
# for each figure, plot the benchmark metric on the y-axis and the model on the x-axis bt using a box plot 
# also plot the cdf of the benchmark metric as another set with the x axis as the benchmark metric and the y-axis as the cdf value with labels with different colors for each model
model_list = ["HBV","BC", "RF", "XGB", "TCN", "LSTM"]

output_dir_box = "/data0/funato/hsp/0_out/1_fig/benchmark/box"
output_dir_cdf = "/data0/funato/hsp/0_out/1_fig/benchmark/cdf"
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_cdf, exist_ok=True)

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

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
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}


for benchmark in benchmark_list:
    print(f"Processing benchmark: {benchmark}")
    cal_data = []
    eva_data = []
    valid_models = []  # Track which models have valid data
    
    for model in model_list:
        print(f"Processing model: {model}")

        if model == "BC":
            input_file_cal = f"/data0/funato/hsp/0_out/BC/results/BC_results_cal.csv"
            input_file_eva = f"/data0/funato/hsp/0_out/BC/results/BC_results_eva.csv"
        else:
            input_file_cal = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/results/{model}_results_cal.csv"
            input_file_eva = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/results/{model}_results_eva.csv"


        if not os.path.exists(input_file_cal):
            print(f"Input file {input_file_cal} does not exist. Skipping model {model}.")
            continue
        if not os.path.exists(input_file_eva):
            print(f"Input file {input_file_eva} does not exist. Skipping model {model}.")
            continue

        try:
            col_name_cal = f"{model}_{benchmark}_cal"
            col_name_eva = f"{model}_{benchmark}_eva"
            cal_df = pd.read_csv(input_file_cal)
            eva_df = pd.read_csv(input_file_eva)

            if col_name_cal not in cal_df.columns or col_name_eva not in eva_df.columns:
                print(f"Benchmark {benchmark} not found in {model} data. Skipping model {model}.")
                continue

            cal_values = cal_df[col_name_cal].dropna().values
            eva_values = eva_df[col_name_eva].dropna().values
            
            if len(cal_values) == 0 or len(eva_values) == 0:
                print(f"No valid data for {benchmark} in {model}. Skipping model {model}.")
                continue
                
            cal_data.append(cal_values)
            eva_data.append(eva_values)
            valid_models.append(model)
            
        except Exception as e:
            print(f"Error processing {model}: {e}. Skipping model {model}.")
            continue

    # Skip this benchmark if no valid models found
    if len(valid_models) == 0:
        print(f"No valid models found for benchmark {benchmark}. Skipping.")
        continue
        
    print(f"Valid models for {benchmark}: {valid_models}")

    # Create box plot for calibration and evaluation data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.boxplot(cal_data, labels=valid_models)
    ax1.set_title(f'Calibration {benchmark} Benchmark Metrics')
    ax1.set_xlabel('Models')
    ax1.set_ylabel(benchmark)
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

    # Create box plot for evaluation data
    ax2.boxplot(eva_data, labels=valid_models)
    ax2.set_title(f'Evaluation {benchmark} Benchmark Metrics')
    ax2.set_xlabel('Models')
    ax2.set_ylabel(benchmark)
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_box, f"{benchmark}_box_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create CDF plot for calibration and evaluation data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # CDF for calibration data
    for i, model in enumerate(valid_models):
        if len(cal_data[i]) > 0:
            sorted_cal = np.sort(cal_data[i])
            cdf_cal = np.arange(1, len(sorted_cal) + 1) / len(sorted_cal)
            ax1.plot(sorted_cal, cdf_cal, label=model, linewidth=2)
    ax1.set_title(f'Calibration {benchmark} CDF')
    ax1.set_xlabel(benchmark)
    ax1.set_ylabel('CDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)     
    ax1.set_ylim(0,1)
    ax1.set_xlim(benchmark_limits_cdf[benchmark]['min'], benchmark_limits_cdf[benchmark]['max']) 

    # CDF for evaluation data
    for i, model in enumerate(valid_models):      
        if len(eva_data[i]) > 0:
            sorted_eva = np.sort(eva_data[i])
            cdf_eva = np.arange(1, len(sorted_eva) + 1) / len(sorted_eva)
            ax2.plot(sorted_eva, cdf_eva, label=model, linewidth=2)
    ax2.set_title(f'Evaluation {benchmark} CDF')
    ax2.set_xlabel(benchmark)
    ax2.set_ylabel('CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0,1)
    ax2.set_xlim(benchmark_limits_cdf[benchmark]['min'], benchmark_limits_cdf[benchmark]['max'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_cdf, f"{benchmark}_cdf_plot.png"), dpi=300, bbox_inches='tight')
    plt.close() 

    
print("All benchmark metrics processed and saved as plots.")

