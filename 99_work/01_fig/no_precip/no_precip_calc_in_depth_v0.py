import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('default')

fs = 20

model_list = ["HBV", "RF", "XGB", "TCN", "LSTM"]
file_tot_num = 424
start_date_zero = "2002-01-01"
end_date_eva = "2006-12-31"

output_dir = "/data0/funato/hsp/0_out/analysis/no_precip_discharge"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/no_precip_discharge"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)

color_list = {
    "HBV": "#7B7B7B",    # Kept as requested (Medium Grey)
    "BC": "#628AE8",     # Kept as requested (Steel Blue)
    "RF":   "#FF7F00",   # Red
    "XGB":  "#9C4803",   # Bright Orange
    "TCN":  "#E41A1C",   # Reddish Purple / Magenta
    "LSTM": "#9808B2"    # Pink
}

def create_detailed_analysis():
    """Create detailed analysis of no-precipitation discharge predictions"""
    
    # Initialize storage
    results = []
    all_discharge_data = {model: [] for model in model_list}
    model_file_means = {model: [] for model in model_list}
    
    print("Analyzing no-precipitation discharge predictions...")
    print(f"Analysis period: {start_date_zero} to {end_date_eva}")
    
    for model in model_list:
        print(f"\nProcessing model: {model}")
        
        for file in range(1, file_tot_num + 1):
            if file % 50 == 0 or file == 1:
                print(f"Processing model: {model}, file number: {file}")
            if model != "HBV":
                input_file = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/varssim_{model}/varssim{file:03d}.csv"
            else:
                input_file = f"/data0/funato/hsp/0_out/HBV/varssim_HBV/varssim{file:03d}.csv"
            if not os.path.exists(input_file):
                print(f"File {input_file} does not exist. Skipping.")
                continue
            
            try:
                # Read and process file
                input_df = pd.read_csv(input_file, parse_dates=['Date'])
                input_df.set_index('Date', inplace=True)
                
                # Filter for evaluation period
                eval_period_df = input_df[start_date_zero:end_date_eva]
                
                if 'no_precip' not in eval_period_df.columns:
                    continue
                
                discharge_values = eval_period_df['no_precip'].dropna()
                
                if len(discharge_values) == 0:
                    continue
                
                # Calculate statistics
                mean_discharge = discharge_values.mean()
                std_discharge = discharge_values.std()
                
                # Store data
                all_discharge_data[model].extend(discharge_values.tolist())
                model_file_means[model].append(mean_discharge)
                
                results.append({
                    'Model': model,
                    'File': file,
                    'Mean_Discharge': mean_discharge,
                    'Std_Discharge': std_discharge,
                    'Min_Discharge': discharge_values.min(),
                    'Max_Discharge': discharge_values.max(),
                    'Median_Discharge': discharge_values.median(),
                    'Q25_Discharge': discharge_values.quantile(0.25),
                    'Q75_Discharge': discharge_values.quantile(0.75),
                    'Count': len(discharge_values)
                })
                
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
        
        print(f"  Processed {len(model_file_means[model])} files for {model}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No data found to analyze!")
        return
    
    
    detailed_csv = f"{output_dir}/no_precip_detailed_results.csv"
    results_df.to_csv(detailed_csv, index=False)
    print(f"\nDetailed results saved to: {detailed_csv}")
    
    # Create model summary
    model_summary = results_df.groupby('Model').agg({
        'Mean_Discharge': ['mean', 'std', 'min', 'max', 'median'],
        'File': 'count'
    }).round(6)
    
    model_summary.columns = ['Mean_of_Means', 'Std_of_Means', 'Min_Mean', 'Max_Mean', 'Median_of_Means', 'File_Count']
    
    summary_csv = f"{output_dir}/no_precip_model_summary.csv"
    model_summary.to_csv(summary_csv)
    print(f"Model summary saved to: {summary_csv}")
    
    print("\nModel Summary Statistics:")
    print(model_summary)
    
    # Create comprehensive visualization
    create_comprehensive_plots(all_discharge_data, model_file_means, results_df, output_dir)
    
    # Statistical analysis
    perform_statistical_tests(model_file_means, output_dir)
    
    return results_df, model_summary

def create_comprehensive_plots(all_discharge_data, model_file_means, results_df, output_dir):
    """Create comprehensive visualization plots"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Box plot comparison
    ax1 = plt.subplot(2, 3, 1)
    box_data = [all_discharge_data[model] for model in model_list if len(all_discharge_data[model]) > 0]
    box_labels = [model for model in model_list if len(all_discharge_data[model]) > 0]

    
def perform_statistical_tests(model_file_means, output_dir):
    """Perform statistical tests to compare models"""
    
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    # Prepare data for statistical tests
    valid_models = [model for model in model_list if len(model_file_means[model]) > 0]
    
    if len(valid_models) < 2:
        print("Not enough models with data for statistical comparison.")
        return
    
    # ANOVA test
    anova_data = [model_file_means[model] for model in valid_models]
    
    if all(len(data) > 1 for data in anova_data):
        f_stat, p_value_anova = stats.f_oneway(*anova_data)
        print(f"\nANOVA Test:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value_anova:.6f}")
        
        if p_value_anova < 0.05:
            print("Result: Significant difference between models (p < 0.05)")
        else:
            print("Result: No significant difference between models (p >= 0.05)")
    
    # Pairwise t-tests
    print(f"\nPairwise t-tests (mean discharge comparison):")
    print("-" * 40)
    
    for i in range(len(valid_models)):
        for j in range(i+1, len(valid_models)):
            model1, model2 = valid_models[i], valid_models[j]
            data1, data2 = model_file_means[model1], model_file_means[model2]
            
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2)
                print(f"{model1} vs {model2}:")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
                
                if p_value < 0.05:
                    better_model = model1 if np.mean(data1) < np.mean(data2) else model2
                    print(f"  Result: {better_model} significantly better (p < 0.05)")
                else:
                    print(f"  Result: No significant difference (p >= 0.05)")
                print()
    
    # Model ranking by mean discharge (lower is better for zero precipitation)
    print(f"\nModel Ranking (by mean discharge - lower is better):")
    print("-" * 50)
    
    model_rankings = []
    for model in valid_models:
        mean_val = np.mean(model_file_means[model])
        std_val = np.std(model_file_means[model])
        model_rankings.append((model, mean_val, std_val))
    
    model_rankings.sort(key=lambda x: x[1])  # Sort by mean
    
    for rank, (model, mean_val, std_val) in enumerate(model_rankings, 1):
        print(f"{rank}. {model}: {mean_val:.6f} ± {std_val:.6f}")

if __name__ == "__main__":
    results_df, model_summary = create_detailed_analysis()
    print("\nAnalysis completed successfully!")
