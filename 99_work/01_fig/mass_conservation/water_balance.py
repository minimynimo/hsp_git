import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_data = "/data0/funato/hsp/0_out/analysis/mass_conservation/water_balance_data.csv"
input_df = pd.read_csv(input_data)

output_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation"
os.makedirs(output_dir, exist_ok=True)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Creating water balance ET difference visualizations...")
print(f"Dataset shape: {input_df.shape}")
print(f"Models in dataset: {input_df['Model'].unique()}")

# Get observed ET values for each basin
observed_df = input_df[input_df['Model'] == 'Observed'].set_index('File_num')['Daily_Mean_ET']

# Calculate ET differences for each model compared to observed
model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
et_differences = []

for model in model_list:
    model_df = input_df[input_df['Model'] == model].set_index('File_num')['Daily_Mean_ET']
    
    # Find common basins between observed and model data
    common_basins = observed_df.index.intersection(model_df.index)
    
    for basin in common_basins:
        obs_et = observed_df.loc[basin]
        model_et = model_df.loc[basin]
        et_diff = model_et - obs_et  # Model ET - Observed ET
        
        et_differences.append({
            'File_num': basin,
            'Model': model,
            'Observed_ET': obs_et,
            'Model_ET': model_et,
            'ET_Difference': et_diff,
            'Absolute_ET_Difference': abs(et_diff)
        })

# Convert to DataFrame
et_diff_df = pd.DataFrame(et_differences)

print(f"ET difference dataset shape: {et_diff_df.shape}")

# 1. Box plot of ET differences by model
plt.figure(figsize=(8, 8))
sns.boxplot(data=et_diff_df, x='Model', y='ET_Difference', order=model_list)
plt.title('Daily Mean ET Differences: Model - Observed', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('ET Difference (mm/day)', fontsize=14)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.ylim(-5, 15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/et_difference_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()


# 3. Scatter plots: Observed vs Model ET for each model
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, model in enumerate(model_list):
    model_data = et_diff_df[et_diff_df['Model'] == model]
    
    axes[i].scatter(model_data['Observed_ET'], model_data['Model_ET'], alpha=0.6, s=30)
    
    # Add 1:1 line
    min_val = min(model_data['Observed_ET'].min(), model_data['Model_ET'].min())
    max_val = max(model_data['Observed_ET'].max(), model_data['Model_ET'].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 Line')
    
    # Calculate correlation
    correlation = model_data['Observed_ET'].corr(model_data['Model_ET'])
    
    axes[i].set_title(f'{model} (r={correlation:.3f}, n={len(model_data)})', fontsize=14)
    axes[i].set_xlabel('Observed ET (mm/day)', fontsize=12)
    axes[i].set_ylabel('Model ET (mm/day)', fontsize=12)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.suptitle('Observed vs Model Daily Mean ET', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/et_observed_vs_model_scatter.png", dpi=300, bbox_inches='tight')
plt.close()


# 5. Summary statistics
print("\nET Difference Summary Statistics by Model:")
print("="*60)
summary_stats = et_diff_df.groupby('Model')['ET_Difference'].describe().round(4)
print(summary_stats)

# Save summary statistics
summary_stats.to_csv(f"{output_dir}/et_difference_summary_stats.csv")

# Save the ET differences dataset
et_diff_df.to_csv(f"{output_dir}/et_differences_dataset.csv", index=False)

print(f"\nVisualizations saved to: {output_dir}")
print("Files created:")
print("- et_difference_boxplot.png")
print("- et_difference_violin.png") 
print("- et_observed_vs_model_scatter.png")
print("- et_mean_absolute_difference.png")
print("- et_difference_summary_stats.csv")
print("- et_differences_dataset.csv")

# Print model ranking by performance
print(f"\nModel Ranking by Mean Absolute ET Difference (Lower is Better):")
print("-" * 50)
ranking = et_diff_df.groupby('Model')['Absolute_ET_Difference'].mean().sort_values()
for i, (model, mae) in enumerate(ranking.items(), 1):
    print(f"{i}. {model}: {mae:.4f} mm/day")

