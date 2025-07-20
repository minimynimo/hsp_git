import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_data = "/data0/funato/hsp/0_out/analysis/mass_conservation/mass_bias_data.csv"
input_df = pd.read_csv(input_data)

output_dir = "/data0/funato/hsp/0_out/1_fig/mass_bias"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Creating mass bias visualizations...")
print(f"Dataset shape: {input_df.shape}")
print(f"Models in dataset: {input_df['Model'].unique()}")

# 1. Box plot of mass bias by model
plt.figure(figsize=(7, 7))
sns.boxplot(data=input_df, x='Model', y='Mass_Bias', order=model_list)
plt.title('Mass Bias Distribution by Model', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Mass Bias', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(-1,5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()

# 2, CDF plot of mass bias by model
plt.figure(figsize=(6, 6))
for model in model_list:
    model_data = input_df[input_df['Model'] == model]['Mass_Bias']
    sns.ecdfplot(model_data, label=model, stat='proportion', linewidth=2)
plt.title('Cumulative Distribution of Mass Bias by Model', fontsize=16)
plt.xlabel('Mass Bias', fontsize=14)    
plt.ylabel('Cumulative Proportion', fontsize=14)
plt.xlim(0, 3)
plt.ylim(0, 1)
plt.legend(title='Model', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_cdf.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Bar plot of mean mass bias by model with error bars
plt.figure(figsize=(12, 8))
mass_bias_stats = input_df.groupby('Model')['Mass_Bias'].agg(['mean', 'std']).reindex(model_list)
mass_bias_stats['mean'].plot(kind='bar', yerr=mass_bias_stats['std'], capsize=5, color='skyblue', edgecolor='black')
plt.title('Mean Mass Bias by Model (with Standard Deviation)', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Mean Mass Bias', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_mean_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Histogram of mass bias for each model
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, model in enumerate(model_list):
    model_data = input_df[input_df['Model'] == model]['Mass_Bias']
    axes[i].hist(model_data, bins=30, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{model} (n={len(model_data)})', fontsize=14)
    axes[i].set_xlabel('Mass Bias', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = model_data.mean()
    median_val = model_data.median()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    axes[i].axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.3f}')
    axes[i].legend()

plt.suptitle('Mass Bias Distribution by Model', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_histograms.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Summary statistics table
print("\nMass Bias Summary Statistics by Model:")
print("="*60)
summary_stats = input_df.groupby('Model')['Mass_Bias'].describe().round(4)
print(summary_stats)

# Save summary statistics to CSV
summary_stats.to_csv(f"{output_dir}/mass_bias_summary_stats.csv")


print(f"\nVisualizations saved to: {output_dir}")
print("Files created:")
print("- mass_bias_boxplot.png")
print("- mass_bias_violin.png") 
print("- mass_bias_mean_bar.png")
print("- mass_bias_histograms.png")
print("- mass_bias_ranking.png")
print("- mass_bias_summary_stats.csv")
