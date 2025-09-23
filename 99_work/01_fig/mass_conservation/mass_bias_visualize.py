import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

fs = 20
input_data = "/data0/funato/hsp/0_out/analysis/mass_conservation/mass_bias_data.csv"
input_df = pd.read_csv(input_data)

output_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation_use"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "HYPER", "RF", "XGB", "TCN", "LSTM"]

color_list = {
    "HBV": "#7B7B7B",    # Kept as requested (Medium Grey)
    "HYPER": "#628AE8",     # Kept as requested (Steel Blue)
    "RF":   "#FF7F00",   # Red
    "XGB":  "#9C4803",   # Bright Orange
    "TCN":  "#E41A1C",   # Reddish Purple / Magenta
    "LSTM": "#9808B2"    # Pink
}

# Set plotting style
plt.style.use('default')

print("Creating mass bias visualizations...")
print(f"Dataset shape: {input_df.shape}")
print(f"Models in dataset: {input_df['Model'].unique()}")
#change "BC" to "HYPER" in input_df
input_df['Model'] = input_df['Model'].replace('BC', 'HYPER')

# 1. Box plot of mass bias by model
plt.figure(figsize=(7, 7))
import seaborn as sns
sns.boxplot(data=input_df, x='Model', y='Mass_Bias', hue='Model', order=model_list, palette=color_list, legend=False)
plt.title('Mass Bias Distribution by Model', fontsize=fs)
plt.xlabel('Model', fontsize=fs)
plt.ylabel('Mass Bias', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.xticks(rotation=45)
plt.ylim(-1,5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()

# 2, CDF plot of mass bias by model
plt.figure(figsize=(6, 6))
for model in model_list:
    if model != "HYPER":
        model_data = input_df[input_df['Model'] == model]['Mass_Bias']
    else:
        model_data = input_df[input_df['Model'] == 'BC']['Mass_Bias']
    sorted_data = np.sort(model_data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=model, linewidth=2, color=color_list[model])
plt.title('Mass Bias', fontsize=fs)
plt.xlabel('Mass Bias', fontsize=fs)    
plt.ylabel('Cumulative Proportion', fontsize=fs)
plt.tick_params(labelsize=fs*0.8)
plt.xlim(0, 3)
plt.xticks([0.0, 1.0, 2.0, 3.0], ['0.0', '1.0', '2.0', '3.0'])
plt.ylim(0, 1)
plt.legend(fontsize=fs*0.6)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/mass_bias_cdf.png", dpi=300, bbox_inches='tight')
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
