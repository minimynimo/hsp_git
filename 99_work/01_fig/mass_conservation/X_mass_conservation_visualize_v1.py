import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
input_csv = "/data0/funato/hsp/0_out/analysis/mass_conservation/mass_conservation_data_v1.csv"
df = pd.read_csv(input_csv)

os.makedirs("/data0/funato/hsp/0_out/1_fig/mass_conservation", exist_ok=True)
fig_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation"

# Plot: Implied Loss vs PET for each model
for model in df['Model'].unique():
    sub = df[df['Model'] == model]
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='PET', y='Implied_Loss', data=sub, hue='Plausible', palette={True:'blue', False:'red'}, alpha=0.7)
    plt.plot([0, sub['PET'].max()], [0, sub['PET'].max()], 'k--', label='L=PET')
    plt.xlabel('Potential Evapotranspiration (PET) [mm]')
    plt.ylabel('Implied Loss (L=P-Q) [mm]')
    plt.title(f"{model}: Implied Loss vs PET")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/implied_loss_vs_pet_{model}.png")
    plt.close()

# Summary barplot: Fraction of plausible basins per model
summary = df.groupby('Model')['Plausible'].mean().reset_index()
plt.figure(figsize=(7,4))
sns.barplot(x='Model', y='Plausible', data=summary, palette='husl')
plt.ylabel('Fraction of Basins with Physically Plausible Loss')
plt.title('Model Robustness: Plausible Water Balance Closure')
plt.tight_layout()
plt.savefig(f"{fig_dir}/plausible_fraction_by_model.png")
plt.close()

# Save summary table
summary.to_csv(f"{fig_dir}/plausible_fraction_by_model.csv", index=False)
print("Visualization complete. Figures saved.")
