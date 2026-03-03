import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
file_path = '/data0/funato/hsp2/0_out/HBV/GB/results/HBV_results_eva.csv'
gaussian_list = "/data0/funato/hsp2/99_work/01_fig/benchmark/calculated_gaussian_HBV_list.csv"

# Load Pareto list
gaussian_df = pd.read_csv(gaussian_list, index_col=None, header=0)
gaussian_ids = set(gaussian_df['id'].tolist())  # Use a set for faster lookup

# Load Results
df = pd.read_csv(file_path)

# 2. Categorize the data
# Create a new column 'Group' to distinguish between Pareto and Non-Pareto basins
# We assume 'file_num' in the results matches 'id' in the gaussian list
df['Group'] = df['file_num'].apply(lambda x: 'Gaussian' if x in gaussian_ids else 'Pareto')

# 3. Plotting
plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")

# Create the box plot
ax = sns.boxplot(
    x='Group', 
    y='HBV_KGE_eva', 
    data=df, 
    palette="Set2",     # You can change colors here (e.g., "viridis", "Blues")
    showfliers=True     # Set to False if you want to hide outliers
)

# Add swarmplot to see individual points on top of boxes (Optional - helpful for smaller datasets)
# sns.swarmplot(x='Group', y='HBV_KGE_eva', data=df, color=".25", size=3)

# 4. Styling
plt.title('HBV Model Performance Comparison: Pareto vs Gaussian Basins', fontsize=16)
plt.ylabel('KGE Score', fontsize=12)
plt.xlabel('Basin Group', fontsize=12)
plt.ylim(0.5, 1.0) # Optional: Fix y-axis limits if needed

# Add sample size text
gaussian_count = len(df[df['Group']=='Gaussian'])
pareto_count = len(df[df['Group']=='Pareto'])
plt.text(0, df['HBV_KGE_eva'].min(), f'n={gaussian_count}', ha='center', va='bottom', fontweight='bold')
plt.text(1, df['HBV_KGE_eva'].min(), f'n={pareto_count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# 5. Save
plt.savefig('HBV_Pareto_Comparison_Boxplot.png', dpi=300)
plt.show()