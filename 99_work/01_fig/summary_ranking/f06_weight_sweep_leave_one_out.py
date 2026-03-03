import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
import os
from collections import defaultdict
from matplotlib.colors import ListedColormap

loc = "US"

# --- Configuration ---
# Directory to save the output figures
output_dir = f"/data0/funato/hsp2/0_out/1_fig/{loc}/summary_ranking_multi/sensitivity_analysis"
os.makedirs(output_dir, exist_ok=True)

# Number of random weight combinations for the Top-1 Frequency analysis
N_SIMULATIONS = 20000

# VISUALIZATION SETTINGS
# Resolution: How many hexagons to draw along one axis. 
# 100-120 is good for smoothness.
TERNARY_RESOLUTION = 100 

# Scale: The value at the triangle corners (Axes range).
# Set to 1.0 for 0.0 - 1.0 range. Set to 100 for 0 - 100 range.
TERNARY_SCALE = 100.0 

# Model names and the color palette
model_list = ["HBV", "HYPER", "RF", "XGB", "TCN", "LSTM"]
color_list = {
    "HBV": "#7B7B7B",    # Medium Grey
    "HYPER": "#628AE8",   # Steel Blue
    "RF": "#FF7F00",      # Orange
    "XGB": "#9C4803",    # Dark Orange
    "TCN": "#E41A1C",     # Red
    "LSTM": "#9808B2"    # Purple
}
fs = 20 

# --- Input Data: Plausibility Ranks ---
rank_file = "/data0/funato/hsp2/0_out/analysis/summary_ranking_multi/automated_ranks/automated_plausibility_ranks.csv"

if not os.path.exists(rank_file):
    print(f"Error: Rank file not found at {rank_file}")
    # Fallback/Dummy data for testing if file missing
    plausibility_ranks = pd.DataFrame({
        'Model': model_list,
        'Mass_Rank': [6, 3, 1.5, 4, 5, 1.5],
        'Drought_Rank': [1, 5, 6, 2, 4, 3],
        'Flood_Rank': [1, 1, 1, 1, 1, 1] 
    }).set_index('Model')
else:
    rank_df = pd.read_csv(rank_file)
    plausibility_ranks = rank_df.set_index('Model')[['Mass_Rank', 'Drought_Rank', 'Flood_Rank']]

print("--- Input Plausibility Ranks (Lower is better) ---")
print(plausibility_ranks)

# --- Step 1 & 2: Perform Weight Sweep for Frequency Analysis ---
weights = np.random.dirichlet(alpha=[1, 1, 1], size=N_SIMULATIONS)
print(f"\nGenerated {N_SIMULATIONS} random weight combinations for frequency analysis...")

winners = []
for w in weights:
    weighted_scores = plausibility_ranks['Mass_Rank'] * w[0] + \
                      plausibility_ranks['Drought_Rank'] * w[1] + \
                      plausibility_ranks['Flood_Rank'] * w[2]
    winner_model = weighted_scores.idxmin()
    winners.append(winner_model)

# --- Step 3: Summarize Frequency Results ---
top_1_frequency = pd.Series(winners).value_counts(normalize=True) * 100
top_1_frequency = top_1_frequency.reindex(model_list, fill_value=0)

print("\n--- Top-1 Frequency Results ---")
print("Percentage of weightings where each model is the most plausible:")
print(top_1_frequency.round(2).to_string())


# --- Step 4: Visualize the Results ---

## Figure 1: Bar Chart of Top-1 Frequency
print("\nGenerating Top-1 Frequency bar chart...")
fig, ax = plt.subplots(figsize=(12, 8))
plot_data = pd.DataFrame({'Model': top_1_frequency.index, 'Frequency': top_1_frequency.values})
sns.barplot(
    data=plot_data,
    x='Model',
    y='Frequency',
    hue='Model',
    palette=[color_list[model] for model in top_1_frequency.index],
    legend=False,
    ax=ax
)
ax.set_ylabel('Frequency of Ranking #1 (%)', fontsize=fs * 0.9)
ax.set_xlabel('Model', fontsize=fs * 0.9)
ax.set_title('Robustness of Plausibility Ranking to Metric Weights', fontsize=fs)
ax.tick_params(labelsize=fs * 0.8)
ax.set_ylim(0, 100)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "top_1_frequency.png"), dpi=300)
plt.close()

## Figure 2 : Ternary Heatmap for Decision Regions
print("Generating ternary heatmap...")

def get_winner(point, plausibility_ranks_df):
    """Calculates the winning model for a given weight point."""
    w_mass, w_drought, w_flood = point
    weighted_scores = plausibility_ranks_df['Mass_Rank'] * w_mass + \
                      plausibility_ranks_df['Drought_Rank'] * w_drought + \
                      plausibility_ranks_df['Flood_Rank'] * w_flood
    return weighted_scores.idxmin()

# Create a mapping from model name to a numerical ID for the heatmap
active_models = [model for model in model_list if model in top_1_frequency.index[top_1_frequency > 0]]
model_ids = {model: i for i, model in enumerate(active_models)}
color_map_array = [color_list[model] for model in active_models]

print(f"Active models: {active_models}")

# Generate data for the heatmap
# NOTE: We use integer loops for grid generation, but convert to float keys for storage
data = defaultdict(int)

for i in range(TERNARY_RESOLUTION + 1):
    for j in range(TERNARY_RESOLUTION - i + 1):
        k = TERNARY_RESOLUTION - i - j
        
        # Normalize coordinates to range [0, SCALE]
        # This fixes the axis labels to be 0.0 - 1.0 instead of 0 - 120
        p_i = (i / TERNARY_RESOLUTION) * TERNARY_SCALE
        p_j = (j / TERNARY_RESOLUTION) * TERNARY_SCALE
        p_k = (k / TERNARY_RESOLUTION) * TERNARY_SCALE
        
        point = (p_i, p_j, p_k)
        
        winner_model = get_winner(point, plausibility_ranks)
        if winner_model in model_ids:
            # Store data using the normalized (float) tuple as key
            data[(p_i, p_j)] = model_ids[winner_model]

# Create the ternary plot
# Set Scale to 1.0 so axes are 0-1
fig, tax = ternary.figure(scale=TERNARY_SCALE)
fig.set_size_inches(8, 6) 

# Draw Boundary and Grid
tax.boundary(linewidth=3.0)
tax.gridlines(color="gray", multiple=TERNARY_SCALE / 5, linewidth=0.5, ls='-', alpha=0.3)

# Axis Ticks
# multiple: step size for ticks (e.g., 0.2 for 0.0, 0.2, 0.4...)
# fontsize: increased to 16 as requested
tax.ticks(axis='lbr', 
          linewidth=1, 
          multiple=TERNARY_SCALE / 5, 
          tick_formats="%.0f", 
          offset=0.03, 
          fontsize=16)

# Heatmap Drawing
cmap = ListedColormap(color_map_array)
data_dict = dict(data)

tax.heatmap(data_dict, 
            cmap=cmap, 
            colorbar=False, 
            vmin=0, 
            vmax=len(active_models)-1, 
            style="hexagonal") 

# Enhanced Axis Labels
#tax.set_title("Model Ranking Sensitivity Analysis", fontsize=fs, pad=40)
tax.bottom_axis_label("Mass Conservation Importance\n(Low ← → High)", fontsize=fs * 0.8, offset=0.18)
tax.right_axis_label("Drought Performance Importance\n(High ← → Low)", fontsize=fs * 0.8, offset=0.20)
tax.left_axis_label("Flood Performance Importance\n(High ← → Low)", fontsize=fs * 0.8, offset=0.20)

# Legend Configuration
legend_elements = []
for model in model_list:
    freq = top_1_frequency[model] if model in top_1_frequency.index else 0.0
    ranks = plausibility_ranks.loc[model]
    label = f"{model} ({freq:.1f}%)"
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color_list[model], 
                                       edgecolor='black', linewidth=0.5, label=label))

tax.legend(handles=legend_elements, 
           loc='upper left', 
           bbox_to_anchor=(1.05, 0.9), 
           fontsize=fs * 0.6, 
           title="Dominance (%)", 
           title_fontsize=fs * 0.6, 
           frameon=True, 
           fancybox=True)

tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')

plt.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.15)
fig.savefig(os.path.join(output_dir, "f06_plausibility_ternary_plot_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAnalysis complete. Figures saved to:\n{output_dir}/")

# --- Additional Dirichlet Sensitivity Analysis ---

## Figure 3: Weight Distribution Analysis
print("Generating weight distribution analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot weight distributions
weight_names = ['Mass Conservation', 'Drought Performance', 'Flood Performance']
for i, name in enumerate(weight_names):
    ax = axes[0, i] if i < 2 else axes[1, 0]
    ax.hist(weights[:, i], bins=50, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][i], density=True)
    ax.set_title(f'{name} Weight Distribution', fontsize=fs * 0.7)
    ax.set_xlabel('Weight Value', fontsize=fs * 0.6)
    ax.set_ylabel('Density', fontsize=fs * 0.6)
    ax.tick_params(labelsize=fs * 0.5)
    ax.axvline(np.mean(weights[:, i]), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(weights[:, i]):.3f}')
    ax.legend(fontsize=fs * 0.5)

# Correlation matrix of weights
ax = axes[1, 1]
weight_df = pd.DataFrame(weights, columns=weight_names)
corr_matrix = weight_df.corr()
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(weight_names)))
ax.set_yticks(range(len(weight_names)))
ax.set_xticklabels([name.replace(' ', '\n') for name in weight_names], fontsize=fs * 0.5)
ax.set_yticklabels([name.replace(' ', '\n') for name in weight_names], fontsize=fs * 0.5)
ax.set_title('Weight Correlation Matrix\n(Dirichlet Constraint)', fontsize=fs * 0.7)

# Add correlation values
for i in range(len(weight_names)):
    for j in range(len(weight_names)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}', ha='center', va='center', 
               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black', fontsize=fs * 0.5)

plt.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "weight_distribution_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()