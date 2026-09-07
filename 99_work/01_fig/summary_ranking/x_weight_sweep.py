import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
import os
from collections import defaultdict

# --- Configuration ---
# Directory to save the output figures
output_dir = "/data0/funato/hsp/0_out/1_fig/summary_ranking_multi/sensitivity_analysis"
os.makedirs(output_dir, exist_ok=True)

# Number of random weight combinations for the Top-1 Frequency analysis
N_SIMULATIONS = 10000
# Resolution for the ternary heatmap grid. Higher is smoother but slower.
TERNARY_RESOLUTION = 100

# Model names and the color palette
model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
color_list = {
    "HBV": "#7B7B7B",    # Medium Grey
    "BC": "#628AE8",   # Steel Blue
    "RF": "#FF7F00",      # Orange
    "XGB": "#9C4803",    # Dark Orange
    "TCN": "#E41A1C",     # Red
    "LSTM": "#9808B2"    # Purple
}
fs = 24 # Font size for plots

# --- Input Data: Plausibility Ranks ---
rank_df = pd.read_csv("/data0/funato/hsp/0_out/analysis/summary_ranking_multi/automated_ranks/automated_plausibility_ranks.csv")
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
# Only include models that actually appear in the results
active_models = [model for model in model_list if model in top_1_frequency.index[top_1_frequency > 0]]
model_ids = {model: i for i, model in enumerate(active_models)}
# Create a mapping from the numerical ID back to the color (only for active models)
color_map_array = [color_list[model] for model in active_models]
print(f"Active models: {active_models}")
print(f"Model IDs: {model_ids}")
print(f"Color map array: {color_map_array}")
print(f"Active models: {active_models}")
print(f"Color map: {color_map_array}")

# Generate data for the heatmap
data = defaultdict(int)
for i in range(TERNARY_RESOLUTION + 1):
    for j in range(TERNARY_RESOLUTION - i + 1):
        k = TERNARY_RESOLUTION - i - j
        point = (i / TERNARY_RESOLUTION, j / TERNARY_RESOLUTION, k / TERNARY_RESOLUTION)
        winner_model = get_winner(point, plausibility_ranks)
        # Only include if the model is in our active models
        if winner_model in model_ids:
            data[(i, j)] = model_ids[winner_model]

# Create the ternary plot with better tick marks
fig, tax = ternary.figure(scale=TERNARY_RESOLUTION)
tax.boundary(linewidth=2.0)
tax.gridlines(color="gray", multiple=TERNARY_RESOLUTION / 5, linewidth=0.5, ls='-', alpha=0.3)
# Add tick labels to show the scale clearly
tax.ticks(axis='lbr', linewidth=1, multiple=TERNARY_RESOLUTION / 5, tick_formats="%.1f", offset=0.02)

# Draw the heatmap (without colorbar since legend explains the colors)
from matplotlib.colors import ListedColormap
print(f"Data sample: {list(data.items())[:5]}")
print(f"Data values range: {min(data.values())} to {max(data.values())}")
cmap = ListedColormap(color_map_array)
# Convert defaultdict to regular dict to avoid RuntimeError during iteration
data_dict = dict(data)
# Ensure the colormap covers the right range of values
tax.heatmap(data_dict, cmap=cmap, colorbar=False, vmin=0, vmax=len(active_models)-1)

# Enhanced axis labels and title with better interpretability
tax.set_title("Model Ranking Sensitivity Analysis", fontsize=fs * 0.7, pad=15)
tax.bottom_axis_label("\nMass Conservation Importance\n(Low ← → High)", fontsize=fs * 0.7, offset=0.08)
tax.right_axis_label("Drought Performance Importance\n(High ← → Low)", fontsize=fs * 0.7, offset=0.15)
tax.left_axis_label("Flood Performance Importance\n(High ← → Low)", fontsize=fs * 0.7, offset=0.15)

# Add corner annotations outside the triangle boundaries
# Use matplotlib axes to position text outside triangle
"""
fig_ax = tax.get_axes()
# Mass focus - bottom right, outside triangle
fig_ax.text(0.95, 0.0, "Mass\nFocus", transform=fig_ax.transAxes, 
           fontsize=fs * 0.6, ha='center', va='bottom', color='black', weight='bold')
# Drought focus - top, outside triangle
fig_ax.text(0.15, 0.95, "Drought\nFocus", transform=fig_ax.transAxes,
           fontsize=fs * 0.6, ha='center', va='top', color='black', weight='bold')
# Flood focus - bottom left, outside triangle
fig_ax.text(0.0, 0.0, "Flood\nFocus", transform=fig_ax.transAxes,
           fontsize=fs * 0.6, ha='center', va='bottom', color='black', weight='bold')"""

# Enhanced legend with ALL models (including those with 0% dominance) - fix color display
legend_elements = []
for model in model_list:
    freq = top_1_frequency[model] if model in top_1_frequency.index else 0.0
    ranks = plausibility_ranks.loc[model]
    label = f"{model} ({freq:.1f}%)\nRanks: {ranks['Mass_Rank']}|{ranks['Drought_Rank']}|{ranks['Flood_Rank']}"
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color_list[model], 
                                       edgecolor='black', linewidth=0.5, label=label))

# Position legend better and add explanation
tax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5), 
          fontsize=fs * 0.55, title="All Models (Dominance %)\nRanks: Mass|Drought|Flood\n(Lower rank = Better)", 
          title_fontsize=fs * 0.6, frameon=True, fancybox=True)

tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)  # Make room for external labels and legend
fig.savefig(os.path.join(output_dir, "plausibility_ternary_plot_heatmap.png"), dpi=300, bbox_inches='tight')
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

## Figure 4: Sensitivity Metrics and Stability Analysis
print("Generating sensitivity and stability analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate dominance regions (area where each model is dominant)
dominance_areas = {}
total_points = sum(1 for i in range(TERNARY_RESOLUTION + 1) for j in range(TERNARY_RESOLUTION - i + 1))
# Initialize all models with 0 dominance
for model in model_list:
    dominance_areas[model] = 0.0

# Only calculate for active models
for model in active_models:
    if model in model_ids:
        model_id = model_ids[model]
        count = sum(1 for point_data in data.values() if point_data == model_id)
        dominance_areas[model] = (count / total_points) * 100

# Plot dominance areas
ax = axes[0, 0]
models_with_area = [model for model in model_list if dominance_areas[model] > 0]
areas = [dominance_areas[model] for model in models_with_area]
colors = [color_list[model] for model in models_with_area]
bars = ax.bar(models_with_area, areas, color=colors, alpha=0.8)
ax.set_title('Model Dominance Regions\n(% of Weight Space)', fontsize=fs * 0.8)
ax.set_ylabel('Dominance Area (%)', fontsize=fs * 0.7)
ax.set_xlabel('Model', fontsize=fs * 0.7)
ax.tick_params(labelsize=fs * 0.6)
for bar, area in zip(bars, areas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{area:.1f}%', 
           ha='center', va='bottom', fontsize=fs * 0.6)

# Stability analysis - compare frequency and dominance area
ax = axes[0, 1]
freq_data = [top_1_frequency[model] for model in models_with_area]
area_data = [dominance_areas[model] for model in models_with_area]
ax.scatter(area_data, freq_data, c=colors, s=100, alpha=0.8)
for i, model in enumerate(models_with_area):
    ax.annotate(model, (area_data[i], freq_data[i]), xytext=(5, 5), 
               textcoords='offset points', fontsize=fs * 0.6)
ax.plot([0, max(max(area_data), max(freq_data))], [0, max(max(area_data), max(freq_data))], 
        'k--', alpha=0.5, label='Perfect Agreement')
ax.set_xlabel('Dominance Area (% of Weight Space)', fontsize=fs * 0.7)
ax.set_ylabel('Random Sampling Frequency (%)', fontsize=fs * 0.7)
ax.set_title('Consistency: Area vs Frequency', fontsize=fs * 0.8)
ax.legend(fontsize=fs * 0.6)
ax.tick_params(labelsize=fs * 0.6)

# Ranking stability - how often each model appears in top 3
rank_stability = defaultdict(lambda: defaultdict(int))
for w in weights[:1000]:  # Use subset for efficiency
    weighted_scores = plausibility_ranks['Mass_Rank'] * w[0] + \
                      plausibility_ranks['Drought_Rank'] * w[1] + \
                      plausibility_ranks['Flood_Rank'] * w[2]
    ranked_models = weighted_scores.sort_values().index.tolist()
    for rank, model in enumerate(ranked_models[:3]):
        rank_stability[rank + 1][model] += 1

# Convert to percentages
for rank in rank_stability:
    total = sum(rank_stability[rank].values())
    for model in rank_stability[rank]:
        rank_stability[rank][model] = (rank_stability[rank][model] / total) * 100

# Plot ranking stability
ax = axes[1, 0]
rank_data = {model: [rank_stability[1].get(model, 0), rank_stability[2].get(model, 0), 
                    rank_stability[3].get(model, 0)] for model in model_list}
x_pos = np.arange(len(model_list))
width = 0.25
for i, rank in enumerate([1, 2, 3]):
    values = [rank_data[model][i] for model in model_list]
    ax.bar(x_pos + i * width, values, width, label=f'Rank {rank}', 
           alpha=0.8, color=f'C{i}')
ax.set_title('Ranking Stability Analysis', fontsize=fs * 0.8)
ax.set_xlabel('Model', fontsize=fs * 0.7)
ax.set_ylabel('Frequency in Top 3 (%)', fontsize=fs * 0.7)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(model_list, rotation=45, fontsize=fs * 0.6)
ax.legend(fontsize=fs * 0.6)
ax.tick_params(labelsize=fs * 0.6)

# Weight sensitivity - how ranking changes with weight perturbation
ax = axes[1, 1]
base_weights = [1/3, 1/3, 1/3]  # Equal weights baseline
perturbation_range = np.linspace(0.1, 0.9, 20)
sensitivity_scores = {model: [] for model in model_list}

for mass_weight in perturbation_range:
    remaining = 1 - mass_weight
    drought_weight = remaining / 2
    flood_weight = remaining / 2
    w = [mass_weight, drought_weight, flood_weight]
    
    weighted_scores = plausibility_ranks['Mass_Rank'] * w[0] + \
                      plausibility_ranks['Drought_Rank'] * w[1] + \
                      plausibility_ranks['Flood_Rank'] * w[2]
    
    for model in model_list:
        sensitivity_scores[model].append(weighted_scores[model])

for model in model_list:
    ax.plot(perturbation_range, sensitivity_scores[model], 'o-', 
           color=color_list[model], label=model, alpha=0.8, linewidth=2)

ax.set_xlabel('Mass Conservation Weight', fontsize=fs * 0.7)
ax.set_ylabel('Weighted Ranking Score', fontsize=fs * 0.7)
ax.set_title('Ranking Sensitivity to Weight Changes', fontsize=fs * 0.8)
ax.legend(fontsize=fs * 0.5, loc='best')
ax.tick_params(labelsize=fs * 0.6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "sensitivity_stability_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Summary Statistics ---
print("\n" + "="*60)
print("COMPREHENSIVE SENSITIVITY ANALYSIS SUMMARY")
print("="*60)

print(f"\n1. DOMINANCE ANALYSIS (Based on {total_points:,} weight combinations):")
for model in sorted(dominance_areas.keys(), key=lambda x: dominance_areas[x], reverse=True):
    if dominance_areas[model] > 0:
        print(f"   {model:8}: {dominance_areas[model]:6.2f}% of weight space")

print(f"\n2. RANDOM SAMPLING FREQUENCY (Based on {N_SIMULATIONS:,} Dirichlet samples):")
for model in top_1_frequency.sort_values(ascending=False).index:
    if top_1_frequency[model] > 0:
        print(f"   {model:8}: {top_1_frequency[model]:6.2f}% frequency")

print(f"\n3. RANKING CONSISTENCY:")
for model in model_list:
    if model in models_with_area:
        area = dominance_areas[model]
        freq = top_1_frequency[model]
        consistency = min(area, freq) / max(area, freq) * 100 if max(area, freq) > 0 else 0
        print(f"   {model:8}: {consistency:6.1f}% consistency (area vs frequency)")

print(f"\n4. WEIGHT DISTRIBUTION STATISTICS:")
print(f"   Mean weights: Mass={np.mean(weights[:, 0]):.3f}, Drought={np.mean(weights[:, 1]):.3f}, Flood={np.mean(weights[:, 2]):.3f}")
print(f"   Std weights:  Mass={np.std(weights[:, 0]):.3f}, Drought={np.std(weights[:, 1]):.3f}, Flood={np.std(weights[:, 2]):.3f}")

most_robust = max(top_1_frequency.items(), key=lambda x: x[1])
print(f"\n5. MOST ROBUST MODEL: {most_robust[0]} ({most_robust[1]:.1f}% dominance)")

print(f"\nAll analysis figures saved to: {output_dir}/")
print("="*60)