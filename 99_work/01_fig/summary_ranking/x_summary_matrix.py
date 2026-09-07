# Create a matrix of the ranking of models based on their performance metrics of 
# benchmark, and physical plausability in mass conservation, drought condition, and flood condition
# Step 1: Rank Each Model Within Each Plausibility Test
# First, for each of your three plausibility tests, you will rank the models from best (1) to worst (6) based on their performance. This normalizes the results from different tests into a common scale.
### Mass Conservation Rank:
# Calculate the absolute mass bias for each model.
# The model with the value closest to 0 gets Rank 1 (Best). The model with the value furthest from 0 gets Rank 6 (Worst).
### Drought Condition Rank:
# Use the mean discharge during the zero-precipitation test.
# The model with the mean discharge closest to 0 gets Rank 1 (Best). The model with the highest mean discharge gets Rank 6 (Worst).
### Flood Condition Rank:
# This is a qualitative rank based on physical realism.
# Rank 1 (Best): Models that show a plausible, consistently increasing peak discharge with increasing precipitation (like HBV and BC).
# Rank 2-4 (Good/Fair): Models that show some increase but perhaps not as consistently (like TCN and LSTM).
# Rank 5-6 (Worst): Models that show a saturated, flat, or decreasing response to increased precipitation (like RF and XGBoost).
# Step 2: Calculate the Composite Plausibility Score
# Now, for each model, simply sum the ranks from the three tests. This gives you a single, unified score for physical plausibility.
# The summary figure should have 
# Y-Axis: Predictive Accuracy (Median KGE). Higher is better.
# X-Axis: Composite Plausibility Score. Lower is better.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
output_dir = "/data0/funato/hsp/0_out/1_fig/summary_ranking"
os.makedirs(output_dir, exist_ok=True)

model_list = ["HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
fs = 24

# Define colors for each model
color_list = {
    "HBV": "#7B7B7B",    # Medium Grey
    "BC": "#628AE8",  # Steel Blue
    "RF": "#FF7F00",     # Orange
    "XGB": "#9C4803",    # Dark Orange
    "TCN": "#E41A1C",    # Red
    "LSTM": "#9808B2"    # Purple
}

print("Loading and processing model performance data...")

# Step 1: Load KGE performance data (predictive accuracy)
kge_scores = {}

for model in model_list:
    if model == "BC":
        input_file = "/data0/funato/hsp/0_out/BC/results/BC_results_eva.csv"
        kge_col = "BC_KGE_eva"
    else:
        input_file = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/results/{model}_results_eva.csv"
        kge_col = f"{model}_KGE_eva"
    
    try:
        df = pd.read_csv(input_file)
        if kge_col in df.columns:
            kge_values = df[kge_col].dropna().values
            kge_scores[model] = np.median(kge_values)
            print(f"{model} median KGE: {kge_scores[model]:.3f}")
        else:
            print(f"Warning: KGE column not found for {model}")
            kge_scores[model] = np.nan
    except FileNotFoundError:
        print(f"Warning: Results file not found for {model}")
        kge_scores[model] = np.nan

# Step 2: Load Mass Conservation data (absolute mass bias)
mass_bias_scores = {}

mass_bias_file = "/data0/funato/hsp/0_out/analysis/mass_conservation/mass_bias_data.csv"
mass_df = pd.read_csv(mass_bias_file)

for model in model_list:
    model_name = "BC" if model == "BC" else model
    
    # Filter data for this model
    model_data = mass_df[mass_df['Model'] == model_name]
    
    if not model_data.empty:
        bias_values = model_data['Mass_Bias'].dropna().values
        mass_bias_scores[model] = np.median(np.abs(bias_values))
        print(f"{model} median absolute mass bias: {mass_bias_scores[model]:.3f}")
    else:
        print(f"Warning: No mass bias data found for {model}")
        mass_bias_scores[model] = np.nan
            

# Step 3: Load Drought Condition data (mean discharge during zero precipitation)
drought_scores = {}

drought_file = "/data0/funato/hsp/0_out/analysis/no_precip_discharge/no_precip_model_summary.csv"
drought_df = pd.read_csv(drought_file)

for model in model_list:
    model_name = "BC" if model == "BC" else model
    
    # Find the row for this model
    model_row = drought_df[drought_df['Model'] == model_name]
    
    if not model_row.empty:
        drought_scores[model] = model_row['Mean_Mean_Discharge'].iloc[0]
        print(f"{model} mean drought discharge: {drought_scores[model]:.3f}")
    else:
        print(f"Warning: No drought discharge data found for {model}")
        drought_scores[model] = np.nan
# Step 4: Define Flood Condition ranks (simplified assessment)
# Based on user specification: 1 for all models except RF and XGB (which get 5)
flood_ranks = {
    "HBV": 1,
    "BC": 1, 
    "TCN": 1,
    "LSTM": 1,
    "RF": 6,     # Poor response
    "XGB":6     # Poor response
}

print("\nCalculating ranks...")

# Step 5: Calculate ranks for each metric
def rank_values(scores_dict, lower_is_better=True):
    """Rank values where 1 is best, 6 is worst"""
    valid_scores = {k: v for k, v in scores_dict.items() if not np.isnan(v)}
    if not valid_scores:
        return {k: np.nan for k in scores_dict.keys()}
    
    sorted_models = sorted(valid_scores.items(), key=lambda x: x[1], reverse=not lower_is_better)
    ranks = {}
    for i, (model, score) in enumerate(sorted_models):
        ranks[model] = i + 1
    
    # Fill in NaN scores with worst rank
    for model, score in scores_dict.items():
        if np.isnan(score):
            ranks[model] = len(model_list)
    
    return ranks

# Calculate ranks for each metric
kge_ranks = rank_values(kge_scores, lower_is_better=False)  # Higher KGE is better
mass_ranks = rank_values(mass_bias_scores, lower_is_better=True)  # Lower bias is better
drought_ranks = rank_values(drought_scores, lower_is_better=True)  # Lower discharge is better

print("\nRanking Results:")
print("Model\tKGE_Rank\tMass_Rank\tDrought_Rank\tFlood_Rank")
for model in model_list:
    print(f"{model}\t{kge_ranks[model]}\t\t{mass_ranks[model]}\t\t{drought_ranks[model]}\t\t{flood_ranks[model]}")

# Step 6: Calculate composite plausibility score (sum of physical plausibility ranks)
composite_scores = {}
for model in model_list:
    composite_scores[model] = mass_ranks[model] + drought_ranks[model] + flood_ranks[model]

print("\nComposite Plausibility Scores (lower is better):")
for model in model_list:
    print(f"{model}: {composite_scores[model]}")

# Step 7: Create the summary ranking plot
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# Plot each model
for model in model_list:
    if not np.isnan(kge_scores[model]):
        ax.scatter(composite_scores[model], kge_scores[model], 
                  color=color_list[model], s=200, alpha=0.8, 
                  edgecolors='black', linewidth=2, label=model)
        
        # Add model labels offset slightly
        ax.annotate(model, (composite_scores[model], kge_scores[model]),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=fs*0.9)

# Formatting
ax.set_xlabel('Composite Plausibility Rank Score (Min 3, Max 18)', fontsize=fs)
ax.set_ylabel('Median KGE', fontsize=fs)
ax.set_title('Model Performance', fontsize=fs)
ax.tick_params(labelsize=fs*0.8)
ax.grid(True, alpha=0.3)

# Set axis limits with some padding
x_min, x_max = min(composite_scores.values()) - 0.5, max(composite_scores.values()) + 0.5
y_min, y_max = min([v for v in kge_scores.values() if not np.isnan(v)]) - 0.05, \
               max([v for v in kge_scores.values() if not np.isnan(v)]) + 0.05
ax.set_xlim(3,18)
ax.set_ylim(y_min -0.1, 1.0)

# Add quadrant labels
ax.text(0.02, 0.98, 'High Accuracy\nHigh Plausibility', 
        transform=ax.transAxes, fontsize=fs*0.8, ha='left', va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

ax.text(0.98, 0.98, 'High Accuracy\nLow Plausibility', 
        transform=ax.transAxes, fontsize=fs*0.8, ha='right', va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

ax.text(0.02, 0.02, 'Low Accuracy\nHigh Plausibility', 
        transform=ax.transAxes, fontsize=fs*0.8, ha='left', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

ax.text(0.98, 0.02, 'Low Accuracy\nLow Plausibility', 
        transform=ax.transAxes, fontsize=fs*0.8, ha='right', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f"{output_dir}/model_ranking_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# Step 8: Create detailed ranking table
ranking_df = pd.DataFrame({
    'Model': model_list,
    'KGE_Score': [kge_scores[model] for model in model_list],
    'KGE_Rank': [kge_ranks[model] for model in model_list],
    'Mass_Bias': [mass_bias_scores[model] for model in model_list],
    'Mass_Rank': [mass_ranks[model] for model in model_list],
    'Drought_Discharge': [drought_scores[model] for model in model_list],
    'Drought_Rank': [drought_ranks[model] for model in model_list],
    'Flood_Rank': [flood_ranks[model] for model in model_list],
    'Composite_Plausibility': [composite_scores[model] for model in model_list]
})

# Save ranking table
ranking_df.to_csv(f"{output_dir}/model_ranking_table.csv", index=False)

print(f"\nResults saved to {output_dir}/")
print("Files generated:")
print("- model_ranking_matrix.png")
print("- model_ranking_table.csv")

print("\nRanking Summary:")
print(ranking_df.round(3))

# Identify best overall model
best_models = ranking_df.nsmallest(3, 'Composite_Plausibility')
print(f"\nTop 3 most physically plausible models:")
for idx, row in best_models.iterrows():
    print(f"{row['Model']}: Composite Score = {row['Composite_Plausibility']}, KGE = {row['KGE_Score']:.3f}")

print("\nAnalysis complete!")

