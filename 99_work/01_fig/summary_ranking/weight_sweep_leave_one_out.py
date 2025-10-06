import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kendalltau, zscore, rankdata
from collections import defaultdict
from sklearn.preprocessing import QuantileTransformer

# --- Configuration ---
# Directory to save the output figures
output_dir = "/data0/funato/hsp/0_out/1_fig/summary_ranking_multi/leave_one_out_analysis"
os.makedirs(output_dir, exist_ok=True)

# Number of random weight combinations for analysis
N_SIMULATIONS = 10000

# Model names and color palette
model_list = ["HBV", "HYPER", "RF", "XGB", "TCN", "LSTM"]
color_list = {
    "HBV": "#7B7B7B",    # Medium Grey
    "HYPER": "#628AE8",   # Steel Blue
    "RF": "#FF7F00",      # Orange
    "XGB": "#9C4803",    # Dark Orange
    "TCN": "#E41A1C",     # Red
    "LSTM": "#9808B2"    # Purple
}
fs = 18  # Font size for plots

# --- Input Data: Plausibility Ranks ---
# Original data with three metrics: Mass Conservation (A), Drought Performance (I), Flood Performance (P)
plausibility_ranks_raw = pd.DataFrame({
    'Model': model_list,
    'Mass_Rank':    [4, 2, 6, 1, 3, 5],    # A - Ascendant (Mass conservation)
    'Drought_Rank': [1, 3, 6, 5, 2, 4],    # I - Intermediate (Drought performance)
    'Flood_Rank':   [1, 1, 6, 6, 1, 1]     # P - Peak (Flood performance)
}).set_index('Model')

print("--- Input Plausibility Ranks (Lower is better) ---")
print(plausibility_ranks_raw)

# --- Step 1: Create Leave-One-Metric-Out Combinations ---
# Create pairs: A+I, A+P, I+P (leaving out P, I, A respectively)
metric_combinations = {
    'No Flood': ['Mass_Rank', 'Drought_Rank'],     # Leave out P
    'No Drought': ['Mass_Rank', 'Flood_Rank'],     # Leave out I  
    'No Mass': ['Drought_Rank', 'Flood_Rank']      # Leave out A
}

print(f"\n--- Leave-One-Metric-Out Combinations ---")
for name, metrics in metric_combinations.items():
    print(f"{name}: {metrics}")

# --- Step 2: Define Alternative Normalization Methods ---
def apply_normalization(data, method='raw'):
    """Apply different normalization methods to the ranking data."""
    if method == 'raw':
        return data.copy()
    elif method == 'z_score':
        # Z-score normalization (mean=0, std=1)
        return data.apply(zscore, axis=0)
    elif method == 'quantile':
        # Quantile normalization (uniform distribution)
        qt = QuantileTransformer(output_distribution='uniform')
        normalized = pd.DataFrame(
            qt.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        return normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")

normalization_methods = ['raw', 'z_score', 'quantile']
print(f"\nNormalization methods: {normalization_methods}")

# --- Step 3: Compute Rankings for All Combinations ---
def compute_ranking_for_combination(data, metric_cols, weights=None):
    """Compute weighted ranking for a given combination of metrics."""
    if weights is None:
        weights = [1.0] * len(metric_cols)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    weighted_scores = np.zeros(len(data))
    for i, col in enumerate(metric_cols):
        weighted_scores += data[col] * weights[i]
    
    # Create ranking (lower score = better rank)
    ranking = pd.Series(weighted_scores, index=data.index).rank()
    return ranking

def generate_random_weights(n_metrics, n_samples):
    """Generate random weights that sum to 1."""
    if n_metrics == 2:
        # For 2 metrics, use Beta distribution to get (w, 1-w)
        w1 = np.random.beta(1, 1, n_samples)
        return np.column_stack([w1, 1-w1])
    else:
        # For 3+ metrics, use Dirichlet distribution
        return np.random.dirichlet([1] * n_metrics, n_samples)

# --- Step 4: Perform Leave-One-Out Analysis with Different Normalizations ---
results = {}
kendall_tau_results = {}

print(f"\n--- Computing Rankings for All Combinations ---")

# Store baseline ranking (all three metrics, equal weights, raw normalization)
baseline_data = apply_normalization(plausibility_ranks_raw, 'raw')
baseline_ranking = compute_ranking_for_combination(
    baseline_data, 
    ['Mass_Rank', 'Drought_Rank', 'Flood_Rank'], 
    [1/3, 1/3, 1/3]
)

print(f"Baseline ranking (all metrics, equal weights):")
print(baseline_ranking.sort_values().to_string())

for norm_method in normalization_methods:
    print(f"\n  Processing normalization: {norm_method}")
    
    # Apply normalization
    normalized_data = apply_normalization(plausibility_ranks_raw, norm_method)
    print(f"    Normalized data for {norm_method}:")
    print(f"    {normalized_data.round(3)}")
    
    results[norm_method] = {}
    kendall_tau_results[norm_method] = {}
    
    for combo_name, metric_cols in metric_combinations.items():
        print(f"    Computing for combination: {combo_name}")
        
        # Generate random weights for this combination
        n_metrics = len(metric_cols)
        weights = generate_random_weights(n_metrics, N_SIMULATIONS)
        
        # Compute rankings for all weight combinations
        rankings_list = []
        winners_list = []
        
        for w in weights:
            ranking = compute_ranking_for_combination(normalized_data, metric_cols, w)
            rankings_list.append(ranking)
            # Winner is the model with rank 1 (best)
            winner = ranking.idxmin()
            winners_list.append(winner)
        
        # Store results
        results[norm_method][combo_name] = {
            'rankings': rankings_list,
            'winners': winners_list,
            'weights': weights,
            'metric_cols': metric_cols
        }
        
        # Compute Kendall's tau with baseline for each ranking
        tau_values = []
        for ranking in rankings_list:
            tau, _ = kendalltau(baseline_ranking.rank(), ranking.rank())
            tau_values.append(tau)
        
        kendall_tau_results[norm_method][combo_name] = {
            'tau_values': tau_values,
            'mean_tau': np.mean(tau_values),
            'std_tau': np.std(tau_values),
            'median_tau': np.median(tau_values)
        }
        
        print(f"      Mean Kendall's tau: {np.mean(tau_values):.3f} ± {np.std(tau_values):.3f}")

# --- Step 5: Analyze Winner Frequencies ---
print(f"\n--- Winner Frequency Analysis ---")

winner_frequencies = {}
for norm_method in normalization_methods:
    winner_frequencies[norm_method] = {}
    
    print(f"\nNormalization: {norm_method}")
    for combo_name in metric_combinations.keys():
        winners = results[norm_method][combo_name]['winners']
        freq = pd.Series(winners).value_counts(normalize=True) * 100
        freq = freq.reindex(model_list, fill_value=0)
        winner_frequencies[norm_method][combo_name] = freq
        
        print(f"  {combo_name}:")
        for model in model_list:
            print(f"    {model:8}: {freq[model]:6.2f}%")

# --- Step 6: Create Comprehensive Visualizations ---

## Figure 1: Winner Frequency Comparison Across Normalizations
print(f"\nGenerating winner frequency comparison plots...")

fig, axes = plt.subplots(len(normalization_methods), len(metric_combinations), 
                        figsize=(18, 12))

for i, norm_method in enumerate(normalization_methods):
    for j, combo_name in enumerate(metric_combinations.keys()):
        ax = axes[i, j] if len(normalization_methods) > 1 else axes[j]
        
        freq_data = winner_frequencies[norm_method][combo_name]
        colors = [color_list[model] for model in model_list]
        
        bars = ax.bar(model_list, freq_data.values, color=colors, alpha=0.8)
        ax.set_title(f'{combo_name}\n({norm_method})', fontsize=fs*0.9)
        ax.set_ylabel('Winner Frequency (%)', fontsize=fs*0.8)
        ax.set_xlabel('Model', fontsize=fs*0.8)
        ax.tick_params(labelsize=fs*0.7)
        ax.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, freq in zip(bars, freq_data.values):
            if freq > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{freq:.1f}%', ha='center', va='bottom', fontsize=fs*0.6)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "winner_frequency_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

## Figure 2: Kendall's Tau Analysis
print("Generating Kendall's tau analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Box plot of Kendall's tau values
ax = axes[0, 0]
tau_data = []
labels = []
colors_tau = []

for norm_method in normalization_methods:
    for combo_name in metric_combinations.keys():
        tau_values = kendall_tau_results[norm_method][combo_name]['tau_values']
        tau_data.append(tau_values)
        labels.append(f'{combo_name}\n({norm_method})')
        # Use a consistent color scheme
        colors_tau.append(['lightblue', 'lightgreen', 'lightcoral'][list(metric_combinations.keys()).index(combo_name)])

box_plot = ax.boxplot(tau_data, labels=labels, patch_artist=True)
for patch, color in zip(box_plot['boxes'], colors_tau):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Kendall\'s Tau Distribution\n(Similarity to Baseline Ranking)', fontsize=fs)
ax.set_ylabel('Kendall\'s Tau', fontsize=fs*0.9)
ax.tick_params(labelsize=fs*0.7)
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Mean Kendall's tau comparison
ax = axes[0, 1]
tau_means = []
tau_stds = []
combo_labels = []
norm_labels = []

for norm_method in normalization_methods:
    for combo_name in metric_combinations.keys():
        tau_means.append(kendall_tau_results[norm_method][combo_name]['mean_tau'])
        tau_stds.append(kendall_tau_results[norm_method][combo_name]['std_tau'])
        combo_labels.append(combo_name)
        norm_labels.append(norm_method)

# Create grouped bar plot
x_pos = np.arange(len(metric_combinations))
width = 0.25

for i, norm_method in enumerate(normalization_methods):
    indices = [j for j, norm in enumerate(norm_labels) if norm == norm_method]
    means = [tau_means[j] for j in indices]
    stds = [tau_stds[j] for j in indices]
    
    ax.bar(x_pos + i*width, means, width, 
           yerr=stds, capsize=5, alpha=0.8, 
           label=norm_method, color=f'C{i}')

ax.set_title('Mean Kendall\'s Tau by Combination', fontsize=fs)
ax.set_ylabel('Mean Kendall\'s Tau', fontsize=fs*0.9)
ax.set_xlabel('Metric Combination', fontsize=fs*0.9)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(list(metric_combinations.keys()), fontsize=fs*0.8)
ax.legend(fontsize=fs*0.8)
ax.tick_params(labelsize=fs*0.7)
ax.grid(True, alpha=0.3)

# Winner stability across normalizations
ax = axes[1, 0]
stability_matrix = np.zeros((len(model_list), len(metric_combinations)))

for j, combo_name in enumerate(metric_combinations.keys()):
    # Calculate coefficient of variation of winner frequencies across normalizations
    for i, model in enumerate(model_list):
        frequencies = [winner_frequencies[norm][combo_name][model] for norm in normalization_methods]
        cv = np.std(frequencies) / (np.mean(frequencies) + 1e-8)  # Add small epsilon to avoid division by zero
        stability_matrix[i, j] = cv

im = ax.imshow(stability_matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(metric_combinations)))
ax.set_yticks(range(len(model_list)))
ax.set_xticklabels(list(metric_combinations.keys()), fontsize=fs*0.8)
ax.set_yticklabels(model_list, fontsize=fs*0.8)
ax.set_title('Winner Frequency Stability\n(CV across normalizations)', fontsize=fs)
ax.set_xlabel('Metric Combination', fontsize=fs*0.9)
ax.set_ylabel('Model', fontsize=fs*0.9)

# Add text annotations
for i in range(len(model_list)):
    for j in range(len(metric_combinations)):
        ax.text(j, i, f'{stability_matrix[i, j]:.2f}', 
               ha='center', va='center', color='white' if stability_matrix[i, j] > 0.5 else 'black',
               fontsize=fs*0.6)

plt.colorbar(im, ax=ax, shrink=0.6, label='Coefficient of Variation')

# Normalization comparison heatmap
ax = axes[1, 1]
# Create a summary metric: mean winner frequency difference from baseline
baseline_winners = pd.Series(results['raw']['A+I (no Flood)']['winners']).value_counts(normalize=True) * 100
baseline_winners = baseline_winners.reindex(model_list, fill_value=0)

diff_matrix = np.zeros((len(normalization_methods), len(metric_combinations)))
for i, norm_method in enumerate(normalization_methods):
    for j, combo_name in enumerate(metric_combinations.keys()):
        current_winners = winner_frequencies[norm_method][combo_name]
        # Calculate mean absolute difference from baseline
        diff = np.mean(np.abs(current_winners - baseline_winners))
        diff_matrix[i, j] = diff

im = ax.imshow(diff_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(metric_combinations)))
ax.set_yticks(range(len(normalization_methods)))
ax.set_xticklabels(list(metric_combinations.keys()), fontsize=fs*0.8)
ax.set_yticklabels(normalization_methods, fontsize=fs*0.8)
ax.set_title('Winner Frequency Deviation\nfrom Baseline (A+I, raw)', fontsize=fs)
ax.set_xlabel('Metric Combination', fontsize=fs*0.9)
ax.set_ylabel('Normalization Method', fontsize=fs*0.9)

# Add text annotations
for i in range(len(normalization_methods)):
    for j in range(len(metric_combinations)):
        ax.text(j, i, f'{diff_matrix[i, j]:.1f}', 
               ha='center', va='center', color='white' if diff_matrix[i, j] > 10 else 'black',
               fontsize=fs*0.7)

plt.colorbar(im, ax=ax, shrink=0.6, label='Mean Abs. Difference (%)')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "kendall_tau_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

## Figure 3: Detailed Ranking Comparison
print("Generating detailed ranking comparison...")

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

for i, norm_method in enumerate(normalization_methods):
    for j, combo_name in enumerate(metric_combinations.keys()):
        ax = axes[i, j]
        
        # Get a sample of rankings for visualization
        rankings_sample = results[norm_method][combo_name]['rankings'][:100]  # First 100 rankings
        
        # Create ranking matrix
        ranking_matrix = np.array([ranking.reindex(model_list).values for ranking in rankings_sample])
        
        # Plot heatmap
        im = ax.imshow(ranking_matrix.T, cmap='RdYlGn', aspect='auto', vmin=1, vmax=6)
        ax.set_title(f'{combo_name}\n({norm_method})', fontsize=fs*0.9)
        ax.set_ylabel('Model', fontsize=fs*0.8)
        ax.set_xlabel('Sample Rankings', fontsize=fs*0.8)
        ax.set_yticks(range(len(model_list)))
        ax.set_yticklabels(model_list, fontsize=fs*0.7)
        ax.tick_params(labelsize=fs*0.7)
        
        if i == 0 and j == len(metric_combinations) - 1:  # Add colorbar to top-right plot
            plt.colorbar(im, ax=ax, shrink=0.6, label='Rank (1=Best)')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "detailed_ranking_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Step 7: Generate Summary Statistics and Report ---
print(f"\n" + "="*80)
print("LEAVE-ONE-METRIC-OUT & SCALING CHECK ANALYSIS SUMMARY")
print("="*80)

print(f"\n1. KENDALL'S TAU STABILITY ANALYSIS:")
print(f"   (Correlation with baseline: all metrics, equal weights, raw normalization)")
print(f"   Range: -1 (complete disagreement) to +1 (complete agreement)")

for norm_method in normalization_methods:
    print(f"\n   {norm_method.upper()} Normalization:")
    for combo_name in metric_combinations.keys():
        tau_stats = kendall_tau_results[norm_method][combo_name]
        print(f"     {combo_name:20}: τ = {tau_stats['mean_tau']:6.3f} ± {tau_stats['std_tau']:5.3f} "
              f"(median: {tau_stats['median_tau']:6.3f})")

print(f"\n2. WINNER FREQUENCY STABILITY:")
print(f"   (How often each model wins across all weight combinations)")

# Calculate overall stability metrics
most_stable_models = {}
most_unstable_models = {}

for norm_method in normalization_methods:
    print(f"\n   {norm_method.upper()} Normalization:")
    for combo_name in metric_combinations.keys():
        freq = winner_frequencies[norm_method][combo_name]
        winner = freq.idxmax()
        max_freq = freq.max()
        print(f"     {combo_name:20}: {winner} dominates ({max_freq:5.1f}%)")
        
        # Track most stable (highest max frequency) and unstable (lowest max frequency)
        if combo_name not in most_stable_models:
            most_stable_models[combo_name] = (winner, max_freq, norm_method)
            most_unstable_models[combo_name] = (winner, max_freq, norm_method)
        else:
            if max_freq > most_stable_models[combo_name][1]:
                most_stable_models[combo_name] = (winner, max_freq, norm_method)
            if max_freq < most_unstable_models[combo_name][1]:
                most_unstable_models[combo_name] = (winner, max_freq, norm_method)

print(f"\n3. NORMALIZATION SENSITIVITY:")
print(f"   (How much rankings change with different scaling methods)")

for combo_name in metric_combinations.keys():
    print(f"\n   {combo_name}:")
    
    # Calculate range of winner frequencies across normalizations
    for model in model_list:
        frequencies = [winner_frequencies[norm][combo_name][model] for norm in normalization_methods]
        min_freq, max_freq = min(frequencies), max(frequencies)
        range_freq = max_freq - min_freq
        if range_freq > 0:
            print(f"     {model:8}: {min_freq:5.1f}% - {max_freq:5.1f}% (range: {range_freq:5.1f}%)")

print(f"\n4. METRIC IMPORTANCE ANALYSIS:")
print(f"   (Impact of leaving out each metric)")

# Compare the impact of leaving out each metric
baseline_tau_all = np.mean([kendall_tau_results[norm]['A+I (no Flood)']['mean_tau'] + 
                           kendall_tau_results[norm]['A+P (no Drought)']['mean_tau'] + 
                           kendall_tau_results[norm]['I+P (no Mass)']['mean_tau'] 
                           for norm in normalization_methods]) / 3

print(f"   Average Kendall's tau across all combinations: {baseline_tau_all:.3f}")

# Find which metric's removal has the least impact (highest tau)
metric_impact = {}
for combo_name in metric_combinations.keys():
    avg_tau = np.mean([kendall_tau_results[norm][combo_name]['mean_tau'] for norm in normalization_methods])
    removed_metric = "Flood" if "no Flood" in combo_name else "Drought" if "no Drought" in combo_name else "Mass"
    metric_impact[removed_metric] = avg_tau
    print(f"   Removing {removed_metric:8} metric: τ = {avg_tau:.3f}")

least_important = max(metric_impact.items(), key=lambda x: x[1])
most_important = min(metric_impact.items(), key=lambda x: x[1])

print(f"\n5. KEY FINDINGS:")
print(f"   • Least important metric: {least_important[0]} (removal has minimal impact, τ = {least_important[1]:.3f})")
print(f"   • Most important metric:  {most_important[0]} (removal has major impact, τ = {most_important[1]:.3f})")

print(f"\n   • Most robust model-combination pairs:")
for combo_name in metric_combinations.keys():
    model, freq, norm = most_stable_models[combo_name]
    print(f"     {combo_name} + {norm}: {model} ({freq:.1f}%)")

print(f"\n   • Rankings are {'STABLE' if min(tau_stats['mean_tau'] for norm in normalization_methods for combo_name in metric_combinations.keys() for tau_stats in [kendall_tau_results[norm][combo_name]]) > 0.5 else 'UNSTABLE'} across normalization methods")
print(f"     (Minimum Kendall's tau: {min(tau_stats['mean_tau'] for norm in normalization_methods for combo_name in metric_combinations.keys() for tau_stats in [kendall_tau_results[norm][combo_name]]):.3f})")

print(f"\nAll analysis files saved to: {output_dir}/")
print("="*80)

# --- Save Detailed Results to CSV ---
print(f"\nSaving detailed results to CSV files...")

# Save Kendall's tau results
tau_summary = []
for norm_method in normalization_methods:
    for combo_name in metric_combinations.keys():
        tau_stats = kendall_tau_results[norm_method][combo_name]
        tau_summary.append({
            'Normalization': norm_method,
            'Combination': combo_name,
            'Mean_Tau': tau_stats['mean_tau'],
            'Std_Tau': tau_stats['std_tau'],
            'Median_Tau': tau_stats['median_tau']
        })

tau_df = pd.DataFrame(tau_summary)
tau_df.to_csv(os.path.join(output_dir, "kendall_tau_results.csv"), index=False)

# Save winner frequencies
freq_summary = []
for norm_method in normalization_methods:
    for combo_name in metric_combinations.keys():
        freq = winner_frequencies[norm_method][combo_name]
        for model in model_list:
            freq_summary.append({
                'Normalization': norm_method,
                'Combination': combo_name,
                'Model': model,
                'Winner_Frequency': freq[model]
            })

freq_df = pd.DataFrame(freq_summary)
freq_df.to_csv(os.path.join(output_dir, "winner_frequencies.csv"), index=False)

print(f"CSV files saved:")
print(f"  • kendall_tau_results.csv")
print(f"  • winner_frequencies.csv")
print(f"\nAnalysis complete!")
