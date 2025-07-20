import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
model_list = ["Observed", "HBV", "RF", "XGB", "TCN", "LSTM"]
input_dir = "/data0/funato/hsp/0_out/analysis/budyko_framework"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/budyko_framework"

# Ensure output directory exists
os.makedirs(output_fig_dir, exist_ok=True)

def theoretical_budyko_curve(phi, w=2.6):
    """
    Calculate theoretical Budyko curve
    phi: aridity index (PET/P)
    w: parameter (default 2.6 for global average)
    Returns: evapotranspiration ratio (ET/P)
    """
    return (phi * np.tanh(1/phi) * (1 - np.exp(-phi)))**0.5

# Load the data
print("Loading Budyko framework data...")
budyko_csv = f"{input_dir}/budyko_framework_data.csv"

# Check if file exists
if not os.path.exists(budyko_csv):
    print(f"Error: {budyko_csv} not found!")
    print("Please run budyko_framework.py first to generate the data.")
    exit(1)

# Load Budyko data
budyko_df = pd.read_csv(budyko_csv)
print(f"Loaded Budyko data for {len(budyko_df)} basins")

# Create observed data entries to treat as another "model"
obs_data_template = budyko_df[budyko_df['Model'] == budyko_df['Model'].iloc[0]].copy()
obs_data_template['Model'] = 'Observed'
obs_data_template['Runoff_Ratio'] = obs_data_template['Runoff_Ratio_Obs']
obs_data_template['ET_Ratio'] = obs_data_template['ET_Ratio_Obs']

# Add observed data to the main dataframe
budyko_df_extended = pd.concat([budyko_df, obs_data_template], ignore_index=True)

# Prepare data for plotting
valid_models = [model for model in model_list if model in budyko_df_extended['Model'].unique()]
print(f"Models with data: {valid_models}")

# Create Budyko framework plots
print("Creating Budyko framework visualizations...")

# 1. Individual model plots (including Observed as a model)
fig1, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

colors = ['black', 'blue', 'green', 'red', 'orange', 'purple']

for idx, model in enumerate(valid_models):
    if idx < len(axes):
        ax = axes[idx]
        
        # Get model data
        model_data = budyko_df_extended[budyko_df_extended['Model'] == model]
        
        if not model_data.empty:
            # Plot theoretical Budyko curve for runoff (Q/P = 1 - ET/P)
            phi_range = np.linspace(0.1, 1.5, 1000)
            et_theoretical = theoretical_budyko_curve(phi_range)
            runoff_theoretical = 1 - et_theoretical
            runoff_theoretical[runoff_theoretical < 0] = 0  # Can't have negative runoff
            ax.plot(phi_range, runoff_theoretical, 'k--', linewidth=2, label='Theoretical Budyko', alpha=0.7)
            
            # Plot model data points
            if model == 'Observed':
                scatter = ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                                   alpha=0.8, s=40, color=colors[idx], label=f'{model}', 
                                   marker='s', edgecolor='white', linewidth=1)
            else:
                scatter = ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                                   alpha=0.7, s=30, color=colors[idx], label=f'{model} Simulated', marker='o')
            
            # Add theoretical limits
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Water Limit (Q/P ≤ 1)')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Humid/Arid (PET/P = 1)')
            
            # Add energy limit line (PET/P = Q/P, or phi = runoff_ratio)
            limit_range = np.linspace(0, 1.5, 100)
            ax.plot(limit_range, limit_range, 'r:', alpha=0.5, linewidth=1, label='Energy Limit (PET/P = Q/P)')
            
            ax.set_xlabel('Aridity Index (PET/P)')
            ax.set_ylabel('Runoff Ratio (Q/P)')
            ax.set_title(f'{model} - Budyko Framework\n({len(model_data)} basins)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1.5)

# Hide unused subplots
for idx in range(len(valid_models), len(axes)):
    axes[idx].set_visible(False)

output_fig1 = f"{output_fig_dir}/budyko_individual_models.png"
plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
print(f"Individual model Budyko plots saved to: {output_fig1}")

# 2. Combined model comparison (All models including Observed)
fig2, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot theoretical Budyko curve for runoff (Q/P = 1 - ET/P)
phi_range = np.linspace(0.1, 1.5, 1000)
et_theoretical = theoretical_budyko_curve(phi_range)
runoff_theoretical = 1 - et_theoretical
runoff_theoretical[runoff_theoretical < 0] = 0  # Can't have negative runoff
ax.plot(phi_range, runoff_theoretical, 'k-', linewidth=3, label='Theoretical Budyko', alpha=0.8)

# Plot all models including observed
for idx, model in enumerate(valid_models):
    model_data = budyko_df_extended[budyko_df_extended['Model'] == model]
    
    if not model_data.empty:
        if model == 'Observed':
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                      alpha=0.9, s=60, color=colors[idx], label=f'{model} ({len(model_data)} basins)', 
                      marker='s', edgecolor='white', linewidth=1)
        else:
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                      alpha=0.7, s=40, color=colors[idx], label=f'{model} ({len(model_data)} basins)', marker='o')

# Add theoretical limits
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Water Limit (Q/P ≤ 1)')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Humid/Arid Boundary (PET/P = 1)')

# Add energy limit line (PET/P = Q/P)
limit_range = np.linspace(0, 1.5, 100)
ax.plot(limit_range, limit_range, 'r:', alpha=0.5, linewidth=2, label='Energy Limit (PET/P = Q/P)')

ax.set_xlabel('Aridity Index (PET/P)', fontsize=12)
ax.set_ylabel('Runoff Ratio (Q/P)', fontsize=12)
ax.set_title('Budyko Framework - All Models Comparison\nGreat Britain Humid Climate', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)

output_fig2 = f"{output_fig_dir}/budyko_all_models_including_observed.png"
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"All models including observed plot saved to: {output_fig2}")

# 3. Model performance comparison (simulated models vs observed)
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Get simulated models only (exclude observed)
sim_models = [model for model in valid_models if model != 'Observed']

# Calculate performance metrics for each simulated model
model_performance = []
for model in sim_models:
    model_data = budyko_df[budyko_df['Model'] == model]
    if not model_data.empty:
        bias = model_data['Runoff_Ratio'].mean() - model_data['Runoff_Ratio_Obs'].mean()
        correlation = model_data['Runoff_Ratio'].corr(model_data['Runoff_Ratio_Obs'])
        rmse = np.sqrt(((model_data['Runoff_Ratio'] - model_data['Runoff_Ratio_Obs'])**2).mean())
        mae = abs(model_data['Runoff_Ratio'] - model_data['Runoff_Ratio_Obs']).mean()
        
        model_performance.append({
            'Model': model,
            'Bias': bias,
            'Correlation': correlation,
            'RMSE': rmse,
            'MAE': mae
        })

perf_df = pd.DataFrame(model_performance)
sim_colors = colors[1:]  # Exclude black (reserved for observed)

# 3.1 Bias comparison
bars1 = axes[0].bar(perf_df['Model'], perf_df['Bias'], color=sim_colors[:len(perf_df)])
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[0].set_ylabel('Bias (Sim - Obs)')
axes[0].set_title('Model Bias in Runoff Ratio', fontweight='bold')
axes[0].grid(True, alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
             f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

# 3.2 Correlation comparison
bars2 = axes[1].bar(perf_df['Model'], perf_df['Correlation'], color=sim_colors[:len(perf_df)])
axes[1].set_ylabel('Correlation Coefficient')
axes[1].set_title('Model Correlation with Observed', fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3.3 RMSE comparison
bars3 = axes[2].bar(perf_df['Model'], perf_df['RMSE'], color=sim_colors[:len(perf_df)])
axes[2].set_ylabel('RMSE')
axes[2].set_title('Root Mean Square Error', fontweight='bold')
axes[2].grid(True, alpha=0.3)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3.4 Scatter plot: Correlation vs Bias
for idx, row in perf_df.iterrows():
    axes[3].scatter(row['Bias'], row['Correlation'], s=200, color=sim_colors[idx], 
               alpha=0.8, edgecolor='black', linewidth=2)
    axes[3].text(row['Bias'] + 0.02, row['Correlation'] + 0.01, row['Model'], 
            fontsize=10, fontweight='bold')

axes[3].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[3].set_xlabel('Bias (Sim - Obs)')
axes[3].set_ylabel('Correlation Coefficient')
axes[3].set_title('Model Performance Overview', fontweight='bold')
axes[3].grid(True, alpha=0.3)

output_fig3 = f"{output_fig_dir}/budyko_model_performance.png"
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
print(f"Model performance analysis saved to: {output_fig3}")

# 4. Detailed Budyko space analysis
fig4, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot theoretical Budyko curve
phi_range = np.linspace(0.1, 1.5, 1000)
et_theoretical = theoretical_budyko_curve(phi_range)
runoff_theoretical = 1 - et_theoretical
runoff_theoretical[runoff_theoretical < 0] = 0
ax.plot(phi_range, runoff_theoretical, 'k-', linewidth=3, label='Theoretical Budyko', alpha=0.8)

# Plot all models with transparency
for idx, model in enumerate(valid_models):
    model_data = budyko_df_extended[budyko_df_extended['Model'] == model]
    
    if not model_data.empty:
        if model == 'Observed':
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                      alpha=0.9, s=50, color=colors[idx], label=f'{model}', 
                      marker='s', edgecolor='white', linewidth=1, zorder=5)
        else:
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                      alpha=0.5, s=25, color=colors[idx], label=f'{model}', marker='o')

# Add comprehensive theoretical limits
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.8, linewidth=2, label='Water Limit (Q/P ≤ 1)')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Humid/Arid Boundary (PET/P = 1)')

# Add energy limit line (PET/P = Q/P)
limit_range = np.linspace(0, 1.5, 100)
ax.plot(limit_range, limit_range, 'r:', alpha=0.7, linewidth=2, label='Energy Limit (PET/P = Q/P)')

# Add climate zone annotations
ax.text(0.3, 1.3, 'Humid\n(Energy Limited)\nPET/P < 1', fontsize=10, ha='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
ax.text(1.2, 0.3, 'Arid\n(Water Limited)\nPET/P > 1', fontsize=10, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

ax.set_xlabel('Aridity Index (PET/P)', fontsize=12)
ax.set_ylabel('Runoff Ratio (Q/P)', fontsize=12)
ax.set_title('Comprehensive Budyko Framework Analysis\nGB Basins with Theoretical Limits', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)

output_fig4 = f"{output_fig_dir}/budyko_comprehensive_analysis.png"
plt.savefig(output_fig4, dpi=300, bbox_inches='tight')
print(f"Comprehensive Budyko analysis saved to: {output_fig4}")

# Print summary statistics (updated to include observed as a model)
print("\n" + "="*60)
print("BUDYKO FRAMEWORK ANALYSIS SUMMARY")
print("="*60)

for model in valid_models:
    model_data = budyko_df_extended[budyko_df_extended['Model'] == model]
    
    if not model_data.empty:
        print(f"\n{model} ({'Observed Data' if model == 'Observed' else 'Model'}) ({len(model_data)} basins):")
        print(f"  Aridity Index (PET/P):")
        print(f"    Mean: {model_data['Aridity_Index'].mean():.3f}")
        print(f"    Std:  {model_data['Aridity_Index'].std():.3f}")
        print(f"    Range: {model_data['Aridity_Index'].min():.3f} - {model_data['Aridity_Index'].max():.3f}")
        
        print(f"  Runoff Ratio (Q/P):")
        print(f"    Mean: {model_data['Runoff_Ratio'].mean():.3f}")
        print(f"    Std:  {model_data['Runoff_Ratio'].std():.3f}")
        print(f"    Range: {model_data['Runoff_Ratio'].min():.3f} - {model_data['Runoff_Ratio'].max():.3f}")
        
        if model != 'Observed':
            # Calculate bias and correlation for simulated models
            original_model_data = budyko_df[budyko_df['Model'] == model]
            bias = original_model_data['Runoff_Ratio'].mean() - original_model_data['Runoff_Ratio_Obs'].mean()
            correlation = original_model_data['Runoff_Ratio'].corr(original_model_data['Runoff_Ratio_Obs'])
            print(f"  Performance vs Observed:")
            print(f"    Bias (Sim - Obs): {bias:.3f}")
            print(f"    Correlation: {correlation:.3f}")

print("\nVisualization completed!")
print(f"Generated figures:")
print(f"  - Individual models (including Observed): {output_fig1}")
print(f"  - All models comparison: {output_fig2}")
print(f"  - Model performance metrics: {output_fig3}")
print(f"  - Comprehensive analysis: {output_fig4}")
print(f"\nData source: {budyko_csv}")
print(f"Total basins analyzed: {len(budyko_df)}")
print(f"Models analyzed: {valid_models}")

# Additional theoretical limits explanation
print("\n" + "="*60)
print("BUDYKO FRAMEWORK THEORETICAL LIMITS EXPLANATION")
print("="*60)
print("1. Water Limit (Q/P ≤ 1): Runoff cannot exceed precipitation")
print("2. Energy Limit (PET/P = Q/P): At this line, all available energy goes to runoff")
print("3. Theoretical Budyko Curve: Optimal water-energy balance")
print("4. Humid/Arid Boundary (PET/P = 1): Energy vs water limitation threshold")
print("   - PET/P < 1: Humid, energy-limited (typical for GB)")
print("   - PET/P > 1: Arid, water-limited")
print("="*60)
