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
model_list = ["HBV", "RF", "XGB", "TCN", "LSTM"]
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

# Prepare data for plotting
valid_models = [model for model in model_list if model in budyko_df['Model'].unique()]
print(f"Models with data: {valid_models}")

# Create Budyko framework plots
print("Creating Budyko framework visualizations...")

# 1. Individual model plots
fig1, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

colors = ['blue', 'green', 'red', 'orange', 'purple']

for idx, model in enumerate(model_list):
    if idx < len(axes):
        ax = axes[idx]
        
        # Get model data
        model_data = budyko_df[budyko_df['Model'] == model]
        
        if not model_data.empty:
            # Plot theoretical Budyko curve for runoff (Q/P = 1 - ET/P)
            phi_range = np.linspace(0.1, 1.5, 1000)
            et_theoretical = theoretical_budyko_curve(phi_range)
            runoff_theoretical = 1 - et_theoretical
            runoff_theoretical[runoff_theoretical < 0] = 0  # Can't have negative runoff
            ax.plot(phi_range, runoff_theoretical, 'k--', linewidth=2, label='Theoretical Budyko', alpha=0.7)
            
            # Plot model data points (simulated)
            scatter_sim = ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                               alpha=0.6, s=30, color=colors[idx], label=f'{model} Simulated', marker='o')
            
            # Plot observed data points
            scatter_obs = ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio_Obs'], 
                               alpha=0.6, s=30, color='black', label='Observed', marker='s', linewidths=1)
            
            # Add water limit and climate boundary
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='Water Limit (Q/P≤1)')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Humid/Arid')
            
            ax.set_xlabel('Aridity Index (PET/P)')
            ax.set_ylabel('Runoff Ratio (Q/P)')
            ax.set_title(f'{model} Model - Budyko Framework\n({len(model_data)} basins)', fontweight='bold')
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

# 2. Combined model comparison (Simulated vs Observed)
fig2, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot theoretical Budyko curve for runoff (Q/P = 1 - ET/P)
phi_range = np.linspace(0.1, 1.5, 1000)
et_theoretical = theoretical_budyko_curve(phi_range)
runoff_theoretical = 1 - et_theoretical
runoff_theoretical[runoff_theoretical < 0] = 0  # Can't have negative runoff
ax.plot(phi_range, runoff_theoretical, 'k--', linewidth=3, label='Theoretical Budyko', alpha=0.8)

# Plot observed data first (using data from any model since it's the same)
if valid_models:
    obs_data = budyko_df[budyko_df['Model'] == valid_models[0]]
    if not obs_data.empty:
        ax.scatter(obs_data['Aridity_Index'], obs_data['Runoff_Ratio_Obs'], 
                  alpha=0.8, s=60, color='black', label=f'Observed ({len(obs_data)} basins)', 
                  marker='s', edgecolor='white', linewidth=1)

# Plot simulated data for each model
for idx, model in enumerate(valid_models):
    model_data = budyko_df[budyko_df['Model'] == model]
    
    if not model_data.empty:
        # Simulated data only
        ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                  alpha=0.7, s=40, color=colors[idx], label=f'{model} Simulated', marker='o')

# Add water limit and climate zone indicators
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Water Limit (Q/P≤1)')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Humid/Arid Boundary')

ax.set_xlabel('Aridity Index (PET/P)', fontsize=12)
ax.set_ylabel('Runoff Ratio (Q/P)', fontsize=12)
ax.set_title('Budyko Framework - Model Performance Analysis\nGreat Britain Humid Climate', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)

output_fig2 = f"{output_fig_dir}/budyko_models_vs_observed.png"
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"Models vs Observed Budyko plot saved to: {output_fig2}")

# 3. Individual model detailed comparison with observed
fig3, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, model in enumerate(valid_models):
    if idx < len(axes):
        ax = axes[idx]
        
        # Get model data
        model_data = budyko_df[budyko_df['Model'] == model]
        
        if not model_data.empty:
            # Plot theoretical Budyko curve for runoff
            phi_range = np.linspace(0.1, 1.5, 1000)
            et_theoretical = theoretical_budyko_curve(phi_range)
            runoff_theoretical = 1 - et_theoretical
            runoff_theoretical[runoff_theoretical < 0] = 0
            ax.plot(phi_range, runoff_theoretical, 'k--', linewidth=2, label='Theoretical', alpha=0.8)
            
            # Plot observed data (black squares)
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio_Obs'], 
                      alpha=0.8, s=50, color='black', label='Observed', marker='s', 
                      edgecolor='white', linewidth=1)
            
            # Plot simulated data (colored circles)
            ax.scatter(model_data['Aridity_Index'], model_data['Runoff_Ratio'], 
                      alpha=0.7, s=40, color=colors[idx], label=f'{model} Simulated', marker='o')
            
            # Add connecting lines to show bias
            for i, row in model_data.iterrows():
                ax.plot([row['Aridity_Index'], row['Aridity_Index']], 
                       [row['Runoff_Ratio'], row['Runoff_Ratio_Obs']], 
                       color='gray', alpha=0.3, linewidth=0.5)
            
            # Add climate indicators
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Aridity Index (PET/P)')
            ax.set_ylabel('Runoff Ratio (Q/P)')
            ax.set_title(f'{model} vs Observed\n({len(model_data)} basins)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1.5)

# Hide unused subplots
for idx in range(len(valid_models), len(axes)):
    axes[idx].set_visible(False)

output_fig3 = f"{output_fig_dir}/budyko_individual_vs_observed.png"
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
print(f"Individual models vs observed plots saved to: {output_fig3}")

# 4. Model performance metrics visualization
fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Calculate performance metrics for each model
model_performance = []
for model in valid_models:
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

# 4.1 Bias comparison
bars1 = ax1.bar(perf_df['Model'], perf_df['Bias'], color=colors[:len(perf_df)])
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_ylabel('Bias (Sim - Obs)')
ax1.set_title('Model Bias in Runoff Ratio', fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
             f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

# 4.2 Correlation comparison
bars2 = ax2.bar(perf_df['Model'], perf_df['Correlation'], color=colors[:len(perf_df)])
ax2.set_ylabel('Correlation Coefficient')
ax2.set_title('Model Correlation with Observed', fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 4.3 RMSE comparison
bars3 = ax3.bar(perf_df['Model'], perf_df['RMSE'], color=colors[:len(perf_df)])
ax3.set_ylabel('RMSE')
ax3.set_title('Root Mean Square Error', fontweight='bold')
ax3.grid(True, alpha=0.3)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 4.4 Scatter plot: Correlation vs Bias
for idx, row in perf_df.iterrows():
    ax4.scatter(row['Bias'], row['Correlation'], s=200, color=colors[idx], 
               alpha=0.8, edgecolor='black', linewidth=2)
    ax4.text(row['Bias'] + 0.02, row['Correlation'] + 0.01, row['Model'], 
            fontsize=10, fontweight='bold')

ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Bias (Sim - Obs)')
ax4.set_ylabel('Correlation Coefficient')
ax4.set_title('Model Performance Overview', fontweight='bold')
ax4.grid(True, alpha=0.3)

output_fig4 = f"{output_fig_dir}/budyko_model_performance.png"
plt.savefig(output_fig4, dpi=300, bbox_inches='tight')
print(f"Model performance analysis saved to: {output_fig4}")

# 5. Observed data characteristics
fig5, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Get observed data (using first model since observed data is the same across all models)
if valid_models:
    obs_data = budyko_df[budyko_df['Model'] == valid_models[0]]
    
    # 5.1 Budyko space with observed data only
    phi_range = np.linspace(0.1, 1.5, 1000)
    et_theoretical = theoretical_budyko_curve(phi_range)
    runoff_theoretical = 1 - et_theoretical
    runoff_theoretical[runoff_theoretical < 0] = 0
    ax1.plot(phi_range, runoff_theoretical, 'k--', linewidth=3, label='Theoretical Budyko', alpha=0.8)
    
    scatter = ax1.scatter(obs_data['Aridity_Index'], obs_data['Runoff_Ratio_Obs'], 
                         alpha=0.7, s=50, c=obs_data['Aridity_Index'], cmap='viridis', 
                         edgecolor='black', linewidth=0.5)
    
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Water Limit')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Humid/Arid Boundary')
    ax1.set_xlabel('Aridity Index (PET/P)')
    ax1.set_ylabel('Runoff Ratio (Q/P)')
    ax1.set_title('Observed Data in Budyko Space\nGreat Britain Basins', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.5)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Aridity Index', rotation=270, labelpad=15)
    
    # 5.2 Aridity index distribution
    ax2.hist(obs_data['Aridity_Index'], bins=25, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Humid/Arid Boundary')
    ax2.axvline(x=obs_data['Aridity_Index'].mean(), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean = {obs_data["Aridity_Index"].mean():.3f}')
    ax2.set_xlabel('Aridity Index (PET/P)')
    ax2.set_ylabel('Number of Basins')
    ax2.set_title('Distribution of Aridity Index', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 5.3 Runoff ratio distribution
    ax3.hist(obs_data['Runoff_Ratio_Obs'], bins=25, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=obs_data['Runoff_Ratio_Obs'].mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Mean = {obs_data["Runoff_Ratio_Obs"].mean():.3f}')
    ax3.set_xlabel('Runoff Ratio (Q/P)')
    ax3.set_ylabel('Number of Basins')
    ax3.set_title('Distribution of Runoff Ratio (Observed)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 5.4 Climate summary
    ax4.axis('off')
    
    # Calculate climate statistics
    humid_count = (obs_data['Aridity_Index'] < 1.0).sum()
    total_count = len(obs_data)
    mean_aridity = obs_data['Aridity_Index'].mean()
    mean_runoff = obs_data['Runoff_Ratio_Obs'].mean()
    
    summary_text = f"""
    GREAT BRITAIN CLIMATE CHARACTERISTICS
    
    Total Basins: {total_count}
    
    Aridity Index (PET/P):
    • Mean: {mean_aridity:.3f}
    • Range: {obs_data['Aridity_Index'].min():.3f} - {obs_data['Aridity_Index'].max():.3f}
    • Std: {obs_data['Aridity_Index'].std():.3f}
    
    Climate Classification:
    • Humid basins (PET/P < 1): {humid_count} ({humid_count/total_count*100:.1f}%)
    • Arid basins (PET/P ≥ 1): {total_count-humid_count} ({(total_count-humid_count)/total_count*100:.1f}%)
    
    Runoff Ratio (Q/P):
    • Mean: {mean_runoff:.3f}
    • Range: {obs_data['Runoff_Ratio_Obs'].min():.3f} - {obs_data['Runoff_Ratio_Obs'].max():.3f}
    • Std: {obs_data['Runoff_Ratio_Obs'].std():.3f}
    
    Regional Characteristics:
    • Temperate maritime climate
    • Energy-limited hydrology
    • High precipitation/low evaporation
    • Consistent runoff generation
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))

output_fig5 = f"{output_fig_dir}/budyko_observed_characteristics.png"
plt.savefig(output_fig5, dpi=300, bbox_inches='tight')
print(f"Observed data characteristics saved to: {output_fig5}")

# Print summary statistics
print("\n" + "="*60)
print("BUDYKO FRAMEWORK ANALYSIS SUMMARY")
print("="*60)

for model in model_list:
    model_data = budyko_df[budyko_df['Model'] == model]
    
    if not model_data.empty:
        print(f"\n{model} Model ({len(model_data)} basins):")
        print(f"  Aridity Index (PET/P):")
        print(f"    Mean: {model_data['Aridity_Index'].mean():.3f}")
        print(f"    Std:  {model_data['Aridity_Index'].std():.3f}")
        print(f"    Range: {model_data['Aridity_Index'].min():.3f} - {model_data['Aridity_Index'].max():.3f}")
        
        print(f"  ET Ratio (ET/P):")
        print(f"    Mean: {model_data['ET_Ratio'].mean():.3f}")
        print(f"    Std:  {model_data['ET_Ratio'].std():.3f}")
        print(f"    Range: {model_data['ET_Ratio'].min():.3f} - {model_data['ET_Ratio'].max():.3f}")
        
        print(f"  Runoff Ratio (Q/P) - Simulated:")
        print(f"    Mean: {model_data['Runoff_Ratio'].mean():.3f}")
        print(f"    Std:  {model_data['Runoff_Ratio'].std():.3f}")
        print(f"    Range: {model_data['Runoff_Ratio'].min():.3f} - {model_data['Runoff_Ratio'].max():.3f}")
        
        print(f"  Runoff Ratio (Q/P) - Observed:")
        print(f"    Mean: {model_data['Runoff_Ratio_Obs'].mean():.3f}")
        print(f"    Std:  {model_data['Runoff_Ratio_Obs'].std():.3f}")
        print(f"    Range: {model_data['Runoff_Ratio_Obs'].min():.3f} - {model_data['Runoff_Ratio_Obs'].max():.3f}")
        
        # Calculate bias and correlation
        bias = model_data['Runoff_Ratio'].mean() - model_data['Runoff_Ratio_Obs'].mean()
        correlation = model_data['Runoff_Ratio'].corr(model_data['Runoff_Ratio_Obs'])
        print(f"  Runoff Ratio Bias (Sim - Obs): {bias:.3f}")
        print(f"  Runoff Ratio Correlation: {correlation:.3f}")

print("\nVisualization completed!")
print(f"Generated figures:")
print(f"  - Individual models: {output_fig1}")
print(f"  - Models vs Observed: {output_fig2}")
print(f"  - Individual vs Observed: {output_fig3}")
print(f"  - Model Performance: {output_fig4}")
print(f"  - Observed Characteristics: {output_fig5}")
print(f"\nData source: {budyko_csv}")
print(f"Total basins analyzed: {len(budyko_df)}")
print(f"Models analyzed: {valid_models}")
