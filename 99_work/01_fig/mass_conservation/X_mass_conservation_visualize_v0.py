import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
model_list = ["Observed", "HBV", "BC", "RF", "XGB", "TCN", "LSTM"]
input_dir = "/data0/funato/hsp/0_out/analysis/mass_conservation"
output_fig_dir = "/data0/funato/hsp/0_out/1_fig/mass_conservation"

# Ensure output directory exists
os.makedirs(output_fig_dir, exist_ok=True)

# Load the mass conservation data
print("Loading mass conservation data...")
mass_conservation_csv = f"{input_dir}/mass_conservation_data.csv"

if not os.path.exists(mass_conservation_csv):
    print(f"Error: {mass_conservation_csv} not found!")
    print("Please run mass_conservation_dataset.py first to generate the data.")
    exit(1)

# Load data
mass_df = pd.read_csv(mass_conservation_csv)
print(f"Loaded mass conservation data for {len(mass_df)} records")

# Prepare data for plotting
valid_models = [model for model in model_list if model in mass_df['Model'].unique()]
print(f"Models with data: {valid_models}")

colors = ['black','pink', 'blue', 'green', 'red', 'orange', 'purple']

print("Creating mass conservation visualizations...")

# 1. Precipitation vs Storm Flow for all models
fig1, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, model in enumerate(valid_models):
    if idx < len(axes):
        ax = axes[idx]
        
        model_data = mass_df[mass_df['Model'] == model]
        
        if not model_data.empty:
            # Create scatter plot
            if model == 'Observed':
                scatter = ax.scatter(model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'],
                                   alpha=0.8, s=40, color=colors[idx], edgecolor='white', linewidth=1,
                                   marker='s', label=f'{model}')
            else:
                scatter = ax.scatter(model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'],
                                   alpha=0.7, s=30, color=colors[idx], marker='o', label=f'{model}')
            
            # Add 1:1 line (perfect mass conservation)
            # Use 95th percentile to avoid outliers
            max_val = max(np.percentile(model_data['Daily_Mean_Precip'], 95), 
                         np.percentile(model_data['Daily_Mean_Storm_Flow'], 95))
            min_val = min(model_data['Daily_Mean_Precip'].min(), model_data['Daily_Mean_Storm_Flow'].min())
            line_range = np.linspace(min_val, max_val, 100)
            ax.plot(line_range, line_range, 'k--', alpha=0.7, linewidth=2, label='1:1 Line (Perfect Conservation)')
            
            # Calculate correlation and R²
            correlation = model_data['Daily_Mean_Precip'].corr(model_data['Daily_Mean_Storm_Flow'])
            r2 = r2_score(model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'])
            
            # Calculate regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'])
            regression_line = slope * line_range + intercept
            ax.plot(line_range, regression_line, color=colors[idx], linestyle='-', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Daily Mean Precipitation (mm/day)')
            ax.set_ylabel('Daily Mean Storm Flow (mm/day)')
            ax.set_title(f'{model} - Mass Conservation\n(R² = {r2:.3f}, r = {correlation:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Set axis limits using 95th percentile to exclude outliers
            ax.set_xlim(0, max_val * 1.05)
            ax.set_ylim(0, max_val * 1.05)
            
            # Set equal aspect ratio for better comparison
            ax.set_aspect('equal', adjustable='box')

# Hide unused subplots
for idx in range(len(valid_models), len(axes)):
    axes[idx].set_visible(False)

output_fig1 = f"{output_fig_dir}/mass_conservation_individual_models.png"
plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
print(f"Individual model mass conservation plots saved to: {output_fig1}")

# 2. Combined comparison of all models
fig2, ax = plt.subplots(1, 1, figsize=(12, 10))

for idx, model in enumerate(valid_models):
    model_data = mass_df[mass_df['Model'] == model]
    
    if not model_data.empty:
        if model == 'Observed':
            ax.scatter(model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'],
                      alpha=0.9, s=60, color=colors[idx], edgecolor='white', linewidth=1,
                      marker='s', label=f'{model} ({len(model_data)} basins)')
        else:
            ax.scatter(model_data['Daily_Mean_Precip'], model_data['Daily_Mean_Storm_Flow'],
                      alpha=0.7, s=40, color=colors[idx], marker='o', 
                      label=f'{model} ({len(model_data)} basins)')

# Add 1:1 line
# Use 95th percentile to avoid outliers for axis limits
max_val_95 = max(np.percentile(mass_df['Daily_Mean_Precip'], 95), 
                 np.percentile(mass_df['Daily_Mean_Storm_Flow'], 95))
min_val = min(mass_df['Daily_Mean_Precip'].min(), mass_df['Daily_Mean_Storm_Flow'].min())
line_range = np.linspace(min_val, max_val_95, 100)
ax.plot(line_range, line_range, 'k--', alpha=0.8, linewidth=3, label='1:1 Line (Perfect Conservation)')

# Set axis limits based on 95th percentile to exclude outliers
ax.set_xlim(0, max_val_95 * 1.05)
ax.set_ylim(0, max_val_95 * 1.05)

ax.set_xlabel('Daily Mean Precipitation (mm/day)', fontsize=12)
ax.set_ylabel('Daily Mean Storm Flow (mm/day)', fontsize=12)
ax.set_title('Mass Conservation Analysis - All Models\nPrecipitation vs Storm Flow', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal', adjustable='box')

output_fig2 = f"{output_fig_dir}/mass_conservation_all_models.png"
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"Combined mass conservation plot saved to: {output_fig2}")

# 3. Mass balance ratio analysis
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 3.1 Mass balance ratio distribution (filter outliers for better visualization)
ratio_95th = np.percentile(mass_df['Mass_Balance_Ratio'].dropna(), 95)
ratio_5th = np.percentile(mass_df['Mass_Balance_Ratio'].dropna(), 5)

for idx, model in enumerate(valid_models):
    model_data = mass_df[mass_df['Model'] == model]
    if not model_data.empty:
        # Filter outliers for visualization
        filtered_ratios = model_data['Mass_Balance_Ratio'][(model_data['Mass_Balance_Ratio'] >= ratio_5th) & 
                                                          (model_data['Mass_Balance_Ratio'] <= ratio_95th)]
        axes[0, 0].hist(filtered_ratios, bins=30, alpha=0.6, 
                       color=colors[idx], label=model, edgecolor='black', linewidth=0.5)

axes[0, 0].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Perfect Balance (Ratio = 1)')
axes[0, 0].set_xlabel('Mass Balance Ratio (Storm Flow / Precipitation)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Mass Balance Ratios\n(5th-95th percentile)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(ratio_5th * 0.9, ratio_95th * 1.1)

# 3.2 Box plot of mass balance ratios (with outlier handling)
mass_balance_data = []
mass_balance_labels = []
for model in valid_models:
    model_data = mass_df[mass_df['Model'] == model]
    if not model_data.empty:
        mass_balance_data.append(model_data['Mass_Balance_Ratio'].dropna())
        mass_balance_labels.append(model)

# Create box plot with outlier control
bp = axes[0, 1].boxplot(mass_balance_data, labels=mass_balance_labels, patch_artist=True,
                       showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
for patch, color in zip(bp['boxes'], colors[:len(mass_balance_labels)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Balance')
axes[0, 1].set_ylabel('Mass Balance Ratio')
axes[0, 1].set_title('Mass Balance Ratio by Model\n(outliers shown as points)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Set y-axis limits to focus on main distribution
axes[0, 1].set_ylim(ratio_5th * 0.8, ratio_95th * 1.2)

# 3.3 Base flow index vs mass balance ratio
for idx, model in enumerate(valid_models):
    model_data = mass_df[mass_df['Model'] == model]
    if not model_data.empty:
        if model == 'Observed':
            axes[1, 0].scatter(model_data['Base_Flow_Index'], model_data['Mass_Balance_Ratio'],
                              alpha=0.8, s=50, color=colors[idx], marker='s', label=model)
        else:
            axes[1, 0].scatter(model_data['Base_Flow_Index'], model_data['Mass_Balance_Ratio'],
                              alpha=0.6, s=30, color=colors[idx], marker='o', label=model)

axes[1, 0].axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Base Flow Index')
axes[1, 0].set_ylabel('Mass Balance Ratio')
axes[1, 0].set_title('Base Flow Index vs Mass Balance Ratio', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3.4 Model performance summary
model_performance = []
for model in valid_models:
    model_data = mass_df[mass_df['Model'] == model]
    if not model_data.empty:
        mean_ratio = model_data['Mass_Balance_Ratio'].mean()
        std_ratio = model_data['Mass_Balance_Ratio'].std()
        median_ratio = model_data['Mass_Balance_Ratio'].median()
        
        model_performance.append({
            'Model': model,
            'Mean_Ratio': mean_ratio,
            'Std_Ratio': std_ratio,
            'Median_Ratio': median_ratio
        })

perf_df = pd.DataFrame(model_performance)

bars = axes[1, 1].bar(perf_df['Model'], perf_df['Mean_Ratio'], 
                     color=colors[:len(perf_df)], alpha=0.7, edgecolor='black')
axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Balance')
axes[1, 1].set_ylabel('Mean Mass Balance Ratio')
axes[1, 1].set_title('Average Mass Balance Performance', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, perf_df['Mean_Ratio']):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)

output_fig3 = f"{output_fig_dir}/mass_conservation_analysis.png"
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
print(f"Mass conservation analysis saved to: {output_fig3}")

# 4. Detailed statistical analysis
fig4, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate performance metrics for each model
performance_metrics = []
for model in valid_models:
    model_data = mass_df[mass_df['Model'] == model]
    if not model_data.empty:
        # Calculate how close to 1:1 line (perfect conservation)
        precip = model_data['Daily_Mean_Precip']
        storm_flow = model_data['Daily_Mean_Storm_Flow']
        
        # RMSE from 1:1 line
        rmse_conservation = np.sqrt(np.mean((precip - storm_flow)**2))
        
        # Mean absolute error from 1:1 line
        mae_conservation = np.mean(np.abs(precip - storm_flow))
        
        # Correlation with 1:1 line
        correlation = precip.corr(storm_flow)
        
        # R² score
        r2 = r2_score(precip, storm_flow)
        
        # Mean bias (how much models over/under-estimate)
        bias = storm_flow.mean() - precip.mean()
        
        performance_metrics.append({
            'Model': model,
            'RMSE': rmse_conservation,
            'MAE': mae_conservation,
            'Correlation': correlation,
            'R²': r2,
            'Bias': bias,
            'Mean_Ratio': model_data['Mass_Balance_Ratio'].mean()
        })

metrics_df = pd.DataFrame(performance_metrics)

# Create performance overview plot
x_pos = np.arange(len(metrics_df))
width = 0.15

ax.bar(x_pos - 2*width, metrics_df['Correlation'], width, label='Correlation', alpha=0.8)
ax.bar(x_pos - width, metrics_df['R²'], width, label='R²', alpha=0.8)
ax.bar(x_pos, 1 - np.abs(metrics_df['Mean_Ratio'] - 1), width, label='Conservation Score', alpha=0.8)
ax.bar(x_pos + width, 1 / (1 + metrics_df['RMSE']), width, label='RMSE Score', alpha=0.8)
ax.bar(x_pos + 2*width, 1 / (1 + metrics_df['MAE']), width, label='MAE Score', alpha=0.8)

ax.set_xlabel('Models')
ax.set_ylabel('Performance Score (0-1, higher is better)')
ax.set_title('Mass Conservation Performance Metrics', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_df['Model'])
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

output_fig4 = f"{output_fig_dir}/mass_conservation_performance.png"
plt.savefig(output_fig4, dpi=300, bbox_inches='tight')
print(f"Performance metrics plot saved to: {output_fig4}")

# Print detailed summary
print("\n" + "="*80)
print("MASS CONSERVATION ANALYSIS SUMMARY")
print("="*80)

for _, row in metrics_df.iterrows():
    print(f"\n{row['Model']} Model:")
    print(f"  Mass Balance Ratio: {row['Mean_Ratio']:.3f}")
    print(f"  Correlation (P vs SF): {row['Correlation']:.3f}")
    print(f"  R² Score: {row['R²']:.3f}")
    print(f"  RMSE: {row['RMSE']:.3f} mm/day")
    print(f"  MAE: {row['MAE']:.3f} mm/day")
    print(f"  Bias: {row['Bias']:.3f} mm/day")

print("\n" + "="*80)
print("INTERPRETATION GUIDE:")
print("="*80)
print("• Mass Balance Ratio close to 1.0 indicates good mass conservation")
print("• High correlation indicates good relationship between P and storm flow")
print("• Low RMSE and MAE indicate better conservation accuracy")
print("• Positive bias: model overestimates storm flow")
print("• Negative bias: model underestimates storm flow")

print("\nVisualization completed!")
print(f"Generated figures:")
print(f"  - Individual models: {output_fig1}")
print(f"  - All models comparison: {output_fig2}")
print(f"  - Mass conservation analysis: {output_fig3}")
print(f"  - Performance metrics: {output_fig4}")
print(f"\nData source: {mass_conservation_csv}")
print(f"Total records analyzed: {len(mass_df)}")
