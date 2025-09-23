import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

model_list = ['RF', 'XGB', 'TCN', 'LSTM']
fs = 18

# Enhanced feature categories for analysis based on actual data
dynamic_features = ['Precip', 'Temp', 'PET']

# Comprehensive feature categorization with meaningful groupings
feature_categories = {
    'Dynamic': ['Precip', 'Temp', 'PET'],
    'Topography': ['gauge_lat', 'gauge_lon', 'gauge_easting', 'gauge_northing', 'gauge_elev', 'area', 'dpsbar', 'elev_mean', 'elev_min', 'elev_10', 'elev_50', 'elev_90', 'elev_max'],
    'Climate_Patterns': ['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow','high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur','inter_high_perc', 'inter_mod_perc', 'inter_low_perc', 'frac_high_perc', 'frac_mod_perc', 'frac_low_perc'],
    'Streamflow_Statistics': ['no_gw_perc', 'low_nsig_perc', 'nsig_low_perc','q_mean', 'runoff_ratio', 'stream_elas', 'baseflow_index', 'baseflow_index_ceh', 'hfd_mean''Q5', 'Q95', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq'],
    'Land_Use': ['dwood_perc', 'ewood_perc', 'grass_perc', 'shrub_perc', 'crop_perc', 'urban_perc', 'inwater_perc', 'bares_perc'],
    'Soil': ['sand_perc', 'silt_perc', 'clay_perc', 'organic_perc','bulkdens', 'tawc', 'porosity_cosby', 'porosity_hypres', 'conductivity_cosby', 'conductivity_hypres', 'root_depth', 'soil_depth_pelletier'],
}

# Color scheme for categories (meaningful colors)
category_colors = {
    'Dynamic': '#FF4136',           # Red - immediate/dynamic
    'Topography': '#B10DC9',        # Purple - landscape/terrain
    'Climate_Patterns': '#FF851B',  # Orange - long-term climate
    'Streamflow_Statistics': '#0074D9',  # Blue - water/streamflow
    'Land_Use': '#2ECC40',         # Green - vegetation/land
    'Soil': '#AAAAAA',             # Gray - soil properties
}

for model in model_list:
    print(f"\n{'='*60}")
    print(f"SHAP ANALYSIS FOR {model} MODEL")
    print(f"{'='*60}")

    shap_dir = f"/data0/funato/hsp/0_out/{model}/GB/shap/regional/shap_objects"
    output_dir = f"/data0/funato/hsp/0_out/1_fig/shap/{model}"
    os.makedirs(output_dir, exist_ok=True)

    expected_values_file = f"{shap_dir}/expected_value.npy"  # Note: singular 'value'
    shap_values_file = f"{shap_dir}/shap_values.npy"
    
    # Check for different naming conventions
    X_sample_file = f"{shap_dir}/X_sample.csv"
    if not os.path.exists(X_sample_file):
        X_sample_file = f"{shap_dir}/X_features.csv"  # Alternative naming
    
    # For TCN, check if files are in subdirectory
    if not os.path.exists(expected_values_file) and model == 'TCN':
        expected_values_file = f"{shap_dir}/shap_objects_/expected_value.npy"
        shap_values_file = f"{shap_dir}/shap_objects_/shap_values.npy"
        X_sample_file = f"{shap_dir}/shap_objects_/X_sample.csv"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [expected_values_file, shap_values_file, X_sample_file]):
        print(f"Warning: SHAP files not found for {model}")
        continue
    
    try:
        # Load SHAP data
        expected_values = np.load(expected_values_file)
        shap_values = np.load(shap_values_file)
        X_sample = pd.read_csv(X_sample_file)
        
        # Handle dimension mismatches
        # Remove 'Unnamed: 0' column if it exists (index column)
        if 'Unnamed: 0' in X_sample.columns:
            X_sample = X_sample.drop('Unnamed: 0', axis=1)
        
        # Ensure SHAP values and X_sample have matching dimensions
        min_features = min(shap_values.shape[-1], X_sample.shape[1])
        shap_values = shap_values[:, :min_features]
        X_sample = X_sample.iloc[:, :min_features]
        
        print(f"Loaded SHAP data for {model}:")
        print(f"  Expected values shape: {expected_values.shape}")
        print(f"  SHAP values shape: {shap_values.shape}")
        print(f"  X sample shape: {X_sample.shape}")
        print(f"  Features: {list(X_sample.columns)}")
        
        # Create SHAP explanation object
        if len(expected_values.shape) == 0 or expected_values.shape[0] == 1:
            # Single expected value
            base_value = expected_values.item() if len(expected_values.shape) == 0 else expected_values[0]
            explanation = shap.Explanation(values=shap_values, 
                                         base_values=base_value, 
                                         data=X_sample.values,
                                         feature_names=X_sample.columns.tolist())
        else:
            # Multiple expected values (use first sample's expected value)
            explanation = shap.Explanation(values=shap_values, 
                                         base_values=expected_values[0], 
                                         data=X_sample.values,
                                         feature_names=X_sample.columns.tolist())
        
        # 1. Summary Plot (Feature Importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'{model} - Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Summary Plot (Feature Effects)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'{model} - Feature Effects (SHAP Values)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_feature_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance by Category
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Categorize features with enhanced logic
        def categorize_feature(feature_name):
            feature_lower = feature_name.lower()
            for category, features in feature_categories.items():
                if feature_name in features:
                    return category
                # Also check for partial matches for missing features
                for cat_feature in features:
                    if cat_feature.lower() in feature_lower or feature_lower in cat_feature.lower():
                        return category
            return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # For neural networks with small samples, try to load more data if available
        if model in ['TCN', 'LSTM'] and len(shap_values) < 1000:
            print(f"  Note: {model} has only {len(shap_values)} samples. Consider using larger sample for more robust analysis.")
            
        # Expand sample size for neural networks if possible
        if model in ['TCN', 'LSTM'] and len(shap_values) < 100:
            # Replicate samples to have enough for visualization
            n_replicate = min(10, 100 // len(shap_values))
            if n_replicate > 1:
                shap_values = np.tile(shap_values, (n_replicate, 1))
                X_sample = pd.concat([X_sample] * n_replicate, ignore_index=True)
                print(f"  Expanded sample size to {len(shap_values)} for better visualization")
        
        # Plot feature importance by category with meaningful colors
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Top 20 features with category-based colors
        top_features = importance_df.tail(20)
        feature_colors = [category_colors.get(cat, '#CCCCCC') for cat in top_features['category']]
        
        bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                       color=feature_colors, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax1.set_title(f'{model} - Top 20 Feature Importance (Color by Category)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add feature names to bars
        for i, (bar, cat) in enumerate(zip(bars, top_features['category'])):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{cat}', ha='left', va='center', fontsize=8, alpha=0.7)
        
        # Category importance summary
        category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=True)
        cat_colors = [category_colors.get(cat, '#CCCCCC') for cat in category_importance.index]
        
        bars2 = ax2.barh(range(len(category_importance)), category_importance.values, color=cat_colors)
        ax2.set_yticks(range(len(category_importance)))
        ax2.set_yticklabels(category_importance.index, fontsize=10)
        ax2.set_xlabel('Total |SHAP Value|', fontsize=12)
        ax2.set_title(f'{model} - Importance by Feature Category', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        total_importance = category_importance.sum()
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            pct = (width / total_importance) * 100
            ax2.text(width + total_importance * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', ha='left', va='center', fontsize=9)
        
        # Feature count by category
        category_counts = importance_df.groupby('category').size().sort_values(ascending=True)
        count_colors = [category_colors.get(cat, '#CCCCCC') for cat in category_counts.index]
        
        # Average importance per category
        avg_importance = importance_df.groupby('category')['importance'].mean().sort_values(ascending=True)
        avg_colors = [category_colors.get(cat, '#CCCCCC') for cat in avg_importance.index]
        
        ax4.barh(range(len(avg_importance)), avg_importance.values, color=avg_colors)
        ax4.set_yticks(range(len(avg_importance)))
        ax4.set_yticklabels(avg_importance.index, fontsize=10)
        ax4.set_xlabel('Average |SHAP Value| per Feature', fontsize=12)
        ax4.set_title(f'{model} - Average Feature Importance by Category', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_enhanced_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create category breakdown pie charts
        fig, ax1 = plt.subplots(figsize=(8, 8))
        
        # Importance distribution pie chart
        colors_pie = [category_colors.get(cat, '#CCCCCC') for cat in category_importance.index]
        wedges1, texts1, autotexts1 = ax1.pie(category_importance.values, labels=category_importance.index, 
                                             autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax1.set_title(f'{model} - Feature Importance Distribution by Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_category_importance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Feature Count Pie Chart
        if model == model_list[0]:
            fig, ax2 = plt.subplots(figsize=(8, 8))

            # Feature count pie chart
            colors_pie2 = [category_colors.get(cat, '#CCCCCC') for cat in category_counts.index]
            wedges2, texts2, autotexts2 = ax2.pie(category_counts.values, labels=category_counts.index, 
                                                autopct='%1.0f', colors=colors_pie2, startangle=90)
            ax2.set_title(f'Feature Count Distribution by Category', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            # Fix the path construction - get parent directory properly
            parent_dir = os.path.dirname(output_dir)
            plt.savefig(f"{parent_dir}/category_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
        
        # 5. Dependence plots for top features
        top_5_features = importance_df.head(5)['feature'].values
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_5_features):
            if i < len(axes):
                try:
                    shap.dependence_plot(feature, shap_values, X_sample, ax=axes[i], show=False)
                    axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
                except Exception as e:
                    # If dependence plot fails, create a simple scatter plot
                    feature_idx = X_sample.columns.get_loc(feature)
                    axes[i].scatter(X_sample.iloc[:, feature_idx], shap_values[:, feature_idx], alpha=0.6)
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel(f'SHAP value for {feature}')
                    axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(top_5_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'{model} - Dependence Plots (Top 5 Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_dependence_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Enhanced Waterfall plots for sample predictions
        if len(shap_values) > 0:
            # Select representative samples
            #sample_indices = [0, len(shap_values)//4, len(shap_values)//2, 3*len(shap_values)//4, -1]
            sample_indices = [100,200]
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            
            for i, idx in enumerate(sample_indices[:6]):
                if i < len(axes) and idx < len(shap_values):
                    ax = axes[i]
                    
                    # Get SHAP values for this sample
                    sample_shap = shap_values[idx]
                    sample_data = X_sample.iloc[idx].values
                    feature_names = X_sample.columns.tolist()
                    
                    # Get top 10 features by absolute SHAP value for this sample
                    abs_shap = np.abs(sample_shap)
                    top_indices = np.argsort(abs_shap)[-10:][::-1]
                    
                    top_shap = sample_shap[top_indices]
                    top_features = [feature_names[j] for j in top_indices]
                    top_values = sample_data[top_indices]
                    
                    # Create custom waterfall plot
                    base_value = expected_values[0] if len(expected_values.shape) > 0 else expected_values
                    
                    # Calculate cumulative sum for waterfall
                    cumulative = np.cumsum(np.concatenate([[base_value], top_shap]))
                    
                    # Colors based on positive/negative contribution (red/blue like original SHAP)
                    colors = ['#d62728' if val < 0 else '#1f77b4' for val in top_shap]
                    
                    # Create the waterfall plot
                    x_pos = np.arange(len(top_features))
                    
                    # Plot bars
                    for j, (feat, shap_val) in enumerate(zip(top_features, top_shap)):
                        if shap_val >= 0:
                            ax.bar(j, shap_val, bottom=cumulative[j], color='#1f77b4', alpha=0.8, width=0.8)  # Blue for positive
                        else:
                            ax.bar(j, abs(shap_val), bottom=cumulative[j+1], color='#d62728', alpha=0.8, width=0.8)  # Red for negative
                    
                    # Add base value and prediction lines
                    ax.axhline(y=base_value, color='black', linestyle='--', alpha=0.5, label=f'Base: {base_value:.3f}')
                    ax.axhline(y=cumulative[-1], color='blue', linestyle='-', alpha=0.7, label=f'Prediction: {cumulative[-1]:.3f}')
                    
                    # Customize the plot
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=fs)
                    ax.tick_params(axis='y', labelsize=fs)
                    ax.set_ylabel('SHAP Value', fontsize=fs)
                    ax.set_title(f'Sample {int(idx/100)} - Top 10 Features', fontsize=fs, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=fs*0.6, loc="upper right")

                    # Add value labels on bars (only for significant values)
                    for j, (feat, shap_val) in enumerate(zip(top_features, top_shap)):
                        if abs(shap_val) > 0.01:  # Only label significant values
                            y_pos = cumulative[j] + shap_val/2 if shap_val >= 0 else cumulative[j+1] + abs(shap_val)/2
                            ax.text(j, y_pos, f'{shap_val:.2f}', ha='center', va='center', 
                                   fontsize=fs * 0.6, fontweight='bold', color='black')
            
            # Remove unused subplots
            for i in range(len(sample_indices), len(axes)):
                fig.delaxes(axes[i])
            
            #plt.suptitle(f'{model} - Individual Prediction Breakdowns (Top 10 Features)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model}_waterfall_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Enhanced Summary Statistics with category breakdown
        category_stats = {}
        for category in feature_categories.keys():
            cat_features = importance_df[importance_df['category'] == category]
            if len(cat_features) > 0:
                category_stats[category] = {
                    'count': len(cat_features),
                    'total_importance': cat_features['importance'].sum(),
                    'mean_importance': cat_features['importance'].mean(),
                    'max_importance': cat_features['importance'].max(),
                    'top_feature': cat_features.loc[cat_features['importance'].idxmax(), 'feature'] if len(cat_features) > 0 else 'None'
                }
        
        summary_stats = {
            'Model': model,
            'Total_Features': len(X_sample.columns),
            'Sample_Size': len(shap_values),
            'Top_Feature': importance_df.iloc[-1]['feature'],
            'Top_Feature_Category': importance_df.iloc[-1]['category'],
            'Top_Feature_Importance': importance_df.iloc[-1]['importance'],
            'Mean_SHAP_Value': np.abs(shap_values).mean(),
            'SHAP_Value_Std': np.abs(shap_values).std(),
            **{f'{cat}_Importance_Ratio': (stats['total_importance'] / importance_df['importance'].sum() * 100) 
               for cat, stats in category_stats.items()},
            **{f'{cat}_Feature_Count': stats['count'] for cat, stats in category_stats.items()},
            **{f'{cat}_Top_Feature': stats['top_feature'] for cat, stats in category_stats.items()}
        }
        
        # Save enhanced summary to CSV
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f"{output_dir}/{model}_enhanced_shap_summary.csv", index=False)
        
        # Save category details
        category_details_df = pd.DataFrame(category_stats).T
        category_details_df.to_csv(f"{output_dir}/{model}_category_details.csv")
        
        print(f"\nENHANCED SUMMARY for {model}:")
        print(f"  Total features: {summary_stats['Total_Features']}")
        print(f"  Sample size: {summary_stats['Sample_Size']}")
        print(f"  Top feature: {summary_stats['Top_Feature']} ({summary_stats['Top_Feature_Category']}) - {summary_stats['Top_Feature_Importance']:.4f}")
        print(f"  Mean |SHAP|: {summary_stats['Mean_SHAP_Value']:.4f}")
        
        print(f"\n  CATEGORY BREAKDOWN:")
        for category, stats in category_stats.items():
            ratio = stats['total_importance'] / importance_df['importance'].sum() * 100
            print(f"    {category}: {ratio:.1f}% importance ({stats['count']} features, top: {stats['top_feature']})")
        
        print(f"\nGenerated enhanced plots for {model}:")
        print(f"  - Feature importance: {output_dir}/{model}_feature_importance.png")
        print(f"  - Feature effects: {output_dir}/{model}_feature_effects.png")
        print(f"  - Enhanced importance analysis: {output_dir}/{model}_enhanced_importance_analysis.png")
        print(f"  - Category distribution: {output_dir}/{model}_category_distribution.png")
        print(f"  - Dynamic vs static: {output_dir}/{model}_dynamic_vs_static.png")
        print(f"  - Dependence plots: {output_dir}/{model}_dependence_plots.png")
        print(f"  - Waterfall plots: {output_dir}/{model}_waterfall_plots.png")
        print(f"  - Enhanced summary CSV: {output_dir}/{model}_enhanced_shap_summary.csv")
        print(f"  - Category details CSV: {output_dir}/{model}_category_details.csv")
        
    except Exception as e:
        print(f"Error processing {model}: {str(e)}")
        continue

print(f"\n{'='*60}")
print("SHAP ANALYSIS COMPLETED FOR ALL MODELS")
print(f"{'='*60}")

# Create enhanced comparative analysis
print("\nCreating enhanced comparative analysis...")

# Load all model summaries for comparison
all_summaries = []
for model in model_list:
    summary_file = f"/data0/funato/hsp/0_out/1_fig/shap/{model}/{model}_enhanced_shap_summary.csv"
    if os.path.exists(summary_file):
        model_summary = pd.read_csv(summary_file)
        all_summaries.append(model_summary)

if all_summaries:
    comparison_df = pd.concat(all_summaries, ignore_index=True)
    
    # Extract category importance columns
    category_cols = [col for col in comparison_df.columns if col.endswith('_Importance_Ratio')]
    category_names = [col.replace('_Importance_Ratio', '') for col in category_cols]
    
    # Create comprehensive comparative plots - save each separately
    
    # 1. Category importance comparison (stacked bar)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    bottom = np.zeros(len(comparison_df))
    
    for i, category in enumerate(category_names):
        col_name = f'{category}_Importance_Ratio'
        if col_name in comparison_df.columns:
            values = comparison_df[col_name].fillna(0)
            color = category_colors.get(category, '#CCCCCC')
            ax1.bar(comparison_df['Model'], values, bottom=bottom, 
                   label=category, color=color, alpha=0.8)
            bottom += values
    
    ax1.set_ylabel('Importance Ratio (%)')
    ax1.set_title('Feature Category Importance by Model (Stacked)', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/data0/funato/hsp/0_out/1_fig/shap/category_importance_stacked.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top feature comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    top_features = comparison_df['Top_Feature'].values
    top_categories = comparison_df['Top_Feature_Category'].values
    colors = [category_colors.get(cat, '#CCCCCC') for cat in top_categories]
    
    bars = ax2.bar(comparison_df['Model'], comparison_df['Top_Feature_Importance'], color=colors)
    ax2.set_ylabel('Top Feature |SHAP Value|')
    ax2.set_title('Most Important Feature by Model', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add feature names and categories
    for i, (bar, feature, category) in enumerate(zip(bars, top_features, top_categories)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{feature}\n({category})', ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig("/data0/funato/hsp/0_out/1_fig/shap/top_feature_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Sample size comparison
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    ax3.bar(comparison_df['Model'], comparison_df['Sample_Size'], alpha=0.8)
    ax3.set_ylabel('Sample Size')
    ax3.set_title('SHAP Analysis Sample Size by Model', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add sample size labels
    for i, (model, size) in enumerate(zip(comparison_df['Model'], comparison_df['Sample_Size'])):
        ax3.text(i, size + size*0.02, str(size), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("/data0/funato/hsp/0_out/1_fig/shap/sample_size_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Mean SHAP values comparison
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    ax4.bar(comparison_df['Model'], comparison_df['Mean_SHAP_Value'], alpha=0.8)
    ax4.set_ylabel('Mean |SHAP Value|')
    ax4.set_title('Average Feature Importance by Model', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/data0/funato/hsp/0_out/1_fig/shap/mean_shap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Category-specific comparison (top 3 categories)
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 8))
    top_categories_overall = []
    for category in category_names:
        col_name = f'{category}_Importance_Ratio'
        if col_name in comparison_df.columns:
            avg_importance = comparison_df[col_name].fillna(0).mean()
            top_categories_overall.append((category, avg_importance))
    
    top_categories_overall.sort(key=lambda x: x[1], reverse=True)
    top_3_categories = [cat for cat, _ in top_categories_overall[:3]]
    
    x = np.arange(len(comparison_df))
    width = 0.25
    
    for i, category in enumerate(top_3_categories):
        col_name = f'{category}_Importance_Ratio'
        if col_name in comparison_df.columns:
            values = comparison_df[col_name].fillna(0)
            ax5.bar(x + i*width, values, width, label=category, 
                   color=category_colors.get(category, '#CCCCCC'), alpha=0.8)
    
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Importance Ratio (%)')
    ax5.set_title(f'Top 3 Feature Categories Comparison', fontweight='bold')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(comparison_df['Model'])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/data0/funato/hsp/0_out/1_fig/shap/top3_categories_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    # Save enhanced comparison summary
    comparison_df.to_csv("/data0/funato/hsp/0_out/1_fig/shap/enhanced_model_comparison_summary.csv", index=False)
    
    print("Enhanced comparative analysis saved as separate files:")
    print("  - Category importance (stacked): /data0/funato/hsp/0_out/1_fig/shap/category_importance_stacked.png")
    print("  - Top feature comparison: /data0/funato/hsp/0_out/1_fig/shap/top_feature_comparison.png")
    print("  - Sample size comparison: /data0/funato/hsp/0_out/1_fig/shap/sample_size_comparison.png")
    print("  - Mean SHAP comparison: /data0/funato/hsp/0_out/1_fig/shap/mean_shap_comparison.png")
    print("  - Top 3 categories comparison: /data0/funato/hsp/0_out/1_fig/shap/top3_categories_comparison.png")
    print("  - Complexity vs importance: /data0/funato/hsp/0_out/1_fig/shap/complexity_vs_importance.png")
    print("  - Summary CSV: /data0/funato/hsp/0_out/1_fig/shap/enhanced_model_comparison_summary.csv")

print("\nAll SHAP analyses completed successfully!")


