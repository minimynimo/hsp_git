import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

fontsize = 24

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_shap_data():
    """Load SHAP values and feature names for all models."""
    base_path = Path("/data0/funato/hsp2/0_out")
    models = ['RF', 'XGB', 'TCN', 'LSTM']
    
    model_data = {}
    
    for model in models:
        # Load SHAP values
        shap_file = base_path / model / "GB" / "shap" / "regional" / "shap_objects" / "shap_values.npy"
        
        # Try different feature file names
        features_file = base_path / model / "GB" / "shap" / "regional" / "shap_objects" / "X_features.csv"
        if not features_file.exists():
            features_file = base_path / model / "GB" / "shap" / "regional" / "shap_objects" / "X_sample.csv"
        
        if shap_file.exists() and features_file.exists():
            try:
                # Load SHAP values
                shap_values = np.load(shap_file)
                
                # Load features data to get feature names
                if "X_features.csv" in str(features_file):
                    # RF model - no index column
                    features_df = pd.read_csv(features_file)
                    feature_names = features_df.columns.tolist()
                else:
                    # Other models - has index column
                    features_df = pd.read_csv(features_file, index_col=0)
                    feature_names = features_df.columns.tolist()
                
                # Ensure SHAP values and feature names match
                if shap_values.shape[1] != len(feature_names):
                    print(f"Warning: {model} dimension mismatch - SHAP: {shap_values.shape[1]}, Features: {len(feature_names)}")
                    # Try to trim to match
                    min_features = min(shap_values.shape[1], len(feature_names))
                    shap_values = shap_values[:, :min_features]
                    feature_names = feature_names[:min_features]
                
                # Calculate feature importance from SHAP values
                # Importance is typically the mean absolute SHAP value across all samples
                feature_importance_values = np.mean(np.abs(shap_values), axis=0)
                
                # Convert to percentage of total importance
                total_importance = np.sum(feature_importance_values)
                importance_percentages = (feature_importance_values / total_importance) * 100
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_percentages
                }).sort_values('importance', ascending=False)
                
                model_data[model] = feature_importance
                print(f"Loaded {model}: {len(feature_names)} features, {shap_values.shape[0]} samples")
            except Exception as e:
                print(f"Error loading {model}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: SHAP data files not found for {model}")
            print(f"  SHAP file exists: {shap_file.exists()}")
            print(f"  Features file tried: {features_file}")
    
    return model_data

# Load the data
print("Loading SHAP data...")
model_data = load_shap_data()

if not model_data:
    print("No SHAP data found!")
    exit()

# Get top 10 features across all models
print("\nFinding top 10 features across all models...")
all_features = {}

for model, data in model_data.items():
    for _, row in data.iterrows():
        feature = row['feature']
        importance = row['importance']
        if feature not in all_features:
            all_features[feature] = []
        all_features[feature].append(importance)

# Calculate average importance across models for each feature
avg_importance = {}
for feature, importances in all_features.items():
    avg_importance[feature] = np.mean(importances)

# Get top 10 features
top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
top_feature_names = [f[0] for f in top_features]

print(f"Top 10 features: {top_feature_names}")

# Create heatmap matrix
models = list(model_data.keys())
importance_matrix = np.zeros((len(top_feature_names), len(models)))

for j, model in enumerate(models):
    feature_dict = dict(zip(model_data[model]['feature'], model_data[model]['importance']))
    for i, feature in enumerate(top_feature_names):
        if feature in feature_dict:
            importance_matrix[i, j] = feature_dict[feature]

# Create the heatmap
fig, ax = plt.subplots(1, 1, figsize=(13, 9))

# Create heatmap
im = ax.imshow(importance_matrix, cmap='viridis_r', aspect='auto', alpha = 0.8)

# Set ticks and labels
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=fontsize)
ax.set_yticks(range(len(top_feature_names)))
ax.set_yticklabels(top_feature_names, fontsize=fontsize)

# Set title and labels
ax.set_title('Top 10 Feature SHAP Importance by Model',fontsize=fontsize)
ax.set_xlabel('Models', fontsize=fontsize)
ax.set_ylabel('Features', fontsize=fontsize)

# Add text annotations
for i in range(len(top_feature_names)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{importance_matrix[i, j]:.1f}%',
                      ha="center", va="center", color="black", fontsize=fontsize*0.9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Feature Importance (%)', rotation=270, labelpad=24, fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/data0/funato/hsp2/0_out/1_fig/shap/f02_top10_shap_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_path}")

plt.close()

# Print summary table
print("\nTop 10 Feature Summary:")
print("=" * 60)
for i, (feature, avg_imp) in enumerate(top_features, 1):
    print(f"{i:2d}. {feature:<20} | Avg Importance: {avg_imp:.2f}%")
    
print("\nAnalysis complete!")
print("Note: Importance calculated as percentage of total SHAP importance per model")
    