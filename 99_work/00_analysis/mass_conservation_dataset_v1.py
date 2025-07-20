#!/usr/bin/env python3
"""
Mass Conservation Analysis with Loss Term Calculation

This script calculates the implied long-term "loss" term L = P - Q_adjusted for each model
and compares it against independent estimates of potential evapotranspiration (PET) from 
the CAMELS-GB dataset to assess physical plausibility.

Author: AI Assistant
Date: July 18, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_PATH = Path('/data0/funato/hsp')
CAMELS_GB_PATH = BASE_PATH / 'camels_GB' / 'data'
OUTPUT_PATH = BASE_PATH / '0_out' / 'analysis' / 'mass_conservation'
MODEL_RESULTS_PATH = BASE_PATH / '0_out'

class MassConservationAnalyzer:
    """Analyze mass conservation and calculate loss terms for hydrological models."""
    
    def __init__(self):
        self.camels_gb_data = None
        self.basin_gauge_mapping = None
        self.model_results = {}
        self.loss_analysis = pd.DataFrame()
        
    def load_camels_gb_data(self):
        """Load CAMELS-GB climatic attributes containing PET data."""
        print("Loading CAMELS-GB climatic attributes...")
        
        climatic_file = CAMELS_GB_PATH / 'CAMELS_GB_climatic_attributes.csv'
        if not climatic_file.exists():
            raise FileNotFoundError(f"CAMELS-GB climatic data not found: {climatic_file}")
            
        self.camels_gb_data = pd.read_csv(climatic_file)
        print(f"Loaded CAMELS-GB data for {len(self.camels_gb_data)} catchments")
        
        # Convert PET from mm/day to mm/year (assuming pet_mean is in mm/day)
        self.camels_gb_data['pet_annual_mm'] = self.camels_gb_data['pet_mean'] * 365.25
        
        return self.camels_gb_data
    
    def load_existing_mass_conservation_data(self):
        """Load existing mass conservation analysis if available."""
        mass_cons_file = OUTPUT_PATH / 'mass_conservation_data_v1.csv'
        
        if mass_cons_file.exists():
            print("Loading existing mass conservation data...")
            existing_data = pd.read_csv(mass_cons_file)
            print(f"Loaded existing data with {len(existing_data)} records")
            return existing_data
        else:
            print("No existing mass conservation data found.")
            return None
    
    def load_basin_gauge_mapping(self):
        """Load the mapping between basin IDs and CAMELS-GB gauge IDs."""
        mapping_file = Path('/data0/funato/hsp/3_gis_data/GB/0_data/river_basin/dataset_GB/gb_file_num.csv')
        
        if not mapping_file.exists():
            print(f"Warning: Mapping file not found: {mapping_file}")
            return None
            
        mapping_df = pd.read_csv(mapping_file)
        print(f"Loaded gauge mapping for {len(mapping_df)} basins")
        return mapping_df
    
    def map_basin_ids_to_gauge_ids(self, basin_ids):
        """
        Map basin IDs (like varssim001) to CAMELS-GB gauge IDs using the mapping file.
        """
        # Load mapping if not already loaded
        if not hasattr(self, 'basin_gauge_mapping') or self.basin_gauge_mapping is None:
            self.basin_gauge_mapping = self.load_basin_gauge_mapping()
        
        if self.basin_gauge_mapping is None:
            return [None] * len(basin_ids)
        
        gauge_ids = []
        for basin_id in basin_ids:
            if basin_id.startswith('varssim'):
                # Extract file number from varssim ID
                try:
                    file_num = int(basin_id.replace('varssim', '').lstrip('0'))
                    # Look up gauge_id from mapping
                    mapping_row = self.basin_gauge_mapping[self.basin_gauge_mapping['File_num'] == file_num]
                    if not mapping_row.empty:
                        gauge_ids.append(mapping_row.iloc[0]['gauge_id'])
                    else:
                        gauge_ids.append(None)
                except:
                    gauge_ids.append(None)
            else:
                gauge_ids.append(None)
        
        return gauge_ids
    
    def calculate_loss_terms(self, data=None):
        """
        Calculate loss terms L = P - Q_adjusted for each model and basin.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Existing mass conservation data. If None, will attempt to load.
        """
        if data is None:
            data = self.load_existing_mass_conservation_data()
            
        if data is None:
            print("No data available for loss term calculation.")
            return None
        
        # Ensure CAMELS-GB data is loaded
        if self.camels_gb_data is None:
            self.load_camels_gb_data()
        
        # Calculate loss term if not already present
        if 'Implied_Loss' not in data.columns:
            data['Implied_Loss'] = data['Total_Precip'] - data['Total_Flow']
        
        # Map basin IDs to gauge IDs for PET lookup
        gauge_ids = self.map_basin_ids_to_gauge_ids(data['Basin_ID'])
        data['Gauge_ID'] = gauge_ids
        
        print(f"Sample gauge IDs: {gauge_ids[:5]}")
        print(f"CAMELS-GB gauge_id sample: {self.camels_gb_data['gauge_id'].head().tolist()}")
        print(f"Gauge_ID dtype: {data['Gauge_ID'].dtype}")
        print(f"CAMELS gauge_id dtype: {self.camels_gb_data['gauge_id'].dtype}")
        
        # Ensure matching data types for merge
        data['Gauge_ID'] = data['Gauge_ID'].astype('Int64')  # Handle NaN values
        self.camels_gb_data['gauge_id'] = self.camels_gb_data['gauge_id'].astype('Int64')
        
        # Merge with CAMELS-GB PET data
        data_with_pet = data.merge(
            self.camels_gb_data[['gauge_id', 'pet_annual_mm', 'p_mean', 'aridity']],
            left_on='Gauge_ID',
            right_on='gauge_id',
            how='left'
        )
        
        print(f"Successfully merged data. Rows with PET data: {data_with_pet['pet_annual_mm'].notna().sum()}/{len(data_with_pet)}")
        
        # Calculate loss minus PET
        data_with_pet['Loss_minus_PET'] = data_with_pet['Implied_Loss'] - data_with_pet['pet_annual_mm']
        
        # Assess physical plausibility
        data_with_pet['Loss_Positive'] = data_with_pet['Implied_Loss'] > 0
        data_with_pet['Loss_vs_PET_Ratio'] = data_with_pet['Implied_Loss'] / data_with_pet['pet_annual_mm']
        
        # Define plausibility criteria
        # Plausible if: L > 0 and 0.5 < L/PET < 2.0 (reasonable range)
        data_with_pet['Plausible'] = (
            (data_with_pet['Implied_Loss'] > 0) &
            (data_with_pet['Loss_vs_PET_Ratio'] > 0.5) &
            (data_with_pet['Loss_vs_PET_Ratio'] < 2.0)
        )
        
        self.loss_analysis = data_with_pet
        return data_with_pet
    
    def analyze_model_performance(self):
        """Analyze model performance based on physical plausibility of loss terms."""
        if self.loss_analysis.empty:
            print("No loss analysis data available. Run calculate_loss_terms first.")
            return
        
        # Summary by model
        model_summary = self.loss_analysis.groupby('Model').agg({
            'Plausible': ['count', 'sum', 'mean'],
            'Implied_Loss': ['mean', 'std', 'min', 'max'],
            'Loss_vs_PET_Ratio': ['mean', 'std'],
            'Loss_minus_PET': ['mean', 'std']
        }).round(3)
        
        model_summary.columns = [
            'Total_Basins', 'Plausible_Count', 'Plausible_Fraction',
            'Loss_Mean', 'Loss_Std', 'Loss_Min', 'Loss_Max',
            'Loss_PET_Ratio_Mean', 'Loss_PET_Ratio_Std',
            'Loss_minus_PET_Mean', 'Loss_minus_PET_Std'
        ]
        
        print("\nModel Performance Summary:")
        print("=" * 80)
        print(model_summary)
        
        return model_summary
    
    def create_visualizations(self, save_plots=True, show_plots=False):
        """Create visualizations for loss term analysis."""
        if self.loss_analysis.empty:
            print("No data for visualization. Run calculate_loss_terms first.")
            return
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        fig_path = BASE_PATH / '0_out' / '1_fig' / 'mass_conservation'
        fig_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Loss vs PET scatter plot
            plt.figure(figsize=(12, 8))
            
            models = self.loss_analysis['Model'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = self.loss_analysis[self.loss_analysis['Model'] == model]
                valid_data = model_data.dropna(subset=['Implied_Loss', 'pet_annual_mm'])
                
                plt.scatter(valid_data['pet_annual_mm'], valid_data['Implied_Loss'], 
                           alpha=0.6, label=model, c=[colors[i]], s=50)
            
            # Add 1:1 line
            max_val = max(self.loss_analysis['pet_annual_mm'].max(), 
                         self.loss_analysis['Implied_Loss'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='L = PET')
            
            plt.xlabel('PET (mm/year)', fontsize=12)
            plt.ylabel('Implied Loss L = P - Q (mm/year)', fontsize=12)
            plt.title('Implied Loss vs Potential Evapotranspiration', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(fig_path / 'loss_vs_pet_scatter.png', dpi=300, bbox_inches='tight')
                print(f"Saved plot: {fig_path / 'loss_vs_pet_scatter.png'}")
            if show_plots:
                plt.show()
            plt.close()
            
            # 2. Loss/PET ratio distribution
            plt.figure(figsize=(12, 6))
            
            valid_ratios = self.loss_analysis.dropna(subset=['Loss_vs_PET_Ratio'])
            
            for model in models:
                model_ratios = valid_ratios[valid_ratios['Model'] == model]['Loss_vs_PET_Ratio']
                plt.hist(model_ratios, alpha=0.6, label=model, bins=30, density=True)
            
            plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='L/PET = 1')
            plt.axvline(x=0.5, color='orange', linestyle=':', alpha=0.7, label='Plausible range')
            plt.axvline(x=2.0, color='orange', linestyle=':', alpha=0.7)
            
            plt.xlabel('Loss/PET Ratio', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title('Distribution of Loss/PET Ratios by Model', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(fig_path / 'loss_pet_ratio_distribution.png', dpi=300, bbox_inches='tight')
                print(f"Saved plot: {fig_path / 'loss_pet_ratio_distribution.png'}")
            if show_plots:
                plt.show()
            plt.close()
            
            # 3. Model plausibility comparison
            plt.figure(figsize=(10, 6))
            
            plausibility_summary = self.loss_analysis.groupby('Model')['Plausible'].agg(['count', 'sum'])
            plausibility_summary['fraction'] = plausibility_summary['sum'] / plausibility_summary['count']
            
            bars = plt.bar(plausibility_summary.index, plausibility_summary['fraction'])
            plt.ylabel('Fraction of Physically Plausible Results', fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.title('Physical Plausibility by Model\n(L > 0 and 0.5 < L/PET < 2.0)', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, frac in zip(bars, plausibility_summary['fraction']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{frac:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(fig_path / 'model_plausibility_comparison.png', dpi=300, bbox_inches='tight')
                print(f"Saved plot: {fig_path / 'model_plausibility_comparison.png'}")
            if show_plots:
                plt.show()
            plt.close()
            
            print("All visualizations created successfully!")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Continuing with analysis...")
    
    def export_results(self, filename=None):
        """Export the loss analysis results to CSV."""
        if self.loss_analysis.empty:
            print("No data to export. Run calculate_loss_terms first.")
            return
        
        if filename is None:
            filename = 'mass_conservation_loss_analysis.csv'
        
        output_file = OUTPUT_PATH / filename
        self.loss_analysis.to_csv(output_file, index=False)
        print(f"Results exported to: {output_file}")
        
        # Also export model summary
        model_summary = self.analyze_model_performance()
        summary_file = OUTPUT_PATH / 'model_plausibility_summary.csv'
        model_summary.to_csv(summary_file)
        print(f"Model summary exported to: {summary_file}")

def main():
    """Main analysis workflow."""
    print("=" * 80)
    print("MASS CONSERVATION ANALYSIS WITH LOSS TERM CALCULATION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = MassConservationAnalyzer()
    
    try:
        # Load CAMELS-GB data
        analyzer.load_camels_gb_data()
        
        # Calculate loss terms
        print("\nCalculating loss terms...")
        loss_data = analyzer.calculate_loss_terms()
        
        if loss_data is not None:
            print(f"Calculated loss terms for {len(loss_data)} model-basin combinations")
            
            # Analyze model performance
            print("\nAnalyzing model performance...")
            model_summary = analyzer.analyze_model_performance()
            
            # Create visualizations
            print("\nCreating visualizations...")
            analyzer.create_visualizations()
            
            # Export results
            print("\nExporting results...")
            analyzer.export_results()
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE!")
            print("=" * 80)
            
        else:
            print("Failed to load or process mass conservation data.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
