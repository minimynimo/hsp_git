#General increase in peak flow with increased precipitation, and how the different rate scenarios shift this relationship
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

model_name = "HBV"

input_data = "/data0/funato/hsp/0_out/HBV/peak_discharge/peak_discharge_comparison_HBV.csv"

output_dir = "/data0/funato/hsp/0_out/1_fig/peak_discharge/scatter/"
os.makedirs(output_dir, exist_ok=True)

comparison_df = pd.read_csv(input_data, header=0, index_col=None)

sim_discharge_cols = ['peak1_rate0','peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                    'peak3_rate0','peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                    'peak5_rate0','peak5_rate10', 'peak5_rate30', 'peak5_rate50']

# Make sure comparison_df exists and is not empty before attempting to plot
if not comparison_df.empty:
    # --- 1. Relationship of Original Precipitation Max and Peak Discharge (Overall Trend) ---
    # This plot shows how original precipitation affects different simulated peak flows.
    flow_cols_for_plot = {
        'Simulated Flow (Baseline)': 'sim_peak1_rate0_peak_flow', # Changed label
        'Simulated Flow (Up-Peak1_rate10)': 'sim_peak1_rate10_peak_flow',
        'Simulated Flow (Up-Peak1_rate30)': 'sim_peak1_rate30_peak_flow',
        'Simulated Flow (Up-Peak1_rate50)': 'sim_peak1_rate50_peak_flow',
        'Simulated Flow (Up-Peak3_rate0)': 'sim_peak3_rate0_peak_flow',
        'Simulated Flow (Up-Peak3_rate10)': 'sim_peak3_rate10_peak_flow',
        'Simulated Flow (Up-Peak3_rate30)': 'sim_peak3_rate30_peak_flow',
        'Simulated Flow (Up-Peak3_rate50)': 'sim_peak3_rate50_peak_flow',
        'Simulated Flow (Up-Peak5_rate0)': 'sim_peak5_rate0_peak_flow',
        'Simulated Flow (Up-Peak5_rate10)': 'sim_peak5_rate10_peak_flow',
        'Simulated Flow (Up-Peak5_rate30)': 'sim_peak5_rate30_peak_flow',
        'Simulated Flow (Up-Peak5_rate50)': 'sim_peak5_rate50_peak_flow',
    }

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    for label, col_name in flow_cols_for_plot.items():
        if col_name in comparison_df.columns:
            sns.regplot(
                data=comparison_df,
                x='obs_peak_precip', # Use observed peak precipitation as x-axis for all
                y=col_name,
                scatter_kws={'alpha':0.1, 's':8}, # Make points smaller and more transparent
                line_kws={'lw': 2},
                label=label,
                ax=plt.gca()
            )

    plt.xlabel('Observed Annual Peak Precipitation (mm/day)')
    plt.ylabel('Simulated Annual Peak Flow (m³/s)')
    plt.title(f'Relationship between Observed Peak Precipitation and Simulated Peak Flow ({model_name} Model)')
    plt.legend(title='Simulation Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"precip_vs_peak_flow_all_scenarios_{model_name}.png"))
    plt.show()
    plt.close()

    print(f"Relationship plot saved to: {os.path.join(output_dir, f'precip_vs_peak_flow_all_scenarios_{model_name}.png')}")

    # -----------------------------------------------------------------------------

    # --- 2. Distribution of Annual Peak Flow Lag Relative to Original Peak Precipitation (Overall Trend) ---
    # This plot shows how the lag distribution changes across different scenarios.
    lag_cols_for_plot = {
        'Baseline Simulation': 'sim_peak1_rate0_peak_lag_to_obs_precip', # Changed label and column name
        'Up-Peak1_rate10': 'sim_peak1_rate10_peak_lag_to_obs_precip',
        'Up-Peak1_rate30': 'sim_peak1_rate30_peak_lag_to_obs_precip',
        'Up-Peak1_rate50': 'sim_peak1_rate50_peak_lag_to_obs_precip',
        'Up-Peak3_rate0': 'sim_peak3_rate0_peak_lag_to_obs_precip',
        'Up-Peak3_rate10': 'sim_peak3_rate10_peak_lag_to_obs_precip',
        'Up-Peak3_rate30': 'sim_peak3_rate30_peak_lag_to_obs_precip',
        'Up-Peak3_rate50': 'sim_peak3_rate50_peak_lag_to_obs_precip',
        'Up-Peak5_rate0': 'sim_peak5_rate0_peak_lag_to_obs_precip',
        'Up-Peak5_rate10': 'sim_peak5_rate10_peak_lag_to_obs_precip',
        'Up-Peak5_rate30': 'sim_peak5_rate30_peak_lag_to_obs_precip',
        'Up-Peak5_rate50': 'sim_peak5_rate50_peak_lag_to_obs_precip',
    }

    lag_df_melted_list = []
    # Add observed flow lag to original precip for comparison
    if 'obs_flow_lag_to_obs_precip' in comparison_df.columns:
        temp_obs_df = comparison_df[['file_num', 'year', 'obs_flow_lag_to_obs_precip']].copy()
        temp_obs_df['Scenario'] = 'Observed Flow' # Renamed for clearer legend
        temp_obs_df = temp_obs_df.rename(columns={'obs_flow_lag_to_obs_precip': 'Peak Flow Lag (Days)'})
        lag_df_melted_list.append(temp_obs_df)

    for label, col_name in lag_cols_for_plot.items():
        if col_name in comparison_df.columns:
            temp_df = comparison_df[['file_num', 'year', col_name]].copy()
            temp_df['Scenario'] = label
            temp_df = temp_df.rename(columns={col_name: 'Peak Flow Lag (Days)'})
            lag_df_melted_list.append(temp_df)

    if lag_df_melted_list:
        lag_df_plot = pd.concat(lag_df_melted_list)

        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        sns.boxplot(
            data=lag_df_plot,
            x='Scenario',
            y='Peak Flow Lag (Days)',
            palette='viridis',
            showfliers=False # Do not show outliers to keep plot clean
        )
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Lag')
        plt.xlabel('Scenario / Data Type')
        plt.ylabel('Annual Peak Flow Lag (Days relative to Observed Peak Precipitation)')
        plt.title(f'Distribution of Annual Peak Flow Lag by Scenario ({model_name} Model)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"peak_flow_lag_boxplot_{model_name}.png"))
        plt.show()
        plt.close()
        print(f"Peak flow lag box plot saved to: {os.path.join(output_dir, f'peak_flow_lag_boxplot_{model_name}.png')}")
    else:
        print("No lag data available for plotting.")

    # -----------------------------------------------------------------------------

    ## 3. Distribution of Simulated Peak Flow Differences from Original Simulation 📊
    # This plot directly shows the impact of synthetic precipitation on simulated flow peaks.

    # Identify columns representing simulated flow peaks differences from 'peak1_rate0' (baseline)
    synthetic_peak_flow_diff_cols = [
        f'diff_from_baseline_perc_flow_{s}'
        for s in sim_discharge_cols if s != 'peak1_rate0' # Exclude baseline from diff calc
    ]

    diff_data_melted_list = []
    for col in synthetic_peak_flow_diff_cols:
        if col in comparison_df.columns:
            temp_df = comparison_df[['file_num', 'year', col]].copy()
            scenario_name = col.replace('diff_from_baseline_perc_flow_', '')
            if 'peak' in scenario_name:
                parts = scenario_name.split('_')
                peak_len = parts[0].replace('peak', '')
                rate = parts[1].replace('rate', '')
                scenario_name = f'Up-Peak {peak_len} Day Rate {rate}%'
            
            temp_df['Scenario'] = scenario_name
            temp_df = temp_df.rename(columns={col: 'Percentage Difference from Baseline Flow Peak'}) # Updated label
            diff_data_melted_list.append(temp_df)

    if diff_data_melted_list:
        diff_df_plot = pd.concat(diff_data_melted_list)

        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        sns.boxplot(
            data=diff_df_plot,
            x='Scenario',
            y='Percentage Difference from Baseline Flow Peak', # Updated label
            palette='magma',
            showfliers=False
        )
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Difference')
        plt.xlabel('Synthetic Precipitation Scenario')
        plt.ylabel('Percentage Change in Simulated Peak Flow from Baseline (%)') # Updated label
        plt.title(f'Distribution of Simulated Peak Flow Changes due to Synthetic Precipitation ({model_name} Model)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"peak_flow_diff_boxplot_from_baseline_{model_name}.png")) # New filename
        plt.show()
        plt.close()
        print(f"Peak flow difference box plot saved to: {os.path.join(output_dir, f'peak_flow_diff_boxplot_from_baseline_{model_name}.png')}")
    else:
        print("No flow difference data available for plotting.")

    # -----------------------------------------------------------------------------
    # --- 4. General Statistics and Example Figure per file_num ---

    # Aggregate statistics for the entire comparison_df
    aggregated_stats = comparison_df.drop(columns=['file_num', 'year', 'obs_peak_precip_date', 'obs_peak_flow_date',
                                                'sim_peak1_rate0_peak_date'] +
                                                [f'sim_{s}_peak_date' for s in sim_discharge_cols if s != 'peak1_rate0'],
                                                errors='ignore').agg({ # Dropped date columns from aggregation
        col: ['mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        for col in comparison_df.columns if col not in ['file_num', 'year', 'obs_peak_precip_date', 'obs_peak_flow_date'] +
                                                [f'sim_{s}_peak_date' for s in sim_discharge_cols]
    })
    aggregated_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_stats.columns.values]
    aggregated_stats = aggregated_stats.dropna(axis=1, how='all')

    print("\n--- Aggregated Statistics Across All Files and Years ---")
    print(aggregated_stats)

    aggregated_stats.to_csv(os.path.join(output_dir, f"aggregated_peak_stats_{model_name}.csv"))
    print(f"Aggregated statistics saved to: {os.path.join(output_dir, f'aggregated_peak_stats_{model_name}.csv')}")

    # Select a representative file for an example plot (using file_num = 1)
    example_file_num = 1
    example_df = comparison_df[comparison_df['file_num'] == example_file_num].copy()

    if not example_df.empty:
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")

        # Plotting Peak Flows
        plt.plot(example_df['year'], example_df['obs_peak_flow'], marker='o', linestyle='-', color='blue', label='Observed Peak Flow')
        plt.plot(example_df['year'], example_df['sim_peak1_rate0_peak_flow'], marker='x', linestyle='--', color='green', label='Simulated Flow (Baseline)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate10_peak_flow'], marker='^', linestyle='-.', color='orange', label='Simulated Flow (Up-Peak1_rate10 Precip)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate30_peak_flow'], marker='s', linestyle=':', color='red', label='Simulated Flow (Up-Peak1_rate30 Precip)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate50_peak_flow'], marker='d', linestyle=':', color='purple', label='Simulated Flow (Up-Peak1_rate50 Precip)')

        plt.xlabel('Year')
        plt.ylabel('Annual Peak Flow (m³/s)')
        plt.title(f'Annual Peak Flow Comparison for File {example_file_num} ({model_name} Model) - Evaluation Period')
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"example_annual_peak_flow_file_{example_file_num}.png"))
        plt.show()
        plt.close()
        print(f"Example annual peak flow plot saved to: {os.path.join(output_dir, f'example_annual_peak_flow_file_{example_file_num}.png')}")

        # Plotting Peak Lags
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")

        plt.plot(example_df['year'], example_df['obs_flow_lag_to_obs_precip'], marker='o', linestyle='-', color='blue', label='Observed Flow Lag (to Observed Precip)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate0_peak_lag_to_obs_precip'], marker='x', linestyle='--', color='green', label='Simulated Flow Lag (Baseline)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate10_peak_lag_to_obs_precip'], marker='^', linestyle='-.', color='orange', label='Simulated Flow Lag (Up-Peak1_rate10 Precip)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate30_peak_lag_to_obs_precip'], marker='s', linestyle=':', color='red', label='Simulated Flow Lag (Up-Peak1_rate30 Precip)')
        plt.plot(example_df['year'], example_df['sim_peak1_rate50_peak_lag_to_obs_precip'], marker='d', linestyle=':', color='purple', label='Simulated Flow Lag (Up-Peak1_rate50 Precip)')

        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Zero Lag')
        
        plt.xlabel('Year')
        plt.ylabel('Annual Peak Flow Lag (Days relative to Observed Peak Precipitation)')
        plt.title(f'Annual Peak Flow Lag Comparison for File {example_file_num} ({model_name} Model) - Evaluation Period')
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"example_annual_peak_lag_file_{example_file_num}.png"))
        plt.show()
        plt.close()
        print(f"Example annual peak lag plot saved to: {os.path.join(output_dir, f'example_annual_peak_lag_file_{example_file_num}.png')}")

    else:
        print(f"No data for example file {example_file_num} in the evaluation period.")