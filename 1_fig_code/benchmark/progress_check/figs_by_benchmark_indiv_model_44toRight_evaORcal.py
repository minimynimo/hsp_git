# SHOULD USE THE ONE IN THE "0_code" FOLDER INSTEAD
# # Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.ticker as mticker

#########
file_tag = 'r700_s1_sr0.4_rr0.001'
all_mode = False # include all models
loc = "JP"
#loc = "US"
#nocal_buf = "_nocal"
nocal_buf = ""
#########

#file_tag_BC = file_tag.replace('_zerofix', '')

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/BMA_RC_RCHBMA_BC'
if all_mode:
    output_dir += '/all_models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)



#ce = 'cal'
ce = 'eva'
os.makedirs(os.path.join(output_dir, 'box',f'{ce}'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point',f'{ce}'), exist_ok=True)


"""def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    special_columns = ['BMA', 'RC', 'RCH', 'BC']
    remaining_columns = [col for col in plot_df.columns if col not in special_columns]
    remaining_positions = list(range(len(special_columns) + 1, len(special_columns) + len(remaining_columns) + 1))

    if remaining_columns:
        remaining_boxplot = ax.boxplot([plot_df[col].values for col in remaining_columns],
                                       patch_artist=True, positions=remaining_positions,
                                       widths=0.6, labels=remaining_columns)
    boxplot = ax.boxplot([plot_df[col].values for col in special_columns],
                         patch_artist=True, positions=range(1, len(special_columns) + 1),
                         widths=0.6, labels=special_columns)
    
    for box in boxplot['boxes']:
        box.set_facecolor('white')
    if remaining_columns:
        for box in remaining_boxplot['boxes']:
            box.set_facecolor('white')
    
    fs = 22
    ax.set_title(f'Boxplot for {benchmark} {ce_name}\n')
    ax.set_ylabel(benchmark, fontsize=fs)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

buf = "_ave-bma-indivmodel" """

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
#benchmark_list = ["NSE"]   

BMA_file_path = f'/data0/funato/0_out/99_out/{loc}/BMA{nocal_buf}/results/BMA_results_{ce}.csv'
RC_file_path = f'/data0/funato/0_out/99_out/{loc}/RC/results/RC_results_{file_tag}_{ce}.csv'
RCHBMA_file_path = f'/data0/funato/0_out/99_out/{loc}/RCHBMA{nocal_buf}/results/RCHBMA_results_{file_tag}_{ce}.csv'
BC_file_path = f'/data0/funato/0_out/99_out/{loc}/BC{nocal_buf}/results/BC_results_{file_tag}_bma_{ce}.csv'

BMA_df = pd.read_csv(BMA_file_path) 
RC_df = pd.read_csv(RC_file_path)
RCHBMA_df = pd.read_csv(RCHBMA_file_path)
BC_df = pd.read_csv(BC_file_path)

for benchmark in benchmark_list:
    # Load data
    BMA_data = BMA_df[f'BMA_{benchmark}_{ce}']
    RC_data = RC_df[f'RC_{file_tag}_{benchmark}_{ce}']
    RCHBMA_data = RCHBMA_df[f'RCHBMA_{file_tag}_{benchmark}_{ce}']
    BC_data = BC_df[f'BC_{file_tag}_bma_{benchmark}_{ce}']

    # Create DataFrame
    plot_df = pd.DataFrame({
        'BMA': BMA_data,
        'RC': RC_data,
        'RCH': RCHBMA_data,
        'BC': BC_data
    })

    # Debug data
    print(f"Benchmark: {benchmark}, Data summary:")
    print(plot_df.describe())

    # Drop NaN values
    plot_df = plot_df.dropna()

    # Create box plot
    fig_box, ax = plt.subplots(figsize=(7, 8))
    boxplot = ax.boxplot([plot_df[col] for col in plot_df.columns],
                         patch_artist=True, labels=plot_df.columns)

    for box in boxplot['boxes']:
        box.set_facecolor('white')

    # Set y-limits dynamically
    y_min, y_max = plot_df.min().min(), plot_df.max().max()
    y_min = -5
    ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

    # Titles and labels
    ax.set_title(f'Boxplot for {benchmark} ({ce})')
    ax.set_ylabel(benchmark)
    plt.tight_layout()

    ax.grid(True)
    # Save plot
    box_output_path = os.path.join(output_dir, f'box/{ce}/{benchmark}_boxplot_{ce}.jpg')
    print(f"Saving boxplot to: {box_output_path}")
    fig_box.savefig(box_output_path)
    plt.close(fig_box)
