# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
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
#loc = "0_JP_MARRMoT_44"
loc = "US"
nocal_buf = "_nocal"
#########

file_tag_BC = file_tag.replace('_zerofix', '')

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

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]



def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    
    # Define the special columns and filter out any that are missing from plot_df
    special_columns = ['BMA', 'RC', 'RCH', 'RC-BC']
    special_columns = [col for col in special_columns if col in plot_df.columns]

    # Separate the remaining columns
    remaining_columns = [col for col in plot_df.columns if col not in special_columns]
    
    # Create boxplot for special columns
    boxplot = ax.boxplot([plot_df[col].values for col in special_columns],
                         patch_artist=True, positions=range(1, len(special_columns) + 1),
                         widths=0.6, labels=special_columns)
    
    # Create boxplot for remaining columns with no gap between them (tight positions)
    remaining_positions = [len(special_columns) + 1 + i * 0.1 for i in range(len(remaining_columns))]
    remaining_boxplot = ax.boxplot([plot_df[col].values for col in remaining_columns],
                                   patch_artist=True, positions=remaining_positions,
                                   widths=0.1)  # Small width to make the boxes appear tight
    
    fs = 22 ## 22 or over for presentation
    # Set title and labels
    ax.set_title(f'Boxplot for {benchmark} {ce_name}\n')
    ax.set_ylabel(benchmark, fontsize = fs)

    # Set xticks with the remaining columns compressed into a single tick label
    middle_remaining_position = (remaining_positions[0] + remaining_positions[-1]) / 2
    
    # Set xticks with the remaining columns compressed into a single tick label
    ticks = list(range(1, len(special_columns) + 1)) + [middle_remaining_position]    
    ax.set_xticks(ticks)
    ax.set_xticklabels(special_columns + ['44 MARRMoT models'], fontsize  = fs)
    ax.set_yticklabels(ax.get_yticks(), fontsize = fs)

    ax.grid(axis = 'y', linewidth = 0.4)
    
    # Define the box colors for the special columns
    box_colors = ['#66c2a4' if col == 'BMA' else '#fdae6b' if col == 'RC' else '#a6bddb' if col == 'RCH' else '#fb8072' if col == 'RC-BC' else 'lightyellow' if col == 'RC-BCwBMA' else 'gainsboro' for col in special_columns]

    # Set color for the special columns
    for patch, color in zip(boxplot['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # Set all remaining columns' boxes to gray
    for patch in remaining_boxplot['boxes']:
        patch.set_facecolor('gainsboro')
    
    # Set median line properties for all boxes
    for median in boxplot['medians'] + remaining_boxplot['medians']:
        median.set_color('black')  # Set the desired color
        median.set_linewidth(2)         # Optionally, adjust the thickness

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))


def create_point_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')
    ax.set_title(f'Point plot for og {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.columns, rotation=90)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

buf = "_ave-bma-indivmodel"
zoom_list = [True, False]

for zoom in zoom_list:
    if zoom:
        benchmark_list = ["logNSE"]
    else:
        benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
    
    for benchmark in benchmark_list:
        # Create a figure with 2 subplots: one for cal and one for eva
        fig_box, axes_box = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        fig_point, axes_point = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        global_y_min, global_y_max = float('inf'), float('-inf')

        BMA_file_path = f'/data0/funato/0_out/99_out/{loc}/BMA{nocal_buf}/results/BMA_results_{ce}.csv'
        RC_file_path = f'/data0/funato/0_out/99_out/{loc}/RC/results/RC_results_{file_tag}_zerofix_{ce}.csv'
        RCHBMA_file_path = f'/data0/funato/0_out/99_out/{loc}/RCHBMA{nocal_buf}/results/RCHBMA_results_{file_tag}_{ce}.csv'
        BC_file_path = f'/data0/funato/0_out/99_out/{loc}/BC{nocal_buf}/results/BC_results_{file_tag_BC}_bma_{ce}.csv'

        BMA_df = pd.read_csv(BMA_file_path) 
        RC_df = pd.read_csv(RC_file_path)
        RCHBMA_df = pd.read_csv(RCHBMA_file_path)
        BC_df = pd.read_csv(BC_file_path)

        BMA_column = f'BMA_{benchmark}_{ce}'
        RC_column = f'RC_{file_tag}_zerofix_{benchmark}_{ce}'
        RCHBMA_column = f'RCHBMA_{file_tag}_{benchmark}_{ce}'
        BC_column = f'BC_{file_tag_BC}_bma_{benchmark}_{ce}'
                    
        # Extract data for the current benchmark
        BMA_data = BMA_df[BMA_column]
        RC_data = RC_df[RC_column]
        RCHBMA_data = RCHBMA_df[RCHBMA_column]
        BC_data = BC_df[BC_column]

        plot_df = pd.DataFrame({
            'BMA': BMA_data,
            'RC': RC_data,
            'RCH': RCHBMA_data,
            'RC-BC': BC_data
})

        """# adding model-based bias correction
        for model_name in model_list:
            indiv_model_file_path = f'/data0/funato/0_out/indiv_model/indiv_model_{model_name}_results_{ce}.csv'
            indiv_model_df = pd.read_csv(indiv_model_file_path)
            indiv_model_column = f'{model_name}_{benchmark}_{ce}'
            indiv_model_data = indiv_model_df[indiv_model_column]
            
            plot_df[model_name] = indiv_model_data"""
        
        # Update global y-axis limits
        y_min, y_max = plot_df.min().min(), plot_df.max().max()
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
        
        if benchmark == 'NSE' or benchmark == 'KGE':
            global_y_min = max(global_y_min, -1)
            global_y_max = 1
        elif benchmark == 'logNSE':
            if zoom:
                global_y_min = max(global_y_min, -50)
            global_y_max = 1
        elif benchmark == 'E1':
            global_y_min = -0.75
            global_y_max = 1
        elif benchmark == 'VE':
            global_y_max = 1
        elif benchmark == 'd':
            global_y_max = 1
        elif benchmark == 'RMSE':
            global_y_min = 0
            global_y_max = 15
        elif benchmark == 'MAE':
            global_y_min = 0


        
        # Set the same y-axis limits for both subplots
        all_axes = [axes_box, axes_point]

        for ax in all_axes:
            ax.set_ylim(global_y_min, global_y_max) 

        create_box_plot(axes_box, plot_df, benchmark, ce)
        #create_line_plot(axes_line, plot_df, benchmark, ce)
        create_point_plot(axes_point, plot_df, benchmark, ce)
        
        fig_box.tight_layout()
        #fig_line.tight_layout()
        fig_point.tight_layout()


        if benchmark == 'logNSE':
            box_output_path = os.path.join(output_dir, f'box/{ce}/{benchmark}_boxplot{buf}_{ce}.jpg')
            #line_output_path = os.path.join(output_dir, f'line/{ce}/{benchmark}_lineplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{ce}/{benchmark}_pointplot{buf}_{ce}.jpg')

        else:
            box_output_path = os.path.join(output_dir, f'box/{ce}/{benchmark}_boxplot{buf}_{ce}.jpg')
            #line_output_path = os.path.join(output_dir, f'line/{ce}/{benchmark}_lineplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{ce}/{benchmark}_pointplot{buf}_{ce}.jpg')

        fig_box.savefig(box_output_path)
        #fig_line.savefig(line_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        #plt.close(fig_line)
        plt.close(fig_point)

print("DONE!")
