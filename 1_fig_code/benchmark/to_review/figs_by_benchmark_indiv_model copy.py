# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#########
file_tag = 'r700_s1_sr0.4_rr0.001_zerofix'
all_mode = True # include all models
#########

output_dir = f'/data0/funato/0_out/0_fig/benchmark/indiv_model'
if all_mode:
    output_dir += '/all_models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)

if 'zerofix' in file_tag:
    fix = 'zerofix'
elif 'nofix' in file_tag:
    fix = 'nofix'
else:
    raise ValueError(f'Invalid fix: set zerofix or nofix in file_tag') 

cal_eva = ['cal', 'eva']

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    
    # Draw the boxplot using matplotlib directly for more control
    boxplot = ax.boxplot(plot_df.values, patch_artist=True, labels=plot_df.columns, widths=0.6)
    
    ax.set_title(f'Boxplot for {benchmark} {ce_name}\n')
    ax.set_ylabel(benchmark)
    ax.set_yticklabels(ax.get_yticks(), fontsize=15)

    # Define the box colors for specific columns
    box_colors = ['#8dd3c7' if col == 'BMA' else '#ffffb3' if col == 'RC' else '#80b1d3' if col == 'RCH' else '#fb8072' if col == 'RC-BC' else 'gainsboro' for col in plot_df.columns]
    main_boxes = ['BMA', 'RC', 'RCH', 'RC-BC']
    
    # Set facecolor for each box
    for patch, color in zip(boxplot['boxes'], box_colors):
        patch.set_facecolor(color)

    for median in boxplot['medians']:
        median.set_color('mediumblue')  # Set the desired color
        median.set_linewidth(2)  # Optionally, adjust the thickness

    # Adjust the width and positions of the boxes and other elements
    main_box_positions = [i for i, col in enumerate(plot_df.columns) if col in main_boxes]
    other_box_positions = [i for i, col in enumerate(plot_df.columns) if col not in main_boxes]

    # Shift the positions of the main boxes to the left half and other boxes to the right half
    left_half_range = np.linspace(4, 16, len(main_box_positions))
    right_half_range = np.linspace(19, 46, len(other_box_positions))

    for i, patch in enumerate(boxplot['boxes']):
        box = patch.get_path().vertices
        if plot_df.columns[i] in main_boxes:
            box[:, 0] = np.clip(box[:, 0], left_half_range[main_box_positions.index(i)] - 0.5, left_half_range[main_box_positions.index(i)] + 2)  # Adjust main boxes
        else:
            box[:, 0] = np.clip(box[:, 0], right_half_range[other_box_positions.index(i)] - 0.3, right_half_range[other_box_positions.index(i)] + 0.3)  # Adjust other boxes
        patch.set_path(patch.get_path())


    # Adjust the positions of the whiskers, caps, medians, and outliers
    for i, whisker in enumerate(boxplot['whiskers']):
        whisker_pos = whisker.get_xdata()
        if plot_df.columns[i // 2] in main_boxes:
            whisker_pos = np.clip(whisker_pos, left_half_range[main_box_positions.index(i // 2)], left_half_range[main_box_positions.index(i // 2)] + 0.4)
        else:
            whisker_pos = np.clip(whisker_pos, right_half_range[other_box_positions.index(i // 2)], right_half_range[other_box_positions.index(i // 2)] + 0.4)
        whisker.set_xdata(whisker_pos)

    for i, cap in enumerate(boxplot['caps']):
        cap_pos = cap.get_xdata()
        if plot_df.columns[i // 2] in main_boxes:
            cap_pos = np.clip(cap_pos, left_half_range[main_box_positions.index(i // 2)] - 0.2, left_half_range[main_box_positions.index(i // 2)] + 0.2)
        else:
            cap_pos = np.clip(cap_pos, right_half_range[other_box_positions.index(i // 2)] - 0.2, right_half_range[other_box_positions.index(i // 2)] + 0.2)
        cap.set_xdata(cap_pos)

    for i, median in enumerate(boxplot['medians']):
        median_pos = median.get_xdata()
        if plot_df.columns[i] in main_boxes:
            median_pos = np.clip(median_pos, left_half_range[main_box_positions.index(i)], left_half_range[main_box_positions.index(i)] + 0.4)
        else:
            median_pos = np.clip(median_pos, right_half_range[other_box_positions.index(i)], right_half_range[other_box_positions.index(i)] + 0.4)
        median.set_xdata(median_pos)

    for i, flier in enumerate(boxplot['fliers']):
        flier_pos = flier.get_xdata()
        if plot_df.columns[i] in main_boxes:
            flier_pos = np.clip(flier_pos, left_half_range[main_box_positions.index(i)], left_half_range[main_box_positions.index(i)] + 0.4)
        else:
            flier_pos = np.clip(flier_pos, right_half_range[other_box_positions.index(i)], right_half_range[other_box_positions.index(i)] + 0.4)
        flier.set_xdata(flier_pos)

    # Adjust x-tick labels to match the new positions
    new_labels = [plot_df.columns[i] for i in main_box_positions + other_box_positions]
    ax.set_xticks(list(left_half_range) + list(right_half_range))
    ax.set_xticklabels([plot_df.columns[i] for i in main_box_positions] + [''] * len(other_box_positions), rotation=90, fontsize=14)
"""
def create_line_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    positions = range(1, len(plot_df.columns) + 1)  # Positions for BMA, RC, RCH-bma
    lw = 0.4
    if benchmark != 'RMSE' and benchmark != 'MAE': # higher values are better
        for i in range(len(plot_df)):  # Loop through each row (data point)
            if plot_df.iloc[i]['RCH-bma'] > plot_df.iloc[i]['BMA'] and plot_df.iloc[i]['RCH-bma'] > plot_df.iloc[i]['RC']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
            elif plot_df.iloc[i]['RCH-bma'] < plot_df.iloc[i]['BMA'] and plot_df.iloc[i]['RCH-bma'] < plot_df.iloc[i]['RC']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='gray')
        # for legends
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='royalblue', label='Good\nRCH-bma > BMA & RC')
        ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='crimson', label='Bad\nRCH-bma < BMA & RC')        
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='gray', label='Intermediate\nRCH-bma between BMA & RC')
            
    else: # lower values are better
        for i in range(len(plot_df)):  # Loop through each row (data point)
            if plot_df.iloc[i]['RCH-bma'] > plot_df.iloc[i]['BMA'] and plot_df.iloc[i]['RCH-bma'] > plot_df.iloc[i]['RC']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='crimson')
            elif plot_df.iloc[i]['RCH-bma'] < plot_df.iloc[i]['BMA'] and plot_df.iloc[i]['RCH-bma'] < plot_df.iloc[i]['RC']:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='royalblue')
            else:
                ax.plot(positions, plot_df.iloc[i], linestyle='-', marker='.', linewidth=lw, color='gray')
        # for legends
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='royalblue', label='Good\nRCH-bma < BMA & RC')
        ax.plot([],[], linestyle='-', marker='.', linewidth=lw, color='crimson', label='Bad\nRCH-bma > BMA & RC')        
        ax.plot([],[],linestyle='-', marker='.', linewidth=lw, color='gray', label='Intermediate\nRCH-bma between BMA & RC')
    ax.set_title(f'Line plot for og {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.columns, rotation=90)
    ax.legend(loc='lower right')
"""
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

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
#benchmark_list = ["NSE"]
buf = "_ave-bma-indivmodel"
zoom = False
for benchmark in benchmark_list:
    # Create a figure with 2 subplots: one for cal and one for eva
    fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    global_y_min, global_y_max = float('inf'), float('-inf')

    for idx, ce in enumerate(cal_eva):
        BMA_file_path = f'/data0/funato/0_out/BMA/results/BMA_results_{ce}.csv'
        RC_file_path = f'/data0/funato/0_out/RC/results/RC_results_{file_tag}_{ce}.csv'
        RCH_bma_file_path = f'/data0/funato/0_out/RC_hybrid_bma/results/RCH_BMA_results_{file_tag}_{ce}.csv'
        Bias_Cor_file_path = f'/data0/funato/0_out/bias_correction/results/BiasCor_results_{file_tag}_{ce}.csv'

        BMA_df = pd.read_csv(BMA_file_path) 
        RC_df = pd.read_csv(RC_file_path)
        RCH_bma_df = pd.read_csv(RCH_bma_file_path)
        Bias_Cor_df = pd.read_csv(Bias_Cor_file_path)

        BMA_column = f'BMA_{benchmark}_{ce}'
        RC_column = f'RC_{file_tag}_{benchmark}_{ce}'
        RCH_bma_column = f'RCHBMA_{file_tag}_{benchmark}_{ce}'
        Bias_Cor_column = f'BiasCor_{file_tag}_{benchmark}_{ce}'
                    
        # Extract data for the current benchmark
        BMA_data = BMA_df[BMA_column]
        RC_data = RC_df[RC_column]
        RCH_bma_data = RCH_bma_df[RCH_bma_column]
        Bias_Cor_data = Bias_Cor_df[Bias_Cor_column]

        plot_df = pd.DataFrame({
            'BMA': BMA_data,
            'RC': RC_data,
            'RCH': RCH_bma_data,
            'RC-BC': Bias_Cor_data
        })

        # adding model-based bias correction
        for model_name in model_list:
            indiv_model_file_path = f'/data0/funato/0_out/indiv_model/indiv_model_{model_name}_results_{ce}.csv'
            indiv_model_df = pd.read_csv(indiv_model_file_path)
            indiv_model_column = f'{model_name}_{benchmark}_{ce}'
            indiv_model_data = indiv_model_df[indiv_model_column]
            
            plot_df[model_name] = indiv_model_data
        
        # Update global y-axis limits
        y_min, y_max = plot_df.min().min(), plot_df.max().max()
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
        
        create_box_plot(axes_box[idx], plot_df, benchmark, ce)
        #create_line_plot(axes_line[idx], plot_df, benchmark, ce)
        create_point_plot(axes_point[idx], plot_df, benchmark, ce)

    # Set the same y-axis limits for both subplots
    all_axes = [axes_box, axes_point]

    for ax_group in all_axes:
        for ax in ax_group:
            if benchmark == 'logNSE':
                if zoom:
                    global_y_min = max(global_y_min, -50)
                ax.set_ylim(global_y_min, 1)
            elif benchmark == 'NSE':
                ax.set_ylim(-1, 1)
            elif benchmark == 'E1':
                ax.set_ylim(-0.75, 1)
            elif benchmark != 'RMSE' and benchmark != 'MAE':
                ax.set_ylim(global_y_min, 1)
            elif benchmark == 'RMSE':
                ax.set_ylim(0, 10)
            else:  # RMSE and MAE
                ax.set_ylim(0, global_y_max)
            
    fig_box.tight_layout()
    #fig_line.tight_layout()
    fig_point.tight_layout()


    if benchmark == 'logNSE':
        box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
        #line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
        point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

    else:
        box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
        #line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
        point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

    fig_box.savefig(box_output_path)
    #fig_line.savefig(line_output_path)
    fig_point.savefig(point_output_path)

    plt.close(fig_box)
    #plt.close(fig_line)
    plt.close(fig_point)

print("DONE!")
