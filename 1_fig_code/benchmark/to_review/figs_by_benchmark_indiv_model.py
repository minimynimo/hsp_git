# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = f'/data0/funato/0_out/0_fig/benchmark/indiv_model'
os.makedirs(output_dir, exist_ok=True)
    
cal_eva = ['cal', 'eva']

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    
    # Draw the boxplot using matplotlib directly for more control
    boxplot = ax.boxplot(plot_df.values, patch_artist=True, labels=plot_df.columns)
    
    ax.set_title(f'Boxplot for {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xticklabels(plot_df.columns, rotation=90)

    # Define the box colors for specific columns
    box_colors = ['lemonchiffon' if col == 'ave' else 'lemonchiffon' if col == 'BMA' else 'lightpink' if col in ['m01', 'm14', 'm22', 'm23'] else 'paleturquoise' if col in ['m07', 'm16'] else 'gainsboro' for col in plot_df.columns]

    # Set facecolor for each box
    for patch, color in zip(boxplot['boxes'], box_colors):
        patch.set_facecolor(color)

    for median in boxplot['medians']:
        median.set_color('mediumblue')  # Set the desired color
        median.set_linewidth(2)    # Optionally, adjust the thickness


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
    fig_line, axes_line = plt.subplots(nrows=1, ncols=2, figsize=(18, 7)) 
    fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    global_y_min, global_y_max = float('inf'), float('-inf')

    for idx, ce in enumerate(cal_eva):
        ave_file_path = f'/data0/funato/0_out/indiv_model/indiv_model_ave_results_{ce}.csv'
        BMA_file_path = f'/data0/funato/0_out/BMA/results/BMA_results_{ce}.csv'

        ave_df = pd.read_csv(ave_file_path)
        BMA_df = pd.read_csv(BMA_file_path)

        ave_column = f'ave_{benchmark}_{ce}'
        BMA_column = f'BMA_{benchmark}_{ce}'
                
        # Extract data for the current benchmark
        ave_data = ave_df[ave_column]
        BMA_data = BMA_df[BMA_column]

        plot_df = pd.DataFrame({
            'ave': ave_data,
            'BMA': BMA_data
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
    all_axes = [axes_box, axes_line, axes_point]

    for ax_group in all_axes:
        for ax in ax_group:
            if benchmark == 'logNSE':
                if zoom:
                    global_y_min = max(global_y_min, -50)
                ax.set_ylim(global_y_min, 1)
            elif benchmark != 'RMSE' and benchmark != 'MAE':
                ax.set_ylim(global_y_min, 1)
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
