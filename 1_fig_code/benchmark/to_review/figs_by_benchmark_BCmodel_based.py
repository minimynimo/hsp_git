# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

#########
file_tag = 'r700_s1_sr0.4_rr0.001_nofix'
#########

if 'zerofix' in file_tag:
    fix = 'zerofix'
elif 'nofix' in file_tag:
    fix = 'nofix'
else:
    raise ValueError(f'Invalid fix: set zerofix or nofix in file_tag') 

output_dir = f'/data0/funato/0_out/0_fig/benchmark/bma-RC-RCH_bma_BiasCor_modelbased/{fix}'
os.makedirs(output_dir, exist_ok=True)
    
cal_eva = ['cal', 'eva']

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    ax.set_title(f'Boxplot for {fix} {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xticklabels(plot_df.columns, rotation=90)

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
    ax.set_title(f'Line plot for {fix} {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.columns, rotation=90)
    ax.legend(loc='lower right')

def create_point_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')
    ax.set_title(f'Point plot for {fix} {benchmark} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.columns, rotation=90)

buf = f'_{fix}'
buf += '_modelbased'

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
    
for benchmark in benchmark_list:
    # Create a figure with 2 subplots: one for cal and one for eva
    fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_line, axes_line = plt.subplots(nrows=1, ncols=2, figsize=(18, 7)) 
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
            'RCH-bma': RCH_bma_data,
            'Bias Cor': Bias_Cor_data
        })

        # adding model-based bias correction
        for model in model_list:
            Bias_Cor_modelbased_file_path = f'/data0/funato/0_out/bias_correction/model_based/results/BiasCor_results_{file_tag}_{model}_{ce}.csv'
            Bias_Cor_modelbased_df = pd.read_csv(Bias_Cor_modelbased_file_path)
            Bias_Cor_modelbased_column = f'BiasCor_{file_tag}_{model}_{benchmark}_{ce}'
            Bias_Cor_modelbased_data = Bias_Cor_modelbased_df[Bias_Cor_modelbased_column]
            
            plot_df[f'{model}'] = Bias_Cor_modelbased_data
        
        # Update global y-axis limits
        y_min, y_max = plot_df.min().min(), plot_df.max().max()
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
        
        create_box_plot(axes_box[idx], plot_df, benchmark, ce)
        create_line_plot(axes_line[idx], plot_df, benchmark, ce)
        create_point_plot(axes_point[idx], plot_df, benchmark, ce)

    # Set the same y-axis limits for both subplots
    all_axes = [axes_box, axes_line, axes_point]

    for ax_group in all_axes:
        for ax in ax_group:
            if benchmark == 'logNSE':
                ax.set_ylim(global_y_min, 1)
            elif benchmark != 'RMSE' and benchmark != 'MAE':
                ax.set_ylim(global_y_min, 1)
            else:  # RMSE and MAE
                ax.set_ylim(0, global_y_max)
            
    fig_box.tight_layout()
    fig_line.tight_layout()
    fig_point.tight_layout()


    if benchmark == 'logNSE':
        box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
        line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
        point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')
    else:
        box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
        line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_cal_eva.jpg')
        point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

    fig_box.savefig(box_output_path)
    fig_line.savefig(line_output_path)
    fig_point.savefig(point_output_path)

    plt.close(fig_box)
    plt.close(fig_line)
    plt.close(fig_point)

print("DONE!")
