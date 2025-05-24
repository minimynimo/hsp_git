# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

#########
file_tag = 'r700_s1_sr0.4_rr0.001_zerofix'
all_test = True # include BMA, RC, RCH-bma, RCH-bc, and Bias Correction
#########

if 'zerofix' in file_tag:
    fix = 'zerofix'
elif 'nofix' in file_tag:
    fix = 'nofix'
else:
    raise ValueError(f'Invalid fix: set zerofix or nofix in file_tag') 

output_dir = f'/data0/funato/0_out/0_fig/benchmark/comb_BiasCorModels_indiv_model'

if all_test:
    output_dir += '/all_models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)

    
#cal_eva = ['cal', 'eva']
cal_eva = ['eva','eva']

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

def create_box_plot_nocolor(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    ax.set_title(f'Boxplot for {fix} {benchmark} {ce_name}\n')
    ax.set_ylabel(benchmark)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
    ax.set_xticklabels(plot_df.columns, rotation=90, fontsize = 15)

# color good models blue and bad models red
def create_box_plot_indiv_models(ax, plot_df, benchmark, ce):
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

zoomlist  = [True, False]

for zoom in zoomlist:
    buf = f'_{fix}'
    buf += '_comb_BCmodel_indiv_model'
    if zoom:
        buf += '_zoom'
    if all_test:
        buf += '_all_models'

    if zoom == False:
        benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
    else: # zoom == True
        benchmark_list = ["logNSE"]
        
    for benchmark in benchmark_list:
        for idx, ce in enumerate(cal_eva):
            # Create a figure with 2 subplots: one for cal and one for eva
            fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            fig_point, ax_point = plt.subplots(nrows=1, ncols=1, figsize=(25, 8))

            global_y_min, global_y_max = float('inf'), float('-inf')

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


            plot_df_indiv = pd.DataFrame({
                'BMA': BMA_data,
                'RC': RC_data,
                'RCH-bma': RCH_bma_data,
                'RC-BC': Bias_Cor_data
            })

            # Add individual model data to plot_df_indiv
            for model_name in model_list:
                indiv_model_file_path = f'/data0/funato/0_out/indiv_model/indiv_model_{model_name}_results_{ce}.csv'
                indiv_model_df = pd.read_csv(indiv_model_file_path)
                indiv_model_column = f'{model_name}_{benchmark}_{ce}'
                indiv_model_data = indiv_model_df[indiv_model_column]
                
                plot_df_indiv[model_name] = indiv_model_data

            global_y_min = -1

            # Update global y-axis limits
            y_min_indiv, y_max_indiv = plot_df_indiv.min().min(), plot_df_indiv.max().max()
            global_y_max = max(global_y_max,y_max_indiv)
        
            
            # Create box plots and point plots for individual models
            ###Change to plot_indiv_models to color good models blue and bad models red see PPT0919
            #create_box_plot_indiv_models(ax_box[0], plot_df_indiv, benchmark, ce)
            create_box_plot_nocolor(ax_box, plot_df_indiv, benchmark, ce)
            create_point_plot(ax_point, plot_df_indiv, benchmark, ce)
            
            # Set the same y-axis limits for both subplots
            all_ax = [ax_box, ax_point]

            for ax in all_ax:
                if benchmark == 'logNSE':
                    if zoom:
                        global_y_min = max(global_y_min, -50)
                    ax.set_ylim(global_y_min, 1)
                elif benchmark != 'RMSE' and benchmark != 'MAE':
                    ax.set_ylim(global_y_min, 1)
                else:  # RMSE and MAE
                    ax.set_ylim(0, global_y_max)
                    
            fig_box.tight_layout()
            fig_point.tight_layout()


        if benchmark == 'logNSE':
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_{ce}.jpg')
        else:
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_{ce}.jpg')

        fig_box.savefig(box_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        plt.close(fig_point)

print("DONE!")

