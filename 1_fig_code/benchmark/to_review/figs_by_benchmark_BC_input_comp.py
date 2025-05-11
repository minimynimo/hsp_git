# Creates box plots and point plots for each benchmark type (KGE, NSE, logNSE, E1, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

#########
file_tag = 'r700_s1_sr0.4_rr0.001_zerofix'
only_input_comp = True
nocal = True
#########
input_comp_file_tag = file_tag.replace('_nofix', '').replace('_zerofix', '') + '_bma'

if nocal:
    nocal_tag = '_nocal'
else:
    nocal_tag = ''

if 'zerofix' in file_tag:
    fix = 'zerofix'
elif 'nofix' in file_tag:
    fix = 'nofix'
else:
    raise ValueError(f'Invalid fix: set zerofix or nofix in file_tag') 
file_tag_BC = file_tag.replace('_zerofix', '')

if only_input_comp:
    output_dir = f'/data0/funato/0_out/0_fig/benchmark/BC_input_comp_only{nocal_tag}/{fix}'
else:
    output_dir = f'/data0/funato/0_out/0_fig/benchmark/BC_input_comp{nocal_tag}/{fix}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)

    
cal_eva = ['cal', 'eva']
#input_list = ["Pp-T-PET-FlwEr","Pp-T-FlwEr","Pp-PET-FlwEr","T-PET-FlwEr","Pp-FlwEr","T-FlwEr","PET-FlwEr","FlwEr"]
input_list = ["Pp-T-PET","Pp-T","Pp-PET","T-PET","Pp","T","PET"]


def create_box_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    plot_df.boxplot(
        ax=ax, 
        boxprops=dict(linestyle='-'), 
        whiskerprops=dict(linestyle='-'), 
        capprops=dict(linestyle='-'), 
        medianprops=dict(linestyle='-')
    )
    ax.set_title(f'Boxplot for {fix} {benchmark}{nocal_tag} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xticklabels(plot_df.columns, rotation=90)

def create_point_plot(ax, plot_df, benchmark, ce):
    ce_name = 'calibration' if ce == 'cal' else 'evaluation'
    positions = list(range(1, len(plot_df.columns) + 1))
    for i in range(len(plot_df)):
        ax.plot(positions, plot_df.iloc[i], marker='.', linewidth=0, color='royalblue')
    ax.set_title(f'Point plot for {fix} {benchmark}{nocal_tag} {ce_name}')
    ax.set_ylabel(benchmark)
    ax.set_xlabel('Model Type')
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.columns, rotation=90)

zoomlist  = [True, False]

for zoom in zoomlist:
    buf = f'_{fix}'
    buf += '_input_comp'
    if zoom:
        buf += '_zoom'

    if zoom == False:
        benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
    else: # zoom == True
        benchmark_list = ["logNSE"]
        
    for benchmark in benchmark_list:
        # Create a figure with 2 subplots: one for cal and one for eva
        fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
        fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

        global_y_min, global_y_max = float('inf'), float('-inf')

        for idx, ce in enumerate(cal_eva):
            if not only_input_comp:
                BMA_file_path = f'/data0/funato/0_out/BMA{nocal_tag}/results/BMA_results_{ce}.csv'
                RC_file_path = f'/data0/funato/0_out/RC/results/RC_results_{file_tag}_{ce}.csv'
                RCHBMA_file_path = f'/data0/funato/0_out/RCHBMA{nocal_tag}/results/RCHBMA_results_{file_tag}_{ce}.csv'
                BC_file_path = f'/data0/funato/0_out/BC{nocal_tag}/results/BC_results_{file_tag_BC}_bma_{ce}.csv'


                BMA_df = pd.read_csv(BMA_file_path)
                RC_df = pd.read_csv(RC_file_path)
                RCHBMA_df = pd.read_csv(RCHBMA_file_path)
                BC_df = pd.read_csv(BC_file_path)

                BMA_column = f'BMA_{benchmark}_{ce}'
                RC_column = f'RC_{file_tag}_{benchmark}_{ce}'
                RCHBMA_column = f'RCHBMA_{file_tag}_{benchmark}_{ce}'
                BC_column = f'BC_{file_tag}_{benchmark}_{ce}'
                        
                # Extract data for the current benchmark
                BMA_data = BMA_df[BMA_column]
                RC_data = RC_df[RC_column]
                RCHBMA_data = RCHBMA_df[RCHBMA_column]
                Bias_Cor_data = BC_df[BC_column]

                plot_df = pd.DataFrame({
                    'BMA': BMA_data,
                    'RC': RC_data,
                    'RCH': RCHBMA_data,
                    'RC-BC': Bias_Cor_data
                })

            else:
                plot_df = pd.DataFrame()

            # adding model-based bias correction
            for input in input_list:
                BC_input_comp_file_path = f'/data0/funato/0_out/BC-input-comp{nocal_tag}/results/BC_input_comp_results_{file_tag_BC}_bma_{input}_{ce}.csv'
                BC_input_comp_df = pd.read_csv(BC_input_comp_file_path)
                BC_input_comp_column = f'BC_{file_tag_BC}_bma_{input}_{benchmark}_{ce}'
                BC_input_comp_data = BC_input_comp_df[BC_input_comp_column]
                
                plot_df[f'BC_{input}'] = BC_input_comp_data
            
            # Update global y-axis limits
            y_min, y_max = plot_df.min().min(), plot_df.max().max()
            global_y_min = min(global_y_min, y_min)
            global_y_max = max(global_y_max, y_max)
            
            create_box_plot(axes_box[idx], plot_df, benchmark, ce)
            create_point_plot(axes_point[idx], plot_df, benchmark, ce)

        if benchmark == 'logNSE':
            if zoom:
                global_y_min = max(global_y_min, -50)
            global_y_max = 1
        elif benchmark != 'RMSE' and benchmark != 'MAE':
            global_y_max = 1
        else:  # RMSE and MAE
            global_y_min = 0

        # Set the same y-axis limits for both subplots
        all_axes = [axes_box, axes_point]

        for ax_group in all_axes:
            for ax in ax_group:
                ax.set_ylim(global_y_min, global_y_max)

        fig_box.tight_layout()
        fig_point.tight_layout()


        if benchmark == 'logNSE':
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')
        else:
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_cal_eva.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_cal_eva.jpg')

        fig_box.savefig(box_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        plt.close(fig_point)

print("DONE!")
