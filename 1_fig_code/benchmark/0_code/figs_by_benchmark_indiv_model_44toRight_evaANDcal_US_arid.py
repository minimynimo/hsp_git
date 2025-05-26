# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, Erel, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

#########
all_mode = True # include all models

#data_type = 'AVE_BMA'
#data_type = 'AVE & BMA & RC & RCH & BC'
data_type = 'BMA & RC & RCH & BC'
#data_type = 'RCH & BC'
#data_type = 'RC & RCH & BC'
#data_type = 'AVE & BMA & RC & RCH & BC & RCBC-BMA'


#loc = "JP"
loc = "US"
#loc = "AUS"
#loc = "GB"

#nocal_tag = "_nocal"
nocal_tag = ""

LSTM = False
M34h = True

arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
arid_file_list = pd.read_csv(arid_file_list)
arid_file_list = arid_file_list['File_num'].tolist()

#reservoir_size = 200
reservoir_size = 700
ridge_param = 0.001 
#ridge_param = 1.0

fs = 22 ## 22 or over for presentation, 13 for paper
#########
file_tag = f'r{reservoir_size}_s1_sr0.4_rr{ridge_param}'

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

benchmark_limits= {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max':10},
    'MAE': {'min': 0, 'max': 10}
}

ce_names = ['Calibration', 'Evaluation']

def create_box_plot(ax, plot_df_cal, plot_df_eva, benchmark, ce):
    plot_dfs = [plot_df_cal, plot_df_eva]
    
    for i, plot_df in enumerate(plot_dfs):
        ce_name = ce_names[i]
        
        # Define the special columns and filter out any that are missing from plot_df
        special_columns_sub = ['AVE', 'BMA', 'RC', 'RCH', 'BC', 'RCBC-BMA']
        if M34h:
            special_columns_sub.append('m34')
        if LSTM:
            special_columns_sub.append('LSTM')
        
        if data_type == 'AVE_BMA':
            special_columns = ['AVE', 'BMA']    
        elif data_type == 'AVE & BMA & RC & RCH & BC':
            special_columns = ['AVE', 'BMA', 'RC', 'RCH', 'BC']
            if M34h:
                special_columns.insert(0, 'FLEX\n-IS')  
            if LSTM:
                special_columns.append('LSTM')
        elif data_type == 'BMA & RC & RCH & BC':
            special_columns = ['BMA', 'RC', 'RCH', 'BC']
            if M34h:    
                special_columns.insert(0, 'FLEX\n-IS')  
            if LSTM:
                special_columns.append('LSTM')
        elif data_type == 'RCH & BC':   
            special_columns = ['RCH', 'BC']
            if M34h:
                special_columns.insert(0, 'FLEX\n-IS')  
            if LSTM:
                special_columns.append('LSTM')
        elif data_type == 'RC & RCH & BC':
            special_columns = ['RC', 'RCH', 'BC']
            if M34h:
                special_columns.insert(0, 'FLEX\n-IS')  
            if LSTM:
                special_columns.append('LSTM')
        elif data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
            special_columns = ['AVE', 'BMA', 'RC', 'RCH', 'BC', 'RCBC-BMA']
            if M34h:
                special_columns.insert(0, 'FLEX\n-IS')  
            if LSTM:
                special_columns.append('LSTM')
        special_columns = [col for col in special_columns if col in plot_df.columns]

        # Separate the remaining columns
        remaining_columns = [col for col in plot_df.columns if col not in special_columns_sub]
        
        # Create boxplot for special columns
        boxplot = ax[i].boxplot([plot_df[col].values for col in special_columns],
                                patch_artist=True, positions=range(1, len(special_columns) + 1),
                                widths=0.6, labels=special_columns)
        
        # Create boxplot for remaining columns with no gap between them (tight positions)
        remaining_positions = [len(special_columns) + 1 + j * 0.1 for j in range(len(remaining_columns))]
        remaining_boxplot = ax[i].boxplot([plot_df[col].values for col in remaining_columns],
                                          patch_artist=True, positions=remaining_positions,
                                          widths=0.1)  # Small width to make the boxes appear tight
        
        # Set title and labels
        if arid_files:
            ax[i].set_title(f'ARID {benchmark} - {ce_name}\n', fontsize = fs)
        else:
            ax[i].set_title(f'NON-ARID {benchmark} - {ce_name}\n', fontsize = fs)
        ax[i].set_ylabel(benchmark, fontsize = fs)

        # Set xticks with the remaining columns compressed into a single tick label
        middle_remaining_position = (remaining_positions[0] + remaining_positions[-1]) / 2
        
        # Set xticks with the remaining columns compressed into a single tick label
        ticks = list(range(1, len(special_columns) + 1)) + [middle_remaining_position]    
        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(special_columns + [f'{str(len(model_list))} MARRMoT models'], fontsize  = fs * 0.85)
        ax[i].set_yticklabels(ax[i].get_yticks(), fontsize = fs)

        ax[i].grid(axis = 'y', linewidth = 0.4)
        
        # Define the box colors for the special columns
        if data_type == 'AVE_BMA':
            box_colors = ['#a6bddb' if col == 'BMA' else '#e3dcf7' if col == 'AVE' else 'gainsboro' for col in special_columns]
        if LSTM:
            if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'RCH & BC':
                box_colors = ['#e3dcf7' if col == 'AVE' else '#a6bddb' if col == 'BMA' else '#fdae6b' if col == 'RC' else '#66c2a4' if col == 'RCH' else '#fb8072' if col == 'BC' else 'pink' if col == 'LSTM' else '#f0ccb6' if col == 'FLEX\n-IS' else 'gainsboro' for col in special_columns]
            if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                box_colors = ['#e3dcf7' if col == 'AVE' else '#a6bddb' if col == 'BMA' else '#fdae6b' if col == 'RC' else '#66c2a4' if col == 'RCH' else '#fb8072' if col == 'BC' else '#8da0cb' if col == 'RCBC-BMA' else 'pink' if col == 'LSTM' else '#f0ccb6' if col == 'FLEX\n-IS' else 'gainsboro' for col in special_columns]
        else:
            if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'RCH & BC':
                box_colors = ['#e3dcf7' if col == 'AVE' else '#a6bddb' if col == 'BMA' else '#fdae6b' if col == 'RC' else '#66c2a4' if col == 'RCH' else '#fb8072' if col == 'BC' else '#f0ccb6' if col == 'FLEX\n-IS' else 'gainsboro' for col in special_columns]
            if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                box_colors = ['#e3dcf7' if col == 'AVE' else '#a6bddb' if col == 'BMA' else '#fdae6b' if col == 'RC' else '#66c2a4' if col == 'RCH' else '#fb8072' if col == 'BC' else '#8da0cb' if col == 'RCBC-BMA' else '#f0ccb6' if col == 'FLEX\n-IS' else 'gainsboro' for col in special_columns]

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

        ax[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

def create_point_plot(ax, plot_df_cal, plot_df_eva, benchmark, ce):
    plot_dfs = [plot_df_cal, plot_df_eva]
    
    for i, plot_df in enumerate(plot_dfs):
        ce_name = ce_names[i]
        positions = list(range(1, len(plot_df.columns) + 1))
        for j in range(len(plot_df)):
            ax[i].plot(positions, plot_df.iloc[j], marker='.', linewidth=0, color='royalblue')
        if arid_files:
            ax[i].set_title(f'Point plot for ARID {benchmark}\n\n{ce_name}\n', fontsize = fs)
        else:
            ax[i].set_title(f'Point plot for NON-ARID {benchmark}\n\n{ce_name}\n', fontsize = fs)
        ax[i].set_ylabel(benchmark)
        ax[i].set_xlabel('Model Type')
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(plot_df.columns, rotation=90)

        ax[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

#buf = "_ave-bma-indivmodel"
buf = ""

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

for arid_files in [True, False]:
    arid_buf = "_arid" if arid_files else "_nonarid"
    if LSTM:
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/{data_type}_wLSTM_{reservoir_size}_{ridge_param}{arid_buf}'
        if M34h:
            output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/{data_type}_wLSTM_M34h_{reservoir_size}_{ridge_param}{arid_buf}'
    elif M34h:
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/{data_type}_wM34h_{reservoir_size}_{ridge_param}{arid_buf}'
    else:
        output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/{data_type}_{reservoir_size}_{ridge_param}{arid_buf}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)

    for benchmark in benchmark_list:
        for ce in ['cal', 'eva']:
            # Create a figure with 2 subplots: one for cal and one for eva
            if data_type == 'AVE_BMA':
                fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
                fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

            fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

            global_y_min, global_y_max = float('inf'), float('-inf')


            AVE_file_path = f'/data0/funato/0_out/99_out/{loc}/AVE{nocal_tag}/results/AVE_results_{ce}.csv'
            BMA_file_path = f'/data0/funato/0_out/99_out/{loc}/BMA{nocal_tag}/results/BMA_results_{ce}.csv'
            if LSTM:
                LSTM_file_path = f'/data0/funato/0_out/99_out/{loc}/LSTM/results/LSTM_results_h50_lr0.001_e70_{ce}.csv'
            if M34h:
                print("A")
                M34h_file_path = f'/data0/funato/0_out/99_out/{loc}/MARRMoT/m34_results_{ce}.csv'
            if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'RCH & BC':
                RC_file_path = f'/data0/funato/0_out/99_out/{loc}/RC_{reservoir_size}_{ridge_param}/results/RC_results_{file_tag}_{ce}.csv'
                RCH_file_path = f'/data0/funato/0_out/99_out/{loc}/RCHBMA_{reservoir_size}_{ridge_param}/results/RCHBMA_results_{file_tag}_{ce}.csv'
                BC_file_path = f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}_{ridge_param}/results/BC_results_{file_tag}_bma_{ce}.csv'
            if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                RCBC_BMA_file_path = f'/data0/funato/0_out/99_out/{loc}/RCBC-BMA_{reservoir_size}_{ridge_param}/results/RCBC-BMA_results_{file_tag}_{ce}.csv'

            AVE_df = pd.read_csv(AVE_file_path)
            BMA_df = pd.read_csv(BMA_file_path)
            if arid_files:
                AVE_df = AVE_df[AVE_df['file_num'].isin(arid_file_list)]
                BMA_df = BMA_df[BMA_df['file_num'].isin(arid_file_list)]
                AVE_df.reset_index(drop=True, inplace=True)
                BMA_df.reset_index(drop=True, inplace=True)
            else:
                AVE_df = AVE_df[~AVE_df['file_num'].isin(arid_file_list)]
                BMA_df = BMA_df[~BMA_df['file_num'].isin(arid_file_list)]
                AVE_df.reset_index(drop=True, inplace=True)
                BMA_df.reset_index(drop=True, inplace=True)
            if LSTM:
                LSTM_df = pd.read_csv(LSTM_file_path) 
            if M34h:
                M34h_df = pd.read_csv(M34h_file_path)
            if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'RCH & BC':
                RC_df = pd.read_csv(RC_file_path)
                RCH_df = pd.read_csv(RCH_file_path)
                BC_df = pd.read_csv(BC_file_path)
                if arid_files:
                    RC_df = RC_df[RC_df['file_num'].isin(arid_file_list)]
                    RCH_df = RCH_df[RCH_df['file_num'].isin(arid_file_list)]
                    BC_df = BC_df[BC_df['file_num'].isin(arid_file_list)]
                    RC_df.reset_index(drop=True, inplace=True)
                    RCH_df.reset_index(drop=True, inplace=True)
                    BC_df.reset_index(drop=True, inplace=True)
                else:
                    RC_df = RC_df[~RC_df['file_num'].isin(arid_file_list)]
                    RCH_df = RCH_df[~RCH_df['file_num'].isin(arid_file_list)]
                    BC_df = BC_df[~BC_df['file_num'].isin(arid_file_list)]
                    RC_df.reset_index(drop=True, inplace=True)
                    RCH_df.reset_index(drop=True, inplace=True)
                    BC_df.reset_index(drop=True, inplace=True)
            if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                RCBC_BMA_df = pd.read_csv(RCBC_BMA_file_path)

            AVE_column = f'AVE_{benchmark}_{ce}'
            BMA_column = f'BMA_{benchmark}_{ce}'
            LSTM_column = f'LSTM_h50_lr0.001_e70_{benchmark}_{ce}'
            M34h_column = f'm34_{benchmark}_{ce}'
            RC_column = f'RC_{file_tag}_{benchmark}_{ce}'
            RCH_column = f'RCHBMA_{file_tag}_{benchmark}_{ce}'
            BC_column = f'BC_{file_tag}_bma_{benchmark}_{ce}'
            RCBC_BMA_column = f'RCBC-BMA_{file_tag}_{benchmark}_{ce}'
                        
            # Extract data for the current benchmark
            AVE_data = AVE_df[AVE_column]
            BMA_data = BMA_df[BMA_column]
            if LSTM:
                LSTM_data = LSTM_df[LSTM_column]
            if M34h:
                M34h_data = M34h_df[M34h_column]
            if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'RCH & BC':
                RC_data = RC_df[RC_column]
                RCH_data = RCH_df[RCH_column]
                BC_data = BC_df[BC_column]
            if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                RCBC_BMA_data = RCBC_BMA_df[RCBC_BMA_column]

            # Create a DataFrame for the current benchmark
            if ce == 'cal':
                plot_df_cal = pd.DataFrame({})
                if 'AVE' in data_type:
                    plot_df_cal['AVE'] = AVE_data
                plot_df_cal['BMA'] = BMA_data
                if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'RCH & BC':
                    if LSTM:
                        plot_df_cal['LSTM'] = LSTM_data
                    if M34h:
                        plot_df_cal['FLEX\n-IS'] = M34h_data
                    plot_df_cal['RC'] = RC_data
                    plot_df_cal['RCH'] = RCH_data
                    plot_df_cal['BC'] = BC_data
                if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                    plot_df_cal['RCBC-BMA'] = RCBC_BMA_data
                for model_name in model_list:
                    MARRMoT_file_path = f'/data0/funato/0_out/99_out/{loc}/MARRMoT_nocal/{model_name}_results_{ce}.csv'
                    MARRMoT_df = pd.read_csv(MARRMoT_file_path)
                    if arid_files:
                        MARRMoT_df = MARRMoT_df[MARRMoT_df['file_num'].isin(arid_file_list)]
                        MARRMoT_df.reset_index(drop=True, inplace=True)
                    else:
                        MARRMoT_df = MARRMoT_df[~MARRMoT_df['file_num'].isin(arid_file_list)]
                        MARRMoT_df.reset_index(drop=True, inplace=True)
                    MARRMoT_column = f'{model_name}_{benchmark}_{ce}'
                    MARRMoT_data = MARRMoT_df[MARRMoT_column]
                    
                    plot_df_cal[model_name] = MARRMoT_data
            else:
                plot_df_eva = pd.DataFrame({})
                if 'AVE' in data_type:
                    plot_df_eva['AVE'] = AVE_data
                plot_df_eva['BMA'] = BMA_data
                if data_type == 'AVE & BMA & RC & RCH & BC' or data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA' or data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'RCH & BC':
                    if M34h:
                        plot_df_eva['FLEX\n-IS'] = M34h_data
                    if LSTM:
                        plot_df_eva['LSTM'] = LSTM_data
                    plot_df_eva['RC'] = RC_data
                    plot_df_eva['RCH'] = RCH_data
                    plot_df_eva['BC'] = BC_data
                if data_type == 'AVE & BMA & RC & RCH & BC & RCBC-BMA':
                    plot_df_eva['RCBC-BMA'] = RCBC_BMA_data

                for model_name in model_list:
                    MARRMoT_file_path = f'/data0/funato/0_out/99_out/{loc}/MARRMoT_nocal/{model_name}_results_{ce}.csv'
                    MARRMoT_df = pd.read_csv(MARRMoT_file_path)
                    if arid_files:
                        MARRMoT_df = MARRMoT_df[MARRMoT_df['file_num'].isin(arid_file_list)]
                        MARRMoT_df.reset_index(drop=True, inplace=True)
                    else:
                        MARRMoT_df = MARRMoT_df[~MARRMoT_df['file_num'].isin(arid_file_list)]
                        MARRMoT_df.reset_index(drop=True, inplace=True)
                    MARRMoT_column = f'{model_name}_{benchmark}_{ce}'
                    MARRMoT_data = MARRMoT_df[MARRMoT_column]
                    
                    plot_df_eva[model_name] = MARRMoT_data

        create_box_plot(axes_box, plot_df_cal, plot_df_eva, benchmark, ce)
        #create_line_plot(axes_line, plot_df_cal, plot_df_eva, benchmark, ce)
        create_point_plot(axes_point, plot_df_cal, plot_df_eva, benchmark, ce)
        
        # Update global y-axis limits
        y_min, y_max = min(plot_df_cal.min().min(), plot_df_eva.min().min()), max(plot_df_cal.max().max(), plot_df_eva.max().max())
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
        
        if benchmark == 'NSE':
            if data_type == 'AVE_BMA':
                global_y_min = max(global_y_min, -5)
            else:
                global_y_min = max(global_y_min, -1)
            global_y_max = 1
        elif benchmark == 'KGE':
            if data_type == 'AVE_BMA':
                global_y_min = max(global_y_min, -5)
            else:
                global_y_min = max(global_y_min, -1)
            global_y_max = 1
        elif benchmark == 'logNSE':
            global_y_max = 1
        elif benchmark == 'E1':
            if data_type == 'AVE_BMA':
                global_y_min = max(global_y_min, -5)
            else:
                global_y_min = max(global_y_min, -1)
            global_y_max = 1
        elif benchmark == 'VE':
            global_y_max = 1
        elif benchmark == 'd':
            global_y_max = 1
        elif benchmark == 'RMSE' or benchmark == 'MAE':
            global_y_min = 0
            if data_type == 'AVE_BMA':
                global_y_max = min(global_y_max, 10)
            else:
                global_y_max = min(global_y_max, 6)

        
        # Set the same y-axis limits for both subplots
        all_axes = [axes_box[0], axes_point[0], axes_box[1], axes_point[1]]

        for ax in all_axes:
            if data_type == 'AVE_BMA':
                ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
            else:
                ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

        fig_box.tight_layout()
        #fig_line.tight_layout()
        fig_point.tight_layout()


        if benchmark == 'logNSE':
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_{ce}.jpg')
            #line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_{ce}.jpg')

        else:
            box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_{ce}.jpg')
            #line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_{ce}.jpg')
            point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_{ce}.jpg')

        fig_box.savefig(box_output_path)
        #fig_line.savefig(line_output_path)
        fig_point.savefig(point_output_path)

        plt.close(fig_box)
        #plt.close(fig_line)
        plt.close(fig_point)

        print(box_output_path)

print("DONE!")
