# Create boxplots for each benchmark type for calibration and evaluation periods
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

#########
file_tag = 'r700_s1_sr0.4_rr0.001_nofix'
#########

cal_eva = ['cal', 'eva']

# Define benchmark types
benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]

# Create output directory if it doesn't exist
output_dir = '/data0/funato/0_out/fig/box/benchmark/'
os.makedirs(output_dir, exist_ok=True)

# Create boxplots for each benchmark type
for benchmark in benchmark_list:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 9))
    
    y_min, y_max = float('inf'), float('-inf')
    
    for idx, ce in enumerate(cal_eva):
        BMA_file_path = f'/data0/funato/0_out/BMA/results/BMA_results_{ce}.csv'
        RC_file_path = f'/data0/funato/0_out/RC/results/RC_results_{file_tag}_{ce}.csv'
        RCH_bma_file_path = f'/data0/funato/0_out/RC_hybrid_bma/results/RCH_BMA_results_{file_tag}_{ce}.csv'

        # Read CSV files
        BMA_df = pd.read_csv(BMA_file_path)
        RC_df = pd.read_csv(RC_file_path)
        RCH_bma_df = pd.read_csv(RCH_bma_file_path)

        # Define column names
        BMA_column = f'BMA_{benchmark}_{ce}'
        RC_column = f'RC_{file_tag}_{benchmark}_{ce}'
        RCH_bma_column = f'RCHBMA_{file_tag}_{benchmark}_{ce}'
        
        # Extract data for the current benchmark
        BMA_data = BMA_df[BMA_column]
        RC_data = RC_df[RC_column]
        RCH_bma_data = RCH_bma_df[RCH_bma_column]
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'BMA': BMA_data,
            'RC': RC_data,
            'RCH-bma': RCH_bma_data
        })
        
        # Create boxplot without vertical lines
        boxplot = plot_df.boxplot(
            ax=axes[idx], 
            boxprops=dict(linestyle='-'), 
            whiskerprops=dict(linestyle='-'), 
            capprops=dict(linestyle='-'), 
            medianprops=dict(linestyle='-')
        )
        axes[idx].set_title(f'Boxplot for {benchmark}_{ce}')
        axes[idx].set_ylabel(benchmark)
        
        # Update y-axis limits
        current_y_min, current_y_max = axes[idx].get_ylim()
        y_min = min(y_min, current_y_min)
        y_max = max(y_max, current_y_max)
    
    # Set the same y-axis limits for both plots
    for ax in axes:
        if benchmark == 'logNSE':
            y_min = max(y_min, -50)
            ax.set_ylim(y_min, 1)
        elif benchmark == 'Erel':
            ymin = max(y_min, -10)
            ax.set_ylim(ymin, 1)
        elif benchmark != 'RMSE' and benchmark != 'MAE':
            ax.set_ylim(y_min, 1)
        else:# RMSE and MAE
            ax.set_ylim(y_min, y_max)
        
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{benchmark}_boxplot_cal_eva.jpg')
    plt.savefig(output_path)
    plt.close()

print("DONE!")