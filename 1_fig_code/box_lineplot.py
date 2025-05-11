import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#######
#benchmark =  'NSE'
benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]

ticks = False
lineplot = False
#lineplot = False

colors = None

##for benchmarks
#indiv_models,num = True, 2 ### benckmark; 7, benchmark_error: 2
#indiv_models, num = False, 0
indiv_models, num = False, False 
#######
def add_trend_line(ax, x, y, color):
    if len(x) > 0:  # Ensure there are points to fit
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        trend_x = np.linspace(min_val, max_val, 100)
        trend_y = model.predict(trend_x.reshape(-1, 1))
        ax.plot(trend_x, trend_y, color=color, linewidth=2, alpha = 0.6)


# Load the CSV file
file_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/csv'
### MARRMoT ###
#file_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/csv/BMA'
### RCH-bma ###
#file_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/RC_hybrid_bma'
### RC ###
#file_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/RC'

file_name = "ave_indiv_results_all"
data= pd.read_csv(f'{file_path}/{file_name}.csv')

df = data.iloc[:, 1:]

if indiv_models == False and num == True:
    column_names = df.columns.tolist()  # Adjust this if your column names are different
    column_order = [0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13]
    column_order_names = [column_names[i] for i in column_order]
    df = df[column_order_names]


#df['m20'] = df['m20'].fillna(method='ffill') 

# Verify the shape of the DataFrame
print(f"DataFrame shape: {df.shape}")

for benchmark in benchmark_list:
    # Create box plots for each column
    #plt.figure(figsize=(40,13))
    plt.figure(figsize=(4,5))


    """
    boxprops = dict(linestyle='-', linewidth=2, color='blue')
    whiskerprops = dict(linestyle='--', linewidth=2, color='green')
    capprops = dict(linestyle='-', linewidth=2, color='black')
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    flierprops = dict(marker='o', color='orange', markersize=8)

    boxplot = df.boxplot(
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops
    )
    """

    if indiv_models == True:
        sub_colors = [['salmon'] *  num + ['lightblue'] * num + ['gray'] * num + ['lightgreen']*num]
        print(colors)
        colors = sub_colors[0]*11
        boxplot = plt.boxplot(df.values, showfliers=False, patch_artist=True,medianprops=dict(color='olivedrab', linewidth = 3))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
            #patch.set_alpha(0.3) 
    else: 
        if colors == None:
            boxplot = plt.boxplot(df.values, showfliers=False, patch_artist=False,medianprops=dict(color='olivedrab', linewidth = 3))
        else:
            boxplot = plt.boxplot(df.values, showfliers=False, patch_artist=True,medianprops=dict(color='olivedrab', linewidth = 3))
            colors =  ['salmon', 'lightblue']+['gray'] * 44 
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
                #patch.set_alpha(0.3) 


        


    # Change colors: first 44 gray, 45th red, 46th blue
    #colors =  ['salmon', 'lightblue']+['gray'] * 44 


    """
    for whisker, color in zip(boxplot['whiskers'], colors*2):
        whisker.set_color(color)
    for cap, color in zip(boxplot['caps'], colors*2):
        cap.set_color(color)
    for median, color in zip(boxplot['medians'], colors):
        median.set_color('olivedrab')  # Keep the median color same for all
    for flier, color in zip(boxplot['fliers'], colors):
        flier.set_color(color)
    """
    """
    # Define custom colors for each box
    colors = [None, None, 'royalblue','crimson', 'royalblue','royalblue', 'crimson','royalblue']

    # Apply custom colors to the boxes
    for patch, color in zip(boxplot['boxes'], colors):
        if color is None:
            patch.set_facecolor('none')  # No fill color
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

    """

    # Customize the plot
    #plt.ylabel('indices', fontsize=15)
    plt.ylabel(f'{benchmark}', fontsize=17)
    plt.xticks(ticks=np.arange(1, len(df.columns) + 1), labels=df.columns, rotation=0, fontsize=15)  # Add column labels and rotate them if they overlap

    plt.tick_params(labelsize=15)

    if ticks == True:
        #yticks = np.arange(-0.25, 1.125, 0.125)   # Create more y-ticks
        yticks = np.arange(-0.375, 1.125, 0.125)   # Create more y-ticks
        plt.yticks(yticks, fontsize=14)


    plt.grid(True, linestyle='--', linewidth=0.25)
    #plt.ylim(0,10)
    plt.ylim(0,1)


    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/fig/box/{file_name}_box_plt.png')

    plt.close()


if lineplot == True:
    data2 = pd.read_csv(f'{file_path}/{file_name}.csv')
    df2 = data2.iloc[:, 1:]

    colors = []
    labels = []
    improved_list = []
    worsened_list = []
    intermediate_list = []
    improved = 0
    worsened = 0
    intermediate = 0
    RCHBMA_column = df2['RCH-bma']
    RC_column = df2['RC']
    BMA_column = df2['BMA']
    for i in range(df2.shape[0]):
        if RCHBMA_column[i] > BMA_column[i] and RCHBMA_column[i] > RC_column[i]:
            colors.append('royalblue')
            improved += 1
            improved_list.append(i)
        elif RCHBMA_column[i] < BMA_column[i] and RCHBMA_column[i] < RC_column[i]:
            colors.append('crimson')
            worsened += 1
            worsened_list.append(i)
        else:
            colors.append('silver')
            intermediate += 1
            intermediate_list.append(i)

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot each row as a line plot
    for i in range(135):
        ax.plot(df2.columns, df2.iloc[i],color=colors[i], linewidth = 0.9)

    ax.plot([], [], color='blue', label=f'Improved {improved}')
    ax.plot([], [], color='gray', label=f'Intermediate {intermediate}')
    ax.plot([], [], color='red', label=f'Worsened {worsened}')

    ax.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/fig/box/{file_name}_line_plt.png')

    print(worsened_list)

    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))

    improved_x = BMA_column[improved_list]
    improved_y = RC_column[improved_list]
    worsened_x = BMA_column[worsened_list]
    worsened_y = RC_column[worsened_list]
    intermediate_x = BMA_column[intermediate_list]
    intermediate_y = RC_column[intermediate_list]

    # Plot scatter with color coding
    scatter = ax.scatter(BMA_column, RC_column, c=colors)

    # Set axis limits to be the same
    min_val = min(BMA_column.min(), RC_column.min())-0.05
    max_val = max(BMA_column.max(), RC_column.max())+0.05
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add trend lines
    add_trend_line(ax, improved_x, improved_y, 'royalblue')
    add_trend_line(ax, worsened_x, worsened_y, 'crimson')
    add_trend_line(ax, intermediate_x, intermediate_y, 'darkgray')



    # Set axis limits to be the same
    min_val = min(BMA_column.min(), RC_column.min())-0.05
    max_val = max(BMA_column.max(), RC_column.max())+0.05
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add legend
    improved_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=10, label=f'Improved {improved}')
    intermediate_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray', markersize=10, label=f'Intermediate {intermediate}')
    worsened_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=10, label=f'Worsened {worsened}')

    ax.legend(handles=[improved_patch,intermediate_patch ,worsened_patch ],fontsize = 13)
    ax.tick_params(labelsize = 13)

    # Labels and title
    ax.set_xlabel('BMA',fontsize = 15)
    ax.set_ylabel('RC',fontsize = 15)

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/fig/box/{file_name}_scatter_plt.png')


