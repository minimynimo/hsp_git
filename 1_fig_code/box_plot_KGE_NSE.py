import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#######
ticks = False
lineplot = False
#lineplot = False

colors = None
#######


# Load the CSV file
file_path = f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/RC_hybrid_bma'
file_name = "RCH_BMA_results_r700_s1_sr0.4_rr0.001_nofix_star" ##REMOVE _KGE AND _NSE
data_KGE= pd.read_csv(f'{file_path}/{file_name}_KGE.csv')
df_KGE = data_KGE.iloc[:, 1:]
data_NSE= pd.read_csv(f'{file_path}/{file_name}_NSE.csv')
df_NSE = data_NSE.iloc[:, 1:]

# Combine the first columns of both dataframes and then the second columns
df = pd.concat([df_KGE.iloc[:, 0], df_KGE.iloc[:, 1], df_NSE.iloc[:, 0], df_NSE.iloc[:, 1]], axis=1)
df.columns = ['KGE_cal','KGE_eva', 'NSE_cal', 'NSE_eva']


#df['m20'] = df['m20'].fillna(method='ffill') 

# Verify the shape of the DataFrame
print(f"DataFrame shape: {df_KGE.shape}")

# Create box plots for each column
plt.figure(figsize=(8,13))

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


# Customize the plot
plt.ylabel('KGE and NSE', fontsize=15)
plt.xticks(ticks=np.arange(1, len(df.columns) + 1), labels=df.columns, rotation=90, fontsize=12)  # Add column labels and rotate them if they overlap

plt.tick_params(labelsize=15)

if ticks == True:
    #yticks = np.arange(-0.25, 1.125, 0.125)   # Create more y-ticks
    yticks = np.arange(-0.375, 1.125, 0.125)   # Create more y-ticks
    plt.yticks(yticks, fontsize=14)


plt.grid(True, linestyle='--', linewidth=0.25)
plt.ylim(0,1)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(f'/Users/funato/Library/CloudStorage/OneDrive-TheUniversityofTokyo/SawadaLab/model/out/fig/box/{file_name}_bnmk_box_plt.png')

plt.close()

