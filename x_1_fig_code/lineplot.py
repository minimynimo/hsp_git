import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#######
benchmark = 'KGE'

ticks = False
#######

# Load the CSV file
file_path = f'/data0/funato/0_out/csv'
data= pd.read_csv(f'{file_path}/lineplot.csv')
df = data.iloc[:, 1:]

# Verify the shape of the DataFrame
print(f"DataFrame shape: {df.shape}")

# Create box plots for each column
plt.figure(figsize=(10,11))


colors = []
labels = []
improved_list = []
worsened_list = []
same_list = []
improved = 0
worsened = 0
same = 0
BMA_column = df['RCH_BMA_700_1_0.4_KGE']
RC_column = df['RC_700_1_0.1_KGE']
for i in range(df.shape[0]):
    if BMA_column[i] > df['BMA'][i] and BMA_column[i] > RC_column[i]:
        colors.append('royalblue')
        improved += 1
        improved_list.append(i)
    elif BMA_column[i] < df['BMA'][i] and BMA_column[i] < RC_column[i]:
        colors.append('crimson')
        worsened += 1
        worsened_list.append(i)
    else:
        colors.append('dimgray')
        same += 1
        same_list.append(i)

fig, ax = plt.subplots(figsize=(6, 8))

# Plot each row as a line plot
for i in range(135):
    ax.plot(df.columns, df.iloc[i],color=colors[i], linewidth = 0.9)

ax.plot([], [], color='blue', label=f'Improved {improved}')
ax.plot([], [], color='gray', label=f'Same {same}')
ax.plot([], [], color='red', label=f'Worsened {worsened}')

ax.legend()

plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(f'/data0/funato/0_out/fig/box/{benchmark}_line_plt_700.png')

print(worsened_list)
print("DONE")