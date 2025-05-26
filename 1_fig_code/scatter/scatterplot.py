import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter 
from matplotlib import cm, ticker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

#######
#benchmark =  'NSE'
benchmark = 'KGE'
file1 = 'RCHBMA_fix_nofix'
zerockfile = 'RCH_BMA_predict_r700_s1_sr0.4_rr0.001_nofix_star'

#ReservoirSize = 300 
ReservoirSize = 700 
#######


# Load the CSV file
file_path = f'/data0/funato/0_out/csv/{file1}.csv'
zerock_file_path = f'/data0/funato/0_out/BMA/{zerockfile}.csv'

data= pd.read_csv(file_path)
zero_ck_df= pd.read_csv(zerock_file_path)

df = data.iloc[:, 1:]

# Verify the shape of the DataFrame
print(f"DataFrame shape: {df.shape}")


subzero = []
for file in range(1, 136):
    if min(zero_ck_df[f'file_{file}']) < 0:
        subzero.append((zero_ck_df[f'file_{file}'] < 0).sum())
    else:
        subzero.append(0)

# Mask subzero values that are exactly 0 and replace with a small value for coloring
subzero_cp = np.array(subzero)
subzero_cp = np.where(subzero_cp == 0, 1e-6, subzero_cp)  # For 0, use black
min_positive_value = np.min(subzero_cp[subzero_cp > 1e-6])  # Find the smallest positive value
subzero_cp = np.where(subzero_cp == 1e-6, min_positive_value, subzero_cp)  # Replace 1e-6 with the smallest positive value

# Create a gradient colormap from blue to red
cmap = colors.LinearSegmentedColormap.from_list('blue_red', ['black', 'crimson', 'orange'])
norm = colors.LogNorm(vmin=min_positive_value, vmax=np.max(subzero_cp))

# Plotting
plt.figure(figsize=(6, 6))  # Adjust the figure size as needed

CorCoef = np.corrcoef(df['RCH_BMA_700_rr0.001_nofix'], df['RCH_BMA_700_rr0.001_zerofix'])

plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='silver', linestyle='-', linewidth=1.0)

# Plotting with the custom colormap
sc = plt.scatter(df['RCH_BMA_700_rr0.001_nofix'], df['RCH_BMA_700_rr0.001_zerofix'], c=subzero_cp, cmap=cmap, norm=norm)

xval = df['RCH_BMA_700_rr0.001_nofix']
yval = df['RCH_BMA_700_rr0.001_zerofix']

# Customize the plot
plt.title(f'{file1} \nCorCoef: {round(CorCoef[0,1], 3)}')
plt.xlabel(f'{benchmark} of RCH_BMA_700_rr0.001_nofix')
plt.ylabel(f'{benchmark} of RCH_BMA_700_rr0.001_zerofix')

plt.grid(True, linestyle='--', linewidth=0.25)

xymin = min(xval.min(), yval.min()) - 0.05
xymax = max(xval.max(), yval.max()) + 0.05
plt.xlim(xymin, xymax)
plt.ylim(xymin, xymax)

# Adjust colorbar ticks for better distribution
cbar = plt.colorbar(sc)
cbar.ax.set_yticklabels(['0', 'min', '10', '100', f'{int(np.max(subzero_cp))}'])


# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(f'/data0/funato/0_out/fig/scatter/{file1}_scatter_plt.png')
