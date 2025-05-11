import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

# Define the file tag and file list
file_tag = 'r700_s1_sr0.4_rr0.001_bma'  # Replace with your actual file tag
file_list = [1,2,3,4,7,8,9]

tag = ''
if tag == '':
    tag = '-'.join(map(str, file_list))

# Load the CSV file
file_path = f"/data0/funato/0_out/bias_correction/reservoir_{file_tag}.csv"
df = pd.read_csv(file_path, index_col=0)
print("DataFrame:\n", df)
# Select the rows based on file_list
selected_rows = df.reindex([f'file_{file}' for file in file_list])

# Plot the data
plt.figure(figsize=(10, 6))
for index, row in selected_rows.iterrows():
    plt.plot(range(1, len(row) + 1), row.values, label=index, linewidth = 0.4, marker='o')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph of Selected Rows')
plt.legend()
plt.tight_layout()
plt.savefig(f'/data0/funato/0_out/0_fig/reservoir/resevoir_{tag}.png')