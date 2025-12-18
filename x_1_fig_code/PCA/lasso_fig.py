import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os

n_components = 2

fs = 20


#loc, file_path = "JP", "/data0/funato/0_out/99_out/JP/BC-PCA-lasso_hp/spin1/reservoir200/BC-PCA-lasso_r200_s1_sr0.4_rr1.0_is0.5_a0.1/PCA" #JP
loc, file_path = "US",  "/data0/funato/0_out/99_out/US/BC-PCA-lasso_hp/non_arid/spin1/reservoir200/BC-PCA-lasso_r200_s1_sr0.4_rr1.0_is0.5_a0.1/PCA" #US
#loc, file_path = "GB", "/data0/funato/0_out/99_out/GB/BC-PCA-lasso_hp/spin1/reservoir300/BC-PCA-lasso_r300_s1_sr0.4_rr1.0_is0.5_a0.1/PCA" #GB

output_fig_dir = f'/data0/funato/0_out/0_fig/{loc}/PCA/lasso_coefficients'
os.makedirs(output_fig_dir, exist_ok=True)  

for i in range(1, n_components+1):
    input_file = pd.read_csv(f'{file_path}/lasso_coefficients_PC{i}.csv')
    if 'area_fraction_used_for_aggregation' in input_file['Attributes'].values:
        input_file = input_file[input_file['Attributes'] != 'area_fraction_used_for_aggregation']
    sorted_attributes = input_file['Attributes']
    sorted_coefs = input_file['Coefficients']

    # Plot the coefficients
    plt.figure(figsize=(13, 6))
    plt.bar(sorted_attributes, sorted_coefs, color='royalblue')
    plt.xlabel('Attributes', fontsize=fs)
    plt.ylabel('Lasso Coefficients', fontsize=fs)
    plt.title(f'Lasso Coefficients for PC{i}', fontsize=fs)
    plt.xticks(rotation=90, fontsize=fs * 0.8)    
    plt.yticks(fontsize=fs * 0.7)

    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_fig_dir}/lasso_coefficients_PC{i}.png')
    plt.close()

