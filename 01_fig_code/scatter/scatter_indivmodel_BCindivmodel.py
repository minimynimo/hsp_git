import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths and model list
individual_model_dir = '/data0/funato/0_out/indiv_model'
bc_model_dir = '/data0/funato/0_out/BC_modelbased/results'
output_dir = '/data0/funato/0_out/0_fig/scatter'
file_tag = 'r700_s1_sr0.4_rr0.001'
cal_eva = ['cal', 'eva']
model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m41", "m42", "m43", "m44", "m46"]

# Initialize lists to store medians
individual_model_medians = []
bc_model_medians = []

benchmark_list = ["KGE","NSE","logNSE","E1","Erel","VE", "d","RMSE","MAE"]

for benchmark in benchmark_list:
    # Create a figure with 2 subplots: one for cal and one for eva
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    for idx, ce in enumerate(cal_eva):
        # Calculate medians for individual models
        individual_model_medians = []
        bc_model_medians = []

        for model_name in model_list:
            indiv_model_file_path = os.path.join(individual_model_dir, f'indiv_model_{model_name}_results_{ce}.csv')
            indiv_model_df = pd.read_csv(indiv_model_file_path)
            column_name = f'{model_name}_{benchmark}_{ce}'
            if column_name in indiv_model_df.columns:
                individual_model_medians.append(indiv_model_df[column_name].median())

        # Calculate medians for BC using individual models
        for model_name in model_list:
            bc_model_file_path = os.path.join(bc_model_dir, f'BC_results_{file_tag}_{model_name}_{ce}.csv')
            bc_model_df = pd.read_csv(bc_model_file_path)
            column_name = f'BC_{file_tag}_{model_name}_{benchmark}_{ce}'
            if column_name in bc_model_df.columns:
                bc_model_medians.append(bc_model_df[column_name].median())


        ax = axes[idx]
        # Add diagonal line representing y = x
        min_val = min(min(individual_model_medians), min(bc_model_medians))
        max_val = max(max(individual_model_medians), max(bc_model_medians))
        ax.plot([min_val, max_val], [min_val, max_val], '-', color='gray', lw=0.7)

        # Plot scatter plot
        ax.scatter(individual_model_medians, bc_model_medians, color='royalblue')
        ax.set_xlabel(f'Individual Models {benchmark}')
        ax.set_ylabel(f'BC Using Individual Models {benchmark}')
        ax.set_title(f'{benchmark} Median Comparison ({ce})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_tag}_{benchmark}_median_comparison_v1.png'))
    plt.close(fig)

print('Done!')