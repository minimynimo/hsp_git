# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
# normalized the attribute for PCA
import matplotlib
matplotlib.use('Agg')
from esn_BC_PCA_lasso import ESN
from run_BC_PCA_lasso import weight_vector, run_BC_pre_PCA, PCA_lasso, run_BC_post_PCA, plot_og_sim_by_weight, plot_scree_plot, hovmoller_weights, plt_rmse_mae
import os
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## Flow
# 1. Run BC model with 700 reservoirs
# 2. PCA on the weights of the BMA and RC models
# 3. Use lasso regression to revert PCA weights
# 4. Using the reverted weights, run the BC model again

#####################
n_components = 5
fs = 15
fs_hov = 25

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

nexttime = True
arid_files = True

buf = ""

loc, ver = "GB", 2

if loc != "US":
    arid_files = False

input_size = 3
output_size = 1

# RC
reservoir_size = 700
#reservoir_size = 200
spectral_radius = 0.4
washout = 0
ridge_param = 0.001 
spin = 1

#PCA
alpha = 0.1

file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
file_tag = ""
#####################
print(file_tag)

ver_name = "ver1_1" if ver == 1 else "ver2_0" 

file_tot_num = 424
attribute_values = pd.read_csv(f'/data0/funato/hsp/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
columns_drop = ['File_num','gauge_id']
varssim_dir = f"/data0/funato/hsp/2_MERV/HYPER"


file_list = list(range(1, file_tot_num+1))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'


output_dir = f'/data0/funato/hsp/0_out/BCrg'
output_fig_dir = f'/data0/funato/hsp/0_out/1_fig/BCrg/{loc}/{ver_name}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/W_out', exist_ok=True)
os.makedirs(output_dir + '/reservoir', exist_ok=True)
os.makedirs(output_dir + '/PCA', exist_ok=True)
os.makedirs(output_fig_dir, exist_ok=True)
os.makedirs(output_fig_dir + '/weights', exist_ok=True)
os.makedirs(output_fig_dir + '/hovmoller', exist_ok=True)

# Create a parameter text file in output_dir
param_file_path = os.path.join(output_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write(f"spin = {spin}\n")
    param_file.write("\n===PCA===\n")
    param_file.write(f"alpha = {alpha}\n")


precip_cols = ['no_precip',  'peak1_rate0',
                   'peak1_rate10', 'peak1_rate30', 'peak1_rate50',
                   'peak3_rate10', 'peak3_rate30', 'peak3_rate50',
                   'peak5_rate10', 'peak5_rate30', 'peak5_rate50']


model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

# weights
bma_weights_og_df = pd.read_csv(f'/data0/funato/hsp/0_out/BMA/weights/BMA_weights.csv', index_col=0)

# Load BMA predictions
bma_df_cal = pd.read_csv(f'/data0/funato/hsp/0_out/BMA/predict/BMA_predict_cal.csv', index_col=0)
bma_df_eva = pd.read_csv(f'/data0/funato/hsp/0_out/BMA/predict/BMA_predict_eva.csv', index_col=0)

attribute_values = attribute_values[attribute_values['File_num'].isin(file_list)]
attribute_values = attribute_values.drop(columns=columns_drop)  # Drop non-numeric columns
attribute_values = attribute_values.apply(pd.to_numeric, errors='coerce')

#normalize attributes
attribute_scaler = StandardScaler()
attribute_values_scaled = attribute_scaler.fit_transform(attribute_values)
attribute_values_scaled = pd.DataFrame(attribute_values_scaled, columns= attribute_values.columns)

if os.path.exists(output_dir + f'/BCrg{file_tag}_log.txt'):
    open(output_dir + f'/BCrg{file_tag}_log.txt', 'w').close()
log_file = open(output_dir + f'/BCrg{file_tag}_log.txt', 'a')

# 1. Run BC model with 700 reservoirs
#### BC ####
model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

result_cal_og, result_eva_og, W_out_og_rows = run_BC_pre_PCA(model, 
                                                            file_list, 
                                                            loc,
                                                            output_dir,
                                                            bma_df_cal, 
                                                            bma_df_eva, 
                                                            start_date_cal, 
                                                            end_date_cal, 
                                                            start_date_eva, 
                                                            end_date_eva, 
                                                            benchmark_list, 
                                                            file_tag, 
                                                            log_file, 
                                                            washout, 
                                                            ridge_param, 
                                                            spin, 
                                                            nexttime, 
                                                            varssim_dir)

W_out_og_df = pd.DataFrame(W_out_og_rows)
W_out_og_df.set_index(0, inplace=True)
W_out_og_df.to_csv(output_dir + f'/W_out/BC_W_out{file_tag}_bma.csv', header=False)

df_results_cal_og = pd.DataFrame(result_cal_og)
df_results_cal_og.to_csv(output_dir + f'/results/BC_results{file_tag}_cal.csv', index=False)

df_results_eva_og = pd.DataFrame(result_eva_og)
df_results_eva_og.to_csv(output_dir + f'/results/BC_results{file_tag}_eva.csv', index=False)

print(f"BC DONE")
log_file.write(f"BC DONE\n")

#### PCA ####
log_file.write(f"PCA\n")
log_file.flush()
X = weight_vector(bma_weights_og_df, W_out_og_df, reservoir_size, model_list, file_list).values # (file_num, 744)

# Normalize the weight vectors for PCA
weights_scaler = StandardScaler()
X_scaled = weights_scaler.fit_transform(X)

pca = PCA(n_components=n_components)
pc_values = pca.fit_transform(X_scaled) # primary components (PC1~5) of each basins (424, 5)
pc_values = pd.DataFrame(pc_values, columns=[f'PC{i}' for i in range(1, n_components + 1)])
pc_values.index = [f'{i}' for i in file_list]
pc_values.index.name = 'file_num'

### Scree plot ###
explained_variance_ratio = pca.explained_variance_ratio_
plot_scree_plot(explained_variance_ratio, n_components, fs, output_fig_dir)

#### LASSO ####
log_file.write(f"PCA lasso\n")
log_file.flush()

output_dir_PCA = output_dir + '/PCA'
lasso_model, predicted_pcs = PCA_lasso(pc_values, attribute_values_scaled, n_components, alpha, fs, output_dir_PCA, output_fig_dir)
predicted_pcs_df = pd.DataFrame(predicted_pcs, columns=[f'PC{i}' for i in range(1, n_components + 1)])
predicted_pcs_df.index = [f'{file_num}' for file_num in file_list]
predicted_pcs_df.to_csv(output_dir + f'/PCA/PCA_lasso_predicted_pcs.csv') # (file num, PCs)

RMSE_list = []
MAE_list = []

### BCrg ###
for PC_n in range(1, n_components + 1): #4
    log_file.write(f"BCrg{file_tag} {PC_n}\n")
    log_file.flush()
    print(f"PC_n {PC_n}")

    ### reverting PCA ###
    X_pca = predicted_pcs_df.values[:, :PC_n] # (file_num, n components)
    eigenvector_n = pca.components_[:PC_n] # (n_components, 744)

    X_orig = np.dot(X_pca, eigenvector_n)
    X_orig_backscaled = weights_scaler.inverse_transform(X_orig)
    weights_inverted = pd.DataFrame(X_orig_backscaled, index=[f'file_{file_num}' for file_num in file_list])

    """plot_og_sim_by_weight(model_list, 
                          reservoir_size, 
                          X, 
                          X_orig_backscaled,
                          PC_n,
                          output_fig_dir, 
                          fs)

    hovmoller_weights(X,
                    X_orig_backscaled,
                    PC_n,
                    fs_hov,
                    output_fig_dir)"""

    # Ensure X and X_orig_backscaled are numeric for calculation
    X_numeric = np.array(X, dtype=float)
    X_orig_backscaled_numeric = np.array(X_orig_backscaled, dtype=float)

    # calculate the rmse and mae
    RMSE = np.sqrt(np.mean((X_numeric - X_orig_backscaled_numeric)**2))
    MAE = np.mean(np.abs(X_numeric - X_orig_backscaled_numeric))

    print(f"PCA n={PC_n}")
    print(f"RMSE: {RMSE}")
    print(f"MAE: {MAE}")

    RMSE_list.append(RMSE)
    MAE_list.append(MAE)

    result_eva_rev, reservoir_rows = run_BC_post_PCA(model, 
                                                        file_list, 
                                                        loc,
                                                        output_dir,
                                                        PC_n, 
                                                        weights_inverted, 
                                                        model_list, 
                                                        start_date_eva, 
                                                        end_date_eva, 
                                                        benchmark_list, 
                                                        nexttime, 
                                                        file_tag, 
                                                        log_file, 
                                                        varssim_dir,
                                                        data_name="varssim_nocal_camels")

    df_results_eva_rev = pd.DataFrame(result_eva_rev)
    df_results_eva_rev.to_csv(output_dir + f'/results/BCrg_results{file_tag}_PC{PC_n}_eva.csv', index=False)

    print(f"BCrg{file_tag} {PC_n} DONE")

    #also predict for the synthetic data
    if PC_n == 3:  # Only predict for the last PC_n
        for precip_col in precip_cols:
            print(f"Predicting for {precip_col} with PC_n={PC_n}")
            result_predict, predict_rows = run_BC_post_PCA(model, 
                                                            file_list, 
                                                            loc,
                                                            output_dir,
                                                            PC_n, 
                                                            weights_inverted, 
                                                            model_list, 
                                                            start_date_eva, 
                                                            end_date_eva, 
                                                            benchmark_list, 
                                                            nexttime, 
                                                            file_tag, 
                                                            log_file, 
                                                            varssim_dir,
                                                            data_name=f"varssim_nocal_{precip_col}",
                                                            predict_col=f"{precip_col}_")

            #df_results_predict = pd.DataFrame(result_predict)
            #df_results_predict.to_csv(output_dir + f'/predict/BCrg_predict_{precip_col}{file_tag}_PC{PC_n}_eva.csv', index=False)

plt_rmse_mae(RMSE_list, MAE_list, n_components,fs, output_fig_dir)



log_file.write(f"DONE\n")
log_file.close()
print("DONE")