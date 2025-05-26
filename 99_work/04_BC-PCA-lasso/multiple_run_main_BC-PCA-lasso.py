# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
import matplotlib
matplotlib.use('Agg')
from esn_BC_PCA_lasso import ESN
from run_BC_PCA_lasso import weight_vector, run_BC_pre_PCA, PCA_lasso, run_BC_post_PCA, plot_og_sim_by_weight, plot_scree_plot, hovmoller_weights, plt_rmse_mae
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
loc, ver = "US", 2

#reservoir_size_list = [5,10,15,20,25,30,50,60,70,80,90,100,150,200,250,300,350,400,500,700]
reservoir_size_list = [30,50,70,100,150,200,250,300,400,500,700] #5,10,15,20,25,
reservoir_size_list = [5,10,50,100,200,300,400,500,700]

"""if os.path.exists(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/BC-PCA-lasso_explained_variance_ratio.csv'):
    explained_variance_ratio_list = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/BC-PCA-lasso_explained_variance_ratio.csv').values.tolist()
else:
    explained_variance_ratio_list = []"""

explained_variance_csv = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/BC-PCA-lasso_explained_variance_ratio.csv'

for reservoir_size in reservoir_size_list:
    print(f"Running with reservoir_size = {reservoir_size}")
    
    #####################
    n_components = 10
    alpha = 0.1
    fs = 15
    fs_hov = 25
    benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]

    spin = 1

    nexttime = True

    buf = ""

    input_size = 3
    output_size = 1
    reservoir_size = reservoir_size
    spectral_radius = 0.4
    washout = 0
    ridge_param = 0.001 

    file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
    #####################
    print(file_tag)

    ver_name = "ver1_1" if ver == 1 else "ver2_0" 
    attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

    if loc == "JP" and ver == 1:
        file_tot_num  = 135
        attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
        columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
    elif loc == "JP" and ver == 2:
        file_tot_num = 87
        attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
        columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
    elif loc == "US" and ver == 1:
        file_tot_num = 669
        attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
        columns_drop = ['File_num','gauge_id','gauge_name','country']
    elif loc == "US" and ver == 2:
        file_tot_num = 667
        attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
        columns_drop = ['File_num','gauge_id','gauge_name','country']
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

    start_date_cal = '1993-01-01'
    end_date_cal = '2000-12-31'
    start_date_eva = '2001-01-01'
    end_date_eva = '2008-12-31'

    output_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/BC-PCA-lasso_{reservoir_size}'
    output_fig_dir = f'/data0/funato/0_out/0_fig/{loc}/PCA/BC-PCA-lasso/BC-PCA-lasso_{reservoir_size}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/results', exist_ok=True)
    os.makedirs(output_dir + '/predict', exist_ok=True)
    os.makedirs(output_dir + '/W_out', exist_ok=True)
    os.makedirs(output_dir + '/reservoir', exist_ok=True)
    os.makedirs(output_dir + '/PCA', exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_fig_dir + '/weights', exist_ok=True)
    os.makedirs(output_fig_dir + '/hovmoller', exist_ok=True)

    file_list = list(range(1,file_tot_num+1))

    model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
                "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
                "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
                "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
                "m42", "m43", "m44", "m46"]

    # weights
    bma_weights_og_df = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/weights/BMA_weights.csv', index_col=0)

    # loading the sim flow data for the BMA and 44 MARRMoT models
    bma_df_cal_og = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_cal.csv', index_col=0)
    bma_df_eva_og = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_eva.csv', index_col=0)

    attribute_values = attribute_values[attribute_values['File_num'].isin(file_list)]
    attribute_values = attribute_values.drop(columns=columns_drop)  # Drop non-numeric columns
    attribute_values = attribute_values.apply(pd.to_numeric, errors='coerce')

    #normalize
    scaler = StandardScaler()
    attribute_values_scaled = scaler.fit_transform(attribute_values)
    attribute_values_scaled = pd.DataFrame(attribute_values_scaled, columns= attribute_values.columns)

        
    if os.path.exists(output_dir + f'/BC-PCA-lasso{file_tag}_log.txt'):
        open(output_dir + f'/BC-PCA-lasso{file_tag}_log.txt', 'w').close()
    log_file = open(output_dir + f'/BC-PCA-lasso{file_tag}_log.txt', 'a')

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
                                                                bma_df_cal_og, 
                                                                bma_df_eva_og, 
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pc_values = pca.fit_transform(X_scaled) # primary components (PC1~3) of each basins (135, 3)
    pc_values = pd.DataFrame(pc_values, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    pc_values.index = [f'{i}' for i in file_list]
    pc_values.index.name = 'file_num'

    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_entry = np.concatenate(([f'R_{reservoir_size}'], explained_variance_ratio.flatten()))
    with open(explained_variance_csv, mode = 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(explained_variance_entry)

    ### Scree plot ###
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

    ### BC-PCA-lasso ###
    for PC_n in range(1, min(n_components,5) + 1): #4
        log_file.write(f"BC-PCA-lasso{file_tag} {PC_n}\n")
        log_file.flush()
        print(f"PC_n {PC_n}")

        ### reverting PCA ###
        X_pca = predicted_pcs_df.values[:, :PC_n] # (file_num, n components)
        eigenvector_n = pca.components_[:PC_n] # (n_components, 744)

        X_orig = np.dot(X_pca, eigenvector_n)
        X_orig_backscaled = scaler.inverse_transform(X_orig)
        weights_inverted = pd.DataFrame(X_orig_backscaled, index=[f'file_{file_num}' for file_num in range(1, len(file_list) + 1)])

        plot_og_sim_by_weight(model_list, 
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
                        output_fig_dir)

        # Ensure X and X_orig_backscaled are numeric
        X = np.array(X, dtype=float)
        X_orig_backscaled = np.array(X_orig_backscaled, dtype=float)

        # calculate the rmse and mae
        RMSE = np.sqrt(np.mean((X - X_orig_backscaled)**2))
        MAE = np.mean(np.abs(X - X_orig_backscaled))

        print(f"PCA n={PC_n}")
        print(f"RMSE: {RMSE}")
        print(f"MAE: {MAE}")

        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

        result_eva_rev, reservoir_rev_rows = run_BC_post_PCA(model, 
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
                                                            varssim_dir)

        df_results_eva_rev = pd.DataFrame(result_eva_rev)
        df_results_eva_rev.to_csv(output_dir + f'/results/BC-PCA-lasso_results{file_tag}_rev_PC{PC_n}_eva.csv', index=False)

        print(f"BC-PCA-lasso{file_tag} {PC_n} DONE")

    #plt_rmse_mae(RMSE_list, MAE_list, n_components,fs, output_fig_dir)

    # save RMSE and MAE to csv
    RMSE_MAE_df = pd.DataFrame({'RMSE': RMSE_list, 'MAE': MAE_list})
    RMSE_MAE_df.to_csv(output_dir + f'/PCA/BC-PCA-lasso_results{file_tag}_rev_RMSE_MAE.csv', index=False)
    
    log_file.write(f"DONE\n")
    log_file.close()
    print("DONE")

#explained_variance_ratio_df = pd.DataFrame(explained_variance_ratio_list, columns=['Reservoir_Size'] + [f'PC{i}' for i in range(1, 11)])
#os.makedirs(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso', exist_ok=True)

#explained_variance_ratio_df.to_csv(f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso/BC-PCA-lasso_explained_variance_ratio.csv', index=False)
print("DONE")
