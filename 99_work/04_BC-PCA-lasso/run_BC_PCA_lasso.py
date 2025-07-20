# BC-PCA model 
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import SymLogNorm
from matplotlib.colors import Normalize

def run_BC_pre_PCA(model, file_list, loc, output_dir, bma_cal_og, bma_eva_og, start_date_cal, end_date_cal, start_date_eva, end_date_eva, benchmark_list, file_tag, log_file, washout, ridge_param, spin, nexttime, varssim_dir):
    result_cal_og = []
    result_eva_og = []
    W_out_og_rows = []

    for file_num in file_list:
        print(f'file {file_num}')
        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"BC-PCA og FILE_{file_num}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        ### INPUT DATA ###
        df_og = load_data(file_num, loc, start_date_cal, end_date_eva,varssim_dir, data_name = "varssim_nocal_camels")

        df_cal_og = df_og[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva_og = df_og[start_date_eva:end_date_eva].copy()

        df_cal_og['Sim flow'] = bma_cal_og.loc[f'file_{file_num}_cal'].values
        df_eva_og['Sim flow'] = bma_eva_og.loc[f'file_{file_num}_eva'].values

        # Calculate Flow Error
        df_cal_og['Flow Error'] = df_cal_og['Obs flow'] - df_cal_og['Sim flow']
        df_eva_og['Flow Error'] = df_eva_og['Obs flow'] - df_eva_og['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_cal_og = np.hstack([
            df_cal_og['Precip'].values.reshape(-1, 1),
            df_cal_og['Temp'].values.reshape(-1, 1),
            df_cal_og['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
        input_eva_og = np.hstack([
            df_eva_og['Precip'].values.reshape(-1, 1),
            df_eva_og['Temp'].values.reshape(-1, 1),
            df_eva_og['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        obs_error_cal_og = df_cal_og['Flow Error'].values.reshape(-1, 1)
        obs_error_eva_og = df_eva_og['Flow Error'].values.reshape(-1, 1)

        # target error
        error_target_cal_og = np.concatenate([obs_error_cal_og[1:], obs_error_eva_og[0:1]]) #2898
        error_target_eva = obs_error_eva_og[1:]  #1089

        # Target flow
        target_cal_og = np.concatenate((df_cal_og['Obs flow'].values[1:], df_eva_og['Obs flow'].values[0:1])) #2898
        target_eva_og = np.concatenate((df_eva_og['Obs flow'].values[1:],)) #1089

        input_eva_og = input_eva_og[:, :-1]

        #############################
        # CALIBRATION
        W_out_og, reservoir_og = model.train(input_cal_og, target_data=error_target_cal_og, washout=washout, ridge_param=ridge_param, spinup=spin)
        W_out_og_row = np.concatenate(([f'file_{file_num}'], W_out_og.flatten()))
        W_out_og_rows.append(W_out_og_row)
        
        ### PREDICTIONS OF ERRORS ###
        error_cal_og = model.predict(reservoir_og, input_cal_og, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_eva_og = model.predict(reservoir_og, input_eva_og, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        predict_cal_og = error_cal_og + np.concatenate([df_cal_og['Sim flow'].values[1:], df_eva_og['Sim flow'].values[0:1]])
        predict_eva_og = error_eva_og + df_eva_og['Sim flow'].values[1:]

        predict_cal_og[predict_cal_og < 0] = 0
        predict_eva_og[predict_eva_og < 0] = 0

        #print(predict_cal_og)
        #print(predict_eva_og)
        #print(BMK(target_cal_og.T, predict_cal_og, "KGE"), BMK(target_eva_og.T, predict_eva_og, "KGE"))

        file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal_og.flatten())
        file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva_og.flatten())

        if file_num == 1:
            predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal_og[0]))
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva_og[0]))

            date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

            predict_cal_df = pd.DataFrame([date_row_cal])
            predict_eva_df = pd.DataFrame([date_row_eva])

            predict_cal_df.to_csv(output_dir + f"/predict/BC_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
            predict_eva_df.to_csv(output_dir + f"/predict/BC_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/predict/BC_predict{file_tag}_cal.csv", 'a') as file:
            pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/predict/BC_predict{file_tag}_eva.csv", 'a') as file:
            pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)


        file_result_cal_og = {'file_num': file_num}
        file_result_eva_og = {'file_num': file_num}

        for benchmark in benchmark_list:
            file_result_cal_og.update({
                f'BC_og{file_tag}_{benchmark}_cal': BMK(target_cal_og, predict_cal_og, benchmark)
            })
            file_result_eva_og.update({
                f'BC-og{file_tag}_{benchmark}_eva': BMK(target_eva_og, predict_eva_og, benchmark)
            })  

        result_cal_og.append(file_result_cal_og)
        result_eva_og.append(file_result_eva_og)

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")

        del W_out_og, reservoir_og, predict_eva_og, predict_cal_og

    return result_cal_og, result_eva_og, W_out_og_rows

def PCA_lasso(pc_values, attribute_values, n_components, alpha, fs, output_dir, output_fig_dir):
    # Handle non-numeric values
    attribute_values = attribute_values.apply(pd.to_numeric, errors='coerce')
    attribute_values = attribute_values.fillna(attribute_values.mean())

    lasso_models = {}
    predicted_pcs = {}
    for pc_n in range(1, n_components+1):  # Skip the 'file_num' column
        y = pc_values[f'PC{pc_n}']
        lasso = Lasso(alpha=alpha, max_iter=1000000)  # Increase max_iter to address convergence warning
        lasso.fit(attribute_values, y)
        lasso_models[f'PC{pc_n}'] = lasso

        # Get non-zero coefficients and their corresponding attribute names
        non_zero_coefs = lasso.coef_[lasso.coef_ != 0]
        non_zero_attributes = attribute_values.columns[lasso.coef_ != 0]

        # Sort coefficients and attributes by absolute value of coefficients
        sorted_indices = np.argsort(np.abs(non_zero_coefs))[::-1]
        sorted_coefs = non_zero_coefs[sorted_indices]
        sorted_attributes = non_zero_attributes[sorted_indices]

        # save to csv
        lasso_coefs = pd.DataFrame({'Attributes': sorted_attributes, 'Coefficients': sorted_coefs})
        lasso_coefs.to_csv(f'{output_dir}/lasso_coefficients_PC{pc_n}.csv', index=False)

        # Plot the coefficients
        plt.figure(figsize=(20, 6))
        plt.bar(sorted_attributes, sorted_coefs, color = 'royalblue')
        plt.xlabel('Attributes', fontsize=fs)
        plt.ylabel('Lasso Coefficients', fontsize=fs)
        plt.title(f'Lasso Coefficients for PC{pc_n}', fontsize=fs)
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_fig_dir}/lasso_coefficients_PC{pc_n}.png')
        plt.close()

        # Plot the actual vs predicted PC values
        predicted_pc = lasso_models[f'PC{pc_n}'].predict(attribute_values)
        predicted_pcs[f'PC{pc_n}'] = predicted_pc
        plt.figure(figsize=(7, 7))
        plt.scatter(y, predicted_pc, label = 'predicted vs actual', color='royalblue')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='gray', linestyle = '--', label = 'y=x')
        plt.xlabel('Actual PC values', fontsize=fs)
        plt.ylabel('Predicted PC values', fontsize=fs)
        plt.title(f'Actual vs Predicted PC{pc_n}', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()

        plt.savefig(f'{output_fig_dir}/actual_vs_predicted_PC{pc_n}.png')

        plt.close()

    return lasso_models, predicted_pcs

def run_BC_post_PCA(model, file_list, loc, output_dir, PC_n, weights_inverted, model_list, start_date_eva, end_date_eva, benchmark_list, nexttime, file_tag, log_file, varssim_dir,data_name, predict_col = ""):
    result_eva_rev = []
    reservoir_rows = []
    
    for file_num in file_list: # 136
        print(f'    file{file_num}')
        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"PC_{PC_n} FILE_{file_num}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        BMA_weights_rev = weights_inverted.loc[f'file_{file_num}'].values[:len(model_list)] ### NOTE THAT THE SUM OF THE WEIGHTS WILL NOT NECCESSARILY BE 1 IN THIS CASE
        # scale the BMA_weights to sum to 1
        BMA_weights_rev = BMA_weights_rev / np.sum(BMA_weights_rev)
        RC_weights_rev = weights_inverted.loc[f'file_{file_num}'].values[len(model_list):]  ### CHANGE TO Wout instead of reservoir last row

        ### INPUT DATA ###
        df_rev = load_data(file_num, loc, start_date_eva, end_date_eva, varssim_dir, data_name=data_name)
        df_eva_rev = df_rev[start_date_eva:end_date_eva].copy()

        # loading the sim flow data for the BMA and 44 MARRMoT models
        # Load BMA predictions
        bma_df_eva_rev = BMA(df_eva_rev, BMA_weights_rev, model_list)
        df_eva_rev['Sim flow'] = bma_df_eva_rev

        # Calculate Flow Error
        df_eva_rev['Flow Error'] = df_eva_rev['Obs flow'] - df_eva_rev['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_eva_rev = np.hstack([
            df_eva_rev['Precip'].values.reshape(-1, 1),
            df_eva_rev['Temp'].values.reshape(-1, 1),
            df_eva_rev['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        obs_error_eva_rev = df_eva_rev['Flow Error'].values.reshape(-1, 1)

        # Target flow
        target_eva_rev = np.concatenate((df_eva_rev['Obs flow'].values[1:],)) #1089

        input_eva_rev = input_eva_rev[:, :-1]

        ##########
        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
        ### PREDICTIONS OF ERRORS ###
        error_eva_rev, reservoir_last = model.predict_PCA(RC_weights_rev, input_eva_rev, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        reservoir_rows.append(np.concatenate(([f'file_{file_num}'], reservoir_last.flatten())))

        # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
        # predict = error_cal + BMAsim
        # predict のtimestep = t+1
        predict_eva_rev = error_eva_rev + df_eva_rev['Sim flow'].values[1:]
        predict_eva_rev[predict_eva_rev < 0] = 0

        #print(predict_eva_rev)
        #print(BMK(target_eva_rev.T, predict_eva_rev,"KGE"))

        file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva_rev.flatten())
        if file_num == 1:
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva_rev[0]))
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])
            predict_eva_df = pd.DataFrame([date_row_eva])
            predict_eva_df.to_csv(output_dir + f"/predict/BCrg_predict{file_tag}_PC{PC_n}_{predict_col}eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/predict/BCrg_predict{file_tag}_PC{PC_n}_{predict_col}eva.csv", 'a') as file:
            pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

        file_results_eva_rev = {'file_num': file_num}

        # Loop over each benchmark and add results to the current file's dictionary
        for benchmark in benchmark_list:
            file_results_eva_rev.update({
                f'BCrg{file_tag}_{benchmark}_eva': BMK(target_eva_rev, predict_eva_rev, benchmark)
            })  

        # Append the completed dictionary to the results list
        result_eva_rev.append(file_results_eva_rev)

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")
        log_file.flush()

        del predict_eva_rev

    return result_eva_rev, reservoir_rows

def variable_names_list(model_list, reservoir_size):
    variable_names = [f'BMA_{model}' for model in model_list] + [f'RC_{i}' for i in range(1, reservoir_size+1)]
    variable_names = np.array(variable_names)
    return variable_names

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, loc, start_date, end_date, varssim_dir, data_name, ver_name="ver2_0"):
    df = pd.read_csv(f"{varssim_dir}/{data_name}/{ver_name}/varssim{file_name(file_num, 3)}.csv")

    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date, end=end_date)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df


def BMK(obs_data, sim_data, benchmark):    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "logNSE":
        temp_obs_data = np.where((obs_data <= 0) | np.isnan(obs_data), 1e-6, obs_data)
        temp_sim_data = np.where((sim_data <= 0) | np.isnan(sim_data), 1e-6, sim_data)
        obs_ave = np.mean(temp_obs_data)
        numer = np.sum((np.log(temp_sim_data) - np.log(temp_obs_data))**2)
        denom = np.sum((np.log(temp_obs_data) - np.log(obs_ave))**2)
        return 1 - numer / denom
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "Erel":
        obs_ave = np.mean(obs_data)
        temp_obs_data = np.where(obs_data == 0, 1e-6, obs_data)
        numer = np.sum(np.square((temp_obs_data - sim_data) / temp_obs_data))
        denom = np.sum(np.square((obs_data - obs_ave) / obs_ave))
        return 1 - numer / denom
    
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

def weight_vector(bma_weights_df, RC_weights_df, reservoir_size, model_list, file_list):
    variable_names = variable_names_list(model_list, reservoir_size)

    weight_vectors =[]
    for file_num in file_list:
        bma_weights = bma_weights_df.loc[f'file_{file_num}'].values
        #print("bma_weights", bma_weights.shape)
        rc_reservoir_values = RC_weights_df.loc[f'file_{file_num}'].values
        #print("rc_reservoir_values", rc_reservoir_values.shape)
        vector = np.concatenate((bma_weights, rc_reservoir_values))
        weight_vectors.append(vector)

    weight_vectors = pd.DataFrame(weight_vectors) ## 135 basins as rows, 744 variables as columns
    weight_vectors.columns = variable_names
    return weight_vectors

def BMA(model_sim, BMA_weights, model_list):
    ## model_sim includes the sim flow data for the BMA and 44 MARRMoT models with the same date index
    bma_df_eva = np.zeros(len(model_sim))  
    for date in model_sim.index:
        BMA_day_prediction = 0
        for i, model_num in enumerate(model_list):
            model_eva = model_sim.at[date, model_num]
            BMA_day_prediction += BMA_weights[i] * model_eva
        bma_df_eva[model_sim.index.get_loc(date)] = BMA_day_prediction
    return bma_df_eva

def plot_og_sim_by_weight(model_list, reservoir_size, X, X_orig_backscaled, n, output_fig_dir, fs):
    variable_names = variable_names_list(model_list, reservoir_size)

    # Ensure X and X_orig_backscaled are numeric
    X = np.array(X, dtype=float)
    X_orig_backscaled = np.array(X_orig_backscaled, dtype=float)

    ylim_val = np.ceil(max(np.abs(X[0].max()), np.abs(X_orig_backscaled[0].max()))*1.05)

    # plot graph of original data and reverted data BY WEIGHT
    plt.figure(figsize=(10, 8))
    x = np.arange(len(variable_names))
    plt.plot(x, X[0], label='Original Data', marker='o', linestyle='-', markersize=5, color='#6b8fd3')
    plt.plot(x, X_orig_backscaled[0], label='Reverted Data', marker='x', linestyle='--', markersize=5, color='#ff5a4d')
    if len(variable_names) > 100:
        #plt.xticks(np.arange(0, len(variable_names), 50))
        plt.xticks(np.arange(0, len(variable_names), 50), variable_names[::50], rotation=90, fontsize=fs)
    else:
        plt.xticks(np.arange(0, len(variable_names)), variable_names, rotation=90, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Variable', fontsize=fs)
    plt.ylabel('Weight', fontsize=fs)
    #plt.ylim(-10,10)
    plt.ylim(-1 * ylim_val, ylim_val)
    plt.title(f"PCA n={n}\n",fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'weights', f'PCA_weights_n{n}.jpg'))
    plt.close()

def plot_scree_plot(explained_variance_ratio, scree_n_components, fs, output_fig_dir):
    explained_variance_n_components = explained_variance_ratio[:scree_n_components]
    cumulative_variance_n_components = np.cumsum(explained_variance_n_components)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 8)
    ax1.grid(True)

    print("range(1, scree_n_components + 1)") # 1to 5
    print(f'1 to  {str(scree_n_components + 1)}')
    print("explained_variance_n_components", explained_variance_n_components) #5

    ax1.bar(range(1, scree_n_components + 1), explained_variance_n_components*100, label='Individual Explained Variance', color='royalblue')
    ax1.set_xlabel('Principal Components', fontsize=fs)
    ax1.set_ylabel('Variance Explained (%)', fontsize=fs)
    ax1.set_title('Scree Plot with Cumulative Variance (Top 10 Components)\n', fontsize=fs)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(range(0, 101, 10))
    ax1.set_yticklabels(range(0, 101, 10), fontsize=fs)
    ax1.set_xticks(range(1, scree_n_components + 1))
    ax1.set_xticklabels(range(1, scree_n_components + 1), fontsize=fs)

    ax2 = ax1.twinx()
    ax2.plot(range(1, scree_n_components + 1), cumulative_variance_n_components * 100, color='crimson', marker='o', label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=fs)
    ax2.set_ylim(0, 100)
    ax2.set_yticks(range(0, 101, 10))
    ax2.set_yticklabels(range(0, 101, 10), fontsize=fs)
    # ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.grid(which='minor', linestyle=':', linewidth='0.3', color='grey')

    # fig.legend(loc="upper left",  fontsize=fs) #bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes, 
    fig.legend(loc="upper right", fontsize=fs, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f'PCA_scree_plot.jpg'))
    plt.cla()
    plt.close()

    print('PCA Scree Plot saved')
    print('DONE!!')

def hovmoller_weights(X, X_orig_backscaled, n, fs_hov, output_fig_dir):
    X = np.array(X, dtype=float)
    X_orig_backscaled = np.array(X_orig_backscaled, dtype=float)

    # plot HOVMOLLER diagram
    BMA_ylim_val_all = np.ceil(max(np.abs(X[:,:43].max()), np.abs(X_orig_backscaled[:,:43].max())))
    RC_ylim_val_all = np.ceil(max(np.abs(X[:,43:].max()), np.abs(X_orig_backscaled[:,43:].max())))

    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(3, 4, width_ratios=[0.1, 0.9, 0.02, 0.02], height_ratios=[1, 1, 1])

    # Create subplots for each pair
    BMA_ax1 = fig.add_subplot(gs[0, 0])
    RC_ax1 = fig.add_subplot(gs[0, 1], sharey=BMA_ax1)
    BMA_ax2 = fig.add_subplot(gs[1, 0], sharey=BMA_ax1)
    RC_ax2 = fig.add_subplot(gs[1, 1], sharey=BMA_ax1)
    BMA_ax3 = fig.add_subplot(gs[2, 0], sharey=BMA_ax1)
    RC_ax3 = fig.add_subplot(gs[2, 1], sharey=BMA_ax1)

    # Remove y-tick labels for RC plots
    for ax in [RC_ax1, RC_ax2, RC_ax3]:
        ax.tick_params(labelleft=False)

    # Plot the data
    BMA_ax1.imshow(X[:, :43], aspect='auto', cmap='PiYG_r')
    RC_ax1.imshow(X[:, 43:], aspect='auto', cmap='bwr')
    BMA_ax2.imshow(X_orig_backscaled[:, :43], aspect='auto', cmap='PiYG_r')
    RC_ax2.imshow(X_orig_backscaled[:, 43:], aspect='auto', cmap='bwr')
    BMA_ax3.imshow(X[:, :43] - X_orig_backscaled[:, :43], aspect='auto', cmap='PiYG_r')
    RC_ax3.imshow(X[:, 43:] - X_orig_backscaled[:, 43:], aspect='auto', cmap='bwr')

    # Titles for each set of subplots
    BMA_ax1.set_title('BMA', fontsize=fs_hov)
    RC_ax1.set_title('RC', fontsize=fs_hov)
    BMA_ax2.set_title('BMA', fontsize=fs_hov)
    RC_ax2.set_title('RC', fontsize=fs_hov)
    BMA_ax3.set_title('BMA', fontsize=fs_hov)
    RC_ax3.set_title('RC', fontsize=fs_hov)

    # Format BMA axes
    for ax, data, title in zip(
        [BMA_ax1, BMA_ax2, BMA_ax3],
        [X[:, :43], X_orig_backscaled[:, :43], X[:, :43] - X_orig_backscaled[:, :43]],
        ['Original Data', 'Reverted Data', 'Difference']
    ):
        norm = Normalize(vmin=-1 * BMA_ylim_val_all, vmax=BMA_ylim_val_all)
        ax.images[0].set_norm(norm)
        ax.tick_params(axis='x', labelsize=fs_hov)
        ax.set_xlabel('Variables', fontsize=fs_hov)
        if ax == BMA_ax1:
            ax.set_ylabel('Original Data\nBMA', fontsize=fs_hov)
        elif ax == BMA_ax2:
            ax.set_ylabel('Reverted Data\nBMA', fontsize=fs_hov)
        else:
            ax.set_ylabel('Difference\nBMA', fontsize=fs_hov)

    # Format RC axes
    for ax, data in zip([RC_ax1, RC_ax2, RC_ax3], [X[:, 43:], X_orig_backscaled[:, 43:], X[:, 43:] - X_orig_backscaled[:, 43:]]):
        norm = SymLogNorm(linthresh=10.0, linscale=0.8, vmin=-1 * RC_ylim_val_all, vmax=RC_ylim_val_all, base=15)
        ax.images[0].set_norm(norm)
        ax.tick_params(axis='x', labelsize=fs_hov)
        ax.set_xlabel('Variables', fontsize=fs_hov)

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.35, right=0.85)

    # Define colorbar axes using appropriate gridspec indices
    BMA_cbar_ax = fig.add_subplot(gs[:, 2])  # Use all rows for a vertically aligned BMA colorbar
    RC_cbar_ax = fig.add_subplot(gs[:, 3])  # Use all rows for a vertically aligned RC colorbar

    # Add colorbars with consistent scaling
    BMA_cbar = fig.colorbar(BMA_ax1.images[0], cax=BMA_cbar_ax, orientation='vertical')
    BMA_cbar.ax.tick_params(labelsize=fs_hov)
    BMA_cbar.set_label('BMA Weight', fontsize=fs_hov)

    RC_cbar = fig.colorbar(RC_ax1.images[0], cax=RC_cbar_ax, orientation='vertical')
    RC_cbar.ax.tick_params(labelsize=fs_hov)
    RC_cbar.set_label('RC Weight', fontsize=fs_hov)

    # Set the title for the figure
    fig.suptitle(f"PCA n={n}", fontsize=fs_hov)

    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'hovmoller', f'PCA_hovmoller_n{n}.jpg'))
    plt.close()

def plt_rmse_mae(RMSE_list, MAE_list, n_components, fs, output_fig_dir):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, n_components+1), RMSE_list, label='RMSE' , color = 'crimson')
    plt.plot(range(1, n_components+1), MAE_list, label='MAE', color = 'royalblue')
    plt.xlabel('PCs', fontsize=fs)
    plt.xticks(range(1, n_components+1), fontsize=fs)
    #plt.yticks(np.arange(0, max(max(RMSE_list), max(MAE_list)) + 0.025, 0.025), fontsize=fs)
    plt.ylabel('value', fontsize=fs)
    plt.ylim(0, )
    plt.legend(fontsize=fs)
    plt.title('RMSE and MAE', fontsize=fs)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'RMSE_MAE.jpg'))
    plt.close()