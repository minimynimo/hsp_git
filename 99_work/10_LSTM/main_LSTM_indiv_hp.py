# Imports
from pathlib import Path
from typing import Tuple, List
import argparse

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import os
from datetime import datetime

# Local imports
from run_LSTM import CamelsTXT
from run_LSTM import Model
from run_LSTM import train_epoch
from run_LSTM import eval_model
from run_LSTM import calc_nse
from run_LSTM import BMK

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser(description='LSTM Individual Basin Modeling with Hyperparameter Tuning')
parser.add_argument('--hidden_size', type=int, default=16, help='Number of LSTM cells')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
args = parser.parse_args()

# Set hyperparameters from arguments
hidden_size = args.hidden_size
dropout_rate = args.dropout

# Fixed hyperparameters based on Kratzert et al. (2018, 2019)
num_epochs = 20  # Fixed at 20 for individual basins
learning_rate = 1e-3  # Fixed at 1e-3
window_size = 270  # Input sequence length
batch_size = 256  # Keep as is

#loc, ver = "JP", 2
#loc, ver = "US", 2
loc, ver = "GB", 2
#loc, ver = "AUS", 2

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

# Create hyperparameter string for output directory
hp_str = f"h{hidden_size}_d{dropout_rate}"
file_tag = f"_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}"

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP":
    file_tot_num = 87
elif loc == "US":
    file_tot_num = 667
elif loc == "AUS":
    file_tot_num = 84
elif loc == "GB":
    file_tot_num = 424

if loc == "JP" and ver == 2:
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    varssim_dir = f"/data0/funato/hsp/2_MERV/varssim_HBV"

#file_list = range(1, file_tot_num + 1)
file_list = range(1,file_tot_num + 1, 4)

# Globals
#FILE_SYSTEM = gcsfs.core.GCSFileSystem(requester_pays=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available
print(f"Using device: {DEVICE}")

# Create output directory with hyperparameter settings
output_dir = f"/data0/funato/hsp/0_out/LSTM/{loc}/individual/hp/{hp_str}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)

if os.path.exists(output_dir + f'/LSTM_log.txt'):
    open(output_dir + f'/LSTM_log.txt', 'w').close()
log_file = open(output_dir + f'/LSTM_log.txt', 'a')

####MAIN#####

results_cal = []
results_eva = []

for file_num in file_list:
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")

    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ##############
    # Data set up#
    ##############

    # Training data
    start_date = pd.to_datetime("1993-01-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2000-12-31", format="%Y-%m-%d")
    ds_train = CamelsTXT(file_num, seq_length=window_size, period="train", dates=[start_date, end_date], varssim_dir=varssim_dir, loc =loc)
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # Validation data. We use the feature means/stds of the training period for normalization
    ###EQUIVALENT TO THE TRAINING DATA FOR HYPER MODEL
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    start_date = pd.to_datetime("1993-01-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2000-12-31", format="%Y-%m-%d")
    ds_val = CamelsTXT(file_num, seq_length=window_size, period="eval", dates=[start_date, end_date],
                        means=means, stds=stds, varssim_dir=varssim_dir, loc =loc)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # Test data. We use the feature means/stds of the training period for normalization
    start_date = pd.to_datetime("2001-01-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2006-12-31", format="%Y-%m-%d")
    ds_test = CamelsTXT(file_num, seq_length=window_size, period="eval", dates=[start_date, end_date],
                        means=means, stds=stds, varssim_dir=varssim_dir, loc =loc)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)


    #########################
    # Model, Optimizer, Loss#
    #########################

    # Here we create our model, feel free 
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        train_epoch(model, optimizer, tr_loader, loss_func, epoch+1)
        obs, preds = eval_model(model, tr_loader)
        preds = ds_train.local_rescale(preds.cpu().numpy(), variable='output')

        nse = calc_nse(obs.cpu().numpy(), preds)
        tqdm.tqdm.write(f"Validation NSE: {nse:.2f}")

        epoch_end_time = datetime.now()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {nse:.2f}, Elapsed Time: {epoch_elapsed_time}\n")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {nse:.2f}, Elapsed Time: {epoch_elapsed_time}")
        log_file.flush()

    pred_start_time = datetime.now()
    pred_start_time_st = pred_start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"start: {pred_start_time_st}\n")

    # Evaluate on test set
    cal_obs, predict_cal = eval_model(model, val_loader)
    eva_obs, predict_eva = eval_model(model, test_loader)


    # Rescale predictions
    predict_cal = ds_train.local_rescale(predict_cal.cpu().numpy(), variable='output')
    predict_eva = ds_train.local_rescale(predict_eva.cpu().numpy(), variable='output')
    target_cal = cal_obs.cpu().numpy()
    target_eva = eva_obs.cpu().numpy()

    nse = calc_nse(target_eva, predict_eva)

    #save predictions and validation results
    #save the prediction results as csv

    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0
    start_date_cal = ds_train.dates[0]
    start_date_eva = ds_test.dates[0]
    #print(predict_cal)
    #print(predict_eva)
    print(BMK(target_cal,predict_cal,"KGE"), BMK(target_eva, predict_eva,"KGE"))

    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

    # Save predictions with proper indexing
    file_row_cal = pd.DataFrame([file_row_cal], columns=['Index'] + [f'Value_{i}' for i in range(len(predict_cal.flatten()))])
    file_row_cal.set_index('Index', inplace=True)
    file_row_eva = pd.DataFrame([file_row_eva], columns=['Index'] + [f'Value_{i}' for i in range(len(predict_eva.flatten()))])
    file_row_eva.set_index('Index', inplace=True)

    if file_num == 1:
        # Write headers for the first file
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))

        date_row_cal = pd.DataFrame([['Date'] + [str(date.date()) for date in predict_cal_dates]])
        date_row_eva = pd.DataFrame([['Date'] + [str(date.date()) for date in predict_eva_dates]])

        date_row_cal.to_csv(output_dir + f"/predict/LSTM_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
        date_row_eva.to_csv(output_dir + f"/predict/LSTM_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

    # Append rows with proper indexing
    file_row_cal.to_csv(output_dir + f"/predict/LSTM_predict{file_tag}_cal.csv", mode='a', header=False)
    file_row_eva.to_csv(output_dir + f"/predict/LSTM_predict{file_tag}_eva.csv", mode='a', header=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'{benchmark}': BMK(target_cal, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'{benchmark}': BMK(target_eva, predict_eva, benchmark)
        })  

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results to CSV after each file
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/results/LSTM_results{file_tag}_cal.csv', index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/results/LSTM_results{file_tag}_eva.csv', index=False, header=True)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"pred elapsed: {end_time - pred_start_time}\n")
    log_file.write(f"epoch elapsed: {end_time - start_time}\n")
    log_file.write(f"Results saved for file {file_num}\n")
    print(f"Results saved for file {file_num}")

##
log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/LSTM_results{file_tag}_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/LSTM_results{file_tag}_eva.csv', index=False, header=True)

print(f"LSTM{file_tag}")
print("DONE")


"""
# Plot results
start_date = ds_test.dates[0]
end_date = ds_test.dates[1] + pd.DateOffset(days=1)
date_range = pd.date_range(start_date, end_date)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(date_range, obs, label="observation")
ax.plot(date_range, eva_preds, label="prediction")
ax.legend()
ax.set_title(f"file_num {file_num} - Test set NSE: {nse:.3f}")
ax.xaxis.set_tick_params(rotation=90)
ax.set_xlabel("Date")
_ = ax.set_ylabel("Discharge (mm/d)")
"""