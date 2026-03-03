# Imports
from pathlib import Path
from typing import Tuple, List

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

# Local imports (Assumed these exist based on your snippet)
from run_LSTM import CamelsTXT
from run_LSTM import Model
from run_LSTM import train_epoch
from run_LSTM import eval_model
from run_LSTM import calc_nse
from run_LSTM import BMK

# --- Configuration ---
num_epochs = 100
hidden_size = 32
dropout_rate = 0.2
learning_rate = 0.001
window_size = 270
batch_size = 256
checkpoint_interval = 10

# loc, ver = "JP", 2
# loc, ver = "US", 2
loc, ver = "GB", 2

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

file_tag = f"_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}"
ver_name = "ver1_1" if ver == 1 else "ver2_0"

# Set file count based on location
if loc == "JP": file_tot_num = 87
elif loc == "US": file_tot_num = 667
elif loc == "AUS": file_tot_num = 84
elif loc == "GB": file_tot_num = 424

# Set directories
if loc == "JP" and ver == 2:
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Output Directories Setup ---
output_dir = f"/data0/funato/hsp2/0_out/LSTM/{loc}/individual/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/models', exist_ok=True) # Directory for .pth model files

# Initialize log file
if os.path.exists(output_dir + f'/LSTM_log.txt'):
    open(output_dir + f'/LSTM_log.txt', 'w').close()
log_file = open(output_dir + f'/LSTM_log.txt', 'a')

# --- Helper Function for Checkpointing ---
def save_checkpoint(model, epoch, file_num, val_loader, test_loader, ds_train, ds_test, results_store):
    """
    Evaluates model, saves predictions, saves model weights, and updates results CSV.
    """
    current_ep = epoch + 1
    
    # 1. Create directory for this specific epoch to keep things organized
    ep_pred_dir = os.path.join(output_dir, 'predict', f'epoch_{current_ep}')
    ep_res_dir = os.path.join(output_dir, 'results', f'epoch_{current_ep}')
    os.makedirs(ep_pred_dir, exist_ok=True)
    os.makedirs(ep_res_dir, exist_ok=True)
    
    # 2. Save Model Weights (.pth)
    model_path = os.path.join(output_dir, 'models', f'lstm_file{file_num:03d}_ep{current_ep}.pth')
    torch.save(model.state_dict(), model_path)

    # 3. Evaluate
    # Note: 'cal' here refers to validation period, 'eva' to test period per your structure
    cal_obs, predict_cal = eval_model(model, val_loader)
    eva_obs, predict_eva = eval_model(model, test_loader)

    # Rescale
    predict_cal = ds_train.local_rescale(predict_cal.cpu().numpy(), variable='output')
    predict_eva = ds_train.local_rescale(predict_eva.cpu().numpy(), variable='output')
    target_cal = cal_obs.cpu().numpy()
    target_eva = eva_obs.cpu().numpy()

    # Clip negatives
    predict_cal[predict_cal < 0] = 0
    predict_eva[predict_eva < 0] = 0

    # 4. Save Prediction CSVs (Append Mode)
    # Prepare rows
    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())
    
    df_pred_cal = pd.DataFrame([file_row_cal])
    df_pred_eva = pd.DataFrame([file_row_eva])
    
    # Define filenames for this epoch
    pred_file_cal = os.path.join(ep_pred_dir, f"LSTM_predict{file_tag}_ep{current_ep}_cal.csv")
    pred_file_eva = os.path.join(ep_pred_dir, f"LSTM_predict{file_tag}_ep{current_ep}_eva.csv")

    # Handle Headers (only if it's the first file processed)
    if file_num == 1:
        start_date_cal = ds_train.dates[0]
        start_date_eva = ds_test.dates[0]
        
        dates_cal = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        dates_eva = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))
        
        header_cal = pd.DataFrame([['Date'] + [str(d.date()) for d in dates_cal]])
        header_eva = pd.DataFrame([['Date'] + [str(d.date()) for d in dates_eva]])
        
        header_cal.to_csv(pred_file_cal, mode='w', index=False, header=False)
        header_eva.to_csv(pred_file_eva, mode='w', index=False, header=False)

    # Append predictions
    df_pred_cal.to_csv(pred_file_cal, mode='a', index=False, header=False)
    df_pred_eva.to_csv(pred_file_eva, mode='a', index=False, header=False)

    # 5. Calculate Metrics and Update Results Store
    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_cal[benchmark] = BMK(target_cal, predict_cal, benchmark)
        file_results_eva[benchmark] = BMK(target_eva, predict_eva, benchmark)

    # Store results in the dictionary for this epoch
    results_store[current_ep]['cal'].append(file_results_cal)
    results_store[current_ep]['eva'].append(file_results_eva)

    # Save Results CSV (Overwrite with accumulated list)
    res_file_cal = os.path.join(ep_res_dir, f'LSTM_results{file_tag}_ep{current_ep}_cal.csv')
    res_file_eva = os.path.join(ep_res_dir, f'LSTM_results{file_tag}_ep{current_ep}_eva.csv')
    
    pd.DataFrame(results_store[current_ep]['cal']).to_csv(res_file_cal, index=False)
    pd.DataFrame(results_store[current_ep]['eva']).to_csv(res_file_eva, index=False)
    
    return BMK(target_eva, predict_eva, "KGE") # Return KGE for logging


# --- Main Loop ---

# Dictionary to hold results for every checkpoint epoch
# Structure: { 5: {'cal': [], 'eva': []}, 10: {...}, ... }
checkpoint_epochs = range(checkpoint_interval, num_epochs + 1, checkpoint_interval)
checkpoint_results_store = {ep: {'cal': [], 'eva': []} for ep in checkpoint_epochs}

for file_num in range(1, file_tot_num + 1):
    start_time = datetime.now()
    log_file.write(f"FILE_{file_num}\n")
    log_file.write(f"start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()

    # --- Data Setup ---
    start_date = pd.to_datetime("1993-01-01")
    end_date = pd.to_datetime("2000-12-31")
    ds_train = CamelsTXT(file_num, seq_length=window_size, period="train", dates=[start_date, end_date], varssim_dir=varssim_dir, loc=loc)
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    means, stds = ds_train.get_means(), ds_train.get_stds()

    ds_val = CamelsTXT(file_num, seq_length=window_size, period="eval", dates=[start_date, end_date], means=means, stds=stds, varssim_dir=varssim_dir, loc=loc)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    ds_test = CamelsTXT(file_num, seq_length=window_size, period="eval", dates=[pd.to_datetime("2001-01-01"), pd.to_datetime("2006-12-31")], means=means, stds=stds, varssim_dir=varssim_dir, loc=loc)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # --- Model Setup ---
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    # --- Training Loop with Checkpoints ---
    for epoch in range(num_epochs):
        # 1. Train
        train_epoch(model, optimizer, tr_loader, loss_func, epoch+1)
        
        # 2. Checkpoint Evaluation
        if (epoch + 1) % checkpoint_interval == 0:
            kge_val = save_checkpoint(
                model=model, 
                epoch=epoch, 
                file_num=file_num, 
                val_loader=val_loader, 
                test_loader=test_loader, 
                ds_train=ds_train, 
                ds_test=ds_test, 
                results_store=checkpoint_results_store
            )
            print(f"File {file_num} | Epoch {epoch+1} Saved | Test KGE: {kge_val:.3f}")
            log_file.write(f"File {file_num} | Epoch {epoch+1} Checkpoint Saved.\n")

    end_time = datetime.now()
    log_file.write(f"end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"total elapsed: {end_time - start_time}\n")
    log_file.write(f"Results saved for file {file_num}\n")
    print(f"Finished File {file_num}")

log_file.write(f"DONE\n")
log_file.close()
print(f"LSTM{file_tag} - ALL DONE")