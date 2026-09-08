# Individual-basin LSTM hyperparameter search (usually launched in bulk via
# run_lstm_hp_tuning.sh instead of calling this directly).
#
# RUN USING (single HP combo; defaults to --dataset camels; edit `loc` in this file to change
# location):
# python main_LSTM_indiv_hp.py --hidden_size 32 --dropout 0.4 --learning_rate 0.001 --seq_length 270 --batch_size 128 --dataset camels

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
from lstm_indiv_resume import load_completed_basins

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser(description='LSTM Individual Basin Modeling with Hyperparameter Tuning')
parser.add_argument('--hidden_size', type=int, default=16, help='Number of LSTM cells')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                     help='Adam learning rate. Default 1e-3 matches the previously hardcoded '
                          'value / Kratzert et al. (2018, 2019).')
parser.add_argument('--seq_length', type=int, default=30,
                     help='Input sequence length (aka window_size). Default 30 matches the '
                          'previously hardcoded value. This repo\'s own individual-model tests '
                          'on GB basins found seq_length materially affects skill (NSE 0.808 '
                          '/0.83/0.838 at 30/90/270), so it is worth sweeping.')
parser.add_argument('--batch_size', type=int, default=256,
                     help='Training batch size. Default 256 matches the previously hardcoded '
                          'value, for backward compatibility if omitted.')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'nse'],
                     help='Loss function. "mse" (default) matches the previously hardcoded '
                          'nn.MSELoss() on locally-normalized targets. "nse" rescales '
                          'predictions/targets to physical discharge units using this basin\'s '
                          'own training-period mean/std and applies the per-basin-variance-'
                          'weighted NSE loss from ealstm_regional_modeling/papercode/nseloss.py '
                          '(eps=0.1), matching the loss regional/adapted LSTM training already '
                          'uses by default.')
parser.add_argument('--dataset', type=str, default='camels', choices=['camels', 'caravan'],
                     help="'camels' (default): built directly from raw CAMELS-US, full "
                          '1980-2014 record. HP search sub-splits its own train/val window '
                          '1990-1998/1999-2000 strictly inside main_LSTM_indiv.py\'s production '
                          "1987-2000 calibration period, mirroring "
                          'ealstm_regional_modeling/main_lstm_regional_hp.py\'s HP_SPLIT -- so '
                          'HP search never touches the 2001-2006 held-out period. \'caravan\': '
                          'older Caravan-derived extraction (1987-2006 only); HP window stays '
                          'the original 1990-1998/1999-2000, unchanged.')
args = parser.parse_args()

# Set hyperparameters from arguments
hidden_size = args.hidden_size
dropout_rate = args.dropout
learning_rate = args.learning_rate  # was hardcoded 1e-3
window_size = args.seq_length  # was hardcoded 30 (input sequence length)
batch_size = args.batch_size  # was hardcoded 256

# Fixed hyperparameters based on Kratzert et al. (2018, 2019)
num_epochs = 20  # Fixed at 20 for individual basins

#loc, ver = "JP", 2
loc, ver = "US", 2
#loc, ver = "GB", 2
#loc, ver = "AUS", 2

dataset = args.dataset

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

# Create hyperparameter string for output directory
# Includes lr/seq_length (not just h/d) now that both are swept, so each grid combo gets its
# own directory instead of relying solely on file_tag to disambiguate within a shared folder.
# loss_tag is appended (empty string for the original "mse" default) so an NSE-loss rerun of the
# same hyperparameter combo lands in its own directory instead of colliding with -- or being
# silently skipped as "already completed" via -- the existing MSE-loss results.
loss_tag = "" if args.loss == "mse" else f"_loss{args.loss}"
hp_str = f"h{hidden_size}_d{dropout_rate}_lr{learning_rate}_sl{window_size}_b{batch_size}{loss_tag}"
file_tag = f"_h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}{loss_tag}"

ver_name = "ver1_1" if ver == 1 else "ver2_0"

# Basin identity: gauge_id (plain string, no `camels_` prefix), read from the `gauge_id`
# column of {loc.lower()}_{dataset}_file_num.csv for US (availability-filtered separately per
# dataset), or {loc.lower()}_file_num.csv for GB/JP (single source, no dataset split). US's
# gauge_id column is prefixed (`camels_01022500`), normalized here via
# str(g).split('_')[-1]; GB's is already plain (e.g. `96001`).
FILE_NUM_CSV = {
    'US': {
        'camels': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_camels_file_num.csv',
        'caravan': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_caravan_file_num.csv',
    },
    'GB': '/data0/funato/hsp2/3_gis_data/GB/0_data/river_basin/dataset_GB/gb_file_num.csv',
    'JP': '/data0/funato/hsp2/3_gis_data/JP/0_data/river_basin/dataset_JP/jp_file_num.csv',
}
_file_num_entry = FILE_NUM_CSV[loc]
_file_num_csv_path = _file_num_entry[dataset] if isinstance(_file_num_entry, dict) else _file_num_entry
_file_num_df = pd.read_csv(_file_num_csv_path, dtype={'gauge_id': str})
_full_gauge_id_list = [str(g).split('_')[-1] for g in _file_num_df['gauge_id']]

if loc == "JP" and ver == 2:
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/caravan/varssim_GB"
elif loc == "US" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_US"

#gauge_id_list = _full_gauge_id_list
gauge_id_list = _full_gauge_id_list[::4]

# HP-search-only train/val window, kept strictly INSIDE main_LSTM_indiv.py's production
# calibration period so HP search never touches its real held-out evaluation period --
# mirrors ealstm_regional_modeling/main_lstm_regional_hp.py's HP_SPLIT exactly.
#   * 'camels'  -- production calibration is 1987-2000; HP search sub-splits that into
#     1990-1998 (train) / 1999-2000 (val), staying clear of the 2001-2006 held-out period.
#   * 'caravan' -- production calibration is 1990-2000 (this script's own dates -- see
#     main_LSTM_indiv.py); HP search sub-splits that into 1990-1998 (train) / 1999-2000 (val),
#     staying clear of the 2001-2006 held-out period. Unchanged from before this script
#     supported multiple datasets.
if dataset == "camels":
    HP_TRAIN_START = pd.to_datetime("1990-01-01", format="%Y-%m-%d")
    HP_TRAIN_END = pd.to_datetime("1998-12-31", format="%Y-%m-%d")
    HP_VAL_START = pd.to_datetime("1999-01-01", format="%Y-%m-%d")
    HP_VAL_END = pd.to_datetime("2000-12-31", format="%Y-%m-%d")
else:  # caravan
    HP_TRAIN_START = pd.to_datetime("1990-01-01", format="%Y-%m-%d")
    HP_TRAIN_END = pd.to_datetime("1998-12-31", format="%Y-%m-%d")
    HP_VAL_START = pd.to_datetime("1999-01-01", format="%Y-%m-%d")
    HP_VAL_END = pd.to_datetime("2000-12-31", format="%Y-%m-%d")

print(f"[main_LSTM_indiv_hp] loc={loc} dataset={dataset!r} "
      f"train={HP_TRAIN_START.date()}..{HP_TRAIN_END.date()} "
      f"val={HP_VAL_START.date()}..{HP_VAL_END.date()}")

# Globals
#FILE_SYSTEM = gcsfs.core.GCSFileSystem(requester_pays=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available
print(f"Using device: {DEVICE}")

# Create output directory with hyperparameter settings
output_dir = f"/data0/funato/hsp2/0_out/LSTM/{loc}/{dataset}/individual/hp/{hp_str}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/model', exist_ok=True)

results_cal_path = output_dir + f'/results/LSTM_results{file_tag}_cal.csv'
results_eva_path = output_dir + f'/results/LSTM_results{file_tag}_eva.csv'
predict_cal_path = output_dir + f"/predict/LSTM_predict{file_tag}_cal.csv"
predict_eva_path = output_dir + f"/predict/LSTM_predict{file_tag}_eva.csv"

completed_basins, results_cal, results_eva = load_completed_basins(results_cal_path, results_eva_path)

log_file = open(output_dir + f'/LSTM_log.txt', 'a')
if completed_basins:
    log_file.write(f"RESUME: {len(completed_basins)} basins already completed, skipping them.\n")
    log_file.flush()
    print(f"Resuming: {len(completed_basins)} basins already completed, skipping them.")

####MAIN#####

for gauge_id in gauge_id_list:
    if str(gauge_id) in completed_basins:
        continue

    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")

    log_file.write(f"BASIN_{gauge_id}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ##############
    # Data set up#
    ##############

    # Training data: HP-search-only calibration sub-window (see HP_TRAIN_START/END above),
    # strictly inside production's real calibration period. The "eva" split below uses
    # HP_VAL_START/END, a genuinely distinct validation sub-window, not a re-use of these dates.
    ds_train = CamelsTXT(gauge_id, seq_length=window_size, period="train", dates=[HP_TRAIN_START, HP_TRAIN_END], varssim_dir=varssim_dir, loc =loc)
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # "Cal" evaluation: performance on the training period itself. We use the feature
    # means/stds of the training period for normalization.
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    ds_val = CamelsTXT(gauge_id, seq_length=window_size, period="eval", dates=[HP_TRAIN_START, HP_TRAIN_END],
                        means=means, stds=stds, varssim_dir=varssim_dir, loc =loc)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # "Eva" evaluation: genuinely held-out HP-search validation sub-window (distinct from the
    # training sub-window above, but still inside production's real calibration period -- see
    # HP_VAL_START/END comment above). We use the feature means/stds of the training period for
    # normalization.
    ds_test = CamelsTXT(gauge_id, seq_length=window_size, period="eval", dates=[HP_VAL_START, HP_VAL_END],
                        means=means, stds=stds, varssim_dir=varssim_dir, loc =loc)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)


    #########################
    # Model, Optimizer, Loss#
    #########################

    # Here we create our model, feel free
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.loss == "nse":
        # train_epoch() (imported from run_LSTM.py) calls loss_func(y_hat, ys) with a plain
        # 2-argument signature, both tensors already locally-normalized (zero mean, unit std
        # over this basin's own training period) by CamelsTXT. NSELoss itself takes a third
        # q_stds argument and expects physical-scale (rescaled) values -- for a single basin
        # q_std is one constant, so it's folded into a closure here rather than changing
        # train_epoch's call signature. Rescaling is done with plain tensor arithmetic (not
        # CamelsTXT.local_rescale, which is numpy-only and would detach the autograd graph).
        q_mean = float(ds_train.get_means()['Obs flow'])
        q_std = float(ds_train.get_stds()['Obs flow'])
        nse_eps = 0.1

        def loss_func(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            pred_phys = y_pred * q_std + q_mean
            true_phys = y_true * q_std + q_mean
            return torch.mean((pred_phys - true_phys) ** 2) / (q_std + nse_eps) ** 2
    else:
        loss_func = nn.MSELoss()

    # Per-basin checkpoint, saved every 5 epochs (see below) so a restart mid-basin resumes
    # training from the last checkpointed epoch instead of redoing the whole basin from epoch 0.
    # Only basin-level resume existed before this (lstm_indiv_resume.load_completed_basins
    # skips basins that already have a row in both results CSVs); this covers the gap where a
    # basin was killed partway through its num_epochs loop.
    checkpoint_path = output_dir + f"/model/checkpoint_basin{gauge_id}.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_file.write(f"RESUME: basin {gauge_id} resuming from epoch {start_epoch}\n")
        log_file.flush()
        print(f"Resuming basin {gauge_id} from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
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

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

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

    file_row_cal = [f'basin_{gauge_id}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'basin_{gauge_id}_eva'] + list(predict_eva.flatten())

    # Save predictions with proper indexing
    file_row_cal = pd.DataFrame([file_row_cal], columns=['Index'] + [f'Value_{i}' for i in range(len(predict_cal.flatten()))])
    file_row_cal.set_index('Index', inplace=True)
    file_row_eva = pd.DataFrame([file_row_eva], columns=['Index'] + [f'Value_{i}' for i in range(len(predict_eva.flatten()))])
    file_row_eva.set_index('Index', inplace=True)

    # Write headers if the predict CSVs don't exist yet -- keyed off file existence rather than
    # "gauge_id == gauge_id_list[0]" so a resumed run (where basin[0] may already be completed
    # and skipped above) still gets a header written by whichever basin is processed first.
    if not os.path.exists(predict_cal_path):
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal))
        date_row_cal = pd.DataFrame([['Date'] + [str(date.date()) for date in predict_cal_dates]])
        date_row_cal.to_csv(predict_cal_path, mode='w', index=False, header=False)
    if not os.path.exists(predict_eva_path):
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva))
        date_row_eva = pd.DataFrame([['Date'] + [str(date.date()) for date in predict_eva_dates]])
        date_row_eva.to_csv(predict_eva_path, mode='w', index=False, header=False)

    # Append rows with proper indexing
    file_row_cal.to_csv(predict_cal_path, mode='a', header=False)
    file_row_eva.to_csv(predict_eva_path, mode='a', header=False)

    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

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
    df_results_cal.to_csv(results_cal_path, index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(results_eva_path, index=False, header=True)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"pred elapsed: {end_time - pred_start_time}\n")
    log_file.write(f"epoch elapsed: {end_time - start_time}\n")
    log_file.write(f"Results saved for basin {gauge_id}\n")
    print(f"Results saved for basin {gauge_id}")

    # Basin finished (row written to both results CSVs, so load_completed_basins will skip it on
    # any future restart) -- the mid-training checkpoint is no longer needed.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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
ax.set_title(f"gauge_id {gauge_id} - Test set NSE: {nse:.3f}")
ax.xaxis.set_tick_params(rotation=90)
ax.set_xlabel("Date")
_ = ax.set_ylabel("Discharge (mm/d)")
"""