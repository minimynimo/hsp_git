import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting

import numpy as np
import os
import argparse
import copy

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from datetime import datetime

# PyTorch device selection - same convention as TCN_regional_modeling-master/main_tcn_hp.py.
# Ported from Keras/TensorFlow + the `tcn` package, which this environment does not have
# installed (unlike torch) - see DEBUG_README.md. Per-basin methodology, CLI surface, and output
# CSV layout are otherwise unchanged.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


#loc, ver = "JP", 2
#loc, ver = "GB", 2
loc, ver = "US", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"


if loc == "JP" and ver == 2:
    varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
elif loc == "GB" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_GB"
elif loc == "US" and ver == 2:
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_US"

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TCN_regional_modeling-master"))
from papercode.utils import get_basin_list
gauge_id_list = get_basin_list(loc)[::4]

from tcn_indiv_resume import load_completed_basins

input_dim = 3

# Parse command-line arguments for hyperparameters (CLI surface unchanged by the PyTorch port)
parser = argparse.ArgumentParser(description='TCN Individual Basin Modeling with Hyperparameter Tuning')
parser.add_argument('--num_filters', type=int, default=16, help='Number of filters per TCN layer')
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size for TCN')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Adam learning rate (was hardcoded 1e-3)')
# PyTorch's TemporalConvNet derives dilations from num_channels directly (dilation=2**i per level,
# see TemporalConvNet below) - no separate `dilations=` list needed anymore (that was a
# Keras-`tcn`-package-specific workaround from before this port). --num_levels now controls
# num_channels = [num_filters] * num_levels, exactly like main_tcn.py/main_tcn_hp.py in the
# regional pipeline. Fixed at 6 by HYPERPARAMETERS.md's TCN "LAYERS 1 2 4 8 16 32" row (not swept -
# see DEBUG_README.md); default here stays 8 for backward compatibility with direct invocations
# that omit the flag - run_tcn_hp_tuning.sh's grid always passes --num_levels 6 explicitly.
parser.add_argument('--num_levels', type=int, default=8, help='Number of dilation levels; num_channels=[num_filters]*num_levels (was hardcoded 8, i.e. dilations [1..128])')
# Input window length (days), swept in the HP grid (90/180/270) - was previously hardcoded to 90
# based on Xu et al., 2021. Default kept at 90 for backward compat with direct invocations that
# omit the flag.
parser.add_argument('--timesteps', type=int, default=90, help='Input window length in days')
args = parser.parse_args()

# Set hyperparameters from arguments
nb_filters = args.num_filters
kernel_size = args.kernel_size
dropout = args.dropout
learning_rate = args.learning_rate
num_levels = args.num_levels
num_channels = [nb_filters] * num_levels
timesteps = args.timesteps
epochs = 30  # Fixed at 30 for consistency
batch_size = 64  # Keep as is for individual basins
early_stopping_patience = 5  # matches the old Keras EarlyStopping(patience=5)
val_split = 0.2  # matches the old Keras .fit(..., validation_split=0.2)

start_date_cal = '1990-01-01'
end_date_cal = '1998-12-31'
start_date_eva = '1999-01-01'
end_date_eva = '2000-12-31'

# Create output directory with hyperparameter settings
# learning_rate and num_levels are swept/overridable here (unlike main_TCN_indiv.py's hardcoded
# production script) - both included in the tag so two runs differing only in one of them can't
# silently overwrite each other's results (same collision-avoidance principle as
# DEBUG_README.md's TCN/LSTM filename-tag fixes).
hp_str = f"f{nb_filters}_k{kernel_size}_d{dropout}_lr{learning_rate}_lv{num_levels}_w{timesteps}"
output_dir = f"/data0/funato/hsp2/0_out/TCN/{loc}/individual/hp/{hp_str}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/predict", exist_ok=True)
os.makedirs(output_dir + "/results", exist_ok=True)
os.makedirs(output_dir + "/hydrographs", exist_ok=True) # New directory for hydrographs

# Receptive-field sanity check, same formula/spirit as TCN_regional_modeling-master/main_tcn_hp.py.
_rf = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
print(f"TCN Configuration: Kernel={kernel_size}, Layers={num_levels}, Filters={nb_filters}")
print(f"Theoretical Receptive Field: {_rf} days (input window/timesteps={timesteps})")
if _rf < timesteps:
    print(f"WARNING: Receptive field ({_rf}) is shorter than the input window ({timesteps}) - "
          f"the model cannot see the full window.")
elif _rf > timesteps:
    print(f"NOTE: Receptive field ({_rf}) exceeds the input window ({timesteps}) - harmless, but "
          f"the deepest dilation levels add capacity without adding information for this window.")


# ####################### #
# TCN Model Definition (ported verbatim from TCN_regional_modeling-master/main_tcn.py)
# ####################### #

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels_ = len(num_channels)
        for i in range(num_levels_):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


def load_data(gauge_id):
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")

    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)

    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)

    return df


def BMK(obs_data,sim_data,benchmark):
    # Ensure data are flat for calculations if they come in (N, 1) shape
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

    # Handle cases where obs_data might be all zero or very close to zero
    # This prevents division by zero or extremely large values in some metrics
    if np.sum(obs_data) == 0 and benchmark not in ["RMSE", "MAE"]:
        return np.nan # Or some other indicator for undefined metric

    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        # Avoid division by zero if obs_data is constant
        if np.sum(np.square(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan # Perfect match for constant series
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))

    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)

        # Guard against zero standard deviation for constant series
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        # Guard against zero mean for very small values
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave)

        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)

    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        if np.sum(np.abs(obs_data - obs_ave)) == 0:
            return 1.0 if np.all(obs_data == sim_data) else np.nan
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))

    elif benchmark == "VE":
        if np.sum(obs_data) == 0:
             return 1.0 if np.all(obs_data == sim_data) else np.nan # If observed is zero, perfect if sim is also zero
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)

    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return 1.0 if numer == 0 else np.nan
        return 1 - numer / denom

    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))

    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

def create_sequences(data, targets, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)


def train_basin_model(X_cal_seq, y_cal_seq, input_size):
    """Train one TCNModel from scratch on a single basin's calibration-period sequences.

    Manual PyTorch training loop reproducing Keras's old `.fit(..., validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])`: the last 20% of the
    (time-ordered) sequence array is held out as validation, training stops once val loss hasn't
    improved for `early_stopping_patience` epochs, and the best-val-loss weights are restored.
    """
    model = TCNModel(input_size=input_size, num_channels=num_channels,
                      kernel_size=kernel_size, dropout=dropout).to(DEVICE)

    n = len(X_cal_seq)
    n_val = max(1, int(round(n * val_split)))
    n_train = max(1, n - n_val)

    X_train_t = torch.from_numpy(X_cal_seq[:n_train]).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_cal_seq[:n_train]).float().to(DEVICE).unsqueeze(1)
    X_val_t = torch.from_numpy(X_cal_seq[n_train:]).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_cal_seq[n_train:]).float().to(DEVICE).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    n_train_samples = X_train_t.shape[0]

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train_samples)
        for i in range(0, n_train_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_func(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_basin(model, X_seq):
    with torch.no_grad():
        X_t = torch.from_numpy(X_seq).float().to(DEVICE)
        return model(X_t).cpu().numpy().flatten()


results_cal_path = output_dir + f'/results/TCN_results_cal.csv'
results_eva_path = output_dir + f'/results/TCN_results_eva.csv'
predict_cal_path = output_dir + f"/predict/TCN_predict_cal.csv"
predict_eva_path = output_dir + f"/predict/TCN_predict_eva.csv"

completed_basins, results_cal, results_eva = load_completed_basins(results_cal_path, results_eva_path)

log_file = open(output_dir + f'/TCN_log.txt', 'a')
if completed_basins:
    log_file.write(f"RESUME: {len(completed_basins)} basins already completed, skipping them.\n")
    log_file.flush()
    print(f"Resuming: {len(completed_basins)} basins already completed, skipping them.")

start_time_overall = datetime.now()
start_time_overall_st = start_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall start: {start_time_overall_st}\n")
log_file.flush()


# --- Individual Training Loop ---
for gauge_id in gauge_id_list:
    if str(gauge_id) in completed_basins:
        continue

    start_time_basin = datetime.now()
    start_time_st_basin = start_time_basin.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"BASIN_{gauge_id} - Individual Training\n")
    log_file.write(f"start: {start_time_st_basin}\n")
    log_file.flush()

    ### LOAD DATA FOR CURRENT BASIN ###
    df = load_data(gauge_id)

    df_cal = df[start_date_cal:end_date_cal].copy()
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Prepare dynamic features
    input_cal_dyn = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)])
    input_eva_dyn = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)])

    # Apply masks to remove NaN values
    mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(df_cal['Obs flow'].values)
    mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(df_eva['Obs flow'].values)

    input_cal = input_cal_dyn[mask_cal]
    input_eva = input_eva_dyn[mask_eva]

    # Targets
    target_cal = df_cal['Obs flow'].values[mask_cal]
    target_eva = df_eva['Obs flow'].values[mask_eva]

    # Normalize dynamic features using this basin's data only (per-basin, unchanged by the port)
    scaler = StandardScaler()
    input_cal = scaler.fit_transform(input_cal)
    input_eva = scaler.transform(input_eva)

    # Create sequences
    input_cal_seq, target_cal_seq = create_sequences(input_cal, target_cal, timesteps)
    input_eva_seq, target_eva_seq = create_sequences(input_eva, target_eva, timesteps)

    # Skip if insufficient data
    if len(input_cal_seq) == 0 or len(input_eva_seq) == 0:
        log_file.write(f"Insufficient data for basin {gauge_id}, skipping\n")
        continue

    ### BUILD AND TRAIN MODEL FOR THIS BASIN (PyTorch TCNModel, one fresh model per basin) ###
    input_cal_seq = input_cal_seq.astype(np.float32)
    target_cal_seq = target_cal_seq.astype(np.float32)
    input_eva_seq = input_eva_seq.astype(np.float32)

    TCN_model = train_basin_model(input_cal_seq, target_cal_seq, input_size=input_cal.shape[1])

    ### MAKE PREDICTIONS ###
    predict_cal_basin = predict_basin(TCN_model, input_cal_seq)
    predict_eva_basin = predict_basin(TCN_model, input_eva_seq)

    predict_cal_basin[predict_cal_basin < 0] = 0
    predict_eva_basin[predict_eva_basin < 0] = 0

    ### SAVE PREDICTIONS ###
    # Prepare CSV data
    file_row_cal = [f'basin_{gauge_id}_cal'] + list(predict_cal_basin)
    file_row_eva = [f'basin_{gauge_id}_eva'] + list(predict_eva_basin)

    # Write headers if the predict CSVs don't exist yet -- keyed off file existence rather than
    # "gauge_id == gauge_id_list[0]" so a resumed run (where basin[0] may already be completed
    # and skipped above) still gets a header written by whichever basin is processed first.
    if not os.path.exists(predict_cal_path):
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=timesteps+1), periods=len(predict_cal_basin))
        date_row_cal = ['Date'] + [str(date.date()) for date in predict_cal_dates]
        with open(predict_cal_path, 'w') as file:
            pd.DataFrame([date_row_cal]).to_csv(file, header=False, index=False)
    if not os.path.exists(predict_eva_path):
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=timesteps+1), periods=len(predict_eva_basin))
        date_row_eva = ['Date'] + [str(date.date()) for date in predict_eva_dates]
        with open(predict_eva_path, 'w') as file:
            pd.DataFrame([date_row_eva]).to_csv(file, header=False, index=False)

    # Append predictions
    with open(predict_cal_path, 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(predict_eva_path, 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    ### CALCULATE BENCHMARKS ###
    file_results_cal = {'gauge_id': gauge_id}
    file_results_eva = {'gauge_id': gauge_id}

    for benchmark in benchmark_list:
        file_results_cal.update({
            f'TCN_{benchmark}_cal': BMK(target_cal_seq, predict_cal_basin, benchmark)
        })
        file_results_eva.update({
            f'TCN_{benchmark}_eva': BMK(target_eva_seq, predict_eva_basin, benchmark)
        })

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    # Save results incrementally
    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(results_cal_path, index=False, header=True)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(results_eva_path, index=False, header=True)

    print(f"gauge_id: {gauge_id}, NSE_cal: {BMK(target_cal_seq, predict_cal_basin, 'NSE'):.4f}, NSE_eva: {BMK(target_eva_seq, predict_eva_basin, 'NSE'):.4f}")

    end_time_basin = datetime.now()
    log_file.write(f"end: {end_time_basin}\n")
    log_file.write(f"elapsed: {end_time_basin - start_time_basin}\n")
    log_file.write(f"Results saved for basin {gauge_id}\n")
    print(f"Results saved for basin {gauge_id}")
    log_file.flush()

end_time_overall = datetime.now()
end_time_overall_st = end_time_overall.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"Overall end: {end_time_overall_st}\n")
log_file.write(f"Overall elapsed: {end_time_overall - start_time_overall}\n")
log_file.write(f"DONE\n")
log_file.close()

# Final save (redundant but ensures final state is saved)
df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/TCN_results_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/TCN_results_eva.csv', index=False, header=True)

print("DONE")
