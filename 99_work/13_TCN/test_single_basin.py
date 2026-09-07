import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from tcn import TCN
from sklearn.preprocessing import StandardScaler

# Load a single basin (GB gauge_id 96001, the first row of gb_file_num.csv)
gauge_id = "96001"
varssim_dir = "/data0/funato/hsp2/2_MERV/varssim_GB"
df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df_cal = df['1990-01-01':'2000-12-31'].copy()
df_eva = df['2001-01-01':'2006-12-31'].copy()

# Prepare data
dyn_cols = ['Precip', 'PET', 'Temp']
input_cal_dyn = df_cal[dyn_cols].values
input_eva_dyn = df_eva[dyn_cols].values

target_cal_raw = df_cal['Obs flow'].values
target_eva_raw = df_eva['Obs flow'].values

# Masks
mask_cal = ~np.isnan(input_cal_dyn).any(axis=1) & ~np.isnan(target_cal_raw)
mask_eva = ~np.isnan(input_eva_dyn).any(axis=1) & ~np.isnan(target_eva_raw)

input_cal_dyn = input_cal_dyn[mask_cal]
input_eva_dyn = input_eva_dyn[mask_eva]
target_cal_raw = target_cal_raw[mask_cal]
target_eva_raw = target_eva_raw[mask_eva]

print(f"Basin {gauge_id} data shapes:")
print(f"  Cal input: {input_cal_dyn.shape}, target: {target_cal_raw.shape}")
print(f"  Eva input: {input_eva_dyn.shape}, target: {target_eva_raw.shape}")
print(f"  Target cal range: [{target_cal_raw.min():.2f}, {target_cal_raw.max():.2f}], mean: {target_cal_raw.mean():.2f}")

# Normalize separately
scaler_dyn = StandardScaler()
input_cal = scaler_dyn.fit_transform(input_cal_dyn)
input_eva = scaler_dyn.transform(input_eva_dyn)

scaler_q = StandardScaler()
target_cal = scaler_q.fit_transform(target_cal_raw.reshape(-1, 1)).flatten()
target_eva = scaler_q.transform(target_eva_raw.reshape(-1, 1)).flatten()

print(f"\nNormalization stats:")
print(f"  Discharge mean: {scaler_q.mean_[0]:.4f}, std: {scaler_q.scale_[0]:.4f}")
print(f"  Target cal range (normalized): [{target_cal.min():.2f}, {target_cal.max():.2f}]")

# Create sequences
def create_sequences(data, targets, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

timesteps = 270
input_cal_seq, target_cal_seq = create_sequences(input_cal, target_cal, timesteps)
input_eva_seq, target_eva_seq = create_sequences(input_eva, target_eva, timesteps)

print(f"\nSequence shapes:")
print(f"  Cal: {input_cal_seq.shape}, {target_cal_seq.shape}")
print(f"  Eva: {input_eva_seq.shape}, {target_eva_seq.shape}")

# Build model
model_in = Input(shape=(timesteps, 3))
model_out = TCN(
    nb_filters=64,
    kernel_size=4,
    nb_stacks=1,
    dilations=[1, 2, 4, 8, 16, 32, 64, 128],
    padding='causal',
    dropout_rate=0.2,
    return_sequences=False
)(model_in)
model_out = Dense(1, activation='linear')(model_out)
model = Model(inputs=model_in, outputs=model_out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

print(f"\nModel built. Training...")

# Train
val_split_idx = int(len(input_cal_seq) * 0.8)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    input_cal_seq[:val_split_idx], 
    target_cal_seq[:val_split_idx], 
    epochs=100, 
    batch_size=64,
    validation_data=(input_cal_seq[val_split_idx:], target_cal_seq[val_split_idx:]),
    callbacks=[early_stopping], 
    verbose=1
)

print(f"\nTraining complete. Epochs trained: {len(history.history['loss'])}")
print(f"Final loss: {history.history['loss'][-1]:.6f}, val_loss: {history.history['val_loss'][-1]:.6f}")

# Predict
predict_cal = model.predict(input_cal_seq, batch_size=64, verbose=0).flatten()
predict_eva = model.predict(input_eva_seq, batch_size=64, verbose=0).flatten()

# Denormalize
predict_cal_denorm = predict_cal * scaler_q.scale_[0] + scaler_q.mean_[0]
predict_eva_denorm = predict_eva * scaler_q.scale_[0] + scaler_q.mean_[0]
predict_cal_denorm[predict_cal_denorm < 0] = 0
predict_eva_denorm[predict_eva_denorm < 0] = 0

target_cal_denorm = target_cal_seq * scaler_q.scale_[0] + scaler_q.mean_[0]
target_eva_denorm = target_eva_seq * scaler_q.scale_[0] + scaler_q.mean_[0]

print(f"\nPredictions (denormalized):")
print(f"  Cal - Target range: [{target_cal_denorm.min():.2f}, {target_cal_denorm.max():.2f}]")
print(f"  Cal - Pred range: [{predict_cal_denorm.min():.2f}, {predict_cal_denorm.max():.2f}]")
print(f"  Eva - Target range: [{target_eva_denorm.min():.2f}, {target_eva_denorm.max():.2f}]")
print(f"  Eva - Pred range: [{predict_eva_denorm.min():.2f}, {predict_eva_denorm.max():.2f}]")

# Calculate NSE
def nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)

nse_cal = nse(target_cal_denorm, predict_cal_denorm)
nse_eva = nse(target_eva_denorm, predict_eva_denorm)

print(f"\nNSE scores:")
print(f"  Calibration: {nse_cal:.4f}")
print(f"  Evaluation: {nse_eva:.4f}")
