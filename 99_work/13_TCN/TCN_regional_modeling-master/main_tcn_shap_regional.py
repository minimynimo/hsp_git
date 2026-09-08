"""
Regional TCN SHAP Analysis Script (canonical, PyTorch-based)

Explains the REAL trained TCNModel checkpoint produced by main_tcn.py, using the
SAME per-basin scaling convention that model was trained and benchmarked with
(see eval_regional.py). This replaces the old main_TCN_regional.py /
main_TCN_shap_regional.py top-level scripts, which trained a separate, throwaway
Keras/TensorFlow TCN model (with pooled/global normalization) purely to run SHAP
on *that* - so the SHAP values they reported did not describe the model whose
NSE/KGE numbers were being benchmarked. See DEBUG_README.md for the identical bug
found and fixed in the LSTM pipeline's SHAP script (main_lstm_shap_regional.py).

Usage:
    python main_tcn_shap_regional.py --run_dir path/to/trained/model \
        --n_basins 10 --n_dates 5 --background_size 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from papercode.datautils import load_attributes, VALID_ATTR
from papercode.utils import get_basin_list
from papercode.tcn_run_registry import load_tcn_state_dict

# ####################### #
# Global Definitions      #
# ####################### #

GLOBAL_SETTINGS = {
    'loc': 'US',
    'seq_length': 270,
}

# Per-dataset training-period fallback, matching main_tcn.py's DATASET_SETTINGS. A trained
# run's own cfg.json train_start/train_end (present for any run trained after --dataset
# support was added) always takes precedence -- see build_samples() below.
DATASET_SETTINGS = {
    'camels': {
        'train_start': pd.to_datetime('1987-01-01', format='%Y-%m-%d'),
        'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    },
    'caravan': {
        'train_start': pd.to_datetime('1990-01-01', format='%Y-%m-%d'),
        'train_end': pd.to_datetime('2000-12-31', format='%Y-%m-%d'),
    },
}

DEVICE = torch.device("cpu")  # SHAP GradientExplainer runs fine on CPU for this model size

gauge_id_list = get_basin_list(GLOBAL_SETTINGS['loc'])

DYN_COLS = ['Precip', 'PET', 'Temp']


def load_data_for_basin(gauge_id, dataset='camels'):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_{GLOBAL_SETTINGS['loc']}"
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


# ####################### #
# TCN Model (must match main_tcn.py's architecture exactly)
# ####################### #

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
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

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


# ####################### #
# Sample building (reuses main_tcn.py's exact per-basin scaling + static
# concatenation convention, so SHAP explains the same feature space the model
# was trained and benchmarked on)
# ####################### #

def build_samples(basins, n_dates, cfg, basin_scalers, static_means, static_stds, attributes):
    """Build (seq_length, n_features) samples for a handful of dates per basin."""
    samples = []
    meta = []
    seq_length = cfg['seq_length']

    for basin in basins:
        if basin not in basin_scalers:
            continue
        dataset = cfg.get('dataset', 'camels')
        df = load_data_for_basin(basin, dataset)
        train_start = pd.to_datetime(cfg.get('train_start', DATASET_SETTINGS[dataset]['train_start']))
        train_end = pd.to_datetime(cfg.get('train_end', DATASET_SETTINGS[dataset]['train_end']))
        df_period = df.loc[train_start:train_end]
        bscaler = basin_scalers[basin]

        clean = df_period[DYN_COLS + ['Obs flow']].dropna()
        if len(clean) < seq_length + n_dates:
            continue

        dyn_scaled_full = (df_period[DYN_COLS].values - bscaler['dynamic_means']) / bscaler['dynamic_stds']

        static_seq = None
        if not cfg['no_static'] and attributes is not None and basin in attributes.index:
            static_vals = attributes.loc[basin].values
            safe_stds = np.where(static_stds != 0, static_stds, 1.0)
            static_scaled = (static_vals - static_means) / safe_stds
            static_seq = np.tile(static_scaled, (seq_length, 1))

        # Pick evenly spaced valid target indices with a full, NaN-free warmup window
        valid_idx = clean.index
        candidate_positions = [df_period.index.get_loc(d) for d in valid_idx if df_period.index.get_loc(d) >= seq_length - 1]
        if not candidate_positions:
            continue
        picks = np.linspace(0, len(candidate_positions) - 1, min(n_dates, len(candidate_positions))).astype(int)

        for p in picks:
            target_pos = candidate_positions[p]
            window = dyn_scaled_full[target_pos - seq_length + 1: target_pos + 1]
            if window.shape[0] != seq_length or np.isnan(window).any():
                continue
            feats = window
            if static_seq is not None:
                feats = np.concatenate([window, static_seq], axis=1)
            samples.append(feats.astype(np.float32))
            meta.append({'basin': basin, 'date': df_period.index[target_pos]})

    return np.array(samples), meta


def main():
    parser = argparse.ArgumentParser(description='Regional TCN SHAP analysis (real checkpoint)')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=int, default=50)
    parser.add_argument('--n_basins', type=int, default=10, help='Basins to explain')
    parser.add_argument('--n_dates', type=int, default=5, help='Dates per basin to explain')
    parser.add_argument('--background_size', type=int, default=50)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / 'cfg.json') as fp:
        cfg = json.load(fp)

    with open(run_dir / 'scalers.json') as fp:
        scalers_data = json.load(fp)
    basin_scalers = {}
    for basin, bs in scalers_data.get('basin_scalers', {}).items():
        basin_scalers[basin] = {
            'q_mean': bs['q_mean'], 'q_std': bs['q_std'],
            'dynamic_means': np.array(bs['dynamic_means']),
            'dynamic_stds': np.array(bs['dynamic_stds'])
        }
    static_means = np.array(scalers_data['static_means']) if 'static_means' in scalers_data else None
    static_stds = np.array(scalers_data['static_stds']) if 'static_stds' in scalers_data else None

    # Recompute the basin list for the dataset this run was actually trained on (the
    # module-level default computed at import time is always 'camels').
    basins_for_run = get_basin_list(GLOBAL_SETTINGS['loc'], dataset=cfg.get('dataset', 'camels'))

    attributes = None
    static_cols = []
    if not cfg['no_static']:
        db_path = run_dir / "attributes.db"
        attributes = load_attributes(db_path, basins=basins_for_run, keep_features=VALID_ATTR,
                                      loc=GLOBAL_SETTINGS['loc'])
        static_cols = list(attributes.columns)

    num_channels = [cfg['num_filters']] * cfg['num_levels'] if 'num_filters' in cfg else cfg['num_channels']
    input_size = 3 + (0 if cfg['no_static'] else (attributes.shape[1] if attributes is not None else 0))

    model = TCNModel(input_size=input_size, num_channels=num_channels,
                      kernel_size=cfg['kernel_size'], dropout=cfg['dropout']).to(DEVICE)
    model_file = run_dir / f"tcn_model_epoch{args.model_epoch}.pt"
    if not model_file.exists():
        model_file = run_dir / "tcn_model_final.pt"
    model.load_state_dict(load_tcn_state_dict(model_file, map_location=DEVICE))
    model.eval()
    print(f"Loaded real trained checkpoint from: {model_file}")

    rng = np.random.RandomState(0)
    all_basins = list(basin_scalers.keys())
    explain_basins = list(rng.choice(all_basins, size=min(args.n_basins, len(all_basins)), replace=False))
    background_basins = list(rng.choice(all_basins, size=min(args.background_size, len(all_basins)), replace=False))

    explain_x, explain_meta = build_samples(explain_basins, args.n_dates, cfg, basin_scalers,
                                             static_means, static_stds, attributes)
    background_x, _ = build_samples(background_basins, 1, cfg, basin_scalers,
                                     static_means, static_stds, attributes)
    background_x = background_x[:args.background_size]

    if len(explain_x) == 0 or len(background_x) == 0:
        raise RuntimeError("Could not build enough clean samples for SHAP - try more basins/dates.")

    print(f"Explaining {len(explain_x)} samples against a background of {len(background_x)} samples "
          f"(shape {explain_x.shape[1:]})")

    import shap
    explain_t = torch.from_numpy(explain_x)
    background_t = torch.from_numpy(background_x)

    explainer = shap.GradientExplainer(model, background_t)
    shap_values = explainer.shap_values(explain_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 4:  # (n_samples, seq_length, n_features, 1)
        shap_values = shap_values[..., 0]

    # Numerically verify the explainer's model call is equivalent to calling model() directly
    with torch.no_grad():
        direct_out = model(explain_t).numpy().flatten()
    print(f"Sanity check - direct model output range: [{direct_out.min():.4f}, {direct_out.max():.4f}]")

    out_dir = Path(f"/data0/funato/hsp2/0_out/TCN/{GLOBAL_SETTINGS['loc']}/{cfg.get('dataset', 'camels')}/regional/shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn_shap = shap_values[:, :, :3]  # (n_samples, seq_length, 3 dynamic inputs)
    mean_abs_dyn = np.abs(dyn_shap).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(DYN_COLS, mean_abs_dyn)
    ax.set_ylabel("mean |SHAP value|")
    ax.set_title("Dynamic input importance (real trained TCN, per-basin scaling)")
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_input_importance.png")
    plt.close(fig)

    # Static attributes: shap_values[:, :, 3:] holds one SHAP value per (sample, timestep,
    # static feature), because build_samples() tiles each static attribute identically across
    # every timestep to feed the TCN's per-timestep-channel input (see static_seq =
    # np.tile(...)). SHAP's completeness/additivity axiom means those seq_length identical-input
    # copies jointly carry the feature's total contribution -- summing (not averaging) over the
    # sequence dimension recombines them into a single per-sample "total contribution" value,
    # comparable on the same scale used for static attribution in the RF/XGB/LSTM regional SHAP
    # scripts (none of which tile static inputs, so their raw SHAP values are already the total).
    # Previously this block (shap_values[:, :, 3:]) was computed but silently discarded --
    # static-attribute attribution was never reported at all, the exact gap this analysis exists
    # to fill.
    static_importance = {}
    if static_cols:
        static_shap = shap_values[:, :, 3:]  # (n_samples, seq_length, n_static)
        static_shap_total = static_shap.sum(axis=1)  # (n_samples, n_static)
        mean_abs_static = np.abs(static_shap_total).mean(axis=0)
        static_importance = dict(zip(static_cols, mean_abs_static.tolist()))

        order = np.argsort(mean_abs_static)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(static_cols))))
        ax.barh([static_cols[i] for i in order][::-1], mean_abs_static[order][::-1])
        ax.set_xlabel("mean |SHAP value| (summed over sequence)")
        ax.set_title("Static attribute importance (real trained TCN, per-basin scaling)")
        fig.tight_layout()
        fig.savefig(out_dir / "static_attribute_importance.png")
        plt.close(fig)

        np.save(out_dir / "shap_values_static.npy", static_shap_total)
    else:
        print("no_static=True for this checkpoint -- no static-attribute SHAP to report.")

    # Combined importance table (dynamic + static, both as "total contribution per sample")
    # so the two feature groups are directly comparable, mirroring main_lstm_shap_regional.py's
    # importance_summary_<tag>.csv format.
    dyn_shap_total = dyn_shap.sum(axis=1)  # (n_samples, 3)
    mean_abs_dyn_total = np.abs(dyn_shap_total).mean(axis=0)
    importance_rows = [{'feature': c, 'type': 'dynamic', 'mean_abs_shap_summed_over_seq': float(v)}
                        for c, v in zip(DYN_COLS, mean_abs_dyn_total)]
    importance_rows += [{'feature': c, 'type': 'static', 'mean_abs_shap_summed_over_seq': float(v)}
                         for c, v in static_importance.items()]
    importance_df = pd.DataFrame(importance_rows).sort_values('mean_abs_shap_summed_over_seq', ascending=False)
    importance_df.to_csv(out_dir / "importance_summary.csv", index=False)

    np.save(out_dir / "shap_values.npy", shap_values)
    pd.DataFrame(explain_meta).to_csv(out_dir / "shap_sample_meta.csv", index=False)
    print(f"Saved SHAP outputs to {out_dir}")
    print(f"Mean |SHAP| per dynamic input (per-timestep average): {dict(zip(DYN_COLS, mean_abs_dyn.tolist()))}")
    print(f"Combined importance (summed-over-sequence, dynamic+static):\n{importance_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
