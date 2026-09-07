"""
XGBoost Regional SHAP Analysis -- explains the REAL trained checkpoint.

Rewritten per DEBUG_README.md's XGBoost audit (following the identical fix already applied to
LSTM (DEBUG_README.md #7) and TCN (DEBUG_README.md #14.6)): the previous version of this script
did not load any trained checkpoint at all. It trained a brand-new XGBoost model from scratch,
using pooled/global normalization (no per-basin scaling) and no static attributes, purely to run
SHAP on that throwaway model. So the reported SHAP values described a different model -- trained
differently, normalized differently -- than the checkpoint whose NSE/KGE numbers get benchmarked.
(The same throwaway-training bug also existed, undocumented as SHAP, in main_xgboost_regional.py
and main_xgboost_shap.py -- both renamed with an `x_` prefix; see DEBUG_README.md.)

This version:
  - Loads the actual trained xgb_model.json + cfg.json + scalers.json from a real
    XGB_regional_modeling-master/runs_xgb/<loc>/xgb_run_*/ directory (same convention as
    eval_regional.py's --run_dir).
  - Reuses that run's real per-basin scalers (scalers['basin_scalers']) for dynamic inputs/target,
    and the same global static-attribute scaler + VALID_ATTR list used in training -- NOT a
    fresh global pooled scaler.
  - Builds feature vectors with the exact same flattening convention as main_xgb.py/eval_regional.py
    (seq_length x 3 dynamic lags, flattened, then static attributes appended), so column i of X here
    is guaranteed to mean the same thing as column i during training/eval.
  - Uses shap.TreeExplainer directly on the loaded xgb.Booster -- exact SHAP values for tree
    ensembles, no throwaway surrogate, no background-sample approximation needed (unlike
    GradientExplainer/KernelExplainer used for the neural-net families).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Same static-attribute set as main_xgb.py / eval_regional.py -- must match whatever the
# checkpoint being explained was actually trained on.
sys.path.insert(0, str(Path(__file__).parent / "XGB_regional_modeling-master"))
from papercode.datautils import load_attributes, VALID_ATTR

DYN_COLS = ['Precip', 'PET', 'Temp']
Q_COL = 'Obs flow'


def load_data_for_basin(loc, gauge_id):
    varssim_dir = f"/data0/funato/hsp2/2_MERV/varssim_{loc}"
    df = pd.read_csv(f"{varssim_dir}/varssim_{gauge_id}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def build_feature_names(seq_length, no_static, static_cols):
    """Must exactly mirror the flatten order used in main_xgb.py/eval_regional.py: for each
    timestep (oldest -> newest), all dyn_cols in order, then (if static) the static attributes.

    static_cols must be the *actual* columns load_attributes() returned for this location, not
    just VALID_ATTR verbatim -- some locations (e.g. GB) only have a subset of the VALID_ATTR
    columns available in their attribute source data (same data-availability gap documented for
    LSTM in DEBUG_README.md #2), so the real static feature count can be smaller than
    len(VALID_ATTR)."""
    names = []
    for t in range(seq_length):
        lag = seq_length - 1 - t
        for var in DYN_COLS:
            names.append(f"{var}_t-{lag}")
    if not no_static:
        names.extend(static_cols)
    return names


def build_samples(run_dir, cfg, scalers, basins, n_samples_per_basin, seed):
    """Build a random subsample of (basin, date) feature vectors from the calibration period,
    using the exact per-basin-scaled feature pipeline the checkpoint was trained with."""
    rng = np.random.default_rng(seed)
    loc = cfg['loc']
    seq_length = cfg['seq_length']
    no_static = cfg.get('no_static', True)
    train_start = pd.to_datetime(cfg['train_start'])
    train_end = pd.to_datetime(cfg['train_end'])

    attributes = None
    static_cols = []
    X_rows = []
    meta = []

    # Static attributes: read straight from the run's own attributes.db (same DB the checkpoint's
    # training used), via the regional package's loader.
    if not no_static:
        db_path = str(run_dir / 'attributes.db')
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True, keep_features=VALID_ATTR)
        static_cols = list(attributes.columns)
        static_means = np.array(scalers['static_means'])
        static_stds = np.array(scalers['static_stds'])

    basin_scalers = scalers['basin_scalers']

    for basin in basins:
        if basin not in basin_scalers:
            continue
        df = load_data_for_basin(loc, basin)
        df_period = df.loc[train_start:train_end]
        bscaler = basin_scalers[basin]
        dyn_data_scaled = (df_period[DYN_COLS].values - np.array(bscaler['dynamic_means'])) / np.array(bscaler['dynamic_stds'])

        is_nan = df_period[DYN_COLS + [Q_COL]].isnull().any(axis=1).values
        clumps = np.ma.clump_unmasked(np.ma.masked_array(is_nan, is_nan))
        if not isinstance(clumps, list):
            clumps = [clumps] if clumps else []

        candidate_targets = []
        for s in clumps:
            start_loc, end_loc = s.start, s.stop
            block_len = end_loc - start_loc
            if block_len < seq_length:
                continue
            for i in range(block_len - seq_length + 1):
                candidate_targets.append((start_loc + i, start_loc + i + seq_length - 1))

        if not candidate_targets:
            continue

        n_pick = min(n_samples_per_basin, len(candidate_targets))
        picks = rng.choice(len(candidate_targets), size=n_pick, replace=False)

        static_data_scaled = None
        if not no_static:
            static_data = attributes.loc[basin].values
            static_data_scaled = (static_data - static_means) / static_stds

        for p in picks:
            i, target_idx = candidate_targets[p]
            dyn_seq_flat = dyn_data_scaled[i:target_idx + 1].flatten()
            final_features = dyn_seq_flat
            if not no_static:
                final_features = np.concatenate([dyn_seq_flat, static_data_scaled])
            X_rows.append(final_features)
            meta.append({'basin': basin, 'date': df_period.index[target_idx]})

    return np.array(X_rows, dtype=np.float32), meta, static_cols


def main():
    parser = argparse.ArgumentParser(description='Regional XGBoost SHAP analysis on a real trained checkpoint')
    parser.add_argument('--run_dir', type=str, required=True,
                         help='Path to a real XGB_regional_modeling-master/runs_xgb/<loc>/xgb_run_* directory')
    parser.add_argument('--n_basins', type=int, default=30, help='Number of basins to sample explained points from')
    parser.add_argument('--n_samples_per_basin', type=int, default=5, help='Explained (basin, date) samples per basin')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print(f"Loading checkpoint from: {run_dir}")

    with open(run_dir / 'cfg.json') as fp:
        cfg = json.load(fp)
    with open(run_dir / 'scalers.json') as fp:
        scalers = json.load(fp)
    if 'basin_scalers' not in scalers:
        raise ValueError(
            f"{run_dir / 'scalers.json'} has no 'basin_scalers' key -- this run predates the "
            "per-basin-scaling fix and cannot be explained consistently with eval_regional.py. "
            "Retrain with the current main_xgb.py."
        )

    model_file = run_dir / "xgb_model.json"
    if not model_file.exists():
        raise ValueError(f"Model file not found: {model_file}")
    booster = xgb.Booster()
    booster.load_model(str(model_file))
    print(f"Loaded real trained checkpoint: {model_file}")

    loc = cfg['loc']
    no_static = cfg.get('no_static', True)
    seq_length = cfg['seq_length']

    # Sample basins from the checkpoint's own scalers['basin_scalers'] keys -- i.e. only basins
    # this specific checkpoint was actually trained on -- not the full location-wide basin
    # universe. A checkpoint may have been trained on a subset (e.g. a smoke test, or a basin
    # dropped for having no clean data), and explaining a basin the model never saw would silently
    # explain nothing (0 samples) rather than error.
    all_basins = list(scalers['basin_scalers'].keys())
    if not all_basins:
        raise ValueError(f"{run_dir / 'scalers.json'} has no basins in basin_scalers.")

    rng = np.random.default_rng(args.seed)
    sample_basins = list(rng.choice(all_basins, size=min(args.n_basins, len(all_basins)), replace=False))

    print(f"Building explained samples from {len(sample_basins)} basins "
          f"(no_static={no_static}, seq_length={seq_length})...")
    X_explain, meta, static_cols = build_samples(run_dir, cfg, scalers, sample_basins, args.n_samples_per_basin, args.seed)
    print(f"Explained sample matrix: {X_explain.shape}")
    if not no_static:
        print(f"Static attribute columns actually available for loc={loc}: {len(static_cols)}/{len(VALID_ATTR)} "
              f"({static_cols})")

    feature_names = build_feature_names(seq_length, no_static, static_cols)
    assert X_explain.shape[1] == len(feature_names), (
        f"Feature count mismatch: X has {X_explain.shape[1]} columns, "
        f"expected {len(feature_names)} from build_feature_names(). "
        "This means the SHAP feature pipeline has drifted from main_xgb.py's training pipeline."
    )
    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)

    print("Running shap.TreeExplainer on the real checkpoint (exact SHAP for tree ensembles)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_explain_df)

    # Sanity check: TreeExplainer's additive property -- base_value + sum(shap) should reconstruct
    # the model's own raw prediction (up to float tolerance). Verified every run, not assumed.
    raw_pred = booster.predict(xgb.DMatrix(X_explain_df))
    reconstructed = explainer.expected_value + shap_values.sum(axis=1)
    max_err = np.max(np.abs(raw_pred - reconstructed))
    print(f"SHAP additivity check: max |raw_pred - (base_value + sum(shap))| = {max_err:.6g}")
    if max_err > 1e-2:
        raise RuntimeError(
            f"SHAP additivity check failed (max_err={max_err:.4g}) -- explainer may not be "
            "wired to the same model actually being predicted from."
        )
    print(f"Finite SHAP values: {np.isfinite(shap_values).all()}")

    out_base = Path(f"/data0/funato/hsp2/0_out/XGB/{loc}/regional/shap")
    run_tag = run_dir.name
    shap_objects_dir = out_base / "shap_objects" / run_tag
    shap_objects_dir.mkdir(parents=True, exist_ok=True)
    np.save(shap_objects_dir / "shap_values.npy", shap_values)
    np.save(shap_objects_dir / "expected_value.npy", np.array(explainer.expected_value))
    X_explain_df.to_csv(shap_objects_dir / "X_sample.csv")
    pd.DataFrame(meta).to_csv(shap_objects_dir / "sample_meta.csv", index=False)
    print(f"SHAP objects saved to {shap_objects_dir}")

    fig_dir = Path(f"0_out/1_fig/shap/XGB/{loc}/regional_checkpoint") / run_tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate the per-lag dynamic SHAP into per-variable importance (270 lags x 3 vars is too
    # dense to read directly) in addition to the raw per-feature summary plot.
    mean_abs_by_var = {}
    for var in DYN_COLS:
        cols = [c for c in feature_names if c.startswith(f"{var}_t-")]
        idx = [feature_names.index(c) for c in cols]
        mean_abs_by_var[var] = np.abs(shap_values[:, idx]).mean()
    if not no_static:
        for var in static_cols:
            idx = feature_names.index(var)
            mean_abs_by_var[var] = np.abs(shap_values[:, idx]).mean()

    plt.figure(figsize=(8, max(4, 0.3 * len(mean_abs_by_var))))
    items = sorted(mean_abs_by_var.items(), key=lambda kv: kv[1])
    plt.barh([k for k, _ in items], [v for _, v in items], color='#9C4803')
    plt.xlabel("mean |SHAP value| (aggregated over lags for dynamic inputs)")
    plt.title(f"XGBoost regional SHAP importance -- {run_tag}")
    plt.tight_layout()
    plt.savefig(fig_dir / "XGB_shap_regional_importance.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_explain_df, plot_type="bar", show=False, max_display=20)
    plt.savefig(fig_dir / "XGB_shap_regional_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"SHAP plots saved to {fig_dir}")
    print("Dynamic-input aggregated importance:", {k: round(v, 4) for k, v in mean_abs_by_var.items() if k in DYN_COLS})
    print("DONE")


if __name__ == "__main__":
    main()
