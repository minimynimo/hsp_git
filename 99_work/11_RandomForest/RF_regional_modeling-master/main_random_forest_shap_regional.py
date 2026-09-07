"""
Regional Random Forest SHAP script.

Explains the ACTUAL trained regional Random Forest model produced by
`RF_regional_modeling-master/main_rf.py` (loaded via --run_dir), using the exact same
270-day flattened dynamic-input window + per-basin-scaled static attributes that model
was trained on (reusing `eval_regional.py`'s `load_data_for_basin`/`normalize_basin_data`/
`create_sequences_rf` directly, so there is no duplicated/diverging feature-construction
logic). This replaces an earlier version of this script that trained its own separate,
differently-featured (lag-1, not 270-day window) throwaway model purely to run SHAP on --
those SHAP values described a different model than the one whose NSE/KGE benchmark numbers
are reported by eval_regional.py.

Usage:
    python main_random_forest_shap_regional.py --run_dir path/to/RF_regional_modeling-master/runs_rf/<loc>/rf_run_.../
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import shap

sys.path.insert(0, str(Path(__file__).parent / "RF_regional_modeling-master"))
from eval_regional import load_data_for_basin, normalize_basin_data, create_sequences_rf
from papercode.datautils import load_attributes, VALID_ATTR


def build_basin_samples(basin, period_start, period_end, run_cfg, basin_scalers, static_scaled):
    """Build the model's raw input feature matrix for one basin/period (same convention
    main_rf.py trained on: flattened seq_length-day window of [Precip, PET, Temp] + static)."""
    df = load_data_for_basin(basin)
    warmup_start = period_start - pd.DateOffset(days=run_cfg["seq_length"] - 1)
    period_data = df[warmup_start:period_end].copy()
    if len(period_data) < run_cfg["seq_length"]:
        return None, None, None

    input_cols = ['Precip', 'PET', 'Temp', 'Obs flow']
    input_data = period_data[input_cols].values

    bscaler = basin_scalers[basin]
    basin_stats = {
        'mean': np.concatenate([bscaler['dynamic_means'], [bscaler['q_mean']]]),
        'std': np.concatenate([bscaler['dynamic_stds'], [bscaler['q_std']]])
    }
    input_data_norm = normalize_basin_data(input_data, basin_stats)

    sequences, _ = create_sequences_rf(input_data_norm, run_cfg["seq_length"], static_scaled)
    dates = period_data.index[run_cfg["seq_length"]:]

    mask = dates >= period_start
    return sequences[mask], dates[mask], basin_stats


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis on a trained regional RF checkpoint")
    parser.add_argument('--run_dir', type=str, required=True, help='RF_regional_modeling-master run_dir to explain')
    parser.add_argument('--n_basins', type=int, default=10, help='Number of basins to sample for the SHAP explanation set')
    parser.add_argument('--n_dates_per_basin', type=int, default=5, help='Number of evenly-spaced eval-period dates per basin')
    parser.add_argument('--background_basins', type=int, default=30, help='Number of basins to draw the background set from')
    parser.add_argument('--background_size', type=int, default=50, help='Number of background (training-period) samples for SHAP')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / 'cfg.json') as fp:
        run_cfg = json.load(fp)
    run_cfg['run_dir'] = run_dir
    loc = run_cfg['loc']

    with open(run_dir / 'scalers.json') as fp:
        scalers = json.load(fp)
    if 'basin_scalers' not in scalers:
        raise ValueError(f"{run_dir}/scalers.json has no 'basin_scalers' -- this run_dir was not trained with a per-basin-scaling main_rf.py")
    basin_scalers = {b: {k: (np.array(v) if isinstance(v, list) else v) for k, v in bs.items()}
                      for b, bs in scalers['basin_scalers'].items()}

    model = joblib.load(run_dir / "rf_model.joblib")
    print(f"Loaded model from {run_dir / 'rf_model.joblib'}")

    no_static = run_cfg.get("no_static", False)
    static_means = np.array(scalers['static_means']) if 'static_means' in scalers else None
    static_stds = np.array(scalers['static_stds']) if 'static_stds' in scalers else None

    static_attrs_scaled = {}
    if not no_static:
        db_path = run_dir / "attributes.db"
        basins_available = list(basin_scalers.keys())
        attrs = load_attributes(db_path=db_path, basins=basins_available,
                                 drop_lat_lon=True, keep_features=VALID_ATTR, loc=loc)
        for b in basins_available:
            if b in attrs.index:
                static_attrs_scaled[b] = (attrs.loc[b].values - static_means) / static_stds

    rng = np.random.default_rng(args.seed)
    all_basins = [b for b in basin_scalers if no_static or b in static_attrs_scaled]

    dyn_feature_names = [f"{var}_lag{lag}" for lag in range(run_cfg["seq_length"] - 1, -1, -1) for var in ["Precip", "PET", "Temp"]]
    feature_names = dyn_feature_names if no_static else dyn_feature_names + VALID_ATTR

    val_start = pd.to_datetime(run_cfg.get("val_start", "2001-01-01"))
    val_end = pd.to_datetime(run_cfg.get("val_end", "2006-12-31"))
    train_start = pd.to_datetime(run_cfg.get("train_start", "1990-01-01"))
    train_end = pd.to_datetime(run_cfg.get("train_end", "2000-12-31"))

    # --- Explanation set: n_basins x n_dates_per_basin evenly-spaced validation dates ---
    print(f"Building explanation set: {args.n_basins} basins x {args.n_dates_per_basin} dates...")
    explain_basins = rng.choice(all_basins, size=min(args.n_basins, len(all_basins)), replace=False)
    X_explain_list = []
    for basin in explain_basins:
        static_scaled = None if no_static else static_attrs_scaled[basin]
        sequences, dates, _ = build_basin_samples(basin, val_start, val_end, run_cfg, basin_scalers, static_scaled)
        if sequences is None or len(sequences) == 0:
            continue
        idx = np.linspace(0, len(sequences) - 1, min(args.n_dates_per_basin, len(sequences))).astype(int)
        X_explain_list.append(sequences[idx])
    X_explain = np.vstack(X_explain_list)
    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)
    print(f"Explanation set: {X_explain_df.shape}")

    # --- Background set: training-period samples drawn from a wider basin pool ---
    print(f"Building background set from {args.background_basins} basins...")
    background_basins = rng.choice(all_basins, size=min(args.background_basins, len(all_basins)), replace=False)
    X_bg_list = []
    for basin in background_basins:
        static_scaled = None if no_static else static_attrs_scaled[basin]
        sequences, dates, _ = build_basin_samples(basin, train_start, train_end, run_cfg, basin_scalers, static_scaled)
        if sequences is None or len(sequences) == 0:
            continue
        X_bg_list.append(sequences)
    X_bg = np.vstack(X_bg_list)
    if len(X_bg) > args.background_size:
        bg_idx = rng.choice(len(X_bg), size=args.background_size, replace=False)
        X_bg = X_bg[bg_idx]
    X_bg_df = pd.DataFrame(X_bg, columns=feature_names)
    print(f"Background set: {X_bg_df.shape}")

    print("Verifying explained model matches the loaded checkpoint (faithfulness check)...")
    direct_pred = model.predict(X_explain_df.values)

    print("Running SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model, data=X_bg_df, feature_perturbation="interventional")
    shap_values = explainer(X_explain_df)

    shap_pred_check = shap_values.base_values + shap_values.values.sum(axis=1)
    max_err = np.max(np.abs(shap_pred_check - direct_pred))
    print(f"Max |SHAP-reconstructed prediction - direct model prediction| = {max_err:.6g}")

    output_dir = Path(f"/data0/funato/hsp2/0_out/RF/{loc}/regional/shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "shap_values.npy", shap_values.values)
    np.save(output_dir / "expected_value.npy", shap_values.base_values)
    X_explain_df.to_csv(output_dir / "X_explained_features.csv", index=False)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    summary = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
    summary = summary.sort_values('mean_abs_shap', ascending=False)
    summary.to_csv(output_dir / "shap_feature_importance_summary.csv", index=False)

    plot_dir = Path(f"/data0/funato/hsp2/0_out/1_fig/shap/RF/{loc}/regional")
    plot_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_explain_df, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(plot_dir / "RF_shap_regional_summary_bar.png")
    plt.close()

    shap.summary_plot(shap_values, X_explain_df, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(plot_dir / "RF_shap_regional_summary.png")
    plt.close()

    # Per-timestep-lag importance aggregated by dynamic variable
    if run_cfg["seq_length"] * 3 <= shap_values.values.shape[1]:
        dyn_shap = shap_values.values[:, :run_cfg["seq_length"] * 3].reshape(-1, run_cfg["seq_length"], 3)
        lag_importance = np.abs(dyn_shap).mean(axis=0)  # (seq_length, 3)
        fig, ax = plt.subplots(figsize=(10, 5))
        lags = np.arange(run_cfg["seq_length"] - 1, -1, -1)
        for i, var in enumerate(["Precip", "PET", "Temp"]):
            ax.plot(lags, lag_importance[:, i], label=var)
        ax.set_xlabel("Lag (days before prediction)")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(f"Per-timestep-lag SHAP importance ({loc} regional RF)")
        ax.invert_xaxis()
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "RF_shap_regional_lag_importance.png")
        plt.close()

    print(f"SHAP outputs saved to {output_dir} and {plot_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
