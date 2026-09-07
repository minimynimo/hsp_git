import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

loc = "US"

# Regional HP results live under the canonical ealstm_regional_modeling run tree, not 0_out
# (see ANALYSIS_README.md's LSTM section) -- each h/d/lr/sl combo directory holds several
# run_<timestamp>_seed<N> subdirs, one per batch_size (read from that run's cfg.json).
LSTM_REGIONAL_HP_ROOT = f"/data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/runs/hp/{loc}"

# Updated output directory for LSTM
output_dir = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/hyperparameter_analysis/LSTM/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

benchmark_limit = {
    "KGE": (-0.5, 1.0),
    "NSE": (-0.5, 1.0),
    "RMSE": (0, 500),
    "MAE": (0, 300)
}

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() 

    # --- 1. Define Hyperparameters based on model type ---
    # Full grid per HYPERPARAMETERS.md.
    if MODEL_TYPE == "regional":
        HIDDEN_SIZES = [64, 128, 256]
        DROPOUTS = [0.2, 0.4]
        LEARNING_RATES = [0.001, 0.0005, 0.0001]
        SEQ_LENGTHS = [270, 365]
        BATCH_SIZES = [256, 512]
    elif MODEL_TYPE == "individual":
        HIDDEN_SIZES = [16, 32, 64, 128]
        DROPOUTS = [0.4, 0.5]
        LEARNING_RATES = [0.001, 0.0005]
        SEQ_LENGTHS = [270, 365]
        BATCH_SIZES = [64, 128]

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for HIDDEN_SIZE in HIDDEN_SIZES:
        for DROPOUT in DROPOUTS:
            for LEARNING_RATE in LEARNING_RATES:
                for SEQ_LENGTH in SEQ_LENGTHS:
                    for BATCH_SIZE in BATCH_SIZES:

                        if MODEL_TYPE == "regional":
                            # Canonical location: runs/hp/US/h{H}_d{D}_lr{LR}_sl{SL}/run_*/,
                            # one run_* subdir per (batch_size, seed); batch_size isn't in the
                            # directory name so it has to be read from each run's cfg.json.
                            combo_dir = (
                                f"{LSTM_REGIONAL_HP_ROOT}/h{HIDDEN_SIZE}_d{DROPOUT}_lr{LEARNING_RATE}_sl{SEQ_LENGTH}"
                            )
                            if not os.path.isdir(combo_dir):
                                continue

                            matched_run = None
                            for run_dir in sorted(glob.glob(os.path.join(combo_dir, "run_*"))):
                                cfg_path = os.path.join(run_dir, "cfg.json")
                                if not os.path.exists(cfg_path):
                                    continue
                                try:
                                    with open(cfg_path) as f:
                                        cfg = json.load(f)
                                except Exception:
                                    continue
                                if cfg.get("batch_size") == BATCH_SIZE:
                                    matched_run = run_dir  # keep the latest matching run

                            if matched_run is None:
                                continue

                            input_path = os.path.join(matched_run, "hp_metrics_eva.csv")

                        elif MODEL_TYPE == "individual":
                            # Pattern: /data0/funato/hsp2/0_out/LSTM/US/individual/hp/h16_d0.4_lr0.001_sl270_b64/results/LSTM_results_h16_lr0.001_e20_w270_b64_d0.4_eva.csv
                            input_path = (
                                f"/data0/funato/hsp2/0_out/LSTM/{loc}/{MODEL_TYPE}/hp/"
                                f"h{HIDDEN_SIZE}_d{DROPOUT}_lr{LEARNING_RATE}_sl{SEQ_LENGTH}_b{BATCH_SIZE}/results/"
                                f"LSTM_results_h{HIDDEN_SIZE}_lr{LEARNING_RATE}_e20_w{SEQ_LENGTH}_b{BATCH_SIZE}_d{DROPOUT}_eva.csv"
                            )

                        # Check existence
                        if not os.path.exists(input_path):
                            continue

                        try:
                            temp_df = pd.read_csv(input_path, index_col=0)

                            # Strip whitespace from column names
                            temp_df.columns = temp_df.columns.str.strip()

                            # Add hyperparameter columns for grouping later
                            temp_df["HIDDEN_SIZE"] = HIDDEN_SIZE
                            temp_df["DROPOUT"] = DROPOUT
                            temp_df["LEARNING_RATE"] = LEARNING_RATE
                            temp_df["SEQ_LENGTH"] = SEQ_LENGTH
                            temp_df["BATCH_SIZE"] = BATCH_SIZE

                            df = pd.concat([df, temp_df], axis=0)

                        except Exception as e:
                            print(f"Error reading {input_path}: {e}")

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    param_cols = ["HIDDEN_SIZE", "DROPOUT", "LEARNING_RATE", "SEQ_LENGTH", "BATCH_SIZE"]

    # Aggregate per-basin results down to one row per hyperparameter combination: median/mean
    # of KGE, NSE, RMSE across basins, ranked from highest to lowest median NSE.
    agg_metrics = ["KGE", "NSE", "RMSE"]
    metric_cols = {}
    for m in agg_metrics:
        for col in df.columns:
            if m in col:
                metric_cols[m] = col
                break

    agg_kwargs = {}
    for m in agg_metrics:
        if m in metric_cols:
            agg_kwargs[f"{m}_median"] = (metric_cols[m], "median")
            agg_kwargs[f"{m}_mean"] = (metric_cols[m], "mean")
    agg_kwargs["n_basins"] = (metric_cols[agg_metrics[0]], "count")

    agg_df = df.groupby(param_cols).agg(**agg_kwargs).reset_index()
    agg_df = agg_df.sort_values("NSE_median", ascending=False)

    output_path = os.path.join(output_dir, f"LSTM_{loc}_{MODEL_TYPE}_hp_eva.csv")
    agg_df.to_csv(output_path, index=False)
    print(f"Saved per-combination median/mean results (ranked by NSE median) to {output_path}")

    # --- CDF per hyperparameter combination ---
    # Every combination gets its own empirical CDF line (thin grey, unlabeled) so the spread
    # across the whole grid is visible at a glance; the top 5 combinations by median NSE (from
    # agg_df above) are drawn in color and labeled with their hyperparameter values.
    top5 = agg_df.head(5)
    top5_keys = [tuple(row[c] for c in param_cols) for _, row in top5.iterrows()]
    top5_colors = plt.cm.tab10.colors[:5]

    def fmt_label(key):
        h, d, lr, sl, b = key
        return f"h={h}, d={d}, lr={lr}, sl={sl}, b={b}"

    for benchmark in benchmark_list:
        target_col = None
        for col in df.columns:
            if benchmark in col:
                target_col = col
                break

        if target_col is None:
            print(f"  Warning: Could not find column containing '{benchmark}'. Available: {df.columns.tolist()}")
            continue

        plt.figure(figsize=(10, 7))

        for name, group in df.groupby(param_cols):
            values = np.sort(group[target_col].dropna().values)
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values)

            if name in top5_keys:
                rank = top5_keys.index(name)
                plt.plot(values, y, drawstyle='steps-post', color=top5_colors[rank],
                          linewidth=2.2, label=f"#{rank+1}: {fmt_label(name)}", zorder=5)
            else:
                plt.plot(values, y, drawstyle='steps-post', color='gray', alpha=0.15,
                          linewidth=0.7, zorder=1)

        plt.title(f"CDF of {benchmark} across hyperparameter combinations -- {MODEL_TYPE.capitalize()} LSTM")
        plt.xlabel(benchmark)
        plt.ylabel("Cumulative probability (across basins)")
        if benchmark in ("KGE", "NSE"):
            plt.xlim(-1, 1)
        else:
            plt.xlim(benchmark_limit[benchmark])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Top 5 (by NSE median)", fontsize=7, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"LSTM_{loc}_{MODEL_TYPE}_{benchmark}_cdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF plot to {plot_path}")

print("Hyperparameter analysis completed.")