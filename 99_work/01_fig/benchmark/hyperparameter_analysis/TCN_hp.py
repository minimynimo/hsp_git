import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

loc = "US"
#loc = "GB"

MODEL_TYPES = ["regional", "individual"]

# Updated output directory for TCN
output_dir = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/hyperparameter_analysis/TCN"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

benchmark_limit = {
    "KGE": {'min': -2, 'max': 1.0},
    "NSE": {'min': -1, 'max': 1.0},
    "RMSE": {'min': 0, 'max': 100},
    "MAE": {'min': 0, 'max': 50},   
}

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() 

    # --- 1. Define Hyperparameters based on model type ---
    # Full grid actually swept by run_tcn_hp_tuning.sh (layers/lv is fixed at 6, not swept --
    # see HYPERPARAMETERS.md/ANALYSIS_README.md). Window size (seq length) is also part of the
    # grid; each combination is averaged over here (grouped only by filter/kernel/dropout for
    # the boxplot, matching this script's original grouping).
    if MODEL_TYPE == "regional":
        NUM_FILTERS = [32, 64, 128]
        KERNEL_SIZES = [3, 5, 7]
        DROPOUTS = [0.1, 0.2, 0.3]
        LEARNING_RATES = [0.001, 0.0005]
        WINDOW_SIZES = [90, 180, 270]
    elif MODEL_TYPE == "individual":
        NUM_FILTERS = [16, 32]
        KERNEL_SIZES = [3, 5]
        DROPOUTS = [0.3, 0.4, 0.5]
        LEARNING_RATES = [0.001, 0.0005]
        WINDOW_SIZES = [90, 180, 270]

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for NUM_FILTER in NUM_FILTERS:
        for KERNEL_SIZE in KERNEL_SIZES:
            for DROPOUT in DROPOUTS:
                for LEARNING_RATE in LEARNING_RATES:
                    for WINDOW_SIZE in WINDOW_SIZES:

                        # --- 3. Construct File Paths ---
                        if MODEL_TYPE == "regional":
                            # Pattern: /data0/funato/hsp2/0_out/TCN/US/regional/hp/results/TCN_results_f32_k3_d0.1_lr0.001_lv6_w90_nsFalse_eva.csv
                            input_path = (
                                f"/data0/funato/hsp2/0_out/TCN/{loc}/{MODEL_TYPE}/hp/results/"
                                f"TCN_results_f{NUM_FILTER}_k{KERNEL_SIZE}_d{DROPOUT}_lr{LEARNING_RATE}_lv6_w{WINDOW_SIZE}_nsFalse_eva.csv"
                            )

                        elif MODEL_TYPE == "individual":
                            # Pattern: /data0/funato/hsp2/0_out/TCN/US/individual/hp/f16_k3_d0.3_lr0.001_lv6_w90/results/TCN_results_eva.csv
                            input_path = (
                                f"/data0/funato/hsp2/0_out/TCN/{loc}/{MODEL_TYPE}/hp/"
                                f"f{NUM_FILTER}_k{KERNEL_SIZE}_d{DROPOUT}_lr{LEARNING_RATE}_lv6_w{WINDOW_SIZE}/results/"
                                f"TCN_results_eva.csv"
                            )

                        # Check existence
                        if not os.path.exists(input_path):
                            # print(f"  Skipping (not found): {input_path}")
                            continue

                        try:
                            temp_df = pd.read_csv(input_path, index_col=0)

                            # Strip whitespace from column names
                            temp_df.columns = temp_df.columns.str.strip()

                            # Add hyperparameter columns for grouping later
                            temp_df["NUM_FILTER"] = NUM_FILTER
                            temp_df["KERNEL_SIZE"] = KERNEL_SIZE
                            temp_df["DROPOUT"] = DROPOUT
                            temp_df["LEARNING_RATE"] = LEARNING_RATE
                            temp_df["WINDOW_SIZE"] = WINDOW_SIZE

                            df = pd.concat([df, temp_df], axis=0)

                        except Exception as e:
                            print(f"Error reading {input_path}: {e}")

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    param_cols = ["NUM_FILTER", "KERNEL_SIZE", "DROPOUT", "LEARNING_RATE", "WINDOW_SIZE"]

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

    output_path = os.path.join(output_dir, f"TCN_{loc}_{MODEL_TYPE}_hp_eva.csv")
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
        f, k, d, lr, w = key
        return f"f={f}, k={k}, d={d}, lr={lr}, w={w}"

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
                plt.plot(values, y, drawstyle='steps-post', color='gray', alpha=0.1,
                          linewidth=0.6, zorder=1)

        plt.title(f"CDF of {benchmark} across hyperparameter combinations -- {MODEL_TYPE.capitalize()} TCN")
        plt.xlabel(benchmark)
        plt.ylabel("Cumulative probability (across basins)")
        if benchmark in ("KGE", "NSE"):
            plt.xlim(-1, 1)
        else:
            plt.xlim(benchmark_limit[benchmark]['min'], benchmark_limit[benchmark]['max'])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Top 5 (by NSE median)", fontsize=7, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"TCN_{loc}_{MODEL_TYPE}_{benchmark}_cdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF plot to {plot_path}")

print("Hyperparameter analysis completed.")