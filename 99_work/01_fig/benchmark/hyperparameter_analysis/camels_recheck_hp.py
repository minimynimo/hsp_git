"""
CAMELS HP re-check aggregation: RF individual, TCN individual, LSTM individual, LSTM
regional (the models whose camels HP sweeps have fully finished). RF regional, XGB
regional/individual, TCN regional are handled separately once their sweeps finish.
"""
import pandas as pd
import numpy as np
import os
import json
import glob

loc = "US"
base = "/data0/funato/hsp2"
out_root = f"{base}/0_out/1_fig/{loc}/benchmark/hyperparameter_analysis"

agg_metrics = ["KGE", "NSE", "RMSE"]

def agg_and_rank(df, param_cols, model_name):
    if df.empty:
        print(f"Warning: No data for {model_name}")
        return None
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
    outdir = os.path.join(out_root, model_name)
    os.makedirs(outdir, exist_ok=True)
    agg_df.to_csv(os.path.join(outdir, f"{model_name}_hp_eva.csv"), index=False)
    return agg_df

results = {}

# ---------------- RF individual ----------------
N_ESTIMATORS = [300, 500]
MIN_SAMPLES_LEAF = [10, 20]
MAX_DEPTH = [5, 10]
MAX_FEATURES = ["sqrt", 0.33]
df = pd.DataFrame()
for n in N_ESTIMATORS:
    for l in MIN_SAMPLES_LEAF:
        for d in MAX_DEPTH:
            for f in MAX_FEATURES:
                p = f"{base}/0_out/RF/{loc}/camels/individual/hp/n{n}_d{d}_l{l}_f{f}/results/RF_results_eva.csv"
                if not os.path.exists(p):
                    print("missing RF indiv", p); continue
                t = pd.read_csv(p, index_col=0)
                t.columns = t.columns.str.strip()
                t["N_ESTIMATORS"] = n; t["MIN_SAMPLES_LEAF"] = l; t["MAX_DEPTH"] = d; t["MAX_FEATURES"] = f
                df = pd.concat([df, t], axis=0)
results["RF_individual"] = agg_and_rank(df, ["N_ESTIMATORS","MIN_SAMPLES_LEAF","MAX_DEPTH","MAX_FEATURES"], "RF_individual_camels")

# ---------------- TCN individual ----------------
NUM_FILTERS = [16, 32]
KERNEL_SIZE = [3, 5]
DROPOUT = [0.4, 0.5]
LEARNING_RATE = [0.001, 0.0005]
WINDOW_SIZE = [90, 180]
NUM_LEVELS = 6
df = pd.DataFrame()
for nf in NUM_FILTERS:
    for k in KERNEL_SIZE:
        for dr in DROPOUT:
            for lr in LEARNING_RATE:
                for w in WINDOW_SIZE:
                    p = f"{base}/0_out/TCN/{loc}/camels/individual/hp/f{nf}_k{k}_d{dr}_lr{lr}_lv{NUM_LEVELS}_w{w}/results/TCN_results_eva.csv"
                    if not os.path.exists(p):
                        print("missing TCN indiv", p); continue
                    t = pd.read_csv(p, index_col=0)
                    t.columns = t.columns.str.strip()
                    t["NUM_FILTERS"]=nf; t["KERNEL_SIZE"]=k; t["DROPOUT"]=dr; t["LEARNING_RATE"]=lr; t["WINDOW_SIZE"]=w
                    df = pd.concat([df, t], axis=0)
results["TCN_individual"] = agg_and_rank(df, ["NUM_FILTERS","KERNEL_SIZE","DROPOUT","LEARNING_RATE","WINDOW_SIZE"], "TCN_individual_camels")

# ---------------- LSTM individual ----------------
HIDDEN_SIZE = [16, 32, 64]
DROPOUT_L = [0.4, 0.5]
LEARNING_RATE_L = [0.001, 0.0005]
SEQ_LENGTH = [270, 365]
BATCH_SIZE = [64, 128]
df = pd.DataFrame()
for h in HIDDEN_SIZE:
    for d in DROPOUT_L:
        for lr in LEARNING_RATE_L:
            for sl in SEQ_LENGTH:
                for bs in BATCH_SIZE:
                    hp_dir = f"{base}/0_out/LSTM/{loc}/camels/individual/hp/h{h}_d{d}_lr{lr}_sl{sl}_b{bs}/results"
                    if not os.path.isdir(hp_dir):
                        print("missing LSTM indiv dir", hp_dir); continue
                    matches = glob.glob(os.path.join(hp_dir, "LSTM_results_*_eva.csv"))
                    if not matches:
                        print("missing LSTM indiv file", hp_dir); continue
                    t = pd.read_csv(matches[0], index_col=0)
                    t.columns = t.columns.str.strip()
                    t["HIDDEN_SIZE"]=h; t["DROPOUT"]=d; t["LEARNING_RATE"]=lr; t["SEQ_LENGTH"]=sl; t["BATCH_SIZE"]=bs
                    df = pd.concat([df, t], axis=0)
results["LSTM_individual"] = agg_and_rank(df, ["HIDDEN_SIZE","DROPOUT","LEARNING_RATE","SEQ_LENGTH","BATCH_SIZE"], "LSTM_individual_camels")

# ---------------- LSTM regional ----------------
HIDDEN_SIZE_R = [64, 128, 256]
DROPOUT_R = [0.2, 0.4]
LEARNING_RATE_R = [0.001, 0.0005]
SEQ_LENGTH_R = [270, 365]
BATCH_SIZE_R = [256, 512]
df = pd.DataFrame()
for h in HIDDEN_SIZE_R:
    for d in DROPOUT_R:
        for lr in LEARNING_RATE_R:
            for sl in SEQ_LENGTH_R:
                combo_dir = f"{base}/99_work/14_LSTM/ealstm_regional_modeling/runs/hp/{loc}/camels/h{h}_d{d}_lr{lr}_sl{sl}"
                if not os.path.isdir(combo_dir):
                    print("missing LSTM regional dir", combo_dir); continue
                found_bs = {}
                for run_dir in sorted(glob.glob(os.path.join(combo_dir, "*/"))):
                    cfg_path = os.path.join(run_dir, "cfg.json")
                    eva_path = os.path.join(run_dir, "hp_metrics_eva.csv")
                    if not (os.path.exists(cfg_path) and os.path.exists(eva_path)):
                        continue
                    try:
                        cfg = json.load(open(cfg_path))
                    except Exception:
                        continue
                    bs = cfg.get("batch_size")
                    if bs in BATCH_SIZE_R:
                        found_bs[bs] = eva_path  # keep latest (sorted order) per batch_size
                for bs, eva_path in found_bs.items():
                    t = pd.read_csv(eva_path, index_col=0)
                    t.columns = t.columns.str.strip()
                    t["HIDDEN_SIZE"]=h; t["DROPOUT"]=d; t["LEARNING_RATE"]=lr; t["SEQ_LENGTH"]=sl; t["BATCH_SIZE"]=bs
                    df = pd.concat([df, t], axis=0)
results["LSTM_regional"] = agg_and_rank(df, ["HIDDEN_SIZE","DROPOUT","LEARNING_RATE","SEQ_LENGTH","BATCH_SIZE"], "LSTM_regional_camels")

# ---------------- TCN regional ----------------
NUM_FILTERS_R = [32, 64, 128]
KERNEL_SIZE_R = [3, 5]
DROPOUT_R2 = [0.1, 0.2, 0.3]
LEARNING_RATE_R2 = [0.001, 0.0005]
WINDOW_SIZE_R = [180, 270]
NUM_LEVELS_R = 6
df = pd.DataFrame()
for nf in NUM_FILTERS_R:
    for k in KERNEL_SIZE_R:
        for dr in DROPOUT_R2:
            for lr in LEARNING_RATE_R2:
                for w in WINDOW_SIZE_R:
                    p = f"{base}/0_out/TCN/{loc}/camels/regional/hp/results/TCN_results_f{nf}_k{k}_d{dr}_lr{lr}_lv{NUM_LEVELS_R}_w{w}_nsFalse_eva.csv"
                    if not os.path.exists(p):
                        print("missing TCN regional", p); continue
                    t = pd.read_csv(p, index_col=0)
                    t.columns = t.columns.str.strip()
                    t["NUM_FILTERS"]=nf; t["KERNEL_SIZE"]=k; t["DROPOUT"]=dr; t["LEARNING_RATE"]=lr; t["WINDOW_SIZE"]=w
                    df = pd.concat([df, t], axis=0)
results["TCN_regional"] = agg_and_rank(df, ["NUM_FILTERS","KERNEL_SIZE","DROPOUT","LEARNING_RATE","WINDOW_SIZE"], "TCN_regional_camels")

# ---------------- XGB regional ----------------
N_ESTIMATORS_XR = [500, 1000]
MAX_DEPTHS_XR = [5, 7]
LEARNING_RATES_XR = [0.01, 0.05, 0.1]
SUBSAMPLE_XR = [0.7, 0.8, 1.0]
COLSAMPLE_BYTREE_XR = [0.7, 0.8]
MIN_CHILD_WEIGHT_XR = [3, 5]
df = pd.DataFrame()
for n in N_ESTIMATORS_XR:
    for d in MAX_DEPTHS_XR:
        for lr in LEARNING_RATES_XR:
            for ss in SUBSAMPLE_XR:
                for cbt in COLSAMPLE_BYTREE_XR:
                    for mcw in MIN_CHILD_WEIGHT_XR:
                        p = f"{base}/0_out/XGB/{loc}/camels/regional/hp/results/XGB_results_n{n}_d{d}_lr{lr}_ss{ss}_cbt{cbt}_mcw{mcw}_eva.csv"
                        if not os.path.exists(p):
                            print("missing XGB regional", p); continue
                        t = pd.read_csv(p, index_col=0)
                        t.columns = t.columns.str.strip()
                        t["N_ESTIMATORS"]=n; t["MAX_DEPTH"]=d; t["LEARNING_RATE"]=lr; t["SUBSAMPLE"]=ss; t["COLSAMPLE_BYTREE"]=cbt; t["MIN_CHILD_WEIGHT"]=mcw
                        df = pd.concat([df, t], axis=0)
results["XGB_regional"] = agg_and_rank(df, ["N_ESTIMATORS","MAX_DEPTH","LEARNING_RATE","SUBSAMPLE","COLSAMPLE_BYTREE","MIN_CHILD_WEIGHT"], "XGB_regional_camels")

# ---------------- RF regional ----------------
N_ESTIMATORS_RR = [300, 500]
MIN_SAMPLES_LEAF_RR = [5, 10, 20]
MAX_DEPTH_RR = [15, 30]
MAX_FEATURES_RR = ["sqrt", 0.33]
df = pd.DataFrame()
for n in N_ESTIMATORS_RR:
    for l in MIN_SAMPLES_LEAF_RR:
        for d in MAX_DEPTH_RR:
            for f in MAX_FEATURES_RR:
                p = f"{base}/0_out/RF/{loc}/camels/regional/hp/results/RF_results_n{n}_d{d}_l{l}_f{f}_eva.csv"
                if not os.path.exists(p):
                    print("missing RF regional", p); continue
                t = pd.read_csv(p, index_col=0)
                t.columns = t.columns.str.strip()
                t["N_ESTIMATORS"]=n; t["MIN_SAMPLES_LEAF"]=l; t["MAX_DEPTH"]=d; t["MAX_FEATURES"]=f
                df = pd.concat([df, t], axis=0)
results["RF_regional"] = agg_and_rank(df, ["N_ESTIMATORS","MIN_SAMPLES_LEAF","MAX_DEPTH","MAX_FEATURES"], "RF_regional_camels")

# ---------------- XGB individual ----------------
N_ESTIMATORS_XI = [500, 1000]
MAX_DEPTHS_XI = [2, 3]
LEARNING_RATES_XI = [0.01, 0.05]
SUBSAMPLE_XI = [0.5, 0.6]
COLSAMPLE_BYTREE_XI = [0.7, 0.8, 1.0]
MIN_CHILD_WEIGHT_XI = [10, 20]
df = pd.DataFrame()
for n in N_ESTIMATORS_XI:
    for d in MAX_DEPTHS_XI:
        for lr in LEARNING_RATES_XI:
            for ss in SUBSAMPLE_XI:
                for cbt in COLSAMPLE_BYTREE_XI:
                    for mcw in MIN_CHILD_WEIGHT_XI:
                        p = f"{base}/0_out/XGB/{loc}/camels/individual/hp/n{n}_d{d}_lr{lr}_ss{ss}_cbt{cbt}_mcw{mcw}/results/XGB_results_eva.csv"
                        if not os.path.exists(p):
                            print("missing XGB individual", p); continue
                        t = pd.read_csv(p, index_col=0)
                        t.columns = t.columns.str.strip()
                        t["N_ESTIMATORS"]=n; t["MAX_DEPTH"]=d; t["LEARNING_RATE"]=lr; t["SUBSAMPLE"]=ss; t["COLSAMPLE_BYTREE"]=cbt; t["MIN_CHILD_WEIGHT"]=mcw
                        df = pd.concat([df, t], axis=0)
results["XGB_individual"] = agg_and_rank(df, ["N_ESTIMATORS","MAX_DEPTH","LEARNING_RATE","SUBSAMPLE","COLSAMPLE_BYTREE","MIN_CHILD_WEIGHT"], "XGB_individual_camels")

print("\n\n========== SUMMARY (top 5 each) ==========")
for name, adf in results.items():
    print(f"\n--- {name} ---")
    if adf is None:
        print("NO DATA")
        continue
    print(f"total combos found: {len(adf)}")
    print(adf.head(5).to_string(index=False))
