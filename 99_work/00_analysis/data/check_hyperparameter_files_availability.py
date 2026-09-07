"""
Check how much of each model's hyperparameter grid (as defined in HYPERPARAMETERS.md /
run_*_hp_tuning.sh) has actually been run, by looking for the output files each hp-search
script produces on disk.

Grids and output-path/filename conventions below are copied from the actual hp-search
scripts (not just HYPERPARAMETERS.md), since a couple of grids drifted from the nominal
values in HYPERPARAMETERS.md after the scripts were fixed up (e.g. TCN's LAYERS row is not
actually swept -- num_levels is fixed at 6). See ANALYSIS_README.md for the full story per
model.

Usage:
    python check_hyperparameter_files_availability.py            # summary only
    python check_hyperparameter_files_availability.py --verbose  # also list missing combos
"""
import argparse
import itertools
import json
from pathlib import Path

HSP2 = Path("/data0/funato/hsp2")

LOC = "US"  # "JP" or "GB" or "US"


def fmt_float2(x):
    """Matches RF regional's `f"{x:.2f}"` formatting for max_features."""
    return f"{x:.2f}"


def any_glob(path_glob: str) -> bool:
    """path_glob is an absolute glob pattern (may contain '*')."""
    path_glob = Path(path_glob)
    parts = path_glob.parts
    # find first component containing a wildcard to split base dir from pattern
    for i, part in enumerate(parts):
        if any(c in part for c in "*?[]"):
            base = Path(*parts[:i]) if i > 0 else Path("/")
            rel = str(Path(*parts[i:]))
            if not base.exists():
                return False
            return any(base.glob(rel))
    return path_glob.exists()


# --------------------------------------------------------------------------------------
# Section definitions: each entry is
#   (label, grid_dict, hp_str_fn(combo_dict) -> str, eval_check_fn(hp_str, combo) -> bool,
#    train_check_fn(hp_str, combo) -> bool, optional)
#
# eval_check_fn/train_check_fn take the combo dict as well as hp_str because not every swept
# hyperparameter is reflected in hp_str (see lstm_regional_evaluated) -- a plain glob on hp_str
# alone can't always tell two combos apart. Most sections' output path IS fully determined by
# hp_str, so they just wrap a hp_str -> glob-pattern-string function with glob_check().
#
# For RF/XGB/TCN Regional, train_check_fn is set separately since those pipelines' HP
# scripts only train a checkpoint; eval_regional_hp.py must be run afterward, by hand, to
# produce the benchmark CSVs that eval_check_fn checks for. Everywhere else, training and
# evaluation happen inside the same script run, so train_check_fn defaults to eval_check_fn
# (a combo is "trained" the instant it's "evaluated").
# --------------------------------------------------------------------------------------

def rf_indiv_hp_str(c):
    return f"n{c['n_estimators']}_d{c['max_depth']}_l{c['min_samples_leaf']}_f{c['max_features']}"


def rf_regional_hp_str(c):
    d = c['max_depth']
    f = c['max_features']
    f_str = f if isinstance(f, str) else fmt_float2(f)
    return f"n{c['n_estimators']}_d{d}_l{c['min_samples_leaf']}_f{f_str}"


def xgb_hp_str(c):
    return (f"n{c['n_estimators']}_d{c['max_depth']}_lr{c['learning_rate']}"
            f"_ss{c['subsample']}_cbt{c['colsample_bytree']}_mcw{c['min_child_weight']}")


def tcn_indiv_hp_str(c):
    return f"f{c['num_filters']}_k{c['kernel_size']}_d{c['dropout']}_lr{c['learning_rate']}_lv6_w{c['window_size']}"


def tcn_regional_hp_str(c):
    return f"f{c['num_filters']}_k{c['kernel_size']}_d{c['dropout']}_lr{c['learning_rate']}_lv6_w{c['window_size']}_nsFalse"


def lstm_indiv_hp_str(c):
    return f"h{c['hidden_size']}_d{c['dropout']}_lr{c['learning_rate']}_sl{c['seq_length']}_b{c['batch_size']}"


def lstm_regional_hp_str(c):
    return f"h{c['hidden_size']}_d{c['dropout']}_lr{c['learning_rate']}_sl{c['seq_length']}"


def lstm_regional_evaluated(hp_str, combo):
    """batch_size (swept {256,512} per run_lstm_hp_tuning_regional.sh) is NOT part of the
    hp_str folder name (_setup_run tags directories by hidden_size/dropout/learning_rate/
    seq_length only), so two different batch_size runs land in the same hp_str folder,
    distinguishable only via their saved cfg.json -- a plain glob on hp_str alone can't tell
    them apart and would report a combo done as soon as EITHER batch_size finished."""
    hp_dir = HSP2 / f"99_work/14_LSTM/ealstm_regional_modeling/runs/hp/{LOC}/{hp_str}"
    if not hp_dir.exists():
        return False
    for run_dir in hp_dir.glob("*"):
        if not (run_dir / "hp_metrics_eva.csv").exists():
            continue
        cfg_path = run_dir / "cfg.json"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path) as fp:
                saved_cfg = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if saved_cfg.get("batch_size") == combo["batch_size"]:
            return True
    return False


SECTIONS = []


def glob_check(hp_str_glob_fn):
    """Adapt a hp_str -> glob-pattern-string function into the (hp_str, combo) -> bool contract
    every section's eval/train check now uses. Use this for sections whose output path is fully
    determined by hp_str; sections where a hyperparameter is swept but NOT reflected in hp_str
    (see lstm_regional_evaluated) need a real (hp_str, combo) -> bool function instead, since a
    glob on hp_str alone can't distinguish those combos."""
    return lambda hp_str, combo: any_glob(hp_str_glob_fn(hp_str))


def add_section(model, mode, grid, hp_str_fn, eval_check_fn, train_check_fn=None):
    combos = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]
    SECTIONS.append({
        "model": model, "mode": mode, "combos": combos, "hp_str_fn": hp_str_fn,
        "eval_check_fn": eval_check_fn, "train_check_fn": train_check_fn or eval_check_fn,
        "has_separate_train_stage": train_check_fn is not None,
    })


# --- RF ---------------------------------------------------------------------------------
add_section(
    "RF", "Individual",
    {"n_estimators": [100, 300, 500], "min_samples_leaf": [5, 10, 20],
     "max_depth": [5, 10, 15], "max_features": ["sqrt", 0.33]},
    rf_indiv_hp_str,
    glob_check(lambda hp_str: str(HSP2 / f"0_out/RF/{LOC}/individual/hp/{hp_str}/results/RF_results_eva.csv")),
)
add_section(
    "RF", "Regional",
    {"n_estimators": [300, 500], "min_samples_leaf": [5, 10, 20],
     "max_depth": [15, 30], "max_features": ["sqrt", 0.33]},
    rf_regional_hp_str,
    # eval_regional_hp.py, run via run_eval_hp.sh --hp_string {hp_str}, writes here:
    glob_check(lambda hp_str: str(HSP2 / f"0_out/RF/{LOC}/regional/hp/results/RF_results_{hp_str}_eva.csv")),
    train_check_fn=glob_check(lambda hp_str: str(HSP2 / f"99_work/11_RandomForest/RF_regional_modeling-master/runs_rf/hp/{hp_str}/*/rf_model.joblib")),
)

# --- XGB --------------------------------------------------------------------------------
add_section(
    "XGB", "Individual",
    {"n_estimators": [500, 1000], "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.05, 0.1],
     "subsample": [0.5, 0.6, 0.7], "colsample_bytree": [0.7, 0.8, 1.0], "min_child_weight": [5, 10, 20]},
    xgb_hp_str,
    glob_check(lambda hp_str: str(HSP2 / f"0_out/XGB/{LOC}/individual/hp/{hp_str}/results/XGB_results_eva.csv")),
)
add_section(
    "XGB", "Regional",
    {"n_estimators": [500, 1000], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
     "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0], "min_child_weight": [1, 3, 5]},
    xgb_hp_str,
    # eval_regional_hp.py, run via run_eval_hp.sh --hp_string {hp_str}, writes here:
    glob_check(lambda hp_str: str(HSP2 / f"0_out/XGB/{LOC}/regional/hp/results/XGB_results_{hp_str}_eva.csv")),
    train_check_fn=glob_check(lambda hp_str: str(HSP2 / f"99_work/12_XGBoost/XGB_regional_modeling-master/runs_xgb/hp/{hp_str}/*/xgb_model.json")),
)

# --- TCN --------------------------------------------------------------------------------
# num_levels is fixed at 6 (not swept). window_size (--timesteps for indiv, --seq_length for
# regional, days) IS swept: {90, 180, 270} for both, per run_tcn_hp_tuning.sh's WINDOW_SIZE array.
add_section(
    "TCN", "Individual",
    {"num_filters": [16, 32], "kernel_size": [3, 5], "dropout": [0.3, 0.4, 0.5],
     "learning_rate": [0.001, 0.0005], "window_size": [90, 180, 270]},
    tcn_indiv_hp_str,
    glob_check(lambda hp_str: str(HSP2 / f"0_out/TCN/{LOC}/individual/hp/{hp_str}/results/TCN_results_eva.csv")),
)
add_section(
    "TCN", "Regional",
    {"num_filters": [32, 64, 128], "kernel_size": [3, 5, 7], "dropout": [0.1, 0.2, 0.3],
     "learning_rate": [0.001, 0.0005], "window_size": [90, 180, 270]},
    tcn_regional_hp_str,
    # eval_regional_hp.py, run via run_eval_tcn_hp.sh --hp_string {hp_str}, writes here:
    glob_check(lambda hp_str: str(HSP2 / f"0_out/TCN/{LOC}/regional/hp/results/TCN_results_{hp_str}_eva.csv")),
    # main_tcn_hp.py's default --epochs is 50 and only ever *prints* "Final model saved to
    # tcn_model_final.pt" without actually writing that file (a real bug in that script, not
    # worked around here) -- the epoch-50 checkpoint (always saved: `epoch % 5 == 0 or
    # epoch == cfg["epochs"]`) is the actual reliable completion marker.
    train_check_fn=glob_check(lambda hp_str: str(HSP2 / f"99_work/13_TCN/TCN_regional_modeling-master/runs_tcn/hp/{LOC}/{hp_str}/*/tcn_model_epoch50.pt")),
)

# --- LSTM -------------------------------------------------------------------------------
add_section(
    "LSTM", "Individual",
    {"hidden_size": [16, 32, 64, 128], "dropout": [0.4, 0.5], "learning_rate": [0.001, 0.0005],
     "seq_length": [270, 365], "batch_size": [64, 128]},
    lstm_indiv_hp_str,
    glob_check(lambda hp_str: str(HSP2 / f"0_out/LSTM/{LOC}/individual/hp/{hp_str}/results/LSTM_results_*_eva.csv")),
)
add_section(
    "LSTM", "Regional",
    # batch_size is swept ({256,512}, per run_lstm_hp_tuning_regional.sh) but is NOT part of
    # hp_str (see lstm_regional_evaluated) -- included here as a real grid dimension (72 total
    # combos, matching the driver script) even though lstm_regional_hp_str ignores it.
    {"hidden_size": [64, 128, 256], "dropout": [0.2, 0.4], "learning_rate": [0.001, 0.0005, 0.0001],
     "seq_length": [270, 365], "batch_size": [256, 512]},
    lstm_regional_hp_str,
    lstm_regional_evaluated,
)

# --- LSTM Adapted (fine-tuning grid only; no equivalent HP grid exists for TCN adapted -----
# see ANALYSIS_README.md) ----------------------------------------------------------------
add_section(
    "LSTM", "Adapted",
    {"ft_epochs": [3, 5, 10], "ft_lr": [0.0001, 0.00001]},
    lambda c: f"ft{c['ft_epochs']}_ftlr{c['ft_lr']}",
    glob_check(lambda hp_str: str(HSP2 / f"0_out/LSTM/{LOC}/adapted_ealstm/results/LSTM_results_*_{hp_str}_eva.csv")),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="also list missing combinations")
    args = parser.parse_args()

    grand_trained, grand_evaluated, grand_total = 0, 0, 0
    header = f"{'Model':<6} {'Mode':<11} {'Trained':>9} {'Evaluated':>10} {'Total':>6} {'Train%':>8} {'Eval%':>7}"
    print(header)
    print("-" * len(header))
    for section in SECTIONS:
        trained_combos, evaluated_combos = [], []
        missing_train, missing_eval = [], []
        for combo in section["combos"]:
            hp_str = section["hp_str_fn"](combo)
            is_trained = section["train_check_fn"](hp_str, combo)
            is_evaluated = section["eval_check_fn"](hp_str, combo)
            (trained_combos if is_trained else missing_train).append(hp_str)
            (evaluated_combos if is_evaluated else missing_eval).append(hp_str)

        n_trained, n_eval, n_total = len(trained_combos), len(evaluated_combos), len(section["combos"])
        train_pct = (100.0 * n_trained / n_total) if n_total else 0.0
        eval_pct = (100.0 * n_eval / n_total) if n_total else 0.0
        print(f"{section['model']:<6} {section['mode']:<11} {n_trained:>9} {n_eval:>10} "
              f"{n_total:>6} {train_pct:>7.1f}% {eval_pct:>6.1f}%")

        grand_trained += n_trained
        grand_evaluated += n_eval
        grand_total += n_total

        if args.verbose:
            if section["has_separate_train_stage"] and missing_train:
                print(f"    not yet trained ({len(missing_train)}):")
                for m in missing_train:
                    print(f"      - {m}")
            if missing_eval:
                label = "not yet evaluated" if section["has_separate_train_stage"] else "missing"
                print(f"    {label} ({len(missing_eval)}):")
                for m in missing_eval:
                    print(f"      - {m}")

    print("-" * len(header))
    train_pct = (100.0 * grand_trained / grand_total) if grand_total else 0.0
    eval_pct = (100.0 * grand_evaluated / grand_total) if grand_total else 0.0
    print(f"{'TOTAL':<6} {'':<11} {grand_trained:>9} {grand_evaluated:>10} "
          f"{grand_total:>6} {train_pct:>7.1f}% {eval_pct:>6.1f}%")

    print()
    print("Notes:")
    print("  - RF/XGB/TCN Regional: 'Trained' = a checkpoint exists under runs_*/hp/... (from")
    print("    run_*_hp_tuning.sh); 'Evaluated' = eval_regional_hp.py has been run for that combo")
    print("    (--hp_string {combo}) and written benchmark CSVs under 0_out/.../regional/hp/.")
    print("  - Everywhere else, Trained == Evaluated: training and evaluation happen inside the")
    print("    same script run (e.g. LSTM Regional's main_lstm_regional_hp.py does both).")
    print("  - LSTM Adapted has no equivalent grid for TCN (no TCN adapted-hp script exists).")


if __name__ == "__main__":
    main()
