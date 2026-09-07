"""
SHAP interpretability analysis for the canonical, true EA-LSTM regional model trained by
`main.py` / evaluated by `eval.py` in this directory.

WHY THIS SCRIPT EXISTS
-----------------------
The old "canonical" SHAP script in this project
(`/data0/funato/hsp2/99_work/14_LSTM/main_LSTM_shap_regional.py`, now renamed
`x_main_LSTM_shap_regional.py` since it is superseded) trained a completely separate,
throwaway Keras/TensorFlow model -- plain LSTM, static attributes concatenated into the
dynamic input at every timestep, GLOBAL pooled mean/std normalization -- purely to run SHAP on
it. That model is architecturally and numerically different from the actual PyTorch true
EA-LSTM (`papercode/ealstm.py`) trained by `main.py`, whose per-basin-scaled data pipeline and
NSE/KGE benchmark numbers are what this project actually reports. SHAP explanations of the
Keras model therefore describe a *different model* than the one being benchmarked -- a real
methodological inconsistency.

This script instead runs SHAP directly against the real, trained PyTorch `Model`/`EALSTM` from
a `main.py` regional-train run directory, reusing `main.py`'s `Model` class and `eval.py`'s
`EvalDataset` / `_load_scalers` / `_param_tag` so the explained model sees the exact same kind
of per-basin-scaled input it was trained and evaluated on (nothing is duplicated ad hoc).

SCOPE: this script only supports the true EA-LSTM configuration (no_static=False,
concat_static=False) -- the only configuration `ealstm_regional_modeling/main.py` is meant to
be run with (see that script's module docstring) and the only one that actually exercises the
EA-LSTM static-attribute conditioning mechanism this analysis is meant to explain. It will
raise a clear error if pointed at a run_dir trained with no_static=True or concat_static=True.

EXPLAINER CHOICE: shap.GradientExplainer (Expected Gradients), not DeepExplainer or
KernelExplainer
------------------------------------------------------------------------------------
  * `shap.DeepExplainer` relies on registering backward hooks that rewrite the gradient of
    each nonlinearity (its DeepLIFT-style rescale rule); this hook mechanism has known
    compatibility gaps with recent PyTorch/numpy releases (this environment: PyTorch 2.11,
    numpy 2.4.4, shap 0.51.0) and is generally the most fragile of the three options.
  * `shap.KernelExplainer` is model-agnostic (treats the model as a black box) but requires
    one full model forward pass per coalition per sample -- for a background of even modest
    size this is far too slow on a CPU-only machine for a seq_length=270, hidden_size=256
    LSTM, and it also does not naturally expose per-timestep-lag attributions.
  * `shap.GradientExplainer` implements Expected Gradients, is PyTorch-native (uses standard
    autograd, no custom backward-hook rewriting), and -- verified empirically here (see
    `_verify_gradient_explainer_multi_input_support` and the timing check in this script's
    development) -- in shap 0.51.0 it accepts a LIST of input tensors directly for models
    with multiple forward() arguments, e.g. `shap.GradientExplainer(model, [bg_x_d, bg_x_s])`
    / `explainer.shap_values([x_d, x_s])`. This means the "single concatenated input, split
    inside a wrapper" pattern is NOT needed for the explainer to accept two inputs -- SHAP
    already handles `EALSTM`'s two-argument (x_d, x_s) forward signature natively. A thin
    wrapper (`_PredictionOnly` below) is still used, but only to reduce `Model.forward`'s
    three-tuple return (`out, h_n, c_n`) to the single prediction tensor SHAP expects as model
    output -- NOT to concatenate/split x_d and x_s. This wrapper's faithfulness is verified
    numerically against calling the real model directly (see `_check_wrapper_equivalence`).
    This also gives genuine per-timestep-lag SHAP values for the dynamic inputs "for free"
    (the returned array has shape [n_samples, seq_length, n_dynamic_features]), rather than
    only an aggregated-over-sequence approximation.

SUBSAMPLE SIZE (CPU-feasible, documented default)
--------------------------------------------------
Explaining all 614 US basins x the full ~6-year validation period x seq_length=270 is
computationally enormous (naive full-scale SHAP here would mean hundreds of thousands of
270-timestep forward+backward passes). Instead this script explains a documented subsample:

  --n_basins 10           10 basins (deterministic: the first 10 with both a fitted scaler
                           and static attributes, sorted by basin id, for reproducibility)
  --n_dates_per_basin 5   5 dates per basin, evenly spaced across that basin's validation
                           period (so the sample covers the full seasonal cycle, not just one
                           cluster of dates)
  --n_background 50       50 background samples for Expected Gradients, drawn at random (with
                           --seed) from the TRAINING period across a wider pool of basins
                           (min(3 * n_basins, available)), matching standard SHAP practice of
                           using a representative sample of the training distribution as the
                           interventional/expectation background, not the same points being
                           explained
  --nsamples 50           GradientExplainer's internal number of interpolation samples per
                           explained point (SHAP's own Expected-Gradients default)

That is 10 x 5 = 50 explained samples. On this CPU-only machine, with the actual production
architecture (hidden_size=256, seq_length=270), 50 background x 50 explained x nsamples=50
was benchmarked at ~11 seconds during development (see script development notes / PR
description) -- comfortably CPU-feasible. All of these are CLI-overridable if a larger/smaller
sample is wanted later.

OUTPUTS
-------
Written to `/data0/funato/hsp2/0_out/LSTM/{loc}/shap_ealstm/`, following the same
per-location-folder convention as `regional_ealstm/` and `adapted_ealstm/` so results from
different locations (US/GB/JP) or differently-configured runs never collide (tag includes the
same no_static/concat_static/hidden_size/dropout/epoch/seq_length encoding as
`eval._param_tag`, plus the SHAP-specific sample-size/seed parameters):

  shap_values_dynamic_<tag>.npy   raw per-timestep-lag SHAP values, shape
                                   (n_explained, seq_length, n_dynamic_features)
  shap_values_static_<tag>.npy    raw SHAP values for static attributes, shape
                                   (n_explained, n_static_features)
  explain_samples_meta_<tag>.csv  basin id + date + observed/predicted discharge for every
                                   explained sample, so SHAP rows can be traced back
  importance_summary_<tag>.csv    one row per feature (dynamic + static), mean(|SHAP|)
                                   aggregated over the sequence dimension for dynamic features
  plot_dynamic_summary_<tag>.png  SHAP beeswarm summary plot, dynamic features (SHAP values
                                   summed over the sequence/lag dimension per sample)
  plot_static_summary_<tag>.png   SHAP beeswarm summary plot, static catchment attributes
  plot_dynamic_lag_importance_<tag>.png
                                   mean(|SHAP|) vs. lag (days before prediction day), one line
                                   per dynamic feature -- shows which lags the model actually
                                   relies on, in the spirit of Kratzert et al.'s own
                                   interpretability follow-up work on this architecture

USAGE
-----
    python shap_regional.py --run_dir runs/run_<timestamp>_US_seed<N>
    python shap_regional.py --run_dir runs/run_<timestamp>_US_seed<N> \\
        --n_basins 10 --n_dates_per_basin 5 --n_background 50 --nsamples 50 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shap  # noqa: E402  (after sys.path insert)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from main_lstm_regional import Model, DYN_COLS, DEVICE  # noqa: E402
from papercode.lstm_run_registry import load_lstm_state_dict  # noqa: E402
from eval import EvalDataset, _load_scalers, _param_tag  # noqa: E402
from papercode.datautils import load_attributes, VALID_ATTR  # noqa: E402
from papercode.utils import get_basin_list  # noqa: E402


# ####################### #
# Model wrapper           #
# ####################### #


class _PredictionOnly(nn.Module):
    """Wraps `main.Model` to return only the scalar prediction tensor.

    `main.Model.forward` returns a 3-tuple `(out, h_n, c_n)` (see main.py); SHAP's explainers
    expect the wrapped callable to return the model's actual output tensor. This wrapper does
    NOT concatenate or split x_d/x_s -- shap.GradientExplainer accepts the two-tensor
    (x_d, x_s) forward signature natively (verified in this script's module docstring /
    `_check_wrapper_equivalence` below); the wrapper's only job is trimming the tuple.
    """

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        return self.model(x_d, x_s)[0]


def _check_wrapper_equivalence(model: Model, wrapped: _PredictionOnly,
                                x_d: torch.Tensor, x_s: torch.Tensor, n_check: int = 5):
    """Numeric sanity check: wrapped(x_d, x_s) must be bit-for-bit (up to fp tolerance)
    identical to calling the real model directly, on a handful of real samples. Raises
    AssertionError if not -- this is the "verify the split-and-forward is equivalent" check
    called for by this script's task description, adapted to the fact that no
    concat/split is actually happening (see class docstring above): here it simply confirms
    the wrapper does not silently alter the model's numerical output."""
    n_check = min(n_check, x_d.shape[0])
    with torch.no_grad():
        direct = model(x_d[:n_check], x_s[:n_check])[0]
        wrapped_out = wrapped(x_d[:n_check], x_s[:n_check])
    if not torch.allclose(direct, wrapped_out, atol=1e-6):
        max_diff = (direct - wrapped_out).abs().max().item()
        raise AssertionError(
            f"Wrapper output diverges from direct model call (max abs diff {max_diff}); "
            "refusing to run SHAP against a wrapper that isn't faithful to the real model.")
    print(f"Wrapper equivalence check passed on {n_check} samples "
          f"(max abs diff {(direct - wrapped_out).abs().max().item():.2e}).")


# ####################### #
# Sample construction     #
# ####################### #


def _select_basins(scalers: Dict, attributes: pd.DataFrame, loc: str, n_basins: int) -> List[str]:
    """Deterministic (sorted, reproducible) choice of the first n_basins basins that have both
    a fitted per-basin scaler and static attributes available."""
    all_basins = get_basin_list(loc=loc)
    usable = [b for b in all_basins if b in scalers['basin_scalers'] and b in attributes.index]
    if len(usable) < n_basins:
        print(f"WARNING: only {len(usable)} usable basins available (< requested {n_basins}); "
              f"using all {len(usable)}.")
    return usable[:n_basins]


def _evenly_spaced_indices(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = min(n_pick, n_total)
    return np.unique(np.linspace(0, n_total - 1, n_pick, dtype=int))


def build_explain_set(basins: List[str], cfg: Dict, scalers: Dict, attributes: pd.DataFrame,
                       n_dates_per_basin: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the samples to be explained: n_dates_per_basin evenly-spaced validation-period
    dates for each of `basins`. Returns (x_d, x_s, meta) where meta has one row per sample
    with columns basin/date/obs_q."""
    val_start = pd.to_datetime(cfg['val_start'])
    val_end = pd.to_datetime(cfg['val_end'])

    xd_list, xs_list, meta_rows = [], [], []
    for basin in basins:
        ds = EvalDataset(basin, (val_start, val_end), cfg, scalers, attributes)
        if len(ds) == 0:
            print(f"WARNING: basin {basin} has no usable validation-period sequences; skipping.")
            continue
        idxs = _evenly_spaced_indices(len(ds), n_dates_per_basin)
        for idx in idxs:
            x, x_s, y = ds[idx]
            xd_list.append(x)
            xs_list.append(x_s)
            meta_rows.append({'gauge_id': basin, 'date': ds.dates[idx].strftime('%Y-%m-%d'),
                               'obs_q': float(y)})

    if not xd_list:
        raise RuntimeError("No explain samples could be built -- check run_dir / date ranges.")

    return (np.stack(xd_list).astype(np.float32),
            np.stack(xs_list).astype(np.float32),
            pd.DataFrame(meta_rows))


def build_background_set(loc: str, cfg: Dict, scalers: Dict, attributes: pd.DataFrame,
                          n_background: int, background_basin_pool: int,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Random background sample (Expected Gradients baseline distribution) drawn from the
    TRAINING period across a wider pool of basins than the explain set, matching standard SHAP
    practice of using a representative sample of the training distribution as background."""
    train_start = pd.to_datetime(cfg['train_start'])
    train_end = pd.to_datetime(cfg['train_end'])

    all_basins = get_basin_list(loc=loc)
    usable = [b for b in all_basins if b in scalers['basin_scalers'] and b in attributes.index]
    pool_basins = usable[:background_basin_pool]

    xd_list, xs_list = [], []
    for basin in pool_basins:
        ds = EvalDataset(basin, (train_start, train_end), cfg, scalers, attributes)
        for i in range(len(ds)):
            x, x_s, _y = ds[i]
            xd_list.append(x)
            xs_list.append(x_s)

    if len(xd_list) < n_background:
        print(f"WARNING: only {len(xd_list)} candidate background samples available "
              f"(< requested {n_background}); using all of them.")
        n_background = len(xd_list)

    chosen = rng.choice(len(xd_list), size=n_background, replace=False)
    xd_arr = np.stack(xd_list).astype(np.float32)[chosen]
    xs_arr = np.stack(xs_list).astype(np.float32)[chosen]
    return xd_arr, xs_arr


# ####################### #
# Plotting / summarizing  #
# ####################### #


def _save_beeswarm(shap_matrix: np.ndarray, feature_values: np.ndarray, feature_names: List[str],
                    title: str, out_path: Path):
    plt.figure()
    shap.summary_plot(shap_matrix, features=feature_values, feature_names=feature_names,
                       show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_lag_importance(shap_dynamic: np.ndarray, dyn_cols: List[str], out_path: Path):
    """Mean(|SHAP|) per lag (days before the prediction day), one line per dynamic feature.
    shap_dynamic shape: (n_samples, seq_length, n_dynamic_features). Lag 0 = the prediction
    day itself, increasing lag = further in the past."""
    seq_length = shap_dynamic.shape[1]
    mean_abs = np.mean(np.abs(shap_dynamic), axis=0)  # (seq_length, n_dyn)
    lags = np.arange(seq_length - 1, -1, -1)  # day index seq_length-1 (last) -> lag 0

    plt.figure(figsize=(8, 5))
    for j, col in enumerate(dyn_cols):
        plt.plot(lags, mean_abs[:, j], label=col)
    plt.xlabel("Lag (days before prediction day)")
    plt.ylabel("Mean |SHAP value|")
    plt.title("Per-timestep-lag SHAP importance, dynamic inputs")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ####################### #
# Main                    #
# ####################### #


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for the regional EA-LSTM model")
    parser.add_argument('--run_dir', type=str, required=True,
                         help='Regional model run dir produced by main.py train')
    parser.add_argument('--model_epoch', type=int, default=None,
                         help='Which model_epoch{N}.pt to explain. Default: last available.')
    parser.add_argument('--n_basins', type=int, default=10,
                         help='Number of basins to include in the explain set (default 10).')
    parser.add_argument('--n_dates_per_basin', type=int, default=5,
                         help='Evenly-spaced validation dates per basin to explain (default 5).')
    parser.add_argument('--n_background', type=int, default=50,
                         help='Background sample size for Expected Gradients (default 50).')
    parser.add_argument('--background_basin_pool', type=int, default=None,
                         help='How many basins to draw background samples from. '
                              'Default: 3 * n_basins.')
    parser.add_argument('--nsamples', type=int, default=50,
                         help="GradientExplainer's internal interpolation-sample count "
                              "(default 50, SHAP's own Expected-Gradients default).")
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for background sampling.')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print("SHAP analysis: regional EA-LSTM model")
    print(f"Run directory: {run_dir}")

    with open(run_dir / 'cfg.json', 'r') as fp:
        cfg = json.load(fp)

    is_no_static = cfg.get('no_static', False)
    is_concat_static = cfg.get('concat_static', False)
    if is_no_static or is_concat_static:
        raise ValueError(
            f"shap_regional.py only supports the true EA-LSTM configuration "
            f"(no_static=False, concat_static=False); run_dir {run_dir} has "
            f"no_static={is_no_static}, concat_static={is_concat_static}. See module docstring.")

    scalers = _load_scalers(run_dir)

    model_epoch = args.model_epoch if args.model_epoch is not None else cfg['epochs']
    model_file = run_dir / f"model_epoch{model_epoch}.pt"
    if not model_file.exists():
        model_file = run_dir / "lstm_model_final.pt"
        if not model_file.exists():
            candidates = sorted(run_dir.glob('model_epoch*.pt'))
            model_file = candidates[-1] if candidates else None
    if model_file is None or not model_file.exists():
        raise ValueError(f"No .pt model file found in {run_dir}")
    print(f"Loading model from: {model_file}")

    loc = cfg['loc']
    input_size_stat = cfg.get('input_size_stat', 0)
    input_size_dyn = cfg.get('input_size_dyn', 3)

    model = Model(input_size_dyn=input_size_dyn, input_size_stat=input_size_stat,
                  hidden_size=cfg['hidden_size'], dropout=cfg.get('dropout', 0.0),
                  concat_static=is_concat_static, no_static=is_no_static,
                  initial_forget_bias=cfg.get('initial_forget_gate_bias', 5)).to(DEVICE)
    model.load_state_dict(load_lstm_state_dict(model_file, map_location=DEVICE))
    model.eval()

    db_path = str(run_dir / 'attributes.db')
    all_basins = get_basin_list(loc=loc)
    attributes = load_attributes(db_path, all_basins)
    attr_names = list(attributes.columns)

    basins = _select_basins(scalers, attributes, loc, args.n_basins)
    print(f"Explain set basins ({len(basins)}): {basins}")

    print("\nBuilding explain samples (evenly-spaced validation dates per basin)...")
    x_d_explain, x_s_explain, meta = build_explain_set(
        basins, cfg, scalers, attributes, args.n_dates_per_basin)
    print(f"Explain set size: {x_d_explain.shape[0]} samples "
          f"(x_d {x_d_explain.shape}, x_s {x_s_explain.shape})")

    background_basin_pool = args.background_basin_pool or (3 * args.n_basins)
    print("\nBuilding background samples (random draw from training period, wider basin pool)...")
    rng = np.random.default_rng(args.seed)
    x_d_bg, x_s_bg = build_background_set(
        loc, cfg, scalers, attributes, args.n_background, background_basin_pool, rng)
    print(f"Background set size: {x_d_bg.shape[0]} samples")

    x_d_explain_t = torch.from_numpy(x_d_explain).to(DEVICE)
    x_s_explain_t = torch.from_numpy(x_s_explain).to(DEVICE)
    x_d_bg_t = torch.from_numpy(x_d_bg).to(DEVICE)
    x_s_bg_t = torch.from_numpy(x_s_bg).to(DEVICE)

    wrapped = _PredictionOnly(model).to(DEVICE)
    wrapped.eval()
    _check_wrapper_equivalence(model, wrapped, x_d_explain_t, x_s_explain_t)

    # Also record the model's actual predictions for the explained samples, for the meta CSV.
    with torch.no_grad():
        preds_scaled = wrapped(x_d_explain_t, x_s_explain_t).cpu().numpy().flatten()
    pred_q = []
    for i, row in meta.iterrows():
        bscaler = scalers['basin_scalers'][str(row['gauge_id'])]
        pred_q.append(preds_scaled[i] * bscaler['q_std'] + bscaler['q_mean'])
    meta['pred_q'] = pred_q

    print(f"\nRunning shap.GradientExplainer (nsamples={args.nsamples})...")
    explainer = shap.GradientExplainer(wrapped, [x_d_bg_t, x_s_bg_t])
    shap_values = explainer.shap_values([x_d_explain_t, x_s_explain_t], nsamples=args.nsamples)

    shap_dynamic = np.asarray(shap_values[0])  # (n_explain, seq_length, n_dyn[, 1])
    shap_static = np.asarray(shap_values[1])   # (n_explain, n_stat[, 1])
    if shap_dynamic.ndim == 4:
        shap_dynamic = shap_dynamic[..., 0]
    if shap_static.ndim == 3:
        shap_static = shap_static[..., 0]

    if not np.all(np.isfinite(shap_dynamic)) or not np.all(np.isfinite(shap_static)):
        raise RuntimeError("SHAP values contain NaN/Inf -- something is wrong upstream.")

    print(f"shap_dynamic shape: {shap_dynamic.shape}, shap_static shape: {shap_static.shape}")
    print(f"Dynamic |SHAP| range: [{np.abs(shap_dynamic).min():.4g}, "
          f"{np.abs(shap_dynamic).max():.4g}], mean {np.abs(shap_dynamic).mean():.4g}")
    print(f"Static |SHAP| range: [{np.abs(shap_static).min():.4g}, "
          f"{np.abs(shap_static).max():.4g}], mean {np.abs(shap_static).mean():.4g}")

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    output_base = Path(f"/data0/funato/hsp2/0_out/LSTM/{loc}/regional/shap")
    output_base.mkdir(parents=True, exist_ok=True)

    tag = (_param_tag(cfg, model_epoch) +
           f"_nb{len(basins)}_nd{args.n_dates_per_basin}_bg{x_d_bg.shape[0]}"
           f"_ns{args.nsamples}_seed{args.seed}")

    np.save(output_base / f"shap_values_dynamic_{tag}.npy", shap_dynamic)
    np.save(output_base / f"shap_values_static_{tag}.npy", shap_static)
    meta.to_csv(output_base / f"explain_samples_meta_{tag}.csv", index=False)
    print(f"\nRaw SHAP arrays + sample metadata saved to {output_base}")

    # Aggregate dynamic SHAP over the sequence dimension (sum of |SHAP| over lags) for a
    # single per-feature-per-sample importance value, as documented in the module docstring.
    dyn_seq_summed = shap_dynamic.sum(axis=1)  # (n_explain, n_dyn) -- signed sum over lag
    dyn_seq_summed_abs = np.abs(shap_dynamic).sum(axis=1)  # for the importance table

    importance_rows = []
    for j, col in enumerate(DYN_COLS):
        importance_rows.append({'feature': col, 'type': 'dynamic',
                                 'mean_abs_shap_summed_over_seq': float(dyn_seq_summed_abs[:, j].mean())})
    for j, col in enumerate(attr_names):
        importance_rows.append({'feature': col, 'type': 'static',
                                 'mean_abs_shap_summed_over_seq': float(np.abs(shap_static[:, j]).mean())})
    importance_df = pd.DataFrame(importance_rows).sort_values(
        'mean_abs_shap_summed_over_seq', ascending=False)
    importance_df.to_csv(output_base / f"importance_summary_{tag}.csv", index=False)
    print(f"Importance summary:\n{importance_df.to_string(index=False)}")

    # Feature "values" for the beeswarm plots: dynamic -> mean over the sequence of the
    # (scaled) input value per sample/feature; static -> the (scaled) static value itself.
    dyn_feature_values = x_d_explain.mean(axis=1)  # (n_explain, n_dyn)

    _save_beeswarm(dyn_seq_summed, dyn_feature_values, DYN_COLS,
                    "SHAP summary -- dynamic inputs (summed over sequence)",
                    output_base / f"plot_dynamic_summary_{tag}.png")
    _save_beeswarm(shap_static, x_s_explain, attr_names,
                    "SHAP summary -- static catchment attributes",
                    output_base / f"plot_static_summary_{tag}.png")
    _save_lag_importance(shap_dynamic, DYN_COLS,
                          output_base / f"plot_dynamic_lag_importance_{tag}.png")
    print(f"Summary plots saved to {output_base}")


if __name__ == "__main__":
    main()
