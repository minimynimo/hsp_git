"""Task graph of the modification plan, consumed by run/driver.py.

Phase 3: tuning (RF, XGB regional + individual; regional TCN filters, then 180 vs 365 window)
Phase 4: final runs, every seed (each run also produces the Phase 5 scenario outputs from its final
         model, so no model is ever retrained for the scenarios)
Phase 5: completeness check of every scenario output
Phase 6-8: evaluation set, scoring, figures, SHAP (tasks stay blocked until their script exists)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from config import experiment as E

PY = "/data0/funato/.venv/bin/python"
ROOT = E.PIPELINE_ROOT
TUNE = E.OUT_ROOT / "tuning"
MARK = E.MARK_DIR


@dataclass
class Task:
    name: str
    cmd: Union[str, Callable[[], Optional[str]]]
    done: Callable[[], bool]
    deps: List[str] = field(default_factory=list)
    cpu: int = 1
    gpu: int = 0

    def command(self):
        return self.cmd() if callable(self.cmd) else self.cmd


def run_done(model, strategy, seed, tag=""):
    return lambda: (E.run_dir(model, strategy, seed, tag) / "_DONE").exists()


def parts_done(model, strategy, seed, worker, n_workers, tag=""):
    def f():
        from hsp.io import load_basin_pool
        rd = E.run_dir(model, strategy, seed, tag)
        return all((rd / "parts" / f"{b}.parquet").exists() for b in load_basin_pool()[worker::n_workers])
    return f


def exists(p):
    return lambda: Path(p).exists()


def script(rel, args=""):
    """Command for a pipeline script, or None while the script does not exist yet."""
    return lambda: f"{PY} {rel}{' ' + args if args else ''}" if (ROOT / rel).exists() else None


def tcn_selected():
    f = TUNE / "TCN_regional" / "selected.json"
    return json.loads(f.read_text()) if f.exists() else None


def build_tasks():
    T = []
    # ---------------- Phase 3: tuning ----------------
    T.append(Task("tune_xgb_regional", f"{PY} train/trees.py --model XGB --strategy regional --mode tune --device cuda --n_jobs 8",
                  exists(TUNE / "XGB_regional" / "selected.json"), cpu=8, gpu=1))
    T.append(Task("tune_rf_regional", f"{PY} train/trees.py --model RF --strategy regional --mode tune --n_jobs 32",
                  exists(TUNE / "RF_regional" / "selected.json"), cpu=32))
    for m in ["RF", "XGB"]:
        T.append(Task(f"tune_{m.lower()}_individual", f"{PY} train/trees.py --model {m} --strategy individual --mode tune --n_jobs 8",
                      exists(TUNE / f"{m}_individual" / "selected.json"), cpu=8))
    w0 = E.TCN_REGIONAL["seq_length"]
    for f in E.TCN_FILTER_GRID:
        T.append(Task(f"tcn_regional_f{f}_w{w0}", f"{PY} train/nets.py regional --model TCN --seed 42 --num_filters {f} --seq_length {w0}",
                      run_done("TCN", "regional", 42, f"f{f}_w{w0}"), cpu=1, gpu=1))
    T.append(Task("tcn_select_filters", f"{PY} run/select_tcn.py filters", exists(TUNE / "TCN_regional" / "filters.json"),
                  deps=[f"tcn_regional_f{f}_w{w0}" for f in E.TCN_FILTER_GRID]))

    def tcn_alt_cmd():
        fj = TUNE / "TCN_regional" / "filters.json"
        if not fj.exists():
            return None
        f = json.loads(fj.read_text())["num_filters"]
        return f"{PY} train/nets.py regional --model TCN --seed 42 --num_filters {f} --seq_length {E.TCN_SEQ_ALTERNATIVE}"

    def tcn_alt_done():
        fj = TUNE / "TCN_regional" / "filters.json"
        return fj.exists() and run_done("TCN", "regional", 42, f"f{json.loads(fj.read_text())['num_filters']}_w{E.TCN_SEQ_ALTERNATIVE}")()

    T.append(Task("tcn_regional_w365", tcn_alt_cmd, tcn_alt_done, deps=["tcn_select_filters"], cpu=1, gpu=1))
    T.append(Task("tcn_select_final", f"{PY} run/select_tcn.py final", exists(TUNE / "TCN_regional" / "selected.json"),
                  deps=["tcn_regional_w365"]))
    phase3 = [t.name for t in T]

    # ---------------- Phase 4: final runs ----------------
    for s in E.SEEDS:
        T.append(Task(f"lstm_regional_s{s}", f"{PY} train/nets.py regional --model LSTM --seed {s}",
                      run_done("LSTM", "regional", s), deps=phase3, cpu=1, gpu=1))
        T.append(Task(f"lstm_adapted_s{s}", f"{PY} train/nets.py adapted --model LSTM --seed {s}",
                      run_done("LSTM", "adapted", s), deps=[f"lstm_regional_s{s}"], cpu=1, gpu=1))

        def tcn_cmd(s=s):
            sel = tcn_selected()
            return None if sel is None else (f"{PY} train/nets.py regional --model TCN --seed {s} "
                                             f"--num_filters {sel['num_filters']} --seq_length {sel['seq_length']}")

        def tcn_ad_cmd(s=s):
            sel = tcn_selected()
            return None if sel is None else f"{PY} train/nets.py adapted --model TCN --seed {s} --regional_tag {sel['tag']}"

        def tcn_done(strategy, s=s):
            return lambda: tcn_selected() is not None and run_done("TCN", strategy, s, tcn_selected()["tag"])()

        T.append(Task(f"tcn_regional_s{s}", tcn_cmd, tcn_done("regional"), deps=phase3, cpu=1, gpu=1))
        T.append(Task(f"tcn_adapted_s{s}", tcn_ad_cmd, tcn_done("adapted"), deps=[f"tcn_regional_s{s}"], cpu=1, gpu=1))
        T.append(Task(f"xgb_regional_s{s}", f"{PY} train/trees.py --model XGB --strategy regional --mode final --seed {s} --device cuda --n_jobs 8",
                      run_done("XGB", "regional", s), deps=phase3, cpu=8, gpu=1))
        T.append(Task(f"rf_regional_s{s}", f"{PY} train/trees.py --model RF --strategy regional --mode final --seed {s} --n_jobs 32",
                      run_done("RF", "regional", s), deps=phase3, cpu=32))
    NW = 3
    for s in E.SEEDS_INDIVIDUAL_NN:
        for m in ["LSTM", "TCN"]:
            ws = []
            for w in range(NW):
                n = f"{m.lower()}_individual_s{s}_w{w}"
                ws.append(n)
                T.append(Task(n, f"{PY} train/nets.py individual --model {m} --seed {s} --worker {w} --n_workers {NW}",
                              parts_done(m, "individual", s, w, NW), deps=phase3, cpu=1, gpu=1))
            T.append(Task(f"{m.lower()}_individual_s{s}", f"{PY} train/nets.py individual --model {m} --seed {s} --n_workers {NW} --finalize",
                          run_done(m, "individual", s), deps=ws))
    for s in E.SEEDS_INDIVIDUAL_TREE:
        for m in ["RF", "XGB"]:
            T.append(Task(f"{m.lower()}_individual_s{s}", f"{PY} train/trees.py --model {m} --strategy individual --mode final --seed {s} --n_jobs 16",
                          run_done(m, "individual", s), deps=phase3, cpu=16))
    hbv_mat = E.run_dir("HBV", "individual", E.HBV_OPTIMISER_SEED) / "matlab"
    T.append(Task("hbv_postprocess", lambda: f"{PY} hbv/postprocess.py" if (hbv_mat / "_MATLAB_FINISHED").exists() else None,
                  run_done("HBV", "individual", E.HBV_OPTIMISER_SEED), cpu=8))
    phase4 = [t.name for t in T if t.name not in phase3]

    # ---------------- Phase 5-8 ----------------
    T.append(Task("check_scenarios", script("checks/check_runs.py"), exists(MARK / "check_runs.done"), deps=phase4, cpu=8))
    T.append(Task("eval_set", script("analysis/eval_set.py"), exists(MARK / "eval_set.done"), deps=["check_scenarios"], cpu=4))
    T.append(Task("scoring", script("analysis/scoring.py"), exists(MARK / "scoring.done"), deps=["eval_set"], cpu=32))
    T.append(Task("figures", script("figures/make_all.py"), exists(MARK / "figures.done"), deps=["scoring"], cpu=4))
    T.append(Task("shap", script("analysis/shap_attribution.py"), exists(MARK / "shap.done"), deps=["scoring"], cpu=16, gpu=1))
    T.append(Task("shap_figure", script("figures/f02_shap.py"), exists(MARK / "shap_figure.done"), deps=["shap"], cpu=2))
    return T
