"""Run directories and outputs, identical for every model and strategy (4.2).

run_dir/
  config.json        hyperparameters, features, loss, dates, seed, git HEAD + code hash
  sim/og.parquet     daily simulated discharge (mm/d), DATA rows TRAIN[0]..TEST[1], columns = basins
  sim/<scen>.parquet one per synthetic scenario, rows = that scenario's window
  metrics.csv        basin, period (cal/val/test), KGE, NSE, r, alpha, beta, n
  parts/<basin>.parquet  per-basin outputs of individual / adapted runs before merging
  _DONE              written last
"""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import experiment as E
from hsp.io import load_forcing, to_jsonable
from hsp.metrics import period_metrics

SIM_COLS = ["og"] + E.SCENARIOS


def code_fingerprint() -> Dict:
    root = E.PIPELINE_ROOT
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    digest = h.hexdigest()[:16]
    snap = E.CODE_SNAPSHOT_DIR / f"{digest}.tar.gz"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in files:
                tar.add(p, arcname=str(p.relative_to(root)))
        tmp = snap.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(snap)
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "."],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:  # noqa: BLE001
        head, dirty = "unknown", True
    return {"git_head": head, "git_dirty": dirty, "code_sha256": digest, "code_snapshot": str(snap)}


def init_run(run_dir: Path, cfg: Dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    full = {
        **cfg,
        "periods": {"data_start": E.DATA_START, "warmup": E.WARMUP, **E.PERIODS},
        "dyn_inputs": E.DYN_COLS,
        "scenarios": E.SCENARIOS,
        "code": code_fingerprint(),
    }
    (run_dir / "config.json").write_text(json.dumps(to_jsonable(full), indent=2))
    return run_dir


def is_done(run_dir: Path) -> bool:
    return (run_dir / "_DONE").exists()


def save_part(run_dir: Path, basin: str, sims: Dict[str, pd.Series], extra: Dict = None):
    """Per-basin outputs (individual / adapted runs): long table Date x scenario."""
    d = run_dir / "parts"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: v.astype("float32") for k, v in sims.items()})
    df.index.name = "Date"
    tmp = d / f"{basin}.tmp.parquet"
    df.to_parquet(tmp)
    tmp.replace(d / f"{basin}.parquet")
    if extra is not None:
        (d / f"{basin}.json").write_text(json.dumps(to_jsonable(extra)))


def part_done(run_dir: Path, basin: str) -> bool:
    return (run_dir / "parts" / f"{basin}.parquet").exists()


def finalize(run_dir: Path, sims: Dict[str, Dict[str, pd.Series]] = None, basins=None):
    """Write sim/*.parquet + metrics.csv, then _DONE. `sims[scenario][basin]` -> Series, or None to
    assemble from parts/."""
    if sims is None:
        sims = {c: {} for c in SIM_COLS}
        for f in sorted((run_dir / "parts").glob("*.parquet")):
            df = pd.read_parquet(f)
            for c in SIM_COLS:
                if c in df:
                    sims[c][f.stem] = df[c].dropna()
    simdir = run_dir / "sim"
    simdir.mkdir(exist_ok=True)
    for scen, per_basin in sims.items():
        if not per_basin:
            continue
        wide = pd.DataFrame(per_basin).sort_index().astype("float32")
        wide.index.name = "Date"
        wide.to_parquet(simdir / f"{scen}.parquet")
    rows = []
    for basin, s in sims["og"].items():
        obs = load_forcing(basin)[E.Q_COL]
        for r in period_metrics(obs, s):
            rows.append({"basin": basin, **r})
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "_DONE").write_text("ok\n")
    m = pd.DataFrame(rows)
    return m.groupby("period")[["KGE", "NSE"]].median()


def load_sim(run_dir: Path, scenario: str = "og") -> pd.DataFrame:
    f = Path(run_dir) / "sim" / f"{scenario}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"missing model output {f}")
    return pd.read_parquet(f)


def clip_nonneg(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)
