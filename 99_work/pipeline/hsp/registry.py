"""The run matrix (plan 4.1): every model x strategy x seed whose outputs the analysis needs."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import experiment as E


@dataclass(frozen=True)
class Run:
    model: str
    strategy: str
    seed: int
    tag: str = ""

    @property
    def dir(self) -> Path:
        return E.run_dir(self.model, self.strategy, self.seed, self.tag)

    @property
    def key(self) -> str:
        return f"{self.model}-{self.strategy}"


def tcn_tag() -> str:
    f = E.OUT_ROOT / "tuning" / "TCN_regional" / "selected.json"
    if not f.exists():
        raise FileNotFoundError(f"TCN selection missing: {f}")
    return json.loads(f.read_text())["tag"]


def expected_runs() -> List[Run]:
    runs = [Run("HBV", "individual", E.HBV_OPTIMISER_SEED)]
    for m in ["RF", "XGB"]:
        runs += [Run(m, "regional", s) for s in E.SEEDS]
        runs += [Run(m, "individual", s) for s in E.SEEDS_INDIVIDUAL_TREE]
    tag = tcn_tag()
    for m, t in [("LSTM", ""), ("TCN", tag)]:
        runs += [Run(m, "regional", s, t) for s in E.SEEDS]
        runs += [Run(m, "adapted", s, t) for s in E.SEEDS]
        runs += [Run(m, "individual", s) for s in E.SEEDS_INDIVIDUAL_NN]
    return runs


def combos() -> List[str]:
    """model-strategy keys in display order."""
    out = []
    for m in E.MODELS:
        for s in E.STRATEGIES[m]:
            out.append(f"{m}-{s}")
    return out


def runs_of(key: str) -> List[Run]:
    return [r for r in expected_runs() if r.key == key]
