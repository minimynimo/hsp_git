"""
Resume support for main_rf.py / main_rf_hp.py.

Both scripts name each run's output directory with a timestamp
(`rf_run_<MMDD_HHMM>_seed<N>`), so a restart after being killed always creates
a brand-new directory and retrains from scratch even if an identical
hyperparameter combination already finished. Since the hyperparameters ARE
saved to that run's `cfg.json`, a prior completed run can still be found by
scanning sibling `cfg.json` files for one whose relevant fields match and
whose model file exists alongside it -- without needing to change the
directory-naming scheme itself (which would orphan already-completed runs).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def find_completed_run(search_root: Path, cfg: Dict, match_keys: List[str],
                        model_filename: str = 'rf_model.joblib') -> Optional[Path]:
    """Search immediate subdirectories of `search_root` for one whose saved
    cfg.json matches `cfg` on every key in `match_keys` and that has
    `model_filename` present -- i.e. a completed run of the same hyperparameters.

    Returns the matching run directory, or None if no match is found.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None

    for cfg_path in sorted(search_root.glob('*/cfg.json')):
        run_dir = cfg_path.parent
        if not (run_dir / model_filename).exists():
            continue
        try:
            with open(cfg_path) as fp:
                saved_cfg = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue

        if all(saved_cfg.get(k) == cfg.get(k) for k in match_keys):
            return run_dir

    return None
