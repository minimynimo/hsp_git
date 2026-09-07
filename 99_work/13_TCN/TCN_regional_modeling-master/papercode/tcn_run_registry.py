"""
Resume support for main_tcn.py / main_tcn_hp.py.

Both scripts name each run's output directory with a timestamp
(`tcn_run_<MMDD_HHMM>_seed<N>`), so a restart after being killed always creates
a brand-new directory and retrains from scratch even if an identical
hyperparameter combination already finished ALL of its epochs. Since the
hyperparameters ARE saved to that run's `cfg.json`, and per-epoch checkpoints
are saved as `tcn_model_epoch<N>.pt`, a prior completed run can be found by
scanning sibling `cfg.json` files for one whose relevant fields match and that
has a checkpoint for the REQUIRED epoch count -- without needing to change the
directory-naming scheme itself (which would orphan already-completed runs).
(Same pattern as RF's/XGB's rf_run_registry.py/xgb_run_registry.py.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


def load_tcn_state_dict(path, map_location=None):
    """Load a TCN regional checkpoint saved by main_tcn.py's train(), tolerating both the
    checkpoint format (dict with 'model_state_dict'/'optimizer_state_dict'/'epoch', used since
    checkpoint-resume was added) and the older bare-state_dict format every checkpoint used
    before that. Returns just the model state_dict, ready for model.load_state_dict(...)."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def find_completed_run(search_root: Path, cfg: Dict, match_keys: List[str],
                        model_filename: str) -> Optional[Path]:
    """Search immediate subdirectories of `search_root` for one whose saved
    cfg.json matches `cfg` on every key in `match_keys` and that has
    `model_filename` (e.g. `tcn_model_epoch{target_epochs}.pt`) present --
    i.e. a run that already completed training through that exact epoch count.

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


def find_resumable_run(search_root: Path, cfg: Dict, match_keys: List[str],
                        checkpoint_glob: str, epoch_from_stem) -> "tuple[Optional[Path], int]":
    """Like `find_completed_run`, but for a run that matches on `match_keys` and has SOME
    epoch checkpoint (via `checkpoint_glob`, e.g. "tcn_model_epoch*.pt") but not the full
    `cfg['epochs']` count -- i.e. a run that was killed partway through and can be resumed
    from its highest saved checkpoint instead of retraining from epoch 0.

    `epoch_from_stem(path) -> int` extracts the epoch number from a checkpoint's filename
    stem (family-specific, since TCN/LSTM name checkpoints differently).

    Returns (run_dir, last_epoch) for the match with the highest last_epoch found across all
    matching sibling directories, or (None, 0) if no resumable run exists. A run already at or
    past cfg['epochs'] is not resumable (that's `find_completed_run`'s job).
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None, 0

    best_dir, best_epoch = None, 0
    for cfg_path in sorted(search_root.glob('*/cfg.json')):
        run_dir = cfg_path.parent
        try:
            with open(cfg_path) as fp:
                saved_cfg = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if not all(saved_cfg.get(k) == cfg.get(k) for k in match_keys):
            continue

        epochs_found = []
        for p in run_dir.glob(checkpoint_glob):
            try:
                epochs_found.append(epoch_from_stem(p))
            except (ValueError, TypeError):
                continue
        if not epochs_found:
            continue
        last_epoch = max(epochs_found)
        if last_epoch >= cfg['epochs']:
            continue  # fully done -- find_completed_run's territory, not resumable
        if last_epoch > best_epoch:
            best_epoch = last_epoch
            best_dir = run_dir

    return best_dir, best_epoch
