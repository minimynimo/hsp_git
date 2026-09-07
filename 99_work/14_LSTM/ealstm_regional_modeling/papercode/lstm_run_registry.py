"""
Resume support for main_lstm_regional.py / main_lstm_regional_hp.py.

Both scripts name each run's output directory with a timestamp
(`run_<MMDD_HHMM%S>_<loc>_seed<N>`), so a restart after being killed always
creates a brand-new directory and retrains from scratch even if an identical
hyperparameter combination already finished ALL of its epochs. Since the
hyperparameters ARE saved to that run's `cfg.json` (written via `_dump_cfg`
before training starts), a prior completed run can be found by scanning
sibling `cfg.json` files for one whose relevant fields match (including
`epochs`, so a completed 10-epoch run never satisfies a request for 30) and
that has `lstm_model_final.pt` present -- which is only written once the full
epoch loop finishes -- without needing to change the directory-naming scheme
itself (which would orphan already-completed runs).
(Same pattern as RF's/XGB's/TCN's *_run_registry.py.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch


def load_lstm_state_dict(path, map_location=None):
    """Load an LSTM regional checkpoint saved by main_lstm_regional.py's train(), tolerating
    both the per-epoch checkpoint format (dict with 'model_state_dict'/'optimizer_state_dict'/
    'epoch', used since checkpoint-resume was added) and the older bare-state_dict format every
    per-epoch checkpoint used before that -- lstm_model_final.pt is unaffected (always saved as
    a bare state_dict). Returns just the model state_dict, ready for model.load_state_dict(...)."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def find_completed_run(search_root: Path, cfg: Dict, match_keys: List[str],
                        model_filename: Union[str, List[str]] = 'lstm_model_final.pt') -> Optional[Path]:
    """Search immediate subdirectories of `search_root` for one whose saved
    cfg.json matches `cfg` on every key in `match_keys` and that has every
    file in `model_filename` present -- i.e. a run that already completed
    training (and, for main_lstm_regional_hp.py, evaluation) through the
    required epoch count. Pass a list to also require the eval-stage output
    (e.g. `['lstm_model_final.pt', 'hp_metrics_cal.csv', 'hp_metrics_eva.csv']`)
    so a run killed after saving the model but before finishing evaluation is
    correctly NOT treated as complete.

    Returns the matching run directory, or None if no match is found.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None

    required_files = [model_filename] if isinstance(model_filename, str) else model_filename

    for cfg_path in sorted(search_root.glob('*/cfg.json')):
        run_dir = cfg_path.parent
        if not all((run_dir / fname).exists() for fname in required_files):
            continue
        try:
            with open(cfg_path) as fp:
                saved_cfg = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue

        if all(saved_cfg.get(k) == cfg.get(k) for k in match_keys):
            return run_dir

    return None


def find_resumable_run(search_root: Path, cfg: Dict, match_keys: List[str]) -> "tuple[Optional[Path], int]":
    """Like `find_completed_run`, but for a run that matches on `match_keys` and has SOME
    `model_epoch{N}.pt` checkpoint but not `lstm_model_final.pt` (i.e. hasn't finished
    `cfg['epochs']` epochs) -- a run that was killed partway through and can be resumed from
    its highest saved checkpoint instead of retraining from epoch 0.

    Returns (run_dir, last_epoch) for the match with the highest last_epoch across all
    matching sibling directories, or (None, 0) if no resumable run exists.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None, 0

    best_dir, best_epoch = None, 0
    for cfg_path in sorted(search_root.glob('*/cfg.json')):
        run_dir = cfg_path.parent
        if (run_dir / 'lstm_model_final.pt').exists():
            continue  # fully done -- find_completed_run's territory
        try:
            with open(cfg_path) as fp:
                saved_cfg = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if not all(saved_cfg.get(k) == cfg.get(k) for k in match_keys):
            continue

        epochs_found = []
        for p in run_dir.glob("model_epoch*.pt"):
            try:
                epochs_found.append(int(p.stem.replace("model_epoch", "")))
            except ValueError:
                continue
        if not epochs_found:
            continue
        last_epoch = max(epochs_found)
        if last_epoch >= cfg['epochs']:
            continue
        if last_epoch > best_epoch:
            best_epoch = last_epoch
            best_dir = run_dir

    return best_dir, best_epoch
