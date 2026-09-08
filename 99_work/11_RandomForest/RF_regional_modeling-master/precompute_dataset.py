"""
Build (and cache) the regional RF feature matrix once, before launching
run_rf_hp_tuning.sh's parallel hyperparameter jobs.

X/y/scalers do not depend on any RF hyperparameter, only on
(loc, ver, seq_length, train period, no_static, basin list). Running this
once means all N parallel main_rf_hp.py jobs load the cached arrays instead
of each redoing the (previously very slow, now vectorized) feature-window
construction from scratch.

Usage:
    python3 precompute_dataset.py --no_static False
"""

import argparse
from pathlib import Path

import pandas as pd

from papercode.datautils import add_basin_attributes, load_attributes, VALID_ATTR
from papercode.utils import get_basin_list
from papercode.rf_features import load_or_build_dataset
from main_rf_hp import GLOBAL_SETTINGS, HP_SPLIT, ver_name, DATASET_CACHE_ROOT, str2bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_static', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='camels', choices=['camels', 'caravan'])
    args = parser.parse_args()

    cfg = dict(GLOBAL_SETTINGS)
    cfg['dataset'] = args.dataset
    cfg.update(HP_SPLIT[args.dataset])
    cfg['no_static'] = args.no_static

    basins = get_basin_list(cfg['loc'], dataset=cfg['dataset'])

    db_dir = Path(__file__).parent / "cache"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "attributes.db")
    add_basin_attributes(db_path=db_path, loc=cfg['loc'], ver_name=ver_name)

    attributes = None
    if not cfg['no_static']:
        attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True,
                                      keep_features=VALID_ATTR)

    X, y, sample_info_df, basin_scalers, static_means, static_stds = load_or_build_dataset(
        cfg, basins, DATASET_CACHE_ROOT, attributes)
    print(f"Dataset ready: X shape {X.shape}, y shape {y.shape}")


if __name__ == "__main__":
    main()
