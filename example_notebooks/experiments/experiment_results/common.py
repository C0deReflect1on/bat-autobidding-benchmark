from __future__ import annotations

import sys
from pathlib import Path

from config import (
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_METADATA,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_RESULTS_ROOT = Path(__file__).resolve().parent
SUBSAMPLE_ROOT = FPA_EXPERIMENT_SUBSAMPLE_METADATA.parent


def ensure_repo_root_on_path() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def require_subsample_ready() -> None:
    missing = [
        str(path)
        for path in required_subsample_paths()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The permanent experiment subsample is missing. "
            "Run prepare_subsample.py first. Missing files: "
            f"{missing}"
        )


def required_subsample_paths() -> list[Path]:
    return [
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
        FPA_EXPERIMENT_SUBSAMPLE_METADATA,
    ]
