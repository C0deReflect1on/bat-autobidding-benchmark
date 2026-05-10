from __future__ import annotations

from typing import Any

from config import (
    FPA_CAMPAIGNS_HOLDOUT_TEST,
    FPA_CAMPAIGNS_TRAIN_VAL,
    FPA_CAMPAIGNS_VAL_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
    FPA_STATS_HOLDOUT_TEST,
    FPA_STATS_TRAIN_VAL,
    FPA_STATS_VAL_VAL,
)


_SPLIT_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "full_train_val_holdout": {
        "train": {
            "campaigns_path": str(FPA_CAMPAIGNS_TRAIN_VAL),
            "stats_path": str(FPA_STATS_TRAIN_VAL),
        },
        "val": {
            "campaigns_path": str(FPA_CAMPAIGNS_VAL_VAL),
            "stats_path": str(FPA_STATS_VAL_VAL),
        },
        "test_holdout": {
            "campaigns_path": str(FPA_CAMPAIGNS_HOLDOUT_TEST),
            "stats_path": str(FPA_STATS_HOLDOUT_TEST),
        },
    },
    "subsample_train_val_holdout": {
        "train": {
            "campaigns_path": str(FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN),
            "stats_path": str(FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN),
        },
        "val": {
            "campaigns_path": str(FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL),
            "stats_path": str(FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL),
        },
        "test_holdout": {
            "campaigns_path": str(FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT),
            "stats_path": str(FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT),
        },
    },
}


def resolve_split_set(split_set: str) -> dict[str, dict[str, str]]:
    split_key = str(split_set).strip()
    if split_key not in _SPLIT_REGISTRY:
        raise ValueError(
            f"Unknown split_set '{split_key}'. Supported values: {sorted(_SPLIT_REGISTRY.keys())}"
        )
    return {
        role: dict(paths)
        for role, paths in _SPLIT_REGISTRY[split_key].items()
    }


def list_split_sets() -> list[str]:
    return sorted(_SPLIT_REGISTRY.keys())


def split_registry_summary() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        split_set: {
            role: dict(paths)
            for role, paths in split_map.items()
        }
        for split_set, split_map in _SPLIT_REGISTRY.items()
    }
