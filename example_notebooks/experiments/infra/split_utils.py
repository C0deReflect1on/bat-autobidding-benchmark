from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .split_registry import resolve_split_set


_NORMALIZED_SPLIT_ALIASES = {
    "train": ("train", "train_val"),
    "val": ("val", "val_val", "test"),
    "test_holdout": ("test_holdout", "holdout_test", "test"),
}


def resolve_normalized_splits(config) -> dict[str, dict[str, str]]:
    data_config = _resolve_data_config(config)
    splits: dict[str, dict[str, str]] = {}
    for normalized_role in ("train", "val", "test_holdout"):
        campaigns_path, stats_path = _resolve_split_paths(config, data_config, normalized_role)
        _ensure_split_paths_exist(config, normalized_role, campaigns_path, stats_path)
        splits[normalized_role] = {
            "campaigns_path": campaigns_path,
            "stats_path": stats_path,
        }
    return splits


def split_fingerprint(
    normalized_splits: dict[str, dict[str, str]],
) -> str:
    hasher = hashlib.sha256()
    for role in ("train", "val", "test_holdout"):
        split = normalized_splits[role]
        hasher.update(role.encode("utf-8"))
        for key in ("campaigns_path", "stats_path"):
            path = Path(split[key])
            hasher.update(str(path).encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_split_manifest(config, normalized_splits: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "family": config.family,
        "run_name": config.run_name,
        "split_set": config.split_set,
        "fingerprint": split_fingerprint(normalized_splits),
        "splits": {
            role: {
                "campaigns_path": split["campaigns_path"],
                "stats_path": split["stats_path"],
            }
            for role, split in normalized_splits.items()
        },
    }


def _resolve_data_config(config) -> dict[str, Any]:
    if config.data_config:
        return config.data_config
    if config.split_set:
        return resolve_split_set(config.split_set)
    raise ValueError(
        f"Config '{config.experiment_name}' must define either data_config or split_set."
    )


def _resolve_split_paths(
    config,
    data_config: dict[str, Any],
    normalized_role: str,
) -> tuple[str, str]:
    if normalized_role not in _NORMALIZED_SPLIT_ALIASES:
        raise ValueError(
            f"Unknown normalized split role '{normalized_role}'. "
            f"Supported roles: {sorted(_NORMALIZED_SPLIT_ALIASES.keys())}"
        )

    for alias in _NORMALIZED_SPLIT_ALIASES[normalized_role]:
        split_cfg = data_config.get(alias)
        if isinstance(split_cfg, dict):
            campaigns_path = split_cfg.get("campaigns_path")
            stats_path = split_cfg.get("stats_path")
            if campaigns_path and stats_path:
                return str(campaigns_path), str(stats_path)

    available = sorted(data_config.keys())
    raise ValueError(
        f"Missing split role '{normalized_role}' for config '{config.experiment_name}'. "
        f"Checked aliases {_NORMALIZED_SPLIT_ALIASES[normalized_role]}; available keys: {available}"
    )


def _ensure_split_paths_exist(
    config,
    normalized_role: str,
    campaigns_path: str,
    stats_path: str,
) -> None:
    campaigns_file = Path(campaigns_path)
    stats_file = Path(stats_path)
    missing = [str(path) for path in (campaigns_file, stats_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Split '{normalized_role}' is missing required file(s) for config "
            f"'{config.experiment_name}': {missing}"
        )
