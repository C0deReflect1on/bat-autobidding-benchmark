from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .baselines.profiles import build_config as build_baseline_config
from .drlb.profiles import build_config as build_drlb_config
from .drlb.profiles import get_profile as get_drlb_profile
from .rlb.profiles import build_config as build_rlb_config
from .shared_runner import run_experiment_inprocess


def run_profile_inprocess(
    family: str,
    run_name: str,
    *,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
    max_train_steps: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    config = build_family_config(
        family,
        run_name,
        split_set=split_set,
        artifacts_root=artifacts_root,
        n_trials=n_trials,
        max_train_steps=max_train_steps,
    )
    return run_experiment_inprocess(
        config,
        verbose=verbose,
        **build_family_runner_kwargs(_normalize_family(family), run_name, config=config),
    )


def run_baseline_profile_inprocess(
    run_name: str = "linear_default",
    *,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    return run_profile_inprocess(
        "baselines",
        run_name,
        split_set=split_set,
        artifacts_root=artifacts_root,
        n_trials=n_trials,
    )


def run_rlb_profile_inprocess(
    run_name: str = "rlb_default",
    *,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    return run_profile_inprocess(
        "rlb",
        run_name,
        split_set=split_set,
        artifacts_root=artifacts_root,
        n_trials=n_trials,
        verbose=verbose,
    )


def run_drlb_profile_inprocess(
    run_name: str = "drlb_smooth",
    *,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
    max_train_steps: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    return run_profile_inprocess(
        "drlb",
        run_name,
        split_set=split_set,
        artifacts_root=artifacts_root,
        n_trials=n_trials,
        max_train_steps=max_train_steps,
        verbose=verbose,
    )


def build_family_config(
    family: str,
    run_name: str,
    *,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
    max_train_steps: int | None = None,
):
    normalized_family = _normalize_family(family)
    experiments_data_dir = None if artifacts_root is None else Path(artifacts_root)

    if normalized_family == "baselines":
        config = build_baseline_config(
            run_name,
            split_set=split_set,
            experiments_data_dir=experiments_data_dir,
        )
    elif normalized_family == "rlb":
        config = build_rlb_config(
            run_name,
            split_set=split_set,
            experiments_data_dir=experiments_data_dir,
        )
    elif normalized_family == "drlb":
        config = build_drlb_config(
            run_name,
            split_set=split_set,
            experiments_data_dir=experiments_data_dir,
        )
    else:
        raise ValueError(f"Unsupported experiment family '{family}'")

    if n_trials is not None:
        config = replace(config, n_trials=int(n_trials))
    if max_train_steps is not None and normalized_family == "drlb":
        config = replace(config, max_steps=int(max_train_steps))
    return config


def build_family_runner_kwargs(
    family: str,
    run_name: str,
    *,
    config,
) -> dict[str, Any]:
    if family == "drlb":
        profile = get_drlb_profile(run_name)
        return {
            "base_drlb_params": profile["base_drlb_params"],
            "baseline_model_params": profile["baseline_model_params"],
            "exp_type": profile["exp_type"],
            "objective": profile["objective"],
            "search_space_fn": profile["search_space_fn"],
            "n_trials": config.n_trials,
            "max_train_steps": config.max_steps,
        }
    return {}


def _normalize_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized == "baseline":
        return "baselines"
    return normalized
