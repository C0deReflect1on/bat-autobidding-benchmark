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
    drlb_profile: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    family_key = str(family).strip().lower()
    experiments_data_dir = None if artifacts_root is None else Path(artifacts_root)

    if family_key in {"baseline", "baselines"}:
        config = build_baseline_config(
            run_name,
            split_set=split_set,
            experiments_data_dir=experiments_data_dir,
        )
        if n_trials is not None:
            config = replace(config, n_trials=int(n_trials))
        return run_experiment_inprocess(config, verbose=verbose)

    if family_key == "rlb":
        config = build_rlb_config(
            run_name,
            split_set=split_set,
            experiments_data_dir=experiments_data_dir,
        )
        if n_trials is not None:
            config = replace(config, n_trials=int(n_trials))
        return run_experiment_inprocess(config, verbose=verbose)

    if family_key != "drlb":
        raise ValueError(f"Unsupported experiment family '{family}'")

    config = build_drlb_config(
        run_name,
        profile=drlb_profile,
        split_set=split_set,
        experiments_data_dir=experiments_data_dir,
    )
    if n_trials is not None:
        config = replace(config, n_trials=int(n_trials))
    if max_train_steps is not None:
        config = replace(config, max_steps=int(max_train_steps))

    profile = get_drlb_profile(config.drlb_profile or config.run_name)
    return run_experiment_inprocess(
        config,
        verbose=verbose,
        base_drlb_params=profile["base_drlb_params"],
        reference_model_params=profile["reference_model_params"],
        state_type=profile["state_type"],
        objective=profile["objective"],
        search_space_fn=profile["search_space_fn"],
        n_trials=config.n_trials,
        max_train_steps=config.max_steps,
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
    profile: str | None = None,
    split_set: str = "subsample_train_val_holdout",
    artifacts_root: str | Path | None = None,
    n_trials: int | None = None,
    max_train_steps: int | None = None,
    use_all_train_timesteps: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run DRLB in-process.

    ``run_name`` names the output directory under ``.../drlb/``.
    ``profile`` selects a built-in hyperparameter preset; if omitted, defaults to ``run_name``.
    """
    config = build_drlb_config(
        run_name,
        profile=profile,
        split_set=split_set,
        experiments_data_dir=None if artifacts_root is None else Path(artifacts_root),
    )
    if n_trials is not None:
        config = replace(config, n_trials=int(n_trials))
    if max_train_steps is not None:
        config = replace(config, max_steps=int(max_train_steps))
    if use_all_train_timesteps:
        config = replace(config, max_steps=None)

    profile_data = get_drlb_profile(config.drlb_profile or config.run_name)
    return run_experiment_inprocess(
        config,
        verbose=verbose,
        base_drlb_params=profile_data["base_drlb_params"],
        reference_model_params=profile_data["reference_model_params"],
        state_type=profile_data["state_type"],
        objective=profile_data["objective"],
        search_space_fn=profile_data["search_space_fn"],
        n_trials=config.n_trials,
        max_train_steps=config.max_steps,
    )
