from __future__ import annotations

from typing import Any

from .adapters.baseline_adapter import run_baseline_experiment, run_baseline_experiment_inprocess
from .adapters.drlb_adapter import run_drlb_experiment, run_drlb_experiment_inprocess
from .adapters.rlb_adapter import run_rlb_experiment, run_rlb_experiment_inprocess
from .infra.reproducibility import initialize_runtime_seeds
from .infra.split_utils import resolve_normalized_splits


def run_experiment(
    config,
    *,
    verbose: bool = False,
    **family_kwargs: Any,
) -> dict[str, Any]:
    normalized_splits = resolve_normalized_splits(config)
    initialize_runtime_seeds(config.seeds)

    if config.family == "drlb":
        return run_drlb_experiment(
            config,
            normalized_splits,
            verbose=verbose,
            **family_kwargs,
        )
    if config.family in {"baseline", "baselines"}:
        return run_baseline_experiment(
            config,
            normalized_splits,
            verbose=verbose,
        )
    if config.family == "rlb":
        return run_rlb_experiment(
            config,
            normalized_splits,
            verbose=verbose,
            **family_kwargs,
        )

    raise ValueError(f"Unsupported experiment family '{config.family}'")


def run_experiment_inprocess(
    config,
    *,
    verbose: bool = False,
    **family_kwargs: Any,
) -> dict[str, Any]:
    normalized_splits = resolve_normalized_splits(config)
    initialize_runtime_seeds(config.seeds)

    if config.family == "drlb":
        return run_drlb_experiment_inprocess(
            config,
            normalized_splits,
            verbose=verbose,
            **family_kwargs,
        )
    if config.family in {"baseline", "baselines"}:
        return run_baseline_experiment_inprocess(
            config,
            normalized_splits,
            verbose=verbose,
        )
    if config.family == "rlb":
        return run_rlb_experiment_inprocess(
            config,
            normalized_splits,
            verbose=verbose,
            **family_kwargs,
        )

    raise ValueError(f"Unsupported experiment family '{config.family}'")
