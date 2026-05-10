from __future__ import annotations

from typing import Any

from .adapters.baseline_adapter import run_baseline_experiment, run_baseline_experiment_inprocess
from .adapters.drlb_adapter import run_drlb_experiment, run_drlb_experiment_inprocess
from .adapters.rlb_adapter import run_rlb_experiment, run_rlb_experiment_inprocess
from .infra.reproducibility import initialize_runtime_seeds
from .infra.split_utils import resolve_normalized_splits


_FAMILY_RUNNERS = {
    "baselines": {
        "summary": run_baseline_experiment,
        "inprocess": run_baseline_experiment_inprocess,
    },
    "drlb": {
        "summary": run_drlb_experiment,
        "inprocess": run_drlb_experiment_inprocess,
    },
    "rlb": {
        "summary": run_rlb_experiment,
        "inprocess": run_rlb_experiment_inprocess,
    },
}


def run_family(
    config,
    *,
    mode: str,
    verbose: bool,
    family_kwargs: dict[str, Any],
) -> dict[str, Any]:
    family_key = "baselines" if config.family in {"baseline", "baselines"} else config.family
    runners = _FAMILY_RUNNERS.get(family_key)
    if runners is None:
        raise ValueError(f"Unsupported experiment family '{config.family}'")

    normalized_splits = resolve_normalized_splits(config)
    initialize_runtime_seeds(config.seeds)
    return runners[mode](
        config,
        normalized_splits,
        verbose=verbose,
        **family_kwargs,
    )


def run_experiment(
    config,
    *,
    verbose: bool = False,
    **family_kwargs: Any,
) -> dict[str, Any]:
    return run_family(
        config,
        mode="summary",
        verbose=verbose,
        family_kwargs=family_kwargs,
    )


def run_experiment_inprocess(
    config,
    *,
    verbose: bool = False,
    **family_kwargs: Any,
) -> dict[str, Any]:
    return run_family(
        config,
        mode="inprocess",
        verbose=verbose,
        family_kwargs=family_kwargs,
    )
