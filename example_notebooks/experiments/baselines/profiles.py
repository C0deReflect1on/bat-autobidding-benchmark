from __future__ import annotations

from pathlib import Path

from ..base_exp_config import ExperimentConfig


_BASELINE_PROFILES: dict[str, dict[str, object]] = {
    "linear_default": {"model_name": "linear", "n_trials": 1},
    "ta_pid_default": {"model_name": "ta_pid", "n_trials": 1},
    "m_pid_default": {"model_name": "m_pid", "n_trials": 1},
    "mystique_default": {"model_name": "mystique", "n_trials": 1},
    "broi_default": {"model_name": "broi", "n_trials": 1},
}


def list_profiles() -> list[str]:
    return sorted(_BASELINE_PROFILES.keys())


def get_profile(run_name: str) -> dict[str, object]:
    if run_name not in _BASELINE_PROFILES:
        raise ValueError(
            f"Unknown baseline run_name '{run_name}'. Supported values: {list_profiles()}"
        )
    return dict(_BASELINE_PROFILES[run_name])


def build_config(
    run_name: str,
    *,
    split_set: str = "subsample_train_val_holdout",
    experiments_data_dir: Path | None = None,
) -> ExperimentConfig:
    profile = get_profile(run_name)
    return ExperimentConfig(
        experiment_name=run_name,
        run_name=run_name,
        n_trials=int(profile["n_trials"]),
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="baselines",
        split_set=split_set,
        model_config={"model_name": str(profile["model_name"])},
        checkpoint_policy="none",
        experiments_data_dir=experiments_data_dir or Path(__file__).resolve().parents[1],
    )
