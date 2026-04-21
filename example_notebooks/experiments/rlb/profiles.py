from __future__ import annotations

from pathlib import Path

from ..base_exp_config import ExperimentConfig


_RLB_PROFILES: dict[str, dict[str, object]] = {
    "rlb_default": {
        "n_trials": 1,
        "model_config": {
            "base_params": {
                "max_bid": 300,
                "gamma": 1.0,
                "N_bound": 72,
                "B_bound": 10000,
                "use_smoothing": False,
            }
        },
    }
}


def list_profiles() -> list[str]:
    return sorted(_RLB_PROFILES.keys())


def get_profile(run_name: str) -> dict[str, object]:
    if run_name not in _RLB_PROFILES:
        raise ValueError(
            f"Unknown RLB run_name '{run_name}'. Supported values: {list_profiles()}"
        )
    return dict(_RLB_PROFILES[run_name])


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
        family="rlb",
        split_set=split_set,
        refit_on="train",
        model_config=dict(profile["model_config"]),
        experiments_data_dir=experiments_data_dir or Path(__file__).resolve().parents[1],
    )
