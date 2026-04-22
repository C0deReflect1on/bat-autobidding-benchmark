from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base_exp_config import ExperimentConfig


_COMMON_BASE_PARAMS = {
    "max_bid": 100.0,
    "T": 72,
    "lambda_min": 1e-6,
    "lambda_max": 10.0,
    "bids_per_timestep": 1,
    "dqn_soft_update_tau": 0.01,
    "dqn_loss_type": "smooth_l1",
    "dqn_grad_clip_norm": 5.0,
    "dqn_reward_clip_value": 10.0,
}

_COMMON_MODEL_PARAMS = {
    "dqn_gamma": 1.0,
    "dqn_lr": 1e-4,
    "dqn_target_update_interval": 100,
    "reward_net_lr": 1e-3,
}

_WIDE_LAMBDA_ACTION_BETAS = (-0.18, -0.10, -0.04, 0.0, 0.04, 0.10, 0.18)


def _base_search_space(trial) -> dict[str, Any]:
    return {
        "dqn_gamma": trial.suggest_float("dqn_gamma", 0.90, 1.0),
        "dqn_lr": trial.suggest_float("dqn_lr", 1e-5, 5e-3, log=True),
        "dqn_target_update_interval": trial.suggest_int("dqn_target_update_interval", 10, 300, step=10),
        "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-2, log=True),
    }


_DRLB_PROFILES: dict[str, dict[str, Any]] = {
    "drlb_smooth": {
        "exp_type": "improved_hybrid_drlb_smooth_eval",
        "objective": "clicks",
        "base_drlb_params": dict(_COMMON_BASE_PARAMS),
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "drlb_hypgrid_v2": {
        "exp_type": "hypgrid_v2_drlb_eval",
        "objective": "clicks",
        "base_drlb_params": dict(_COMMON_BASE_PARAMS),
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "drlb_hypgrid_v3": {
        "exp_type": "hypgrid_v3_drlb_eval",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "reward_net_loss_type": "smooth_l1",
            "reward_net_grad_clip_norm": 5.0,
            "reward_net_reward_clip_value": 10.0,
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "drlb_subsample_multi_lambda": {
        "exp_type": "improved_hybrid_drlb_smooth_eval",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": _WIDE_LAMBDA_ACTION_BETAS,
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
}


def list_profiles() -> list[str]:
    return sorted(_DRLB_PROFILES.keys())


def get_profile(profile: str) -> dict[str, Any]:
    if profile not in _DRLB_PROFILES:
        raise ValueError(
            f"Unknown DRLB profile '{profile}'. Supported values: {list_profiles()}"
        )
    return dict(_DRLB_PROFILES[profile])


def build_config(
    run_name: str,
    *,
    profile: str | None = None,
    split_set: str = "subsample_train_val_holdout",
    experiments_data_dir: Path | None = None,
) -> ExperimentConfig:
    """Build DRLB experiment config.

    ``run_name`` sets artifact directory names (``.../drlb/<run_name>/``).
    ``profile`` selects hyperparameters from ``_DRLB_PROFILES``; if omitted, ``run_name`` is used.
    """
    profile_key = profile if profile is not None else run_name
    prof = get_profile(profile_key)
    return ExperimentConfig(
        experiment_name=run_name,
        run_name=run_name,
        drlb_profile=profile_key,
        n_trials=int(prof["n_trials"]),
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set=split_set,
        max_steps=int(prof["max_steps"]),
        experiments_data_dir=experiments_data_dir or Path(__file__).resolve().parents[1],
    )
