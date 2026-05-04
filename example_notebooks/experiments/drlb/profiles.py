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
    "dqn_epsilon_start": 0.95,
    "dqn_epsilon_end": 0.05,
    "dqn_epsilon_anneal": 2e-5,
}

DEFAULT_LAMBDA_ACTION_BETAS = (-0.08, -0.03, -0.01, 0.0, 0.01, 0.03, 0.08)
WIDE_LAMBDA_ACTION_BETAS_1 = (-0.18, -0.10, -0.04, 0.0, 0.04, 0.10, 0.18)
WIDE_LAMBDA_ACTION_BETAS_2 = (-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
MAY03_LINEAR_LAMBDA_INIT = 0.0028423174374845716


def _base_search_space(trial) -> dict[str, Any]:
    return {
        "dqn_gamma": trial.suggest_float("dqn_gamma", 0.90, 1.0),
        "dqn_lr": trial.suggest_float("dqn_lr", 1e-5, 5e-3, log=True),
        "dqn_target_update_interval": trial.suggest_int("dqn_target_update_interval", 10, 300, step=10),
        "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-2, log=True),
    }


def _may03_lr_bid_clip_search_space(trial) -> dict[str, Any]:
    lr_grid = [1e-2, 1e-3, 3e-4, 1e-4]
    return {
        "dqn_lr": trial.suggest_categorical("dqn_lr", lr_grid),
        "reward_net_lr": trial.suggest_categorical("reward_net_lr", lr_grid),
        "bid_lower_clip": trial.suggest_int("bid_lower_clip", 1, 10),
        "bid_upper_clip": trial.suggest_int("bid_upper_clip", 1, 10),
    }


_DRLB_PROFILES: dict[str, dict[str, Any]] = {
    "drlb_improved": {
        "state_type": "improved",
        "objective": "clicks",
        "base_drlb_params": dict(_COMMON_BASE_PARAMS),
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "drlb_hybrid": {
        "state_type": "hybrid",
        "objective": "clicks",
        "base_drlb_params": dict(_COMMON_BASE_PARAMS),
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "drlb_improved_smooth": {
        "state_type": "improved",
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
    "drlb_imporved_smooth": {
        "state_type": "improved",
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
    "drlb_smooth": {
        "state_type": "improved",
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
    "default_drlb_default_lambda": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": DEFAULT_LAMBDA_ACTION_BETAS,
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "default_drlb_wide_lambda_1": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": WIDE_LAMBDA_ACTION_BETAS_1,
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "default_drlb_wide_lambda_2": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": WIDE_LAMBDA_ACTION_BETAS_2,
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 64,
    },
    "default_drlb_best_episode_return": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": DEFAULT_LAMBDA_ACTION_BETAS,
            "reward_net_target_mode": "best_episode_return",
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 10000,
    },
    "default_drlb_instant_episode_return": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": DEFAULT_LAMBDA_ACTION_BETAS,
            "reward_net_target_mode": "immediate_step",
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 10000,
    },
    "improved_drlb_best_episode_return": {
        "state_type": "improved",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": 1e-5,
            "lambda_max": 5.0,
            "lambda_action_betas": DEFAULT_LAMBDA_ACTION_BETAS,
            "reward_net_target_mode": "best_episode_return",
        },
        "reference_model_params": dict(_COMMON_MODEL_PARAMS),
        "search_space_fn": _base_search_space,
        "n_trials": 1,
        "max_steps": 10000,
    },
    "may03_default_linear_lambda_legacy": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
        "max_steps": 64,
    },
    "may03_ratio_bat_linear_lambda_legacy": {
        "state_type": "ratio_bat",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
        "max_steps": 64,
    },
    "may03_ta_ratio_bat_linear_lambda_legacy": {
        "state_type": "ta_ratio_bat",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
        "max_steps": 64,
    },
    # May 04: same as may03_* but no finite λ clipping (see notebooks may_04/).
    "may04_default_linear_lambda_legacy": {
        "state_type": "default",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": float("-inf"),
            "lambda_max": float("inf"),
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
        "max_steps": 64,
    },
    "may04_ratio_bat_linear_lambda_legacy": {
        "state_type": "ratio_bat",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": float("-inf"),
            "lambda_max": float("inf"),
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
        "max_steps": 64,
    },
    "may04_ta_ratio_bat_linear_lambda_legacy": {
        "state_type": "ta_ratio_bat",
        "objective": "clicks",
        "base_drlb_params": {
            **_COMMON_BASE_PARAMS,
            "lambda_min": float("-inf"),
            "lambda_max": float("inf"),
            "init_lambda": MAY03_LINEAR_LAMBDA_INIT,
            "init_lambda_mode": "constant",
        },
        "reference_model_params": {
            **_COMMON_MODEL_PARAMS,
            "dqn_gamma": 1.0,
        },
        "search_space_fn": _may03_lr_bid_clip_search_space,
        "n_trials": 10,
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
