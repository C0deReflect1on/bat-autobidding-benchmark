from dataclasses import dataclass, field
from pathlib import Path
from .base_exp_config import ExperimentConfig

from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST
from config import (
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
    FPA_CAMPAIGNS_HOLDOUT_TEST,
    FPA_CAMPAIGNS_TRAIN_VAL,
    FPA_CAMPAIGNS_VAL_VAL,
    FPA_STATS_HOLDOUT_TEST,
    FPA_STATS_TRAIN_VAL,
    FPA_STATS_VAL_VAL,
)
from config import FPA_SUBSAMPLE_CAMPAIGNS_TRAIN, FPA_SUBSAMPLE_STATS_TRAIN


# ---------------------------------------------------------------------------
# Configs that point to the global data/ directory (genuinely unique paths)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RND42N10Config(ExperimentConfig):
    experiment_name: str = "exp_1_rnd_42_n10"
    n_trials: int = 10
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    family: str = "drlb"
    split_set: str = "legacy_train_test"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(FPA_CAMPAIGNS_TRAIN),
                "stats_path": str(FPA_STATS_TRAIN),
            },
            "test": {
                "campaigns_path": str(FPA_CAMPAIGNS_TEST),
                "stats_path": str(FPA_STATS_TEST),
            },
        }
    )


@dataclass(frozen=True)
class SubsampleRND42N10Config(ExperimentConfig):
    experiment_name: str = "exp_subsample_rnd_42_n10"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    family: str = "drlb"
    split_set: str = "legacy_train_test"
    eval_campaign_fraction: float | None = 0.1
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(FPA_SUBSAMPLE_CAMPAIGNS_TRAIN),
                "stats_path": str(FPA_SUBSAMPLE_STATS_TRAIN),
            },
            "test": {
                "campaigns_path": str(FPA_SUBSAMPLE_CAMPAIGNS_TRAIN),
                "stats_path": str(FPA_SUBSAMPLE_STATS_TRAIN),
            }
        }
    )


@dataclass(frozen=True)
class RND42TrainTestHybridSmoothConfig(ExperimentConfig):
    """Uses global train/test split (not a local val split)."""
    experiment_name: str = "exp_train_test_drlb_dqn_smooth"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    family: str = "drlb"
    split_set: str = "legacy_train_test"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(FPA_CAMPAIGNS_TRAIN),
                "stats_path": str(FPA_STATS_TRAIN),
            },
            "test": {
                "campaigns_path": str(FPA_CAMPAIGNS_TEST),
                "stats_path": str(FPA_STATS_TEST),
            },
        }
    )


# ---------------------------------------------------------------------------
# Train/val split configs -- use the factory to avoid path duplication
# ---------------------------------------------------------------------------

def _fixed_fpa_split_data_config() -> dict:
    return {
        "train_val": {
            "campaigns_path": str(FPA_CAMPAIGNS_TRAIN_VAL),
            "stats_path": str(FPA_STATS_TRAIN_VAL),
        },
        "val_val": {
            "campaigns_path": str(FPA_CAMPAIGNS_VAL_VAL),
            "stats_path": str(FPA_STATS_VAL_VAL),
        },
        "holdout_test": {
            "campaigns_path": str(FPA_CAMPAIGNS_HOLDOUT_TEST),
            "stats_path": str(FPA_STATS_HOLDOUT_TEST),
        },
    }


def _experiment_results_subsample_data_config() -> dict:
    return {
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
    }


def RND42TrainValConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_drlb_dqn",
        n_trials=10,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def RND42TrainValHybridConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_drlb_dqn_hybrid",
        n_trials=1,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def RND42TrainValHybridSmoothConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_drlb_dqn_smooth",
        n_trials=1,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def RND42TrainValHybridSmoothLambdaTrainConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_smooth_drlb_dqn_lambda_train",
        n_trials=10,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def RND42TrainValHypgridV2Config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_drlb_dqn_hypgrid_v2",
        n_trials=1,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def RND42TrainValHypgridV3Config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_tune_drlb_dqn_hypgrid_v3",
        n_trials=1,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="canonical_train_val_holdout",
        data_config=_fixed_fpa_split_data_config(),
    )


def BaselineLinearTrainValConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp_baseline_linear_trainval",
        n_trials=10,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="baseline",
        split_set="canonical_train_val_holdout",
        checkpoint_policy="none",
        model_config={"model_name": "linear"},
        data_config=_fixed_fpa_split_data_config(),
    )


def _experiment_results_root():
    return Path(__file__).resolve().parent / "experiment_results"


def ExperimentResultsLinearConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="linear",
        n_trials=5,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="baseline",
        split_set="fpa_subsample_experiment_results",
        checkpoint_policy="none",
        model_config={"model_name": "linear"},
        experiments_data_dir=_experiment_results_root(),
        data_config=_experiment_results_subsample_data_config(),
    )


def ExperimentResultsRlbConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="rlb",
        n_trials=5,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="rlb",
        split_set="fpa_subsample_experiment_results",
        refit_on="train",
        model_config={
            "base_params": {
                "max_bid": 300,
                "gamma": 1.0,
                "N_bound": 72,
                "B_bound": 10000,
                "use_smoothing": False,
            }
        },
        experiments_data_dir=_experiment_results_root(),
        data_config=_experiment_results_subsample_data_config(),
    )


def ExperimentResultsDrlbConfig() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="drlb",
        n_trials=2,
        random_seed=42,
        auction_mode="FPA",
        metric="SCR",
        family="drlb",
        split_set="fpa_subsample_experiment_results",
        refit_on="train",
        max_steps=128,
        experiments_data_dir=_experiment_results_root(),
        data_config=_experiment_results_subsample_data_config(),
    )
