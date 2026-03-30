from dataclasses import dataclass, field
from pathlib import Path
from .base_exp_config import ExperimentConfig

from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST
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

def RND42TrainValConfig() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_drlb_dqn", n_trials=10)

def RND42TrainValHybridConfig() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_drlb_dqn_hybrid")

def RND42TrainValHybridSmoothConfig() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_drlb_dqn_smooth")

def RND42TrainValHybridSmoothLambdaTrainConfig() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_smooth_drlb_dqn_lambda_train", n_trials=10)

def RND42TrainValHypgridV2Config() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_drlb_dqn_hypgrid_v2")

def RND42TrainValHypgridV3Config() -> ExperimentConfig:
    return ExperimentConfig.train_val("exp_tune_drlb_dqn_hypgrid_v3")
