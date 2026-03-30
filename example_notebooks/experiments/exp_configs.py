from dataclasses import dataclass, field
from pathlib import Path
from .base_exp_config import ExperimentConfig

try:
    from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST
    from config import FPA_SUBSAMPLE_CAMPAIGNS_TRAIN, FPA_SUBSAMPLE_STATS_TRAIN
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST
    from config import FPA_SUBSAMPLE_CAMPAIGNS_TRAIN, FPA_SUBSAMPLE_STATS_TRAIN


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
class RND42TrainValConfig(ExperimentConfig):
    experiment_name: str = "exp_tune_drlb_dqn"
    n_trials: int = 10
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn" / "config" / "val_stats.csv"
                ),
            },
        }
    )


@dataclass(frozen=True)
class RND42TrainValHybridConfig(ExperimentConfig):
    experiment_name: str = "exp_tune_drlb_dqn_hybrid"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hybrid" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hybrid" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hybrid" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hybrid" / "config" / "val_stats.csv"
                ),
            },
        }
    )


@dataclass(frozen=True)
class RND42TrainValHybridSmoothConfig(ExperimentConfig):
    experiment_name: str = "exp_tune_drlb_dqn_smooth"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_smooth" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_smooth" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_smooth" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_smooth" / "config" / "val_stats.csv"
                ),
            },
        }
    )


@dataclass(frozen=True)
class RND42TrainValHybridSmoothLambdaTrainConfig(ExperimentConfig):
    experiment_name: str = "exp_tune_smooth_drlb_dqn_lambda_train"
    n_trials: int = 10
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_smooth_drlb_dqn_lambda_train" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_smooth_drlb_dqn_lambda_train" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_smooth_drlb_dqn_lambda_train" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_smooth_drlb_dqn_lambda_train" / "config" / "val_stats.csv"
                ),
            },
        }
    )


@dataclass(frozen=True)
class RND42TrainTestHybridSmoothConfig(ExperimentConfig):
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


@dataclass(frozen=True)
class RND42TrainValHypgridV2Config(ExperimentConfig):
    experiment_name: str = "exp_tune_drlb_dqn_hypgrid_v2"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v2" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v2" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v2" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v2" / "config" / "val_stats.csv"
                ),
            },
        }
    )


@dataclass(frozen=True)
class RND42TrainValHypgridV3Config(ExperimentConfig):
    experiment_name: str = "exp_tune_drlb_dqn_hypgrid_v3"
    n_trials: int = 1
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
    eval_campaign_fraction: float | None = None
    eval_slice_seed: int = 42
    data_config: dict = field(
        default_factory=lambda: {
            "train": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v3" / "config" / "train_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v3" / "config" / "train_stats.csv"
                ),
            },
            "test": {
                "campaigns_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v3" / "config" / "val_campaigns.csv"
                ),
                "stats_path": str(
                    Path(__file__).resolve().parent / "exp_tune_drlb_dqn_hypgrid_v3" / "config" / "val_stats.csv"
                ),
            },
        }
    )
