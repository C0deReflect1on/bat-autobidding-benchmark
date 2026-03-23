from dataclasses import dataclass, field
from .base_exp_config import ExperimentConfig

try:
    from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import FPA_CAMPAIGNS_TRAIN, FPA_CAMPAIGNS_TEST, FPA_STATS_TRAIN, FPA_STATS_TEST


@dataclass(frozen=True)
class RND42N10Config(ExperimentConfig):
    experiment_name: str = "exp_1_rnd_42_n10"
    n_trials: int = 10
    random_seed: int = 42
    auction_mode: str = "FPA"
    metric: str = "SCR"
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
