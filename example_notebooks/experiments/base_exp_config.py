from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    n_trials: int
    random_seed: int
    auction_mode: str
    metric: str
    data_config: dict
    eval_campaign_fraction: Optional[float] = None
    eval_slice_seed: int = 42
    experiments_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    experiment_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    best_models_dir: Path = field(init=False)
    best_params_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if not self.experiment_name or "/" in self.experiment_name:
            raise ValueError("experiment_name must be non-empty and must not contain '/'")

        experiment_dir = self.experiments_data_dir / self.experiment_name
        object.__setattr__(self, "experiment_dir", experiment_dir)
        object.__setattr__(self, "config_dir", experiment_dir / "config")
        object.__setattr__(self, "best_models_dir", experiment_dir / "best_models")
        object.__setattr__(self, "best_params_dir", experiment_dir / "best_params")
        object.__setattr__(self, "outputs_dir", experiment_dir / "outputs")

    def ensure_artifact_dirs(self) -> None:
        self.best_models_dir.mkdir(parents=True, exist_ok=True)
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def best_params_path(self, model_name: str) -> Path:
        return self.best_params_dir / f"{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl"


def assert_unique_experiment_names(configs: Iterable[ExperimentConfig]) -> None:
    names = [config.experiment_name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment_name values: {duplicates}")
