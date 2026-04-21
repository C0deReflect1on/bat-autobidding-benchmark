from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional

from config import (
    FPA_CAMPAIGNS_HOLDOUT_TEST,
    FPA_CAMPAIGNS_TRAIN_VAL,
    FPA_CAMPAIGNS_VAL_VAL,
    FPA_STATS_HOLDOUT_TEST,
    FPA_STATS_TRAIN_VAL,
    FPA_STATS_VAL_VAL,
)

@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    n_trials: int
    random_seed: int
    auction_mode: str
    metric: str
    data_config: dict
    family: str = "drlb"
    objective_type: str = "clicks"
    split_set: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    refit_on: str = "train"
    checkpoint_policy: str = "best_val"
    optimize_split: str = "val"
    optimize_metric: Optional[str] = None
    master_seed: Optional[int] = None
    data_seed: Optional[int] = None
    optuna_seed: Optional[int] = None
    model_seed: Optional[int] = None
    replay_buffer_seed: Optional[int] = None
    train_seed: Optional[int] = None
    eval_seed: Optional[int] = None
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
        if self.family not in {"baseline", "drlb", "rlb"}:
            raise ValueError(f"Unsupported family '{self.family}'")
        if self.refit_on not in {"train", "train_plus_val"}:
            raise ValueError(f"Unsupported refit_on '{self.refit_on}'")
        if self.optimize_split not in {"val"}:
            raise ValueError(f"Unsupported optimize_split '{self.optimize_split}'")

        experiment_dir = self.experiments_data_dir / self.experiment_name
        object.__setattr__(self, "experiment_dir", experiment_dir)
        object.__setattr__(self, "config_dir", experiment_dir / "config")
        object.__setattr__(self, "best_models_dir", experiment_dir / "best_models")
        object.__setattr__(self, "best_params_dir", experiment_dir / "best_params")
        object.__setattr__(self, "outputs_dir", experiment_dir / "outputs")
        master_seed = int(self.random_seed if self.master_seed is None else self.master_seed)
        object.__setattr__(self, "master_seed", master_seed)
        object.__setattr__(self, "data_seed", int(self.data_seed if self.data_seed is not None else master_seed + 1000))
        object.__setattr__(self, "optuna_seed", int(self.optuna_seed if self.optuna_seed is not None else master_seed + 2000))
        object.__setattr__(self, "model_seed", int(self.model_seed if self.model_seed is not None else master_seed + 3000))
        object.__setattr__(
            self,
            "replay_buffer_seed",
            int(self.replay_buffer_seed if self.replay_buffer_seed is not None else master_seed + 4000),
        )
        object.__setattr__(self, "train_seed", int(self.train_seed if self.train_seed is not None else master_seed + 5000))
        object.__setattr__(self, "eval_seed", int(self.eval_seed if self.eval_seed is not None else master_seed + 6000))
        object.__setattr__(self, "optimize_metric", self.metric if self.optimize_metric is None else self.optimize_metric)
        if not self.split_set:
            object.__setattr__(self, "split_set", self._infer_split_set())

    def ensure_artifact_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.best_models_dir.mkdir(parents=True, exist_ok=True)
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def best_params_path(self, model_name: str) -> Path:
        return self.best_params_dir / f"{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl"

    @property
    def objective_metric(self) -> str:
        return self.metric

    @property
    def seeds(self) -> dict[str, int]:
        return {
            "master_seed": int(self.master_seed),
            "data_seed": int(self.data_seed),
            "optuna_seed": int(self.optuna_seed),
            "model_seed": int(self.model_seed),
            "replay_buffer_seed": int(self.replay_buffer_seed),
            "train_seed": int(self.train_seed),
            "eval_seed": int(self.eval_seed),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "experiment_dir": str(self.experiment_dir),
                "config_dir": str(self.config_dir),
                "best_models_dir": str(self.best_models_dir),
                "best_params_dir": str(self.best_params_dir),
                "outputs_dir": str(self.outputs_dir),
            }
        )
        return _json_ready(payload)

    def _infer_split_set(self) -> str:
        keys = set(self.data_config.keys())
        if {"train_val", "val_val", "holdout_test"} <= keys:
            return "canonical_train_val_holdout"
        if {"train", "test"} <= keys:
            return "legacy_train_test"
        return "custom"

    @classmethod
    def train_val(
        cls,
        experiment_name: str,
        *,
        n_trials: int = 1,
        random_seed: int = 42,
        auction_mode: str = "FPA",
        metric: str = "SCR",
        base_dir: Optional[Path] = None,
    ) -> "ExperimentConfig":
        """Factory for experiments that use canonical fixed train/val/holdout splits."""
        data_config = {
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
        return cls(
            experiment_name=experiment_name,
            n_trials=n_trials,
            random_seed=random_seed,
            auction_mode=auction_mode,
            metric=metric,
            split_set="canonical_train_val_holdout",
            data_config=data_config,
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def assert_unique_experiment_names(configs: Iterable[ExperimentConfig]) -> None:
    names = [config.experiment_name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment_name values: {duplicates}")
