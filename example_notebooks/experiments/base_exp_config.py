from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from typing import Optional

from .infra.artifacts import json_ready
from .infra.reproducibility import derive_seed_map
from .infra.split_registry import resolve_split_set


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    n_trials: int
    random_seed: int
    auction_mode: str
    metric: str
    data_config: dict[str, Any] = field(default_factory=dict)
    family: str = "drlb"
    run_name: str = ""
    drlb_profile: Optional[str] = None
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
    family_dir: Path = field(init=False)
    experiment_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    best_models_dir: Path = field(init=False)
    best_params_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if not self.experiment_name or "/" in self.experiment_name:
            raise ValueError("experiment_name must be non-empty and must not contain '/'")
        if not self.family or "/" in self.family:
            raise ValueError("family must be non-empty and must not contain '/'")
        if self.refit_on not in {"train", "train_plus_val"}:
            raise ValueError(f"Unsupported refit_on '{self.refit_on}'")
        if self.optimize_split not in {"val"}:
            raise ValueError(f"Unsupported optimize_split '{self.optimize_split}'")

        run_name = self.run_name or self.experiment_name
        if "/" in run_name:
            raise ValueError("run_name must not contain '/'")
        object.__setattr__(self, "run_name", run_name)

        data_config = dict(self.data_config)
        split_set = self.split_set
        if not data_config:
            split_set = split_set or "subsample_train_val_holdout"
            data_config = resolve_split_set(split_set)
        if not split_set:
            split_set = self._infer_split_set(data_config)

        object.__setattr__(self, "data_config", data_config)
        object.__setattr__(self, "split_set", split_set)

        family_dir = self.experiments_data_dir / self.family
        experiment_dir = family_dir / self.run_name
        object.__setattr__(self, "family_dir", family_dir)
        object.__setattr__(self, "experiment_dir", experiment_dir)
        object.__setattr__(self, "config_dir", experiment_dir / "config")
        object.__setattr__(self, "best_models_dir", experiment_dir / "best_models")
        object.__setattr__(self, "best_params_dir", experiment_dir / "best_params")
        object.__setattr__(self, "outputs_dir", experiment_dir / "outputs")

        seed_map = derive_seed_map(int(self.random_seed if self.master_seed is None else self.master_seed))
        object.__setattr__(self, "master_seed", int(self.master_seed if self.master_seed is not None else seed_map["master_seed"]))
        object.__setattr__(self, "data_seed", int(self.data_seed if self.data_seed is not None else seed_map["data_seed"]))
        object.__setattr__(self, "optuna_seed", int(self.optuna_seed if self.optuna_seed is not None else seed_map["optuna_seed"]))
        object.__setattr__(self, "model_seed", int(self.model_seed if self.model_seed is not None else seed_map["model_seed"]))
        object.__setattr__(
            self,
            "replay_buffer_seed",
            int(self.replay_buffer_seed if self.replay_buffer_seed is not None else seed_map["replay_buffer_seed"]),
        )
        object.__setattr__(self, "train_seed", int(self.train_seed if self.train_seed is not None else seed_map["train_seed"]))
        object.__setattr__(self, "eval_seed", int(self.eval_seed if self.eval_seed is not None else seed_map["eval_seed"]))
        object.__setattr__(self, "optimize_metric", self.metric if self.optimize_metric is None else self.optimize_metric)

    def ensure_artifact_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.best_models_dir.mkdir(parents=True, exist_ok=True)
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def best_params_path(self, model_name: str) -> Path:
        return self.best_params_dir / f"{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl"

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
                "family_dir": str(self.family_dir),
                "experiment_dir": str(self.experiment_dir),
                "config_dir": str(self.config_dir),
                "best_models_dir": str(self.best_models_dir),
                "best_params_dir": str(self.best_params_dir),
                "outputs_dir": str(self.outputs_dir),
            }
        )
        return json_ready(payload)

    @staticmethod
    def _infer_split_set(data_config: dict[str, Any]) -> str:
        keys = set(data_config.keys())
        if {"train", "val", "test_holdout"} <= keys:
            return "normalized_train_val_holdout"
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
        """Factory for experiments that use the shared full train/val/holdout split."""
        return cls(
            experiment_name=experiment_name,
            n_trials=n_trials,
            random_seed=random_seed,
            auction_mode=auction_mode,
            metric=metric,
            split_set="full_train_val_holdout",
            experiments_data_dir=base_dir or Path(__file__).resolve().parent,
        )
