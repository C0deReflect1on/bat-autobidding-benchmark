from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DrlbModelParams:
    exp_type: str
    T: int
    bids_per_timestep: int
    lambda_min: float
    lambda_max: float


@dataclass(frozen=True)
class DqnParams:
    gamma: float
    lr: float
    target_update_interval: int
    soft_update_tau: float
    loss_type: str
    grad_clip_norm: float | None
    reward_clip_value: float | None


@dataclass(frozen=True)
class RewardNetParams:
    lr: float
    loss_type: str
    grad_clip_norm: float | None
    reward_clip_value: float | None


@dataclass(frozen=True)
class DrlbRuntimeParams:
    min_bid: float
    max_bid: float
    objective: str
    eval_mode: bool
    inference_lambda_init_mode: str
    verbose: bool
    use_tqdm: bool
    debug_logs: bool
    fit_log_every: int
    inference_log_every: int
    auction_mode: str


@dataclass(frozen=True)
class DrlbConfig:
    model: DrlbModelParams
    dqn: DqnParams
    reward_net: RewardNetParams
    runtime: DrlbRuntimeParams

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DrlbConfigParser:
    _DEFAULTS = {
        "exp_type": "improved_drlb_eval",
        "T": 72,
        "bids_per_timestep": 1,
        "lambda_min": 1e-6,
        "lambda_max": 10.0,
        "dqn_gamma": 1.0,
        "dqn_lr": 1e-4,
        "dqn_target_update_interval": 100,
        "dqn_soft_update_tau": 0.0,
        "dqn_loss_type": "mse",
        "dqn_grad_clip_norm": None,
        "dqn_reward_clip_value": None,
        "reward_net_lr": 1e-3,
        "reward_net_loss_type": "mse",
        "reward_net_grad_clip_norm": None,
        "reward_net_reward_clip_value": None,
        "min_bid": 0.0,
        "max_bid": 500.0,
        "objective": "clicks",
        "eval_mode": True,
        "inference_lambda_init_mode": "train_derived",
        "verbose": False,
        "use_tqdm": True,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
        "auction_mode": "VCG",
    }
    _VALID_OBJECTIVES = {"clicks", "contacts"}
    _VALID_LOSS_TYPES = {"mse", "smooth_l1"}
    _VALID_LAMBDA_INIT_MODES = {"train_derived", "checkpoint_final", "legacy"}
    _VALID_AUCTION_MODES = {"VCG", "FPA"}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DrlbConfig:
        raw = dict(raw or {})
        values = cls._flatten_raw(raw)
        return cls._build(values)

    @classmethod
    def from_checkpoint(cls, checkpoint_payload: Mapping[str, Any]) -> DrlbConfig:
        if not isinstance(checkpoint_payload, Mapping):
            raise ValueError("Checkpoint payload must be a mapping.")
        cfg_raw = checkpoint_payload.get("config")
        if not isinstance(cfg_raw, Mapping):
            raise ValueError("Checkpoint payload must contain 'config' mapping.")
        cls._validate_checkpoint_config_shape(cfg_raw)
        values = cls._flatten_raw(dict(cfg_raw))
        return cls._build(values, require_all=True)

    @classmethod
    def _validate_checkpoint_config_shape(cls, cfg_raw: Mapping[str, Any]) -> None:
        required_top_level = {"model", "dqn", "reward_net", "runtime"}
        missing_top_level = required_top_level - set(cfg_raw.keys())
        if missing_top_level:
            raise ValueError(
                f"Checkpoint config is missing top-level sections: {sorted(missing_top_level)}"
            )

        required_sections = {
            "model": {"exp_type", "T", "bids_per_timestep", "lambda_min", "lambda_max"},
            "dqn": {
                "gamma",
                "lr",
                "target_update_interval",
                "soft_update_tau",
                "loss_type",
                "grad_clip_norm",
                "reward_clip_value",
            },
            "reward_net": {"lr", "loss_type", "grad_clip_norm", "reward_clip_value"},
            "runtime": {
                "min_bid",
                "max_bid",
                "objective",
                "eval_mode",
                "inference_lambda_init_mode",
                "verbose",
                "use_tqdm",
                "debug_logs",
                "fit_log_every",
                "inference_log_every",
                "auction_mode",
            },
        }
        for section_name, required_keys in required_sections.items():
            section = cfg_raw.get(section_name)
            if not isinstance(section, Mapping):
                raise ValueError(f"Checkpoint config section '{section_name}' must be a mapping.")
            missing = required_keys - set(section.keys())
            if missing:
                raise ValueError(
                    f"Checkpoint config section '{section_name}' is missing keys: {sorted(missing)}"
                )

    @classmethod
    def _flatten_raw(cls, raw: dict[str, Any]) -> dict[str, Any]:
        if {"model", "dqn", "reward_net", "runtime"} <= set(raw.keys()):
            flattened = {}
            flattened.update(raw.get("model", {}))
            dqn = raw.get("dqn", {})
            flattened.update(
                {
                    "dqn_gamma": dqn.get("gamma"),
                    "dqn_lr": dqn.get("lr"),
                    "dqn_target_update_interval": dqn.get("target_update_interval"),
                    "dqn_soft_update_tau": dqn.get("soft_update_tau"),
                    "dqn_loss_type": dqn.get("loss_type"),
                    "dqn_grad_clip_norm": dqn.get("grad_clip_norm"),
                    "dqn_reward_clip_value": dqn.get("reward_clip_value"),
                }
            )
            reward_net = raw.get("reward_net", {})
            flattened.update(
                {
                    "reward_net_lr": reward_net.get("lr"),
                    "reward_net_loss_type": reward_net.get("loss_type"),
                    "reward_net_grad_clip_norm": reward_net.get("grad_clip_norm"),
                    "reward_net_reward_clip_value": reward_net.get("reward_clip_value"),
                }
            )
            flattened.update(raw.get("runtime", {}))
            # Normalize aliases.
            if "max_bid" not in flattened and "maxBid" in flattened:
                flattened["max_bid"] = flattened["maxBid"]
            if "min_bid" not in flattened and "minBid" in flattened:
                flattened["min_bid"] = flattened["minBid"]
            return flattened
        return raw

    @classmethod
    def _build(cls, values: dict[str, Any], require_all: bool = False) -> DrlbConfig:
        def get(name: str) -> Any:
            val = values.get(name, cls._DEFAULTS[name])
            if require_all and val is None and cls._DEFAULTS[name] is not None:
                raise ValueError(f"Missing required config key: {name}")
            return val

        exp_type = str(get("exp_type"))
        T = cls._as_int(get("T"), "T", min_value=1)
        bids_per_timestep = cls._as_int(get("bids_per_timestep"), "bids_per_timestep", min_value=1)
        lambda_min = cls._as_float(get("lambda_min"), "lambda_min", min_value=0.0, strictly_positive=True)
        lambda_max = cls._as_float(get("lambda_max"), "lambda_max", min_value=lambda_min, strictly_positive=True)

        dqn_loss_type = str(get("dqn_loss_type"))
        cls._ensure_in(dqn_loss_type, cls._VALID_LOSS_TYPES, "dqn_loss_type")
        reward_loss_type = str(get("reward_net_loss_type"))
        cls._ensure_in(reward_loss_type, cls._VALID_LOSS_TYPES, "reward_net_loss_type")

        min_bid = cls._as_float(get("min_bid"), "min_bid", min_value=0.0)
        max_bid = cls._as_float(get("max_bid"), "max_bid", min_value=0.0)
        if max_bid < min_bid:
            raise ValueError("max_bid must be >= min_bid")

        objective = str(get("objective"))
        cls._ensure_in(objective, cls._VALID_OBJECTIVES, "objective")
        inference_lambda_init_mode = str(get("inference_lambda_init_mode"))
        cls._ensure_in(
            inference_lambda_init_mode,
            cls._VALID_LAMBDA_INIT_MODES,
            "inference_lambda_init_mode",
        )
        auction_mode = str(get("auction_mode")).upper()
        cls._ensure_in(auction_mode, cls._VALID_AUCTION_MODES, "auction_mode")

        return DrlbConfig(
            model=DrlbModelParams(
                exp_type=exp_type,
                T=T,
                bids_per_timestep=bids_per_timestep,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            ),
            dqn=DqnParams(
                gamma=cls._as_float(get("dqn_gamma"), "dqn_gamma"),
                lr=cls._as_float(get("dqn_lr"), "dqn_lr", min_value=0.0, strictly_positive=True),
                target_update_interval=cls._as_int(
                    get("dqn_target_update_interval"),
                    "dqn_target_update_interval",
                    min_value=1,
                ),
                soft_update_tau=cls._as_float(get("dqn_soft_update_tau"), "dqn_soft_update_tau", min_value=0.0),
                loss_type=dqn_loss_type,
                grad_clip_norm=cls._as_optional_float(get("dqn_grad_clip_norm"), "dqn_grad_clip_norm"),
                reward_clip_value=cls._as_optional_float(
                    get("dqn_reward_clip_value"),
                    "dqn_reward_clip_value",
                    min_value=0.0,
                ),
            ),
            reward_net=RewardNetParams(
                lr=cls._as_float(get("reward_net_lr"), "reward_net_lr", min_value=0.0, strictly_positive=True),
                loss_type=reward_loss_type,
                grad_clip_norm=cls._as_optional_float(
                    get("reward_net_grad_clip_norm"),
                    "reward_net_grad_clip_norm",
                ),
                reward_clip_value=cls._as_optional_float(
                    get("reward_net_reward_clip_value"),
                    "reward_net_reward_clip_value",
                    min_value=0.0,
                ),
            ),
            runtime=DrlbRuntimeParams(
                min_bid=min_bid,
                max_bid=max_bid,
                objective=objective,
                eval_mode=cls._as_bool(get("eval_mode"), "eval_mode"),
                inference_lambda_init_mode=inference_lambda_init_mode,
                verbose=cls._as_bool(get("verbose"), "verbose"),
                use_tqdm=cls._as_bool(get("use_tqdm"), "use_tqdm"),
                debug_logs=cls._as_bool(get("debug_logs"), "debug_logs"),
                fit_log_every=cls._as_int(get("fit_log_every"), "fit_log_every", min_value=1),
                inference_log_every=cls._as_int(
                    get("inference_log_every"),
                    "inference_log_every",
                    min_value=1,
                ),
                auction_mode=auction_mode,
            ),
        )

    @staticmethod
    def _ensure_in(value: str, allowed: set[str], name: str) -> None:
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}; got '{value}'")

    @staticmethod
    def _as_float(
        value: Any,
        name: str,
        min_value: float | None = None,
        strictly_positive: bool = False,
    ) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a float-like value.") from exc
        if strictly_positive and out <= 0:
            raise ValueError(f"{name} must be > 0")
        if min_value is not None and out < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return out

    @classmethod
    def _as_optional_float(
        cls,
        value: Any,
        name: str,
        min_value: float | None = None,
    ) -> float | None:
        if value is None:
            return None
        return cls._as_float(value, name, min_value=min_value)

    @staticmethod
    def _as_int(value: Any, name: str, min_value: int | None = None) -> int:
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an int-like value.") from exc
        if min_value is not None and out < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return out

    @staticmethod
    def _as_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"{name} must be a boolean-like value.")
