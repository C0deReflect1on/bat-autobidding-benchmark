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
    lambda_action_betas: tuple[float, ...]


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
        "lambda_action_betas": (-0.08, -0.03, -0.01, 0.0, 0.01, 0.03, 0.08),
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
        values = dict(raw or {})
        if {"model", "dqn", "reward_net", "runtime"} <= set(values):
            return cls._from_sectioned_config(values, require_all=False)
        return cls._from_flat_config(values)

    @classmethod
    def from_checkpoint(cls, checkpoint_payload: Mapping[str, Any]) -> DrlbConfig:
        if not isinstance(checkpoint_payload, Mapping):
            raise ValueError("Checkpoint payload must be a mapping.")
        config = checkpoint_payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Checkpoint payload must contain 'config' mapping.")
        return cls._from_sectioned_config(config, require_all=True)

    @classmethod
    def _from_flat_config(cls, values: Mapping[str, Any]) -> DrlbConfig:
        defaults = cls._DEFAULTS

        exp_type = str(values.get("exp_type", defaults["exp_type"]))

        T = int(values.get("T", defaults["T"]))
        if T < 1:
            raise ValueError("T must be >= 1")

        bids_per_timestep = int(values.get("bids_per_timestep", defaults["bids_per_timestep"]))
        if bids_per_timestep < 1:
            raise ValueError("bids_per_timestep must be >= 1")

        lambda_min = float(values.get("lambda_min", defaults["lambda_min"]))
        if lambda_min <= 0:
            raise ValueError("lambda_min must be > 0")

        lambda_max = float(values.get("lambda_max", defaults["lambda_max"]))
        if lambda_max < lambda_min or lambda_max <= 0:
            raise ValueError("lambda_max must be >= lambda_min and > 0")

        raw_betas = values.get("lambda_action_betas", defaults["lambda_action_betas"])
        if raw_betas is None:
            raw_betas = defaults["lambda_action_betas"]
        if not isinstance(raw_betas, (list, tuple)):
            raise ValueError(
                "lambda_action_betas must be a list/tuple of floats (DQN discrete actions)."
            )
        lambda_action_betas = tuple(float(beta) for beta in raw_betas)

        dqn_loss_type = str(values.get("dqn_loss_type", defaults["dqn_loss_type"]))
        if dqn_loss_type not in cls._VALID_LOSS_TYPES:
            raise ValueError(
                f"dqn_loss_type must be one of {sorted(cls._VALID_LOSS_TYPES)}; got '{dqn_loss_type}'"
            )
        reward_loss_type = str(
            values.get("reward_net_loss_type", defaults["reward_net_loss_type"])
        )
        if reward_loss_type not in cls._VALID_LOSS_TYPES:
            raise ValueError(
                "reward_net_loss_type must be one of "
                f"{sorted(cls._VALID_LOSS_TYPES)}; got '{reward_loss_type}'"
            )

        min_bid = float(values.get("min_bid", values.get("minBid", defaults["min_bid"])))
        if min_bid < 0:
            raise ValueError("min_bid must be >= 0")

        max_bid = float(values.get("max_bid", values.get("maxBid", defaults["max_bid"])))
        if max_bid < min_bid or max_bid < 0:
            raise ValueError("max_bid must be >= min_bid")

        objective = str(values.get("objective", defaults["objective"]))
        if objective not in cls._VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(cls._VALID_OBJECTIVES)}; got '{objective}'"
            )

        inference_lambda_init_mode = str(
            values.get(
                "inference_lambda_init_mode",
                defaults["inference_lambda_init_mode"],
            )
        )
        if inference_lambda_init_mode not in cls._VALID_LAMBDA_INIT_MODES:
            raise ValueError(
                "inference_lambda_init_mode must be one of "
                f"{sorted(cls._VALID_LAMBDA_INIT_MODES)}; got '{inference_lambda_init_mode}'"
            )

        auction_mode = str(values.get("auction_mode", defaults["auction_mode"])).upper()
        if auction_mode not in cls._VALID_AUCTION_MODES:
            raise ValueError(
                f"auction_mode must be one of {sorted(cls._VALID_AUCTION_MODES)}; got '{auction_mode}'"
            )

        dqn_gamma = float(values.get("dqn_gamma", defaults["dqn_gamma"]))

        dqn_lr = float(values.get("dqn_lr", defaults["dqn_lr"]))
        if dqn_lr <= 0:
            raise ValueError("dqn_lr must be > 0")

        dqn_target_update_interval = int(
            values.get(
                "dqn_target_update_interval",
                defaults["dqn_target_update_interval"],
            )
        )
        if dqn_target_update_interval < 1:
            raise ValueError("dqn_target_update_interval must be >= 1")

        dqn_soft_update_tau = float(
            values.get("dqn_soft_update_tau", defaults["dqn_soft_update_tau"])
        )
        if dqn_soft_update_tau < 0:
            raise ValueError("dqn_soft_update_tau must be >= 0")

        dqn_grad_clip_norm_raw = values.get(
            "dqn_grad_clip_norm",
            defaults["dqn_grad_clip_norm"],
        )
        dqn_grad_clip_norm = (
            None if dqn_grad_clip_norm_raw is None else float(dqn_grad_clip_norm_raw)
        )

        dqn_reward_clip_value_raw = values.get(
            "dqn_reward_clip_value",
            defaults["dqn_reward_clip_value"],
        )
        dqn_reward_clip_value = (
            None
            if dqn_reward_clip_value_raw is None
            else float(dqn_reward_clip_value_raw)
        )
        if dqn_reward_clip_value is not None and dqn_reward_clip_value < 0:
            raise ValueError("dqn_reward_clip_value must be >= 0")

        reward_net_lr = float(values.get("reward_net_lr", defaults["reward_net_lr"]))
        if reward_net_lr <= 0:
            raise ValueError("reward_net_lr must be > 0")

        reward_grad_clip_norm_raw = values.get(
            "reward_net_grad_clip_norm",
            defaults["reward_net_grad_clip_norm"],
        )
        reward_grad_clip_norm = (
            None if reward_grad_clip_norm_raw is None else float(reward_grad_clip_norm_raw)
        )

        reward_clip_value_raw = values.get(
            "reward_net_reward_clip_value",
            defaults["reward_net_reward_clip_value"],
        )
        reward_clip_value = (
            None if reward_clip_value_raw is None else float(reward_clip_value_raw)
        )
        if reward_clip_value is not None and reward_clip_value < 0:
            raise ValueError("reward_net_reward_clip_value must be >= 0")

        fit_log_every = int(values.get("fit_log_every", defaults["fit_log_every"]))
        if fit_log_every < 1:
            raise ValueError("fit_log_every must be >= 1")

        inference_log_every = int(
            values.get(
                "inference_log_every",
                defaults["inference_log_every"],
            )
        )
        if inference_log_every < 1:
            raise ValueError("inference_log_every must be >= 1")

        return DrlbConfig(
            model=DrlbModelParams(
                exp_type=exp_type,
                T=T,
                bids_per_timestep=bids_per_timestep,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                lambda_action_betas=lambda_action_betas,
            ),
            dqn=DqnParams(
                gamma=dqn_gamma,
                lr=dqn_lr,
                target_update_interval=dqn_target_update_interval,
                soft_update_tau=dqn_soft_update_tau,
                loss_type=dqn_loss_type,
                grad_clip_norm=dqn_grad_clip_norm,
                reward_clip_value=dqn_reward_clip_value,
            ),
            reward_net=RewardNetParams(
                lr=reward_net_lr,
                loss_type=reward_loss_type,
                grad_clip_norm=reward_grad_clip_norm,
                reward_clip_value=reward_clip_value,
            ),
            runtime=DrlbRuntimeParams(
                min_bid=min_bid,
                max_bid=max_bid,
                objective=objective,
                eval_mode=cls._parse_bool(
                    values.get("eval_mode", defaults["eval_mode"]),
                    "eval_mode",
                ),
                inference_lambda_init_mode=inference_lambda_init_mode,
                verbose=cls._parse_bool(
                    values.get("verbose", defaults["verbose"]),
                    "verbose",
                ),
                use_tqdm=cls._parse_bool(
                    values.get("use_tqdm", defaults["use_tqdm"]),
                    "use_tqdm",
                ),
                debug_logs=cls._parse_bool(
                    values.get("debug_logs", defaults["debug_logs"]),
                    "debug_logs",
                ),
                fit_log_every=fit_log_every,
                inference_log_every=inference_log_every,
                auction_mode=auction_mode,
            ),
        )

    @classmethod
    def _from_sectioned_config(
        cls,
        raw: Mapping[str, Any],
        *,
        require_all: bool,
    ) -> DrlbConfig:
        required_sections = ("model", "dqn", "reward_net", "runtime")
        sections: dict[str, Mapping[str, Any]] = {}

        for section_name in required_sections:
            section = raw.get(section_name)
            if section is None and not require_all:
                sections[section_name] = {}
                continue
            if not isinstance(section, Mapping):
                raise ValueError(f"Checkpoint config section '{section_name}' must be a mapping.")
            sections[section_name] = section

        if require_all:
            required_keys = {
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
            for section_name, keys in required_keys.items():
                missing = keys - set(sections[section_name].keys())
                if missing:
                    raise ValueError(
                        f"Checkpoint config section '{section_name}' is missing keys: {sorted(missing)}"
                    )

        model = sections["model"]
        dqn = sections["dqn"]
        reward_net = sections["reward_net"]
        runtime = sections["runtime"]
        defaults = cls._DEFAULTS

        flat_values = {
            "exp_type": model.get("exp_type", defaults["exp_type"]),
            "T": model.get("T", defaults["T"]),
            "bids_per_timestep": model.get("bids_per_timestep", defaults["bids_per_timestep"]),
            "lambda_min": model.get("lambda_min", defaults["lambda_min"]),
            "lambda_max": model.get("lambda_max", defaults["lambda_max"]),
            "lambda_action_betas": model.get(
                "lambda_action_betas",
                defaults["lambda_action_betas"],
            ),
            "dqn_gamma": dqn.get("gamma", defaults["dqn_gamma"]),
            "dqn_lr": dqn.get("lr", defaults["dqn_lr"]),
            "dqn_target_update_interval": dqn.get(
                "target_update_interval",
                defaults["dqn_target_update_interval"],
            ),
            "dqn_soft_update_tau": dqn.get(
                "soft_update_tau",
                defaults["dqn_soft_update_tau"],
            ),
            "dqn_loss_type": dqn.get("loss_type", defaults["dqn_loss_type"]),
            "dqn_grad_clip_norm": dqn.get(
                "grad_clip_norm",
                defaults["dqn_grad_clip_norm"],
            ),
            "dqn_reward_clip_value": dqn.get(
                "reward_clip_value",
                defaults["dqn_reward_clip_value"],
            ),
            "reward_net_lr": reward_net.get("lr", defaults["reward_net_lr"]),
            "reward_net_loss_type": reward_net.get(
                "loss_type",
                defaults["reward_net_loss_type"],
            ),
            "reward_net_grad_clip_norm": reward_net.get(
                "grad_clip_norm",
                defaults["reward_net_grad_clip_norm"],
            ),
            "reward_net_reward_clip_value": reward_net.get(
                "reward_clip_value",
                defaults["reward_net_reward_clip_value"],
            ),
            "min_bid": runtime.get("min_bid", runtime.get("minBid", defaults["min_bid"])),
            "max_bid": runtime.get("max_bid", runtime.get("maxBid", defaults["max_bid"])),
            "objective": runtime.get("objective", defaults["objective"]),
            "eval_mode": runtime.get("eval_mode", defaults["eval_mode"]),
            "inference_lambda_init_mode": runtime.get(
                "inference_lambda_init_mode",
                defaults["inference_lambda_init_mode"],
            ),
            "verbose": runtime.get("verbose", defaults["verbose"]),
            "use_tqdm": runtime.get("use_tqdm", defaults["use_tqdm"]),
            "debug_logs": runtime.get("debug_logs", defaults["debug_logs"]),
            "fit_log_every": runtime.get("fit_log_every", defaults["fit_log_every"]),
            "inference_log_every": runtime.get(
                "inference_log_every",
                defaults["inference_log_every"],
            ),
            "auction_mode": runtime.get("auction_mode", defaults["auction_mode"]),
        }
        return cls._from_flat_config(flat_values)

    @staticmethod
    def _parse_bool(value: Any, name: str) -> bool:
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
