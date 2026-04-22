from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DrlbModelParams:
    state_type: str
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
        "state_type": "improved",
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
        d = cls._DEFAULTS

        state_type = values.get("state_type", d["state_type"])
        T = int(values.get("T", d["T"]))
        bids_per_timestep = int(values.get("bids_per_timestep", d["bids_per_timestep"]))
        lambda_min = values.get("lambda_min", d["lambda_min"])
        lambda_max = values.get("lambda_max", d["lambda_max"])
        lambda_betas = values.get("lambda_action_betas", d["lambda_action_betas"])
        if lambda_betas is None:
            lambda_betas = d["lambda_action_betas"]
        lambda_action_betas = tuple(lambda_betas)

        dqn_loss_type = values.get("dqn_loss_type", d["dqn_loss_type"])
        reward_loss_type = values.get("reward_net_loss_type", d["reward_net_loss_type"])

        min_bid = values.get("min_bid", values.get("minBid", d["min_bid"]))
        max_bid = values.get("max_bid", values.get("maxBid", d["max_bid"]))
        objective = values.get("objective", d["objective"])
        inference_lambda_init_mode = values.get(
            "inference_lambda_init_mode",
            d["inference_lambda_init_mode"],
        )
        auction_mode = str(values.get("auction_mode", d["auction_mode"])).upper()

        dqn_gamma = values.get("dqn_gamma", d["dqn_gamma"])
        dqn_lr = values.get("dqn_lr", d["dqn_lr"])
        dqn_target_update_interval = values.get(
            "dqn_target_update_interval",
            d["dqn_target_update_interval"],
        )
        dqn_soft_update_tau = values.get("dqn_soft_update_tau", d["dqn_soft_update_tau"])
        dqn_grad_clip_norm = values.get("dqn_grad_clip_norm", d["dqn_grad_clip_norm"])
        dqn_reward_clip_value = values.get(
            "dqn_reward_clip_value",
            d["dqn_reward_clip_value"],
        )

        reward_net_lr = values.get("reward_net_lr", d["reward_net_lr"])
        reward_grad_clip_norm = values.get(
            "reward_net_grad_clip_norm",
            d["reward_net_grad_clip_norm"],
        )
        reward_clip_value = values.get(
            "reward_net_reward_clip_value",
            d["reward_net_reward_clip_value"],
        )

        fit_log_every = int(values.get("fit_log_every", d["fit_log_every"]))
        inference_log_every = int(values.get("inference_log_every", d["inference_log_every"]))

        return DrlbConfig(
            model=DrlbModelParams(
                state_type=state_type,
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
                eval_mode=values.get("eval_mode", d["eval_mode"]),
                inference_lambda_init_mode=inference_lambda_init_mode,
                verbose=values.get("verbose", d["verbose"]),
                use_tqdm=values.get("use_tqdm", d["use_tqdm"]),
                debug_logs=values.get("debug_logs", d["debug_logs"]),
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
                "model": {"state_type", "T", "bids_per_timestep", "lambda_min", "lambda_max"},
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
            "state_type": model.get("state_type", defaults["state_type"]),
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
