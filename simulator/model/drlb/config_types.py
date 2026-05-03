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
    epsilon_start: float
    epsilon_end: float
    epsilon_anneal: float
    loss_type: str
    loss: Any
    scheduler_factory: Any
    grad_clip_norm: float | None
    reward_clip_value: float | None


@dataclass(frozen=True)
class RewardNetParams:
    lr: float
    loss_type: str
    loss: Any
    scheduler_factory: Any
    grad_clip_norm: float | None
    reward_clip_value: float | None
    target_mode: str
    state_action_bucket_size: float


@dataclass(frozen=True)
class DrlbRuntimeParams:
    min_bid: float
    max_bid: float
    traffic_path: str
    bid_lower_clip: float
    bid_upper_clip: float
    objective: str
    eval_mode: bool
    fit_lambda_init: float | None
    inference_lambda_init: float | None
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
        return {
            "model": asdict(self.model),
            "dqn": {
                "gamma": self.dqn.gamma,
                "lr": self.dqn.lr,
                "target_update_interval": self.dqn.target_update_interval,
                "soft_update_tau": self.dqn.soft_update_tau,
                "epsilon_start": self.dqn.epsilon_start,
                "epsilon_end": self.dqn.epsilon_end,
                "epsilon_anneal": self.dqn.epsilon_anneal,
                "loss_type": self.dqn.loss_type,
                "grad_clip_norm": self.dqn.grad_clip_norm,
                "reward_clip_value": self.dqn.reward_clip_value,
            },
            "reward_net": {
                "lr": self.reward_net.lr,
                "loss_type": self.reward_net.loss_type,
                "grad_clip_norm": self.reward_net.grad_clip_norm,
                "reward_clip_value": self.reward_net.reward_clip_value,
                "target_mode": self.reward_net.target_mode,
                "state_action_bucket_size": self.reward_net.state_action_bucket_size,
            },
            "runtime": asdict(self.runtime),
        }


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
        "dqn_epsilon_start": 0.95,
        "dqn_epsilon_end": 0.05,
        "dqn_epsilon_anneal": 2e-5,
        "dqn_loss_type": "mse",
        "dqn_loss": None,
        "dqn_scheduler_factory": None,
        "dqn_grad_clip_norm": None,
        "dqn_reward_clip_value": None,
        "reward_net_lr": 1e-3,
        "reward_net_loss_type": "mse",
        "reward_net_loss": None,
        "reward_net_scheduler_factory": None,
        "reward_net_grad_clip_norm": None,
        "reward_net_reward_clip_value": None,
        "reward_net_target_mode": "monte_carlo_return",
        "reward_net_state_action_bucket_size": 0.01,
        "min_bid": 0.0,
        "max_bid": 500.0,
        "traffic_path": "data/traffic_share.csv",
        "bid_lower_clip": 5.0,
        "bid_upper_clip": 5.0,
        "objective": "clicks",
        "eval_mode": True,
        "fit_lambda_init": None,
        "inference_lambda_init": None,
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
        dqn_loss = values.get("dqn_loss", d["dqn_loss"])
        dqn_scheduler_factory = values.get(
            "dqn_scheduler_factory",
            values.get("dqn_scheduler_fn", values.get("dqn_lr_scheduler", d["dqn_scheduler_factory"])),
        )
        reward_loss_type = values.get("reward_net_loss_type", d["reward_net_loss_type"])
        reward_loss = values.get("reward_net_loss", d["reward_net_loss"])
        reward_scheduler_factory = values.get(
            "reward_net_scheduler_factory",
            values.get(
                "reward_net_scheduler_fn",
                values.get("reward_net_lr_scheduler", d["reward_net_scheduler_factory"]),
            ),
        )

        min_bid = values.get("min_bid", values.get("minBid", d["min_bid"]))
        max_bid = values.get("max_bid", values.get("maxBid", d["max_bid"]))
        bid_lower_clip = values.get("bid_lower_clip", d["bid_lower_clip"])
        bid_upper_clip = values.get("bid_upper_clip", d["bid_upper_clip"])
        objective = values.get("objective", d["objective"])
        fit_lambda_init = values.get("fit_lambda_init", d["fit_lambda_init"])
        inference_lambda_init_mode = values.get(
            "inference_lambda_init_mode",
            d["inference_lambda_init_mode"],
        )
        inference_lambda_init = values.get("inference_lambda_init", d["inference_lambda_init"])
        auction_mode = str(values.get("auction_mode", d["auction_mode"])).upper()

        dqn_gamma = values.get("dqn_gamma", d["dqn_gamma"])
        dqn_lr = values.get("dqn_lr", d["dqn_lr"])
        dqn_target_update_interval = values.get(
            "dqn_target_update_interval",
            d["dqn_target_update_interval"],
        )
        dqn_soft_update_tau = values.get("dqn_soft_update_tau", d["dqn_soft_update_tau"])
        dqn_epsilon_start = values.get("dqn_epsilon_start", d["dqn_epsilon_start"])
        dqn_epsilon_end = values.get("dqn_epsilon_end", d["dqn_epsilon_end"])
        dqn_epsilon_anneal = values.get("dqn_epsilon_anneal", d["dqn_epsilon_anneal"])
        dqn_grad_clip_norm = values.get("dqn_grad_clip_norm", d["dqn_grad_clip_norm"])
        dqn_reward_clip_value = values.get(
            "dqn_reward_clip_value",
            d["dqn_reward_clip_value"],
        )

        traffic_path = str(values.get("traffic_path", d["traffic_path"]))
        reward_net_lr = values.get("reward_net_lr", d["reward_net_lr"])
        reward_grad_clip_norm = values.get(
            "reward_net_grad_clip_norm",
            d["reward_net_grad_clip_norm"],
        )
        reward_clip_value = values.get(
            "reward_net_reward_clip_value",
            d["reward_net_reward_clip_value"],
        )
        reward_net_target_mode = values.get(
            "reward_net_target_mode",
            d["reward_net_target_mode"],
        )
        reward_net_state_action_bucket_size = values.get(
            "reward_net_state_action_bucket_size",
            d["reward_net_state_action_bucket_size"],
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
                epsilon_start=dqn_epsilon_start,
                epsilon_end=dqn_epsilon_end,
                epsilon_anneal=dqn_epsilon_anneal,
                loss_type=dqn_loss_type,
                loss=dqn_loss,
                scheduler_factory=dqn_scheduler_factory,
                grad_clip_norm=dqn_grad_clip_norm,
                reward_clip_value=dqn_reward_clip_value,
            ),
            reward_net=RewardNetParams(
                lr=reward_net_lr,
                loss_type=reward_loss_type,
                loss=reward_loss,
                scheduler_factory=reward_scheduler_factory,
                grad_clip_norm=reward_grad_clip_norm,
                reward_clip_value=reward_clip_value,
                target_mode=reward_net_target_mode,
                state_action_bucket_size=reward_net_state_action_bucket_size,
            ),
            runtime=DrlbRuntimeParams(
                min_bid=min_bid,
                max_bid=max_bid,
                traffic_path=traffic_path,
                bid_lower_clip=bid_lower_clip,
                bid_upper_clip=bid_upper_clip,
                objective=objective,
                eval_mode=values.get("eval_mode", d["eval_mode"]),
                fit_lambda_init=fit_lambda_init,
                inference_lambda_init=inference_lambda_init,
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
                "reward_net": {
                    "lr",
                    "loss_type",
                    "grad_clip_norm",
                    "reward_clip_value",
                },
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
            "dqn_epsilon_start": dqn.get(
                "epsilon_start",
                defaults["dqn_epsilon_start"],
            ),
            "dqn_epsilon_end": dqn.get(
                "epsilon_end",
                defaults["dqn_epsilon_end"],
            ),
            "dqn_epsilon_anneal": dqn.get(
                "epsilon_anneal",
                defaults["dqn_epsilon_anneal"],
            ),
            "dqn_loss_type": dqn.get("loss_type", defaults["dqn_loss_type"]),
            "dqn_loss": dqn.get("loss", defaults["dqn_loss"]),
            "dqn_scheduler_factory": dqn.get(
                "scheduler_factory",
                defaults["dqn_scheduler_factory"],
            ),
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
            "reward_net_loss": reward_net.get("loss", defaults["reward_net_loss"]),
            "reward_net_scheduler_factory": reward_net.get(
                "scheduler_factory",
                defaults["reward_net_scheduler_factory"],
            ),
            "reward_net_grad_clip_norm": reward_net.get(
                "grad_clip_norm",
                defaults["reward_net_grad_clip_norm"],
            ),
            "reward_net_reward_clip_value": reward_net.get(
                "reward_clip_value",
                defaults["reward_net_reward_clip_value"],
            ),
            "reward_net_target_mode": reward_net.get(
                "target_mode",
                defaults["reward_net_target_mode"],
            ),
            "reward_net_state_action_bucket_size": reward_net.get(
                "state_action_bucket_size",
                defaults["reward_net_state_action_bucket_size"],
            ),
            "min_bid": runtime.get("min_bid", runtime.get("minBid", defaults["min_bid"])),
            "max_bid": runtime.get("max_bid", runtime.get("maxBid", defaults["max_bid"])),
            "traffic_path": runtime.get("traffic_path", defaults["traffic_path"]),
            "bid_lower_clip": runtime.get("bid_lower_clip", defaults["bid_lower_clip"]),
            "bid_upper_clip": runtime.get("bid_upper_clip", defaults["bid_upper_clip"]),
            "objective": runtime.get("objective", defaults["objective"]),
            "eval_mode": runtime.get("eval_mode", defaults["eval_mode"]),
            "fit_lambda_init": runtime.get(
                "fit_lambda_init",
                defaults["fit_lambda_init"],
            ),
            "inference_lambda_init_mode": runtime.get(
                "inference_lambda_init_mode",
                defaults["inference_lambda_init_mode"],
            ),
            "inference_lambda_init": runtime.get(
                "inference_lambda_init",
                defaults["inference_lambda_init"],
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
