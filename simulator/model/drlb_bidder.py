import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from simulator.model.bidder import _Bidder
from simulator.model.drlb.rl_bid_agent_alibaba import RlBidAgent
from simulator.simulation.modules import History


class DRLBBidder(_Bidder):
    """
    DRLB adapter for BAT simulator interfaces.

    The class wraps the original DRLB agent and exposes BAT-compatible API:
    - place_bid(bidding_input_params, history)
    - fit(stats_df, ...)
    - save_model(path) / load_model(path)
    """

    default_params = {
        "max_bid": 500.0,
        "min_bid": 0.0,
        "exp_type": "improved_drlb_eval",
        "T": 72,
        "bids_per_timestep": 1,
        "objective": "clicks",  # clicks | contacts
        "model_path": None,
        "eval_mode": True,
        "lambda_min": 1e-6,
        "lambda_max": 10.0,
        "inference_lambda_init_mode": "train_derived",
        "verbose": False,
        "use_tqdm": True,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
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
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        params = params or {}

        self.max_bid = float(params.get("max_bid", self.default_params["max_bid"]))
        self.min_bid = float(params.get("min_bid", self.default_params["min_bid"]))
        self.exp_type = str(params.get("exp_type", self.default_params["exp_type"]))
        self.T = int(params.get("T", self.default_params["T"]))
        # One BAT hour is treated as one DRLB step.
        self.bids_per_timestep = int(params.get("bids_per_timestep", self.default_params["bids_per_timestep"]))
        self.objective = str(params.get("objective", self.default_params["objective"]))
        self.eval_mode = bool(params.get("eval_mode", self.default_params["eval_mode"]))
        self.lambda_min = float(params.get("lambda_min", self.default_params["lambda_min"]))
        self.lambda_max = float(params.get("lambda_max", self.default_params["lambda_max"]))
        self.inference_lambda_init_mode = str(
            params.get("inference_lambda_init_mode", self.default_params["inference_lambda_init_mode"])
        )
        self.verbose = bool(params.get("verbose", self.default_params["verbose"]))
        self.use_tqdm = bool(params.get("use_tqdm", self.default_params["use_tqdm"]))
        self.debug_logs = bool(params.get("debug_logs", self.default_params["debug_logs"]))
        self.fit_log_every = int(params.get("fit_log_every", self.default_params["fit_log_every"]))
        self.inference_log_every = int(params.get("inference_log_every", self.default_params["inference_log_every"]))
        self.dqn_gamma = float(params.get("dqn_gamma", self.default_params["dqn_gamma"]))
        self.dqn_lr = float(params.get("dqn_lr", self.default_params["dqn_lr"]))
        self.dqn_target_update_interval = int(
            params.get("dqn_target_update_interval", self.default_params["dqn_target_update_interval"])
        )
        self.dqn_soft_update_tau = float(
            params.get("dqn_soft_update_tau", self.default_params["dqn_soft_update_tau"])
        )
        self.dqn_loss_type = str(params.get("dqn_loss_type", self.default_params["dqn_loss_type"]))
        dqn_grad_clip = params.get("dqn_grad_clip_norm", self.default_params["dqn_grad_clip_norm"])
        self.dqn_grad_clip_norm = None if dqn_grad_clip is None else float(dqn_grad_clip)
        dqn_reward_clip = params.get("dqn_reward_clip_value", self.default_params["dqn_reward_clip_value"])
        self.dqn_reward_clip_value = None if dqn_reward_clip is None else float(dqn_reward_clip)
        self.reward_net_lr = float(params.get("reward_net_lr", self.default_params["reward_net_lr"]))
        self.reward_net_loss_type = str(
            params.get("reward_net_loss_type", self.default_params["reward_net_loss_type"])
        )
        reward_net_grad_clip = params.get(
            "reward_net_grad_clip_norm",
            self.default_params["reward_net_grad_clip_norm"],
        )
        self.reward_net_grad_clip_norm = (
            None if reward_net_grad_clip is None else float(reward_net_grad_clip)
        )
        reward_net_clip = params.get(
            "reward_net_reward_clip_value",
            self.default_params["reward_net_reward_clip_value"],
        )
        self.reward_net_reward_clip_value = (
            None if reward_net_clip is None else float(reward_net_clip)
        )

        self._agent_params = {
            "exp_type": self.exp_type,
            "T": self.T,
            "bids_per_timestep": self.bids_per_timestep,
            "config_path": params.get("config_path"),
            "dqn_gamma": self.dqn_gamma,
            "dqn_lr": self.dqn_lr,
            "dqn_target_update_interval": self.dqn_target_update_interval,
            "dqn_soft_update_tau": self.dqn_soft_update_tau,
            "dqn_loss_type": self.dqn_loss_type,
            "dqn_grad_clip_norm": self.dqn_grad_clip_norm,
            "dqn_reward_clip_value": self.dqn_reward_clip_value,
            "reward_net_lr": self.reward_net_lr,
            "reward_net_loss_type": self.reward_net_loss_type,
            "reward_net_grad_clip_norm": self.reward_net_grad_clip_norm,
            "reward_net_reward_clip_value": self.reward_net_reward_clip_value,
        }
        self.agent = RlBidAgent(self._agent_params)

        self._campaign_id = None
        self._history_rows_processed = 0
        self._campaign_initialized = False
        self._bid_calls = 0
        self._fit_steps = 0
        self.train_lambda_init: Optional[float] = None
        self.loaded_checkpoint_lambda: Optional[float] = None

        model_path = params.get("model_path", self.default_params["model_path"])
        if model_path:
            self.load_model(model_path)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
            if not np.isfinite(out):
                return default
            return out
        except (TypeError, ValueError):
            return default

    def _hour_index(self, bidding_input_params: Dict[str, Any]) -> int:
        start = int(bidding_input_params["campaign_start_time"])
        curr_time = int(bidding_input_params["curr_time"])
        return max(0, int((curr_time - start) // 3600))

    @staticmethod
    def _campaign_total_steps(start_time: float, end_time: float) -> int:
        duration_seconds = max(float(end_time) - float(start_time), 3600.0)
        return max(1, int(np.ceil(duration_seconds / 3600.0)))

    @classmethod
    def _elapsed_time_ratio(cls, curr_time: float, start_time: float, end_time: float) -> float:
        duration_seconds = max(float(end_time) - float(start_time), 3600.0)
        elapsed_seconds = min(max(float(curr_time) - float(start_time), 0.0), duration_seconds)
        return float(np.clip(elapsed_seconds / duration_seconds, 0.0, 1.0))

    def _log(self, message: str) -> None:
        if self.debug_logs:
            print(f"[DRLBBidder] {message}")

    def _clip_lambda(self, value: float) -> float:
        return float(np.clip(float(value), self.lambda_min, self.lambda_max))

    def _resolve_inference_lambda_init(self) -> Optional[float]:
        if self.inference_lambda_init_mode == "train_derived" and self.train_lambda_init is not None:
            return self._clip_lambda(self.train_lambda_init)
        if self.inference_lambda_init_mode == "checkpoint_final" and self.loaded_checkpoint_lambda is not None:
            return self._clip_lambda(self.loaded_checkpoint_lambda)
        if self.inference_lambda_init_mode == "legacy":
            return None
        # Keep old checkpoints usable: if train-derived is requested but unavailable,
        # fall back to the loaded checkpoint lambda before using the legacy reset value.
        if self.loaded_checkpoint_lambda is not None:
            return self._clip_lambda(self.loaded_checkpoint_lambda)
        return None

    def _uses_campaign_meta_features(self) -> bool:
        return self.agent.state_repr.uses_campaign_meta

    def _init_campaign_runtime(self, bidding_input_params: Dict[str, Any]) -> None:
        self.agent._reset_episode()
        self.agent.bids_per_timestep = max(1, self.bids_per_timestep)

        initial_balance = max(1.0, self._safe_float(bidding_input_params.get("initial_balance"), 1.0))
        balance = max(0.0, self._safe_float(bidding_input_params.get("balance"), initial_balance))
        start_time = self._safe_float(bidding_input_params.get("campaign_start_time"), 0.0)
        end_time = self._safe_float(bidding_input_params.get("campaign_end_time"), start_time + 3600.0)
        total_steps = self._campaign_total_steps(start_time, end_time)
        self.agent.configure_episode(initial_balance, total_steps=total_steps)
        inference_lambda_init = self._resolve_inference_lambda_init()
        if inference_lambda_init is not None:
            self.agent.ctl_lambda = inference_lambda_init
        self.agent.rem_budget = balance
        self.agent.rem_budget_ratio = self.agent.rem_budget / max(self.agent.budget, 1e-9)
        if self._uses_campaign_meta_features():
            self.agent.elapsed_time_ratio = self._elapsed_time_ratio(
                self._safe_float(bidding_input_params.get("curr_time"), start_time),
                start_time,
                end_time,
            )
        self.agent.cur_time_step = float(self._hour_index(bidding_input_params))
        self.agent.cur_state = self.agent._get_state()

        self._campaign_initialized = True
        self._campaign_id = bidding_input_params.get("campaign_id")
        self._history_rows_processed = 0
        self._bid_calls = 0
        self._log(
            f"init campaign_id={self._campaign_id} "
            f"init_balance={self.agent.budget:.2f} hour={int(self.agent.cur_time_step)} "
            f"lambda_init={float(self.agent.ctl_lambda):.6f} mode={self.inference_lambda_init_mode}"
        )

    def _sync_budget(self, bidding_input_params: Dict[str, Any]) -> None:
        balance = max(0.0, self._safe_float(bidding_input_params.get("balance"), 0.0))
        initial_balance = max(1.0, self._safe_float(bidding_input_params.get("initial_balance"), 1.0))
        self.agent.budget = initial_balance
        self.agent.rem_budget = balance
        self.agent.rem_budget_ratio = balance / max(initial_balance, 1e-9)

    def _ingest_history(self, history: History) -> None:
        # Consume all new BAT rows once. Each row corresponds to one completed hour.
        while self._history_rows_processed < len(history.rows):
            row = history.rows[self._history_rows_processed]
            reward = max(0.0, self._safe_float(row.get("clicks_history"), 0.0))

            cost = max(0.0, self._safe_float(row.get("spend_history"), 0.0))
            win = bool(cost > 0.0)
            prev_bid = max(0.0, self._safe_float(row.get("bid"), 0.0))

            # BAT does not expose "potential clicks"; use observed reward as a conservative proxy.
            self.agent._update_reward_cost(
                bid=prev_bid,
                reward=reward,
                potential_reward=reward,
                cost=cost,
                win=win,
            )
            if self.debug_logs and self._history_rows_processed % max(1, self.inference_log_every) == 0:
                self._log(
                    f"ingest hour={self._history_rows_processed} "
                    f"reward={reward:.4f} cost={cost:.4f} win={int(win)}"
                )
            self._history_rows_processed += 1

    def _build_obs(self, bidding_input_params: Dict[str, Any]) -> Dict[str, float]:
        start_time = self._safe_float(bidding_input_params.get("campaign_start_time"), 0.0)
        end_time = self._safe_float(bidding_input_params.get("campaign_end_time"), start_time + 3600.0)
        obs = {
            "timeStepIndex": float(self._hour_index(bidding_input_params)),
            "ctr": max(0.0, self._safe_float(bidding_input_params.get("prev_ctr"), 0.0)),
            "leastWinningCost": max(1e-6, self._safe_float(bidding_input_params.get("prev_bid"), 1e-6)),
        }
        if self._uses_campaign_meta_features():
            obs["elapsedTimeRatio"] = self._elapsed_time_ratio(
                self._safe_float(bidding_input_params.get("curr_time"), start_time),
                start_time,
                end_time,
            )
            obs["initialBudgetScale"] = self.agent.initial_budget_scale
        return obs

    def place_bid(self, bidding_input_params: Dict[str, Any], history: History) -> float:
        campaign_id = bidding_input_params.get("campaign_id")
        if (not self._campaign_initialized) or (campaign_id != self._campaign_id):
            self._init_campaign_runtime(bidding_input_params)

        self._sync_budget(bidding_input_params)
        self._ingest_history(history)
        obs = self._build_obs(bidding_input_params)

        raw_bid = float(self.agent.act(obs, eval_mode=self.eval_mode))
        balance = max(0.0, self._safe_float(bidding_input_params.get("balance"), 0.0))
        upper = min(balance, self.max_bid)
        if upper <= 0:
            return 0.0

        bid = np.clip(raw_bid, self.min_bid, upper)
        self._bid_calls += 1
        if self.debug_logs and self._bid_calls % max(1, self.inference_log_every) == 0:
            self._log(
                f"place_bid campaign_id={campaign_id} hour={self._hour_index(bidding_input_params)} "
                f"ctr={obs['ctr']:.6f} raw_bid={raw_bid:.4f} bid={float(bid):.4f} "
                f"balance={balance:.2f} lambda={float(self.agent.ctl_lambda):.6f}"
            )
        return float(max(0.0, bid))

    def fit(
        self,
        stats_df: pd.DataFrame,
        campaigns_df: Optional[pd.DataFrame] = None,
        max_steps: Optional[int] = None,
        objective: Optional[str] = None,
    ):
        """
        Offline pretraining on BAT stats_df in a campaign-aware hourly loop.
        """
        required_cols = {
            "campaign_id",
            "period",
            "CTRPredicts",
            "AuctionClicksSurplus",
            "AuctionContactsSurplus",
            "AuctionWinBidSurplus",
        }
        missing = required_cols - set(stats_df.columns)
        if missing:
            raise ValueError(f"stats_df is missing required columns: {sorted(missing)}")

        objective = objective or self.objective
        target_col = "AuctionContactsSurplus" if objective == "contacts" else "AuctionClicksSurplus"
        grouped_stats = (
            stats_df.sort_values(["campaign_id", "period"])
            .groupby(["campaign_id", "period"], as_index=False)
            .agg(
                ctr=("CTRPredicts", "mean"),
                reward=(target_col, "sum"),
                spend=("AuctionWinBidSurplus", "sum"),
            )
        )
        if campaigns_df is None:
            raise ValueError("campaigns_df is required for campaign-aware DRLBBidder.fit")
        campaign_required_cols = {"campaign_id", "campaign_start", "campaign_end", "auction_budget"}
        campaign_missing = campaign_required_cols - set(campaigns_df.columns)
        if campaign_missing:
            raise ValueError(f"campaigns_df is missing required columns: {sorted(campaign_missing)}")
        stats = grouped_stats.merge(
            campaigns_df.loc[:, ["campaign_id", "campaign_start", "campaign_end", "auction_budget"]],
            on="campaign_id",
            how="left",
        )
        if stats[["campaign_start", "campaign_end", "auction_budget"]].isna().any().any():
            missing_campaign_ids = (
                stats.loc[
                    stats[["campaign_start", "campaign_end", "auction_budget"]].isna().any(axis=1),
                    "campaign_id",
                ]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                "campaign metadata is missing for campaign_id values: "
                f"{missing_campaign_ids[:10]}"
            )
        stats = stats.sort_values(["campaign_id", "period"]).reset_index(drop=True)

        if stats.empty:
            self._log("fit skipped: empty stats")
            return self

        # Data-driven initialization of lambda in bid = ctr / lambda.
        spend_non_zero = stats["spend"].clip(lower=1e-6)
        implied_lambda = (stats["ctr"] / spend_non_zero).replace([np.inf, -np.inf], np.nan).dropna()
        if not implied_lambda.empty:
            self.agent.ctl_lambda = self._clip_lambda(implied_lambda.median())
        self._log(
            f"fit start rows={len(stats)} campaigns={stats['campaign_id'].nunique()} objective={objective} "
            f"lambda_init={float(self.agent.ctl_lambda):.6f}"
        )
        lambda_init = float(self.agent.ctl_lambda)
        self.train_lambda_init = lambda_init

        prev_eval_mode = self.eval_mode
        self.eval_mode = False
        self._fit_steps = 0
        total_steps = int(min(len(stats), max_steps)) if max_steps is not None else int(len(stats))
        progress = tqdm(total=total_steps, desc="DRLBBidder.fit", unit="step") if self.use_tqdm else None

        try:
            for _, campaign_stats in stats.groupby("campaign_id", sort=False):
                if self._fit_steps >= total_steps:
                    break
                campaign_stats = campaign_stats.sort_values("period").reset_index(drop=True)
                campaign_budget = max(1.0, self._safe_float(campaign_stats["auction_budget"].iloc[0], 1.0))
                campaign_start = self._safe_float(campaign_stats["campaign_start"].iloc[0], 0.0)
                campaign_end = self._safe_float(campaign_stats["campaign_end"].iloc[0], campaign_start + 3600.0)

                self.agent._reset_episode()
                self.agent.bids_per_timestep = max(1, self.bids_per_timestep)
                self.agent.configure_episode(campaign_budget, total_steps=len(campaign_stats))
                self.agent.ctl_lambda = lambda_init
                self.agent.cur_time_step = 0.0
                if self._uses_campaign_meta_features():
                    self.agent.elapsed_time_ratio = self._elapsed_time_ratio(
                        self._safe_float(campaign_stats["period"].iloc[0], campaign_start),
                        campaign_start,
                        campaign_end,
                    )
                self.agent.cur_state = self.agent._get_state()

                for idx, row in enumerate(campaign_stats.itertuples(index=False)):
                    if self._fit_steps >= total_steps:
                        break
                    elapsed_ratio = self._elapsed_time_ratio(
                        self._safe_float(row.period, campaign_start),
                        campaign_start,
                        campaign_end,
                    )
                    obs = {
                        "timeStepIndex": float(idx),
                        "ctr": max(0.0, self._safe_float(row.ctr, 0.0)),
                        "leastWinningCost": max(1e-6, self._safe_float(row.spend, 1e-6)),
                    }
                    if self._uses_campaign_meta_features():
                        obs["elapsedTimeRatio"] = elapsed_ratio
                        obs["initialBudgetScale"] = self.agent.initial_budget_scale
                    bid = float(self.agent.act(obs, eval_mode=False))

                    reward = max(0.0, self._safe_float(row.reward, 0.0))
                    spend = max(0.0, self._safe_float(row.spend, 0.0))
                    self.agent._update_reward_cost(
                        bid=bid,
                        reward=reward,
                        potential_reward=reward,
                        cost=spend,
                        win=bool(spend > 0.0),
                    )
                    self._fit_steps += 1
                    if progress is not None:
                        progress.update(1)
                    if self.verbose and self._fit_steps % max(1, self.fit_log_every) == 0:
                        self._log(
                            f"fit step={self._fit_steps}/{total_steps} campaign_id={int(row.campaign_id)} "
                            f"ctr={obs['ctr']:.6f} reward={reward:.4f} spend={spend:.4f} "
                            f"bid={bid:.4f} lambda={float(self.agent.ctl_lambda):.6f}"
                        )
                if self.agent.bids_processed_in_current_timestep > 0:
                    self.agent.finalize_episode(eval_mode=False)
        finally:
            if progress is not None:
                progress.close()

        self.eval_mode = prev_eval_mode
        self._log(
            f"fit done steps={self._fit_steps} total_rewards={float(self.agent.total_rewards):.4f} "
            f"total_wins={int(self.agent.total_wins)} lambda={float(self.agent.ctl_lambda):.6f}"
        )
        return self

    def get_training_diagnostics(self) -> pd.DataFrame:
        columns = [
            "global_t",
            "rem_budget",
            "lambda",
            "eps",
            "dqn_action",
            "dqn_loss",
            "reward_signal",
            "reward_net_loss",
        ]
        return pd.DataFrame(self.agent.step_memory, columns=columns)

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        payload = {
            "model_type": "drlb_bat_adapter",
            "config": {
                "max_bid": self.max_bid,
                "min_bid": self.min_bid,
                "exp_type": self.exp_type,
                "T": self.T,
                "bids_per_timestep": self.bids_per_timestep,
                "objective": self.objective,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "inference_lambda_init_mode": self.inference_lambda_init_mode,
                "dqn_gamma": self.dqn_gamma,
                "dqn_lr": self.dqn_lr,
                "dqn_target_update_interval": self.dqn_target_update_interval,
                "dqn_soft_update_tau": self.dqn_soft_update_tau,
                "dqn_loss_type": self.dqn_loss_type,
                "dqn_grad_clip_norm": self.dqn_grad_clip_norm,
                "dqn_reward_clip_value": self.dqn_reward_clip_value,
                "reward_net_lr": self.reward_net_lr,
                "reward_net_loss_type": self.reward_net_loss_type,
                "reward_net_grad_clip_norm": self.reward_net_grad_clip_norm,
                "reward_net_reward_clip_value": self.reward_net_reward_clip_value,
            },
            "agent_state": {
                "ctl_lambda": self.agent.ctl_lambda,
                "train_lambda_init": self.train_lambda_init,
                "dqn_local": self.agent.dqn_agent.qnetwork_local.state_dict(),
                "dqn_target": self.agent.dqn_agent.qnetwork_target.state_dict(),
                "dqn_optimizer": self.agent.dqn_agent.optimizer.state_dict(),
                "reward_net": self.agent.reward_net.reward_net.state_dict(),
                "reward_optimizer": self.agent.reward_net.optimizer.state_dict(),
            },
        }
        torch.save(payload, path)
        self._log(f"model saved path={path}")

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint.get("config", {})
        agent_state = checkpoint.get("agent_state", {})

        self.max_bid = float(config.get("max_bid", self.max_bid))
        self.min_bid = float(config.get("min_bid", self.min_bid))
        self.exp_type = str(config.get("exp_type", self.exp_type))
        self.T = int(config.get("T", self.T))
        self.bids_per_timestep = int(config.get("bids_per_timestep", self.bids_per_timestep))
        self.objective = str(config.get("objective", self.objective))
        self.lambda_min = float(config.get("lambda_min", self.lambda_min))
        self.lambda_max = float(config.get("lambda_max", self.lambda_max))
        self.inference_lambda_init_mode = str(
            config.get("inference_lambda_init_mode", self.inference_lambda_init_mode)
        )
        self.dqn_gamma = float(config.get("dqn_gamma", self.dqn_gamma))
        self.dqn_lr = float(config.get("dqn_lr", self.dqn_lr))
        self.dqn_target_update_interval = int(
            config.get("dqn_target_update_interval", self.dqn_target_update_interval)
        )
        self.dqn_soft_update_tau = float(config.get("dqn_soft_update_tau", self.dqn_soft_update_tau))
        self.dqn_loss_type = str(config.get("dqn_loss_type", self.dqn_loss_type))
        dqn_grad_clip = config.get("dqn_grad_clip_norm", self.dqn_grad_clip_norm)
        self.dqn_grad_clip_norm = None if dqn_grad_clip is None else float(dqn_grad_clip)
        dqn_reward_clip = config.get("dqn_reward_clip_value", self.dqn_reward_clip_value)
        self.dqn_reward_clip_value = None if dqn_reward_clip is None else float(dqn_reward_clip)
        self.reward_net_lr = float(config.get("reward_net_lr", self.reward_net_lr))
        self.reward_net_loss_type = str(config.get("reward_net_loss_type", self.reward_net_loss_type))
        reward_net_grad_clip = config.get("reward_net_grad_clip_norm", self.reward_net_grad_clip_norm)
        self.reward_net_grad_clip_norm = (
            None if reward_net_grad_clip is None else float(reward_net_grad_clip)
        )
        reward_net_clip = config.get("reward_net_reward_clip_value", self.reward_net_reward_clip_value)
        self.reward_net_reward_clip_value = (
            None if reward_net_clip is None else float(reward_net_clip)
        )

        dqn_local = agent_state.get("dqn_local", {})
        fc1_weight = dqn_local.get("fc1.weight")
        checkpoint_state_size = int(fc1_weight.shape[1]) if fc1_weight is not None else None
        if self.exp_type in {"improved_drlb", "improved_drlb_eval"} and checkpoint_state_size == 7:
            self.exp_type = "scaled_budget_eval" if self.exp_type.endswith("_eval") else "scaled_budget"

        self._agent_params = {
            "exp_type": self.exp_type,
            "T": self.T,
            "bids_per_timestep": self.bids_per_timestep,
            "dqn_gamma": self.dqn_gamma,
            "dqn_lr": self.dqn_lr,
            "dqn_target_update_interval": self.dqn_target_update_interval,
            "dqn_soft_update_tau": self.dqn_soft_update_tau,
            "dqn_loss_type": self.dqn_loss_type,
            "dqn_grad_clip_norm": self.dqn_grad_clip_norm,
            "dqn_reward_clip_value": self.dqn_reward_clip_value,
            "reward_net_lr": self.reward_net_lr,
            "reward_net_loss_type": self.reward_net_loss_type,
            "reward_net_grad_clip_norm": self.reward_net_grad_clip_norm,
            "reward_net_reward_clip_value": self.reward_net_reward_clip_value,
        }
        self.agent = RlBidAgent(self._agent_params)
        self.agent.bids_per_timestep = max(1, self.bids_per_timestep)

        if "dqn_local" in agent_state:
            self.agent.dqn_agent.qnetwork_local.load_state_dict(agent_state["dqn_local"])
        if "dqn_target" in agent_state:
            self.agent.dqn_agent.qnetwork_target.load_state_dict(agent_state["dqn_target"])
        if "dqn_optimizer" in agent_state:
            self.agent.dqn_agent.optimizer.load_state_dict(agent_state["dqn_optimizer"])
        if "reward_net" in agent_state:
            self.agent.reward_net.reward_net.load_state_dict(agent_state["reward_net"])
        if "reward_optimizer" in agent_state:
            self.agent.reward_net.optimizer.load_state_dict(agent_state["reward_optimizer"])
        self.agent.ctl_lambda = float(agent_state.get("ctl_lambda", self.agent.ctl_lambda))
        self.loaded_checkpoint_lambda = self._clip_lambda(self.agent.ctl_lambda)
        train_lambda_init = agent_state.get("train_lambda_init")
        self.train_lambda_init = None if train_lambda_init is None else self._clip_lambda(train_lambda_init)
        self._log(f"model loaded path={model_path}")
