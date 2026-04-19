import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from simulator.model.bidder import _Bidder
from simulator.model.drlb.config_types import DrlbConfig, DrlbConfigParser
from simulator.model.drlb.rl_bid_agent_alibaba import RlBidAgent
from simulator.simulation.modules import Campaign, History, SimulationResult
from simulator.simulation.simulate import simulate_step


@dataclass(frozen=True)
class StepInput:
    campaign_id: int
    period_start_ts: int
    ctr_hint: float
    balance: float
    initial_balance: float
    prev_bid_proxy: float


class _FitCampaignStepEnvironment:
    """Campaign-hour environment used by DRLBBidder.fit."""

    def __init__(
        self,
        stats_pdf: pd.DataFrame,
        campaign_row: pd.Series,
        auction_mode: str,
    ):
        self.stats_pdf = stats_pdf.sort_values("period").reset_index(drop=True)
        self.auction_mode = auction_mode
        self.campaign = self._build_campaign(campaign_row)

    @staticmethod
    def _build_campaign(campaign_row: pd.Series) -> Campaign:
        start = int(campaign_row["campaign_start"])
        end = int(campaign_row["campaign_end"])
        budget = max(1.0, float(campaign_row["auction_budget"]))
        aligned_start = (start // 3600) * 3600
        return Campaign(
            campaign_id=int(campaign_row["campaign_id"]),
            campaign_start=start,
            campaign_end=end,
            initial_balance=budget,
            balance=budget,
            curr_time=aligned_start,
            prev_time=aligned_start,
            prev_balance=budget,
            prev_bid=0.0,
            prev_clicks=0.0,
            prev_contacts=0.0,
        )

    def _window(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        return self.stats_pdf[
            (self.stats_pdf["period"] >= start_ts)
            & (self.stats_pdf["period"] < end_ts)
            & (self.stats_pdf["campaign_id"] == self.campaign.campaign_id)
        ]

    def _ctr_hint(self) -> float:
        window = self._window(self.campaign.curr_time, self.campaign.curr_time + 3600)
        if not window.empty:
            return float(max(0.0, window["CTRPredicts"].mean()))

        max_lookback = max(1, int((self.campaign.curr_time - self.campaign.campaign_start) // 3600) + 1)
        for i in range(1, max_lookback + 1):
            lookback = self._window(
                self.campaign.curr_time - 3600 * i,
                self.campaign.curr_time - 3600 * (i - 1),
            )
            if not lookback.empty:
                return float(max(0.0, lookback["CTRPredicts"].mean()))
        return 0.0

    def get_step_input(self) -> StepInput:
        return StepInput(
            campaign_id=self.campaign.campaign_id,
            period_start_ts=int(self.campaign.curr_time),
            ctr_hint=self._ctr_hint(),
            balance=float(max(0.0, self.campaign.balance)),
            initial_balance=float(max(1.0, self.campaign.initial_balance)),
            prev_bid_proxy=float(max(0.0, self.campaign.prev_bid)),
        )

    def step(self, bid: float) -> SimulationResult:
        simulation_result = simulate_step(
            stats_pdf=self.stats_pdf,
            campaign=self.campaign,
            bid=float(bid),
            auction_mode=self.auction_mode,
        )

        coef = 1.0
        if simulation_result.spent > self.campaign.balance and simulation_result.spent > 0:
            coef = self.campaign.balance / simulation_result.spent

        adjusted = SimulationResult(
            bid=float(bid),
            spent=float(simulation_result.spent) * coef,
            visibility=float(simulation_result.visibility) * coef,
            clicks=float(simulation_result.clicks) * coef,
            contacts=float(simulation_result.contacts) * coef,
        )

        self.campaign.prev_balance = self.campaign.balance
        self.campaign.prev_clicks = self.campaign.clicks
        self.campaign.prev_contacts = self.campaign.contacts
        self.campaign.prev_time = self.campaign.curr_time
        self.campaign.prev_bid = float(bid)
        self.campaign.balance -= adjusted.spent
        self.campaign.clicks += adjusted.clicks
        self.campaign.contacts += adjusted.contacts
        self.campaign.curr_time += 3600
        return adjusted

    def done(self) -> bool:
        return (
            self.campaign.curr_time >= self.campaign.campaign_end
            or self.campaign.balance <= 0
        )


class DRLBBidder(_Bidder):
    """
    DRLB adapter for BAT simulator interfaces.

    The class wraps the original DRLB agent and exposes BAT-compatible API:
    - place_bid(bidding_input_params, history)
    - fit(stats_df, ...)
    - save_model(path) / load_model(path)
    """

    CHECKPOINT_FORMAT_VERSION = 2

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        params = params or {}

        self._config = DrlbConfigParser.from_dict(params)
        self._apply_config(self._config)
        self.agent = RlBidAgent(self._config)

        self._campaign_id = None
        self._history_rows_processed = 0
        self._campaign_initialized = False
        self._bid_calls = 0
        self._fit_steps = 0
        self.train_lambda_init: Optional[float] = None
        self.loaded_checkpoint_lambda: Optional[float] = None

        model_path = params.get("model_path")
        if model_path:
            self.load_model(str(model_path))

    def _apply_config(self, config: DrlbConfig) -> None:
        self._config = config

        self.exp_type = config.model.exp_type
        self.T = int(config.model.T)
        self.bids_per_timestep = int(config.model.bids_per_timestep)
        self.lambda_min = float(config.model.lambda_min)
        self.lambda_max = float(config.model.lambda_max)

        self.dqn_gamma = float(config.dqn.gamma)
        self.dqn_lr = float(config.dqn.lr)
        self.dqn_target_update_interval = int(config.dqn.target_update_interval)
        self.dqn_soft_update_tau = float(config.dqn.soft_update_tau)
        self.dqn_loss_type = str(config.dqn.loss_type)
        self.dqn_grad_clip_norm = config.dqn.grad_clip_norm
        self.dqn_reward_clip_value = config.dqn.reward_clip_value

        self.reward_net_lr = float(config.reward_net.lr)
        self.reward_net_loss_type = str(config.reward_net.loss_type)
        self.reward_net_grad_clip_norm = config.reward_net.grad_clip_norm
        self.reward_net_reward_clip_value = config.reward_net.reward_clip_value

        self.min_bid = float(config.runtime.min_bid)
        self.max_bid = float(config.runtime.max_bid)
        self.objective = str(config.runtime.objective)
        self.eval_mode = bool(config.runtime.eval_mode)
        self.inference_lambda_init_mode = str(config.runtime.inference_lambda_init_mode)
        self.verbose = bool(config.runtime.verbose)
        self.use_tqdm = bool(config.runtime.use_tqdm)
        self.debug_logs = bool(config.runtime.debug_logs)
        self.fit_log_every = int(config.runtime.fit_log_every)
        self.inference_log_every = int(config.runtime.inference_log_every)
        self.auction_mode = str(config.runtime.auction_mode)

    def _current_config_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "exp_type": self.exp_type,
                "T": self.T,
                "bids_per_timestep": self.bids_per_timestep,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
            },
            "dqn": {
                "gamma": self.dqn_gamma,
                "lr": self.dqn_lr,
                "target_update_interval": self.dqn_target_update_interval,
                "soft_update_tau": self.dqn_soft_update_tau,
                "loss_type": self.dqn_loss_type,
                "grad_clip_norm": self.dqn_grad_clip_norm,
                "reward_clip_value": self.dqn_reward_clip_value,
            },
            "reward_net": {
                "lr": self.reward_net_lr,
                "loss_type": self.reward_net_loss_type,
                "grad_clip_norm": self.reward_net_grad_clip_norm,
                "reward_clip_value": self.reward_net_reward_clip_value,
            },
            "runtime": {
                "min_bid": self.min_bid,
                "max_bid": self.max_bid,
                "objective": self.objective,
                "eval_mode": self.eval_mode,
                "inference_lambda_init_mode": self.inference_lambda_init_mode,
                "verbose": self.verbose,
                "use_tqdm": self.use_tqdm,
                "debug_logs": self.debug_logs,
                "fit_log_every": self.fit_log_every,
                "inference_log_every": self.inference_log_every,
                "auction_mode": self.auction_mode,
            },
        }

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
        while self._history_rows_processed < len(history.rows):
            row = history.rows[self._history_rows_processed]
            reward = max(0.0, self._safe_float(row.get("clicks_history"), 0.0))
            cost = max(0.0, self._safe_float(row.get("spend_history"), 0.0))
            win = bool(cost > 0.0)
            prev_bid = max(0.0, self._safe_float(row.get("bid"), 0.0))

            self.agent._update_reward_cost(
                bid=prev_bid,
                reward=reward,
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
            "prevBidProxy": max(0.0, self._safe_float(bidding_input_params.get("prev_bid"), 0.0)),
        }
        if self._uses_campaign_meta_features():
            obs["elapsedTimeRatio"] = self._elapsed_time_ratio(
                self._safe_float(bidding_input_params.get("curr_time"), start_time),
                start_time,
                end_time,
            )
            obs["initialBudgetScale"] = self.agent.initial_budget_scale
        return obs

    def _build_fit_obs(self, step_input: StepInput, campaign_start: float, campaign_end: float) -> Dict[str, float]:
        obs = {
            "timeStepIndex": float(max(0, (step_input.period_start_ts - int(campaign_start)) // 3600)),
            "ctr": max(0.0, float(step_input.ctr_hint)),
            "prevBidProxy": max(0.0, float(step_input.prev_bid_proxy)),
        }
        if self._uses_campaign_meta_features():
            obs["elapsedTimeRatio"] = self._elapsed_time_ratio(
                float(step_input.period_start_ts),
                campaign_start,
                campaign_end,
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
        Offline pretraining on BAT stats_df with simulator-conditioned transitions.
        """
        required_cols = {
            "campaign_id",
            "period",
            "contact_price_bin",
            "CTRPredicts",
            "AuctionClicksSurplus",
            "AuctionContactsSurplus",
            "AuctionWinBidSurplus",
            "AuctionVisibilitySurplus",
        }
        missing = required_cols - set(stats_df.columns)
        if missing:
            raise ValueError(f"stats_df is missing required columns: {sorted(missing)}")

        if campaigns_df is None:
            raise ValueError("campaigns_df is required for campaign-aware DRLBBidder.fit")
        campaign_required_cols = {"campaign_id", "campaign_start", "campaign_end", "auction_budget"}
        campaign_missing = campaign_required_cols - set(campaigns_df.columns)
        if campaign_missing:
            raise ValueError(f"campaigns_df is missing required columns: {sorted(campaign_missing)}")

        objective = objective or self.objective
        if objective not in {"clicks", "contacts"}:
            raise ValueError("objective must be either 'clicks' or 'contacts'")

        stats = stats_df.sort_values(["campaign_id", "period"]).reset_index(drop=True)
        if stats.empty:
            self._log("fit skipped: empty stats")
            return self

        grouped = (
            stats.groupby(["campaign_id", "period"], as_index=False)
            .agg(
                ctr=("CTRPredicts", "mean"),
                spend=("AuctionWinBidSurplus", "sum"),
            )
        )
        spend_non_zero = grouped["spend"].clip(lower=1e-6)
        implied_lambda = (grouped["ctr"] / spend_non_zero).replace([np.inf, -np.inf], np.nan).dropna()
        if not implied_lambda.empty:
            self.agent.ctl_lambda = self._clip_lambda(implied_lambda.median())
        lambda_init = float(self.agent.ctl_lambda)
        self.train_lambda_init = lambda_init

        campaigns = campaigns_df.sort_values("campaign_id").reset_index(drop=True)
        candidate_steps = int(
            campaigns.apply(
                lambda row: self._campaign_total_steps(row["campaign_start"], row["campaign_end"]),
                axis=1,
            ).sum()
        )
        total_steps = int(min(candidate_steps, max_steps)) if max_steps is not None else candidate_steps
        if total_steps <= 0:
            self._log("fit skipped: no campaign steps available")
            return self

        prev_eval_mode = self.eval_mode
        self.eval_mode = False
        self._fit_steps = 0
        progress = tqdm(total=total_steps, desc="DRLBBidder.fit", unit="step") if self.use_tqdm else None

        try:
            for _, campaign_row in campaigns.iterrows():
                if self._fit_steps >= total_steps:
                    break

                campaign_id = int(campaign_row["campaign_id"])
                campaign_stats = stats[stats["campaign_id"] == campaign_id].copy()
                if campaign_stats.empty:
                    continue

                env = _FitCampaignStepEnvironment(
                    stats_pdf=campaign_stats,
                    campaign_row=campaign_row,
                    auction_mode=self.auction_mode,
                )

                campaign_budget = max(1.0, self._safe_float(campaign_row["auction_budget"], 1.0))
                campaign_start = self._safe_float(campaign_row["campaign_start"], 0.0)
                campaign_end = self._safe_float(campaign_row["campaign_end"], campaign_start + 3600.0)

                self.agent._reset_episode()
                self.agent.bids_per_timestep = max(1, self.bids_per_timestep)
                self.agent.configure_episode(
                    campaign_budget,
                    total_steps=self._campaign_total_steps(campaign_start, campaign_end),
                )
                self.agent.ctl_lambda = lambda_init
                self.agent.cur_time_step = 0.0
                if self._uses_campaign_meta_features():
                    self.agent.elapsed_time_ratio = self._elapsed_time_ratio(
                        float(env.campaign.curr_time),
                        campaign_start,
                        campaign_end,
                    )
                self.agent.cur_state = self.agent._get_state()

                while (not env.done()) and self._fit_steps < total_steps:
                    step_input = env.get_step_input()
                    obs = self._build_fit_obs(step_input, campaign_start, campaign_end)

                    raw_bid = float(self.agent.act(obs, eval_mode=False))
                    upper = min(step_input.balance, self.max_bid)
                    bid = 0.0 if upper <= 0 else float(np.clip(raw_bid, self.min_bid, upper))

                    outcome = env.step(bid)
                    reward = float(outcome.contacts if objective == "contacts" else outcome.clicks)
                    spend = max(0.0, float(outcome.spent))
                    self.agent._update_reward_cost(
                        bid=bid,
                        reward=max(0.0, reward),
                        cost=spend,
                        win=bool(spend > 0.0),
                    )

                    self._fit_steps += 1
                    if progress is not None:
                        progress.update(1)

                    if self.verbose and self._fit_steps % max(1, self.fit_log_every) == 0:
                        self._log(
                            f"fit step={self._fit_steps}/{total_steps} campaign_id={campaign_id} "
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
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "config": self._current_config_dict(),
            "weights": {
                "dqn_local": self.agent.dqn_agent.qnetwork_local.state_dict(),
                "dqn_target": self.agent.dqn_agent.qnetwork_target.state_dict(),
                "reward_net": self.agent.reward_net.reward_net.state_dict(),
            },
            "optimizers": {
                "dqn_optimizer": self.agent.dqn_agent.optimizer.state_dict(),
                "reward_optimizer": self.agent.reward_net.optimizer.state_dict(),
            },
            "runtime_state": {
                "ctl_lambda": float(self.agent.ctl_lambda),
                "train_lambda_init": self.train_lambda_init,
            },
        }
        torch.save(payload, path)
        self._log(f"model saved path={path}")

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint: expected a dict payload.")

        version = checkpoint.get("format_version")
        if version != self.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported DRLB checkpoint format. "
                f"Expected format_version={self.CHECKPOINT_FORMAT_VERSION}, got {version}."
            )

        config = DrlbConfigParser.from_checkpoint(checkpoint)
        self._apply_config(config)
        self.agent = RlBidAgent(config)
        self.agent.bids_per_timestep = max(1, self.bids_per_timestep)

        weights = checkpoint.get("weights")
        optimizers = checkpoint.get("optimizers")
        if not isinstance(weights, dict) or not isinstance(optimizers, dict):
            raise ValueError("Invalid checkpoint: missing 'weights' or 'optimizers' sections.")

        required_weights = {"dqn_local", "dqn_target", "reward_net"}
        required_optimizers = {"dqn_optimizer", "reward_optimizer"}
        missing_weights = required_weights - set(weights)
        missing_optimizers = required_optimizers - set(optimizers)
        if missing_weights:
            raise ValueError(f"Checkpoint is missing weight entries: {sorted(missing_weights)}")
        if missing_optimizers:
            raise ValueError(f"Checkpoint is missing optimizer entries: {sorted(missing_optimizers)}")

        self.agent.dqn_agent.qnetwork_local.load_state_dict(weights["dqn_local"])
        self.agent.dqn_agent.qnetwork_target.load_state_dict(weights["dqn_target"])
        self.agent.reward_net.reward_net.load_state_dict(weights["reward_net"])
        self.agent.dqn_agent.optimizer.load_state_dict(optimizers["dqn_optimizer"])
        self.agent.reward_net.optimizer.load_state_dict(optimizers["reward_optimizer"])

        runtime_state = checkpoint.get("runtime_state", {})
        self.agent.ctl_lambda = float(runtime_state.get("ctl_lambda", self.agent.ctl_lambda))
        self.loaded_checkpoint_lambda = self._clip_lambda(self.agent.ctl_lambda)
        train_lambda_init = runtime_state.get("train_lambda_init")
        self.train_lambda_init = None if train_lambda_init is None else self._clip_lambda(train_lambda_init)
        self._log(f"model loaded path={model_path}")
