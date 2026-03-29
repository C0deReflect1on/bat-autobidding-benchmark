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
        "verbose": False,
        "use_tqdm": True,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
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
        self.verbose = bool(params.get("verbose", self.default_params["verbose"]))
        self.use_tqdm = bool(params.get("use_tqdm", self.default_params["use_tqdm"]))
        self.debug_logs = bool(params.get("debug_logs", self.default_params["debug_logs"]))
        self.fit_log_every = int(params.get("fit_log_every", self.default_params["fit_log_every"]))
        self.inference_log_every = int(params.get("inference_log_every", self.default_params["inference_log_every"]))

        self._agent_params = {
            "exp_type": self.exp_type,
            "T": self.T,
            "bids_per_timestep": self.bids_per_timestep,
            "config_path": params.get("config_path"),
        }
        self.agent = RlBidAgent(self._agent_params)

        self._campaign_id = None
        self._history_rows_processed = 0
        self._campaign_initialized = False
        self._bid_calls = 0
        self._fit_steps = 0

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

    def _log(self, message: str) -> None:
        if self.debug_logs:
            print(f"[DRLBBidder] {message}")

    def _init_campaign_runtime(self, bidding_input_params: Dict[str, Any]) -> None:
        self.agent._reset_episode()
        self.agent.bids_per_timestep = max(1, self.bids_per_timestep)

        initial_balance = max(1.0, self._safe_float(bidding_input_params.get("initial_balance"), 1.0))
        balance = max(0.0, self._safe_float(bidding_input_params.get("balance"), initial_balance))
        self.agent.budget = initial_balance
        self.agent.rem_budget = balance
        self.agent.rem_budget_ratio = self.agent.rem_budget / max(self.agent.budget, 1e-9)
        self.agent.cur_time_step = float(self._hour_index(bidding_input_params))
        self.agent.cur_state = self.agent._get_state()

        self._campaign_initialized = True
        self._campaign_id = bidding_input_params.get("campaign_id")
        self._history_rows_processed = 0
        self._bid_calls = 0
        self._log(
            f"init campaign_id={self._campaign_id} "
            f"init_balance={self.agent.budget:.2f} hour={int(self.agent.cur_time_step)}"
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
        return {
            "timeStepIndex": float(self._hour_index(bidding_input_params)),
            "ctr": max(0.0, self._safe_float(bidding_input_params.get("prev_ctr"), 0.0)),
            "leastWinningCost": max(1e-6, self._safe_float(bidding_input_params.get("prev_bid"), 1e-6)),
        }

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

    def fit(self, stats_df: pd.DataFrame, max_steps: Optional[int] = None, objective: Optional[str] = None):
        """
        Offline pretraining on BAT stats_df in an hourly loop.

        This intentionally keeps training simple and BAT-interface compatible.
        """
        required_cols = {
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

        stats = (
            stats_df.sort_values("period")
            .groupby("period", as_index=False)
            .agg(
                ctr=("CTRPredicts", "mean"),
                reward=(target_col, "sum"),
                spend=("AuctionWinBidSurplus", "sum"),
            )
        )
        if max_steps is not None:
            stats = stats.head(int(max_steps))

        if stats.empty:
            self._log("fit skipped: empty stats")
            return self

        # Data-driven initialization of lambda in bid = ctr / lambda.
        spend_non_zero = stats["spend"].clip(lower=1e-6)
        implied_lambda = (stats["ctr"] / spend_non_zero).replace([np.inf, -np.inf], np.nan).dropna()
        if not implied_lambda.empty:
            self.agent.ctl_lambda = float(np.clip(implied_lambda.median(), self.lambda_min, self.lambda_max))
        self._log(
            f"fit start rows={len(stats)} objective={objective} "
            f"lambda_init={float(self.agent.ctl_lambda):.6f}"
        )

        prev_eval_mode = self.eval_mode
        self.eval_mode = False
        self._fit_steps = 0

        synthetic_budget = max(1.0, float(stats["spend"].mean() * max(1, len(stats))))
        self.agent._reset_episode()
        self.agent.bids_per_timestep = max(1, self.bids_per_timestep)
        self.agent.budget = synthetic_budget
        self.agent.rem_budget = synthetic_budget
        self.agent.rem_budget_ratio = 1.0
        self.agent.cur_time_step = 0.0
        self.agent.cur_state = self.agent._get_state()

        fit_iter = stats.itertuples(index=False)
        if self.use_tqdm:
            fit_iter = tqdm(
                fit_iter,
                total=len(stats),
                desc="DRLBBidder.fit",
                unit="step",
            )

        for idx, row in enumerate(fit_iter):
            obs = {
                "timeStepIndex": float(idx),
                "ctr": max(0.0, self._safe_float(row.ctr, 0.0)),
                "leastWinningCost": max(1e-6, self._safe_float(row.spend, 1e-6)),
            }
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
            if self.verbose and self._fit_steps % max(1, self.fit_log_every) == 0:
                self._log(
                    f"fit step={self._fit_steps}/{len(stats)} ctr={obs['ctr']:.6f} "
                    f"reward={reward:.4f} spend={spend:.4f} bid={bid:.4f} "
                    f"lambda={float(self.agent.ctl_lambda):.6f}"
                )

        self.agent.finalize_episode(eval_mode=False)
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
            },
            "agent_state": {
                "ctl_lambda": self.agent.ctl_lambda,
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

        self._agent_params = {
            "exp_type": self.exp_type,
            "T": self.T,
            "bids_per_timestep": self.bids_per_timestep,
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
        self._log(f"model loaded path={model_path}")
