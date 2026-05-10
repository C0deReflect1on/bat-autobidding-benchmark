import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from simulator.model.bidder import _Bidder
from simulator.model.drlb.config_types import DrlbConfig, DrlbConfigParser
from simulator.model.drlb.rl_bid_agent_bat import RlBidAgent
from simulator.model.traffic import Traffic
from simulator.simulation.bat_step_env import BatStepEnv, StepInput
from simulator.simulation.modules import History, SimulationResult
from simulator.simulation.utils import bin2price, price2bin


class DRLBBidder(_Bidder):
    """
    DRLB adapter for BAT simulator interfaces.

    The class wraps the original DRLB agent and exposes BAT-compatible API:
    - place_bid(bidding_input_params, history)
    - fit(stats_df, ...)
    - save_model(path) / load_model(path)
    """

    CHECKPOINT_FORMAT_VERSION = 3

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        params = params or {}

        self._config = DrlbConfigParser.from_dict(params)
        self._apply_config(self._config)
        self.agent = RlBidAgent(self._config)
        self._init_traffic_model(traffic_path_override=params.get("traffic_path"))

        self._campaign_id = None
        self._history_rows_processed = 0
        self._campaign_initialized = False
        self._bid_calls = 0
        self._fit_steps = 0
        self._runtime_step_memory: list[list[float | int]] = []

        model_path = params.get("model_path")
        if model_path:
            self.load_model(str(model_path))
        elif self.init_lambda_mode == "constant" and self.init_lambda is not None:
            self.agent.ctl_lambda = self._clip_lambda(self.init_lambda)

    def _apply_config(self, config: DrlbConfig) -> None:
        self._config = config

        self.state_type = config.model.state_type
        self.T = int(config.model.T)
        self.lambda_min = config.model.lambda_min
        self.lambda_max = config.model.lambda_max

        self.min_bid = config.runtime.min_bid
        self.max_bid = config.runtime.max_bid
        self.bid_lower_clip = config.runtime.bid_lower_clip
        self.bid_upper_clip = config.runtime.bid_upper_clip
        self.objective = str(config.runtime.objective)
        self.eval_mode = bool(config.runtime.eval_mode)
        mode = config.runtime.init_lambda_mode
        if mode not in ("constant", "rule"):
            raise ValueError("init_lambda_mode must be 'constant' or 'rule'")
        self.init_lambda_mode = mode
        raw_init = config.runtime.init_lambda
        self.init_lambda = None if raw_init is None else self._clip_lambda(float(raw_init))
        self.lambda_init_rule = config.runtime.lambda_init_rule
        self.verbose = bool(config.runtime.verbose)
        self.use_tqdm = bool(config.runtime.use_tqdm)
        self.debug_logs = bool(config.runtime.debug_logs)
        self.fit_log_every = int(config.runtime.fit_log_every)
        self.inference_log_every = int(config.runtime.inference_log_every)
        self.auction_mode = str(config.runtime.auction_mode)
        self.traffic_path = str(config.runtime.traffic_path)

    def _current_config_dict(self) -> dict[str, Any]:
        config_dict = self._config.to_dict()
        config_dict["model"]["lambda_action_betas"] = list(config_dict["model"]["lambda_action_betas"])
        return config_dict

    def _hour_index(self, bidding_input_params: Dict[str, Any]) -> int:
        start = int(bidding_input_params["campaign_start_time"])
        curr_time = int(bidding_input_params["curr_time"])
        return max(0, int((curr_time - start) // 3600))

    @staticmethod
    def _campaign_total_steps(start_time: float, end_time: float) -> int:
        duration_seconds = max(end_time - start_time, 3600.0)
        return max(1, int(np.ceil(duration_seconds / 3600.0)))

    @classmethod
    def _elapsed_time_ratio(cls, curr_time: float, start_time: float, end_time: float) -> float:
        duration_seconds = max(end_time - start_time, 3600.0)
        elapsed_seconds = min(max(curr_time - start_time, 0.0), duration_seconds)
        return np.clip(elapsed_seconds / duration_seconds, 0.0, 1.0)

    @staticmethod
    def _scale_budget(budget: float) -> float:
        return float(np.log1p(max(float(budget), 0.0)) / 10.0)

    def _log(self, message: str) -> None:
        if self.debug_logs:
            print(f"[DRLBBidder] {message}")

    def _clip_lambda(self, value: float) -> float:
        return np.clip(value, self.lambda_min, self.lambda_max)

    def _clip_bid_to_budget(self, raw_bid: float, prev_bid: float, balance: float) -> float:
        budget_cap = max(0.0, balance)
        if budget_cap <= 0:
            return 0.0

        bid = max(float(raw_bid), float(self.min_bid))
        if prev_bid > 0 and bid > 0:
            prev_bin = price2bin(prev_bid)
            raw_bin = price2bin(bid)
            clipped_bin = np.clip(
                raw_bin,
                prev_bin - self.bid_lower_clip,
                prev_bin + self.bid_upper_clip,
            )
            bid = bin2price(clipped_bin)
        return min(max(0.0, bid), budget_cap)

    def _resolve_lambda_from_rule(self, initial_balance: float) -> Optional[float]:
        if self.lambda_init_rule is None:
            return None
        edges = np.asarray(self.lambda_init_rule["edges"], dtype=float)
        values = np.asarray(self.lambda_init_rule["values"], dtype=float)
        if values.size == 0:
            return None
        idx = int(np.searchsorted(edges, float(initial_balance), side="right") - 1)
        idx = int(np.clip(idx, 0, values.size - 1))
        return self._clip_lambda(float(values[idx]))

    def get_lambda(self, initial_balance: float) -> Optional[float]:
        if self.init_lambda_mode == "rule":
            return self._resolve_lambda_from_rule(initial_balance)
        return self.init_lambda

    def _init_traffic_model(self, traffic_path_override: Optional[str] = None) -> None:
        if not self.agent.state_repr.uses_traffic_share:
            self.traffic = None
            return

        path_str = str(traffic_path_override or self.traffic_path)
        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            path_obj = repo_root / path_obj
        self.traffic_path = str(path_obj)
        self.traffic = Traffic(path=self.traffic_path)

    def _resolve_traffic_share(
        self,
        region_id: Optional[int],
        start_time: float,
        curr_time: float,
        end_time: float,
    ) -> Optional[float]:
        if self.traffic is None or region_id is None:
            return None
        traffic_total = max(self.traffic.get_traffic_share(int(region_id), int(start_time), int(end_time)), 1e-9)
        traffic_elapsed = self.traffic.get_traffic_share(int(region_id), int(start_time), int(curr_time))
        return float(np.clip(traffic_elapsed / traffic_total, 0.0, 1.0))

    def _uses_campaign_meta_features(self) -> bool:
        return self.agent.state_repr.uses_campaign_meta

    def _default_history_reward_field(self) -> str:
        return "contacts_history" if self.objective == "contacts" else "clicks_history"

    def _resolve_history_reward_field(self, explicit_field: Optional[str] = None) -> str:
        if explicit_field:
            return str(explicit_field)
        return self._default_history_reward_field()

    def _resolve_history_reward(self, row: Dict[str, Any], reward_field: str) -> float:
        return max(0.0, row.get(reward_field, 0.0))

    def _resolve_outcome_reward(self, outcome: SimulationResult, objective: Optional[str] = None) -> float:
        resolved_objective = str(objective or self.objective)
        reward = outcome.contacts if resolved_objective == "contacts" else outcome.clicks
        return max(0.0, reward)

    def _init_agent_episode(
        self,
        initial_balance: float,
        total_steps: int,
        start_time: float,
        end_time: float,
        curr_time: float,
        region_id: Optional[int] = None,
        lambda_init: Optional[float] = None,
    ) -> None:
        self.agent.reset_episode()
        self.agent.configure_episode(initial_balance, total_steps=total_steps)
        if lambda_init is not None:
            self.agent.ctl_lambda = self._clip_lambda(lambda_init)
        if self._uses_campaign_meta_features():
            self.agent.sync_runtime_context(
                balance=initial_balance,
                initial_budget=initial_balance,
                elapsed_time_ratio=self._elapsed_time_ratio(
                    curr_time,
                    start_time,
                    end_time,
                ),
                traffic_share=self._resolve_traffic_share(
                    region_id=region_id,
                    start_time=start_time,
                    curr_time=curr_time,
                    end_time=end_time,
                ),
            )

    def _init_campaign_runtime(self, bidding_input_params: Dict[str, Any]) -> None:
        initial_balance = max(1.0, bidding_input_params.get("initial_balance", 1.0))
        balance = max(0.0, bidding_input_params.get("balance", initial_balance))
        start_time = bidding_input_params.get("campaign_start_time", 0.0)
        end_time = bidding_input_params.get("campaign_end_time", start_time + 3600.0)
        total_steps = self._campaign_total_steps(start_time, end_time)
        lambda_init = self.get_lambda(initial_balance)

        self._init_agent_episode(
            initial_balance=initial_balance,
            total_steps=total_steps,
            start_time=start_time,
            end_time=end_time,
            curr_time=bidding_input_params.get("curr_time", start_time),
            region_id=bidding_input_params.get("region_id"),
            lambda_init=lambda_init,
        )

        self.agent.sync_runtime_context(
            balance=balance,
            initial_budget=initial_balance,
            elapsed_time_ratio=self._elapsed_time_ratio(
                bidding_input_params.get("curr_time", start_time),
                start_time,
                end_time,
            ) if self._uses_campaign_meta_features() else None,
            traffic_share=self._resolve_traffic_share(
                region_id=bidding_input_params.get("region_id"),
                start_time=start_time,
                curr_time=bidding_input_params.get("curr_time", start_time),
                end_time=end_time,
            ),
        )

        self._campaign_initialized = True
        self._campaign_id = bidding_input_params.get("campaign_id")
        self._history_rows_processed = 0
        self._bid_calls = 0
        self._runtime_step_memory = []
        self._log(
            f"init campaign_id={self._campaign_id} "
            f"init_balance={initial_balance:.2f} hour={self._hour_index(bidding_input_params)} "
            f"lambda_init={self.agent.ctl_lambda:.6f} mode={self.init_lambda_mode}"
        )

    def _sync_budget(self, bidding_input_params: Dict[str, Any]) -> None:
        balance = max(0.0, bidding_input_params.get("balance", 0.0))
        initial_balance = max(1.0, bidding_input_params.get("initial_balance", 1.0))
        elapsed_time_ratio = None
        if self._uses_campaign_meta_features():
            start_time = bidding_input_params.get("campaign_start_time", 0.0)
            end_time = bidding_input_params.get("campaign_end_time", start_time + 3600.0)
            curr_time = bidding_input_params.get("curr_time", start_time)
            elapsed_time_ratio = self._elapsed_time_ratio(curr_time, start_time, end_time)
        self.agent.sync_runtime_context(
            balance=balance,
            initial_budget=initial_balance,
            elapsed_time_ratio=elapsed_time_ratio,
            traffic_share=self._resolve_traffic_share(
                region_id=bidding_input_params.get("region_id"),
                start_time=bidding_input_params.get("campaign_start_time", 0.0),
                curr_time=bidding_input_params.get("curr_time", 0.0),
                end_time=bidding_input_params.get(
                    "campaign_end_time",
                    bidding_input_params.get("campaign_start_time", 0.0) + 3600.0,
                ),
            ),
        )

    def _ingest_history(self, history: History, reward_field: Optional[str] = None) -> None:
        resolved_reward_field = self._resolve_history_reward_field(reward_field)
        while self._history_rows_processed < len(history.rows):
            row = history.rows[self._history_rows_processed]
            reward = self._resolve_history_reward(row, resolved_reward_field)
            cost = max(0.0, row.get("spend_history", 0.0))
            win = bool(cost > 0.0)

            self.agent.state_repr.update_state(
                immediate_reward=reward,
                spend=cost,
                win=win,
            )
            if self.debug_logs and self._history_rows_processed % max(1, self.inference_log_every) == 0:
                self._log(
                    f"ingest hour={self._history_rows_processed} "
                    f"reward_field={resolved_reward_field} reward={reward:.4f} cost={cost:.4f} win={int(win)}"
                )
            self._history_rows_processed += 1

    def _build_agent_obs(
        self,
        time_step_index: float,
        ctr_pred: float,
        balance: float,
        initial_balance: float,
        start_time: float,
        end_time: float,
        curr_time: float,
        region_id: Optional[int] = None,
    ) -> Dict[str, float]:
        obs = {
            "timeStepIndex": time_step_index,
            "ctr": max(0.0, ctr_pred),
            "ctr_pred": max(0.0, ctr_pred),
            "balance": max(0.0, balance),
            "initialBalance": max(1.0, initial_balance),
        }
        if self._uses_campaign_meta_features():
            obs["elapsedTimeRatio"] = self._elapsed_time_ratio(
                curr_time,
                start_time,
                end_time,
            )
            obs["initialBudgetScale"] = self._scale_budget(initial_balance)
            traffic_share = self._resolve_traffic_share(
                region_id=region_id,
                start_time=start_time,
                curr_time=curr_time,
                end_time=end_time,
            )
            if traffic_share is not None:
                obs["trafficShare"] = traffic_share
        return obs

    def place_bid(self, bidding_input_params: Dict[str, Any], history: History) -> float:
        campaign_id = bidding_input_params.get("campaign_id")
        if (not self._campaign_initialized) or (campaign_id != self._campaign_id):
            self._init_campaign_runtime(bidding_input_params)

        self._sync_budget(bidding_input_params)
        self._ingest_history(history, reward_field=bidding_input_params.get("history_reward_field"))

        start_time = bidding_input_params['campaign_start_time']
        end_time = bidding_input_params['campaign_end_time']
        ctr_pred = bidding_input_params['ctr_pred']
    
        obs = self._build_agent_obs(
            time_step_index=self._hour_index(bidding_input_params),
            ctr_pred=ctr_pred,
            balance=bidding_input_params["balance"],
            initial_balance=bidding_input_params["initial_balance"],
            start_time=start_time,
            end_time=end_time,
            curr_time=bidding_input_params["curr_time"],
            region_id=bidding_input_params.get("region_id"),
        )

        raw_bid = self.agent.act(obs, eval_mode=self.eval_mode)
        balance = bidding_input_params['balance']
        bid = self._clip_bid_to_budget(
            raw_bid,
            prev_bid=bidding_input_params.get("prev_bid", 0.0),
            balance=balance,
        )
        self._bid_calls += 1
        self._runtime_step_memory.append(
            [
                int(bidding_input_params.get("curr_time", 0)),
                int(self._hour_index(bidding_input_params)),
                float(max(0.0, bidding_input_params["balance"])),
                float(self.agent.ctl_lambda),
                float(self.agent.eps),
                int(self.agent.dqn_action),
                float(obs["ctr"]),
                float(max(0.0, bid)),
            ]
        )
        if self.debug_logs and self._bid_calls % max(1, self.inference_log_every) == 0:
            self._log(
                f"place_bid campaign_id={campaign_id} hour={self._hour_index(bidding_input_params)} "
                f"ctr_pred={obs['ctr']:.6f} raw_bid={raw_bid:.4f} bid={bid:.4f} "
                f"balance={balance:.2f} lambda={self.agent.ctl_lambda:.6f}"
            )
        return max(0.0, bid)

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

        stats = stats_df.sort_values(["campaign_id", "period"]).reset_index(drop=True)

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
        fit_total_rewards = 0.0
        fit_total_wins = 0
        progress = tqdm(total=total_steps, desc="DRLBBidder.fit", unit="step") if self.use_tqdm else None
        train_prior_recorded = False

        try:
            for _, campaign_row in campaigns.iterrows():
                if self._fit_steps >= total_steps:
                    break

                campaign_id = int(campaign_row["campaign_id"])
                campaign_stats = stats[stats["campaign_id"] == campaign_id].copy()
                if campaign_stats.empty:
                    continue

                env = BatStepEnv(
                    stats_pdf=campaign_stats,
                    campaign_row=campaign_row,
                    auction_mode=self.auction_mode,
                )

                campaign_budget = max(1.0, campaign_row["auction_budget"])
                campaign_start = campaign_row["campaign_start"]
                campaign_end = campaign_row["campaign_end"]
                episode_lambda_init = self.get_lambda(campaign_budget)

                self._init_agent_episode(
                    initial_balance=campaign_budget,
                    total_steps=self._campaign_total_steps(campaign_start, campaign_end),
                    start_time=campaign_start,
                    end_time=campaign_end,
                    curr_time=env.campaign.curr_time,
                    region_id=campaign_row.get("region_id"),
                    lambda_init=episode_lambda_init,
                )
                if not train_prior_recorded:
                    self.train_prior_lambda_init = float(self.agent.ctl_lambda)
                    train_prior_recorded = True

                while (not env.done()) and self._fit_steps < total_steps:
                    # Algorithm 2, step 1: observe simulator inputs and build s_t context.
                    step_input = env.get_step_input()
                    obs = self._build_agent_obs(
                        time_step_index=max(0, (step_input.period_start_ts - int(campaign_start)) // 3600),
                        ctr_pred=step_input.ctr_pred,
                        balance=step_input.balance,
                        initial_balance=campaign_budget,
                        start_time=campaign_start,
                        end_time=campaign_end,
                        curr_time=step_input.period_start_ts,
                        region_id=campaign_row.get("region_id"),
                    )

                    # Algorithm 2, step 2: form s_t and pick action a_t from DQN policy.
                    self.agent.state_repr.sync_runtime_context(
                        balance=obs["balance"],
                        initial_budget=obs["initialBalance"],
                        elapsed_time_ratio=obs.get("elapsedTimeRatio"),
                        initial_budget_scale=obs.get("initialBudgetScale"),
                        traffic_share=obs.get("trafficShare"),
                    )
                    state_before_action = self.agent.state_repr.curr_state.copy()
                    action_idx = self.agent.dqn_agent.act(
                        state_before_action,
                        eps=self.agent.eps,
                        eval_mode=False,
                    )
                    action_beta = self.agent.BETA[action_idx]

                    # Algorithm 2, step 3: execute action in environment.
                    raw_bid = self.agent.calc_bid(
                        obs["ctr"],
                        action_beta,
                        available_budget=step_input.balance,
                    )
                    bid = self._clip_bid_to_budget(
                        raw_bid,
                        prev_bid=env.campaign.prev_bid,
                        balance=step_input.balance,
                    )

                    outcome = env.step(bid)
                    # Algorithm 2, step 4: observe reward r_t and close state transition to s_{t+1}.
                    immediate_reward = self._resolve_outcome_reward(outcome, objective)
                    spend = max(0.0, outcome.spent)
                    done = env.done()
                    fit_total_rewards += max(0.0, immediate_reward)
                    if spend > 0.0:
                        fit_total_wins += 1

                    state_after_outcome = self.agent.state_repr.update_state(
                        immediate_reward=immediate_reward,
                        spend=spend,
                        win=bool(spend > 0.0),
                    )

                    rnet_reward = self.agent.predict_reward(state_before_action, action_beta)
                    # Algorithm 2, step 5: update DQN with RewardNet-predicted reward for (s_t, a_t).
                    self.agent.learn_dqn_transition(
                        state_before_action,
                        action_idx,
                        rnet_reward,
                        state_after_outcome,
                        done=done,
                    )

                    # Algorithm 2, step 6: train RewardNet from immediate reward / episode target mode.
                    self.agent.record_reward_net_step(state_before_action, action_beta, immediate_reward)

                    self._fit_steps += 1
                    if progress is not None:
                        progress.update(1)

                    if self.verbose and self._fit_steps % max(1, self.fit_log_every) == 0:
                        self._log(
                            f"fit step={self._fit_steps}/{total_steps} campaign_id={campaign_id} "
                            f"ctr_pred={obs['ctr']:.6f} reward={immediate_reward:.4f} spend={spend:.4f} "
                            f"bid={bid:.4f} lambda={self.agent.ctl_lambda:.6f}"
                        )

                self.agent.finish_episode()
        finally:
            if progress is not None:
                progress.close()

        self.eval_mode = prev_eval_mode
        self._log(
            f"fit done steps={self._fit_steps} total_rewards={fit_total_rewards:.4f} "
            f"total_wins={fit_total_wins} lambda={self.agent.ctl_lambda:.6f}"
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

    def get_runtime_diagnostics(self) -> pd.DataFrame:
        columns = [
            "curr_time",
            "hour_index",
            "balance",
            "lambda",
            "eps",
            "dqn_action",
            "ctr_pred",
            "bid",
        ]
        return pd.DataFrame(self._runtime_step_memory, columns=columns)

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        config_dict = self._current_config_dict()
        config_dict["runtime"] = dict(config_dict["runtime"])
        config_dict["runtime"]["init_lambda"] = float(self.agent.ctl_lambda)

        payload = {
            "model_type": "drlb_bat_adapter",
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "config": config_dict,
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
                "ctl_lambda": self.agent.ctl_lambda,
            },
        }
        torch.save(payload, path)
        self._log(f"model saved path={path}")

    def load_model(self, model_path: str) -> None:
        # Full training checkpoints (config + optimizers), not weights-only tensors.
        # PyTorch >= 2.6 defaults weights_only=True; numpy scalars etc. require False.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint: expected a dict payload.")

        version = checkpoint.get("format_version")
        if version != self.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported DRLB checkpoint format. "
                f"Expected format_version={self.CHECKPOINT_FORMAT_VERSION}, got {version}."
            )

        runtime_traffic_path = self.traffic_path
        config = DrlbConfigParser.from_checkpoint(checkpoint)
        self._apply_config(config)
        self.agent = RlBidAgent(config)
        self._init_traffic_model(traffic_path_override=runtime_traffic_path)

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
        if "ctl_lambda" in runtime_state:
            self.agent.ctl_lambda = runtime_state["ctl_lambda"]
        elif self.init_lambda_mode == "constant" and self.init_lambda is not None:
            self.agent.ctl_lambda = self.init_lambda
        self._log(f"model loaded path={model_path}")
