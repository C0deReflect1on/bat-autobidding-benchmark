"""
DRLB state representation variants.

Each instance owns mutable campaign state and exposes explicit transition
snapshots: state_before_action is s_t, state_after_outcome is s_{t+1}.
"""
from dataclasses import dataclass

import numpy as np


class BaseStateRepresentation:
    state_size: int = 0
    state_action_size: int = 0
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False
    uses_traffic_share: bool = False

    @staticmethod
    def _scale_budget(budget):
        return float(np.log1p(max(float(budget), 0.0)) / 10.0)

    def begin_episode(self, initial_budget, total_steps=None):
        self.t_step = 0
        self.budget = max(1.0, float(initial_budget))
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.traffic_share = 0
        self.initial_budget_scale = self._scale_budget(self.budget)
        self.elapsed_time_ratio = 0
        self.episode_steps_total = max(1, int(total_steps or 1))
        self.ROL = self.episode_steps_total
        self.ROL_ratio = 1
        self.budget_spent_e = 0
        self.wins_e = 0
        self.rewards_e = 0
        self.total_wins = 0
        self.total_rewards = 0
        self.rewards_prev_t = 0
        self.rewards_prev_t_ratio = 0
        self.BCR = 0
        self.WR = 0
        self.CPI = 0
        self.CPM = 0
        self._reset_step_accumulators()
        self.curr_state = self._build_state()

    def sync_runtime_context(
        self,
        balance,
        initial_budget,
        elapsed_time_ratio=None,
        initial_budget_scale=None,
        traffic_share=None,
    ):
        self.budget = max(1.0, float(initial_budget))
        self.rem_budget = max(0.0, float(balance))
        self.rem_budget_ratio = self.rem_budget / max(self.budget, 1e-9)
        self.initial_budget_scale = self._scale_budget(self.budget)
        if elapsed_time_ratio is not None:
            self.elapsed_time_ratio = float(np.clip(elapsed_time_ratio, 0.0, 1.0))
        if initial_budget_scale is not None:
            self.initial_budget_scale = float(initial_budget_scale)
        if traffic_share is not None:
            self.traffic_share = float(np.clip(traffic_share, 0.0, 1.0))
        self.curr_state = self._build_state()

    def update_state(self, immediate_reward, spend, win):
        self.imp_opps_t += 1

        if win:
            self.budget_spent_t += spend
            self.wins_t += 1
            self.wins_e += 1
            self.total_wins += 1
            self.reward_t += immediate_reward
            self.rewards_e += immediate_reward
            self.total_rewards += immediate_reward
            self.cost_t += spend
        self.t_step += 1
        prev_budget = self.rem_budget
        self.rem_budget = max(prev_budget - self.budget_spent_t, 0)
        self.budget_spent_e += self.budget_spent_t
        self.rewards_prev_t = self.reward_t
        self.ROL = max(self.ROL - 1, 0)
        self.BCR = 0 if prev_budget == 0 else -((self.rem_budget - prev_budget) / prev_budget)
        self.compute_step_metrics(self)
        self.WR = self.wins_t / max(self.imp_opps_t, 1)
        self.curr_state = self._build_state()
        self._reset_step_accumulators()
        return self.curr_state.copy()

    def finish_episode(self):
        return

    def available_budget_for_step(self):
        return max(self.rem_budget - self.budget_spent_t, 0)

    def get_state(self, agent=None) -> np.ndarray:
        target = self if agent is None else agent
        return self._build_state_from(target)

    def _build_state(self) -> np.ndarray:
        return self._build_state_from(self)

    def _build_state_from(self, target) -> np.ndarray:
        raise NotImplementedError

    def _reset_step_accumulators(self):
        self.possible_clicks_t = 0
        self.total_rewards_t = 0
        self.reward_t = 0
        self.cost_t = 0
        self.wins_t = 0
        self.imp_opps_t = 0
        self.budget_spent_t = 0

    def _compute_common_ratios(self, target) -> None:
        target.ROL_ratio = max(target.ROL, 0) / max(target.episode_steps_total, 1)
        target.rem_budget_ratio = max(target.rem_budget, 0) / max(target.budget, 1)

    def _compute_cpi(self, target) -> None:
        target.CPI = 0 if target.wins_t == 0 else (target.cost_t / target.wins_t) / 300

    def _compute_cpm(self, target) -> None:
        target.CPM = 0 if target.wins_t == 0 else (target.cost_t / target.wins_t) * 1000

    def _compute_reward_density(self, target) -> None:
        target.rewards_prev_t_ratio = target.reward_t / max(target.imp_opps_t, 1)

    def compute_step_metrics(self, target=None) -> None:
        self._compute_common_ratios(self if target is None else target)

    def reset_step_fields(self, target=None) -> None:
        return


class CpiRatioState(BaseStateRepresentation):
    def compute_step_metrics(self, target=None) -> None:
        target = self if target is None else target
        self._compute_cpi(target)
        self._compute_reward_density(target)
        self._compute_common_ratios(target)

    def reset_step_fields(self, target=None) -> None:
        target = self if target is None else target
        target.CPI = 0


@dataclass
class ImprovedState(CpiRatioState):
    """Original-style DRLB state adapted for BAT (6-dim)."""

    state_size: int = 6
    state_action_size: int = 7
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False
    uses_traffic_share: bool = False

    def _build_state_from(self, target) -> np.ndarray:
        return np.asarray([
            target.rem_budget_ratio,
            target.ROL_ratio,
            target.BCR,
            target.CPI,
            target.WR,
            target.rewards_prev_t_ratio,
        ], dtype=np.float32)


@dataclass
class ScaledBudgetState(CpiRatioState):
    """Adds elapsed-time ratio and log-scaled budget to improved state (7-dim)."""

    state_size: int = 7
    state_action_size: int = 8
    reward_net_order: str = "learn_first"
    uses_campaign_meta: bool = True
    uses_traffic_share: bool = False

    def _build_state_from(self, target) -> np.ndarray:
        return np.asarray([
            target.rem_budget_ratio,
            target.elapsed_time_ratio,
            target.initial_budget_scale,
            target.BCR,
            target.CPI,
            target.WR,
            target.rewards_prev_t_ratio,
        ], dtype=np.float32)


@dataclass
class HybridState(BaseStateRepresentation):
    """Hybrid state with CPM, absolute rewards, step index (9-dim)."""

    state_size: int = 9
    state_action_size: int = 10
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = True
    uses_traffic_share: bool = False

    def _build_state_from(self, target) -> np.ndarray:
        return np.asarray([
            target.rem_budget_ratio,
            target.elapsed_time_ratio,
            target.initial_budget_scale,
            target.t_step,
            target.ROL,
            target.BCR,
            target.CPM,
            target.WR,
            target.rewards_prev_t,
        ], dtype=np.float32)

    def compute_step_metrics(self, target=None) -> None:
        target = self if target is None else target
        self._compute_cpm(target)
        self._compute_common_ratios(target)

    def reset_step_fields(self, target=None) -> None:
        target = self if target is None else target
        target.CPM = 0


@dataclass
class DefaultState(BaseStateRepresentation):
    """Legacy / vanilla DRLB state (7-dim, absolute budget)."""

    state_size: int = 7
    state_action_size: int = 8
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False
    uses_traffic_share: bool = False

    def _build_state_from(self, target) -> np.ndarray:
        return np.asarray([
            target.t_step,
            target.rem_budget,
            target.ROL,
            target.BCR,
            target.CPM,
            target.WR,
            target.rewards_prev_t,
        ], dtype=np.float32)

    def compute_step_metrics(self, target=None) -> None:
        target = self if target is None else target
        self._compute_cpm(target)

    def reset_step_fields(self, target=None) -> None:
        target = self if target is None else target
        target.CPM = 0


@dataclass
class ImprovedTrafficShareState(CpiRatioState):
    """Improved state extended with elapsed traffic share (7-dim)."""

    state_size: int = 7
    state_action_size: int = 8
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = True
    uses_traffic_share: bool = True

    def _build_state_from(self, target) -> np.ndarray:
        return np.asarray([
            target.rem_budget_ratio,
            target.ROL_ratio,
            target.traffic_share,
            target.BCR,
            target.CPI,
            target.WR,
            target.rewards_prev_t_ratio,
        ], dtype=np.float32)


STATE_REPRESENTATIONS = {
    "improved": ImprovedState,
    "scaled_budget": ScaledBudgetState,
    "hybrid": HybridState,
    "default": DefaultState,
    "improved_traffic_share": ImprovedTrafficShareState,
}


def get_state_repr(state_type: str) -> BaseStateRepresentation:
    """Resolve a canonical ``state_type`` key as a fresh mutable instance."""
    return STATE_REPRESENTATIONS[state_type]()
