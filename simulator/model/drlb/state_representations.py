"""
DRLB state representation variants.

Each class defines how the DQN state vector is built, what per-step metrics
are computed, and in which order the reward network is updated.  Adding a new
state variant means adding one class here and one entry in the lookup dicts at
the bottom of this file -- no other files need to change.
"""
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shared state representation base classes
# ---------------------------------------------------------------------------

class BaseStateRepresentation:
    state_size: int = 0
    state_action_size: int = 0
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False

    def get_state(self, agent) -> np.ndarray:
        raise NotImplementedError

    def _compute_common_ratios(self, agent) -> None:
        agent.ROL_ratio = max(agent.ROL, 0) / max(agent.episode_steps_total, 1)
        agent.rem_budget_ratio = max(agent.rem_budget, 0) / max(agent.budget, 1)

    def _compute_cpi(self, agent) -> None:
        agent.CPI = 0 if agent.wins_t == 0 else (agent.cost_t / agent.wins_t) / 300

    def _compute_cpm(self, agent) -> None:
        agent.CPM = 0 if agent.wins_t == 0 else (agent.cost_t / agent.wins_t) * 1000

    def _compute_reward_density(self, agent) -> None:
        # Use reward density per bid opportunity instead of a synthetic
        # "potential reward" proxy that can be degenerate.
        agent.rewards_prev_t_ratio = agent.reward_t / max(agent.imp_opps_t, 1)

    def compute_step_metrics(self, agent) -> None:
        self._compute_common_ratios(agent)

    def reset_step_fields(self, agent) -> None:
        return


class CpiRatioState(BaseStateRepresentation):
    def compute_step_metrics(self, agent) -> None:
        self._compute_cpi(agent)
        self._compute_reward_density(agent)
        self._compute_common_ratios(agent)

    def reset_step_fields(self, agent) -> None:
        agent.CPI = 0


# ---------------------------------------------------------------------------
# State representation classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImprovedState(CpiRatioState):
    """Original-style DRLB state adapted for BAT (6-dim)."""

    state_size: int = 6
    state_action_size: int = 7
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False

    def get_state(self, agent) -> np.ndarray:
        return np.asarray([
            agent.rem_budget_ratio,
            agent.ROL_ratio,
            agent.BCR,
            agent.CPI,
            agent.WR,
            agent.rewards_prev_t_ratio,
        ], dtype=np.float32)



@dataclass(frozen=True)
class ScaledBudgetState(CpiRatioState):
    """Adds elapsed-time ratio and log-scaled budget to improved state (7-dim)."""

    state_size: int = 7
    state_action_size: int = 8
    reward_net_order: str = "learn_first"
    uses_campaign_meta: bool = True

    def get_state(self, agent) -> np.ndarray:
        return np.asarray([
            agent.rem_budget_ratio,
            agent.elapsed_time_ratio,
            agent.initial_budget_scale,
            agent.BCR,
            agent.CPI,
            agent.WR,
            agent.rewards_prev_t_ratio,
        ], dtype=np.float32)



@dataclass(frozen=True)
class HybridState(BaseStateRepresentation):
    """
    Hybrid state with CPM, absolute rewards, step index (9-dim).

    Covers: hybrid, hybrid_smooth, hypgrid_v2, hypgrid_v3 experiment families.
    The only behavioural difference between smooth/hypgrid variants is the
    reward-net update order, which is set via the constructor default or
    overridden when instantiating.
    """

    state_size: int = 9
    state_action_size: int = 10
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = True

    def get_state(self, agent) -> np.ndarray:
        return np.asarray([
            agent.rem_budget_ratio,
            agent.elapsed_time_ratio,
            agent.initial_budget_scale,
            agent.t_step,
            agent.ROL,
            agent.BCR,
            agent.CPM,
            agent.WR,
            agent.rewards_prev_t,
        ], dtype=np.float32)

    def compute_step_metrics(self, agent) -> None:
        self._compute_cpm(agent)
        self._compute_common_ratios(agent)

    def reset_step_fields(self, agent) -> None:
        agent.CPM = 0


@dataclass(frozen=True)
class DefaultState(BaseStateRepresentation):
    """Legacy / vanilla DRLB state (7-dim, absolute budget)."""

    state_size: int = 7
    state_action_size: int = 8
    reward_net_order: str = "predict_first"
    uses_campaign_meta: bool = False

    def get_state(self, agent) -> np.ndarray:
        return np.asarray([
            agent.t_step,
            agent.rem_budget,
            agent.ROL,
            agent.BCR,
            agent.CPM,
            agent.WR,
            agent.rewards_prev_t,
        ], dtype=np.float32)

    def compute_step_metrics(self, agent) -> None:
        self._compute_cpm(agent)

    def reset_step_fields(self, agent) -> None:
        agent.CPM = 0


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

STATE_REPRESENTATIONS: dict[str, BaseStateRepresentation] = {
    "improved": ImprovedState(),
    "scaled_budget": ScaledBudgetState(),
    "hybrid": HybridState(reward_net_order="learn_first"),
    "hybrid_smooth": HybridState(reward_net_order="learn_first"),
    "hypgrid_v2": HybridState(reward_net_order="predict_first"),
    "hypgrid_v3": HybridState(reward_net_order="predict_first"),
    "default": DefaultState(),
}

EXP_TYPE_TO_STATE_FAMILY: dict[str, str] = {
    "improved_drlb": "improved",
    "improved_drlb_eval": "improved",
    "scaled_budget": "scaled_budget",
    "scaled_budget_eval": "scaled_budget",
    "improved_hybrid_drlb": "hybrid",
    "improved_hybrid_drlb_eval": "hybrid",
    "improved_hybrid_drlb_smooth": "hybrid_smooth",
    "improved_hybrid_drlb_smooth_eval": "hybrid_smooth",
    "hypgrid_v2_drlb": "hypgrid_v2",
    "hypgrid_v2_drlb_eval": "hypgrid_v2",
    "hypgrid_v3_drlb": "hypgrid_v3",
    "hypgrid_v3_drlb_eval": "hypgrid_v3",
}


def get_state_repr(exp_type: str) -> BaseStateRepresentation:
    """Resolve an exp_type string to a StateRepresentation instance."""
    family = EXP_TYPE_TO_STATE_FAMILY.get(exp_type, "default")
    return STATE_REPRESENTATIONS[family]
