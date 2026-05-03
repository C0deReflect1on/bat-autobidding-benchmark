import numpy as np
import torch

from .config_types import DrlbConfig
from .dqn import DQN
from .reward_net import RewardNet
from .state_representations import get_state_repr


class RlBidAgent:
    """BAT-native DRLB controller: networks here, mutable BAT state in state_repr."""

    def __init__(self, config: DrlbConfig):
        self.config = config
        model_cfg = config.model
        dqn_cfg = config.dqn
        reward_cfg = config.reward_net

        self.state_type = model_cfg.state_type
        self.T = int(model_cfg.T)
        self.state_repr = get_state_repr(self.state_type)

        self.BETA = [float(beta) for beta in model_cfg.lambda_action_betas]
        self.lambda_min = model_cfg.lambda_min
        self.lambda_max = model_cfg.lambda_max
        self.eps_start = float(dqn_cfg.epsilon_start)
        self.eps_end = float(dqn_cfg.epsilon_end)
        self.anneal = float(dqn_cfg.epsilon_anneal)
        self.eps = self.eps_start
        self.ctl_lambda = 1.0 / 0.7
        self.dqn_action = max(0, len(self.BETA) // 2)
        self.rnet_r = 0.0
        self.global_T = 0
        self.step_memory = []
        self.episode_memory = []

        self.dqn_agent = DQN(
            state_size=self.state_repr.state_size,
            action_size=len(self.BETA),
            gamma=float(dqn_cfg.gamma),
            lr=float(dqn_cfg.lr),
            target_update_interval=int(dqn_cfg.target_update_interval),
            soft_update_tau=float(dqn_cfg.soft_update_tau),
            loss_type=dqn_cfg.loss_type,
            loss=dqn_cfg.loss,
            scheduler_factory=dqn_cfg.scheduler_factory,
            grad_clip_norm=dqn_cfg.grad_clip_norm,
            reward_clip_value=dqn_cfg.reward_clip_value,
        )
        self.reward_net = RewardNet(
            state_action_size=self.state_repr.state_action_size,
            reward_size=1,
            lr=float(reward_cfg.lr),
            loss_type=reward_cfg.loss_type,
            loss=reward_cfg.loss,
            scheduler_factory=reward_cfg.scheduler_factory,
            grad_clip_norm=reward_cfg.grad_clip_norm,
            reward_clip_value=reward_cfg.reward_clip_value,
            target_mode=reward_cfg.target_mode,
            state_action_bucket_size=reward_cfg.state_action_bucket_size,
        )
        self.state_repr.begin_episode(10000, total_steps=self.T)

    def epsilon_at_step(self, global_t=None):
        step = self.global_T if global_t is None else int(global_t)
        return max(self.eps_start - self.anneal * step, self.eps_end)

    def reset_episode(self):
        self.state_repr.begin_episode(10000, total_steps=self.T)
        # Keep epsilon decay continuous across campaigns instead of resetting
        # exploration to the cold-start value every new episode.
        self.eps = self.epsilon_at_step()
        self.ctl_lambda = 1.0 / 0.7
        self.dqn_action = max(0, len(self.BETA) // 2)
        self.rnet_r = 0.0
        self.reward_net.S = []
        self.reward_net.V = 0

    def configure_episode(self, budget, total_steps=None):
        self.state_repr.begin_episode(budget, total_steps=total_steps or self.T)

    def sync_runtime_context(
        self,
        balance,
        initial_budget,
        elapsed_time_ratio=None,
        initial_budget_scale=None,
        traffic_share=None,
    ):
        self.state_repr.sync_runtime_context(
            balance=balance,
            initial_budget=initial_budget,
            elapsed_time_ratio=elapsed_time_ratio,
            initial_budget_scale=initial_budget_scale,
            traffic_share=traffic_share,
        )

    def calc_bid(self, ctr, action_beta, available_budget=None):
        self.ctl_lambda = float(np.clip(self.ctl_lambda * (1 + action_beta), self.lambda_min, self.lambda_max))
        budget = self.state_repr.available_budget_for_step() if available_budget is None else available_budget
        bid_amt = ctr / self.ctl_lambda
        if bid_amt > budget:
            bid_amt = budget
        return max(0, bid_amt)

    def predict_reward(self, state_before_action, action_beta):
        state_action = np.append(state_before_action, action_beta).astype(np.float32)
        with torch.no_grad():
            return float(self.reward_net.act(state_action).squeeze().cpu().item())

    def learn_dqn_transition(self, state_before_action, action_idx, rnet_reward, state_after_outcome, done=False):
        self.dqn_agent.step(
            state_before_action,
            action_idx,
            rnet_reward,
            state_after_outcome,
            done=done,
        )
        self.dqn_action = int(action_idx)
        self.rnet_r = float(rnet_reward)
        self.global_T += 1
        self.eps = self.epsilon_at_step()
        self._record_step_history()

    def record_reward_net_step(self, state_before_action, action_beta, immediate_reward):
        state_action = np.append(state_before_action, action_beta).astype(np.float32)
        if self.reward_net.target_mode == "immediate_step":
            self.reward_net.add_to_memory(state_action, np.asarray([immediate_reward], dtype=np.float32))
            self.reward_net.step()
        else:
            self.reward_net.step()
            self.reward_net.record_episode_step(state_action, immediate_reward)

    def finish_episode(self):
        self.reward_net.flush_episode_targets()

    def act(self, obs=None, eval_mode=False):
        if obs is not None:
            self.state_repr.sync_runtime_context(
                balance=obs["balance"],
                initial_budget=obs["initialBalance"],
                elapsed_time_ratio=obs.get("elapsedTimeRatio"),
                initial_budget_scale=obs.get("initialBudgetScale"),
                traffic_share=obs.get("trafficShare"),
            )
        state_before_action = self.state_repr.curr_state.copy()
        action_idx = self.dqn_agent.act(state_before_action, eps=self.eps, eval_mode=eval_mode)
        action_beta = self.BETA[action_idx]
        self.dqn_action = int(action_idx)
        return self.calc_bid(obs["ctr"], action_beta) if obs is not None else action_idx

    def _record_step_history(self):
        self.step_memory.append([
            self.global_T,
            int(self.state_repr.rem_budget),
            self.ctl_lambda,
            self.eps,
            self.dqn_action,
            self.dqn_agent.loss,
            self.rnet_r,
            self.reward_net.loss,
        ])
