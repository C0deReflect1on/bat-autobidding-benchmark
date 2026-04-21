import numpy as np
import torch

from .config_types import DrlbConfig
from .dqn import DQN
from .reward_net import RewardNet
from .state_representations import get_state_repr


class RlBidAgent:

    @staticmethod
    def _scale_budget(budget):
        return float(np.log1p(max(float(budget), 0.0)) / 10.0)

    def __init__(self, config: DrlbConfig):
        self.config = config
        self.exp_type = config.model.exp_type
        self.T = int(config.model.T)
        self.bids_per_timestep = int(config.model.bids_per_timestep)
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

        self.state_repr = get_state_repr(self.exp_type)

        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.eps = 0.9
        self.anneal = 2e-5

        self.dqn_agent = DQN(
            state_size=self.state_repr.state_size,
            action_size=7,
            gamma=self.dqn_gamma,
            lr=self.dqn_lr,
            target_update_interval=self.dqn_target_update_interval,
            soft_update_tau=self.dqn_soft_update_tau,
            loss_type=self.dqn_loss_type,
            grad_clip_norm=self.dqn_grad_clip_norm,
            reward_clip_value=self.dqn_reward_clip_value,
        )
        self.reward_net = RewardNet(
            state_action_size=self.state_repr.state_action_size,
            reward_size=1,
            lr=self.reward_net_lr,
            loss_type=self.reward_net_loss_type,
            grad_clip_norm=self.reward_net_grad_clip_norm,
            reward_clip_value=self.reward_net_reward_clip_value,
        )

        self.dqn_action = 3
        self.ctl_lambda = 1.0 / 0.7

        self.step_memory = []
        self.episode_memory = []

        self.global_T = 0
        self.episode_budgets = None
        self.budget = 10000
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.total_wins = 0
        self.total_rewards = 0
        self.rewards_prev_t = 0
        self.rewards_prev_t_ratio = 0
        self.rnet_r = 0
        self.wins_e = 0
        self.rewards_e = 0
        self.elapsed_time_ratio = 0
        self.initial_budget_scale = self._scale_budget(self.budget)
        self.episode_steps_total = max(self.T, 1)
        self.ROL = self.T
        self.ROL_ratio = 1

    def _get_state(self):
        return self.state_repr.get_state(self)

    def sync_runtime_context(
        self,
        balance,
        initial_budget,
        elapsed_time_ratio=None,
        initial_budget_scale=None,
    ):
        self.budget = max(1.0, float(initial_budget))
        self.rem_budget = max(0.0, float(balance))
        self.rem_budget_ratio = self.rem_budget / max(self.budget, 1e-9)
        self.initial_budget_scale = self._scale_budget(self.budget)

        if elapsed_time_ratio is not None:
            self.elapsed_time_ratio = float(np.clip(elapsed_time_ratio, 0.0, 1.0))
        if initial_budget_scale is not None:
            self.initial_budget_scale = float(initial_budget_scale)

    def _reset_episode(self):
        self.t_step = 0
        self._reset_step()

        self.budget = 10000
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.initial_budget_scale = self._scale_budget(self.budget)
        self.elapsed_time_ratio = 0
        self.episode_steps_total = max(self.T, 1)
        self.budget_spent_t = 0
        self.budget_spent_e = 0

        self.ctl_lambda = 1.0 / 0.7
        self.dqn_action = 3

        self.ROL = self.T
        self.ROL_ratio = 1

        self.bids_processed_in_current_timestep = 0

        self.wins_e = 0
        self.rewards_e = 0

        self.reward_net.V = 0
        self.reward_net.S = []

    def configure_episode(self, budget, total_steps=None):
        self.sync_runtime_context(
            balance=float(budget),
            initial_budget=float(budget),
        )
        self.elapsed_time_ratio = 0
        self.episode_steps_total = max(1, int(total_steps or self.T))
        self.ROL = self.episode_steps_total
        self.ROL_ratio = 1

    def _update_step(self):
        self.global_T += 1
        self.t_step += 1

        self.prev_budget = self.rem_budget
        self.rem_budget = max(self.prev_budget - self.budget_spent_t, 0)
        self.budget_spent_e += self.budget_spent_t
        self.rewards_prev_t = self.reward_t
        self.ROL = max(self.ROL - 1, 0)

        self.BCR = 0 if self.prev_budget == 0 else -((self.rem_budget - self.prev_budget) / self.prev_budget)

        self.state_repr.compute_step_metrics(self)

        self.WR = self.wins_t / max(self.imp_opps_t, 1)

        self.eps = max(0.95 - self.anneal * self.global_T, 0.05)

    def _reset_step(self):
        self.possible_clicks_t = 0
        self.total_rewards_t = 0
        self.reward_t = 0
        self.cost_t = 0
        self.wins_t = 0
        self.imp_opps_t = 0
        self.BCR = 0
        self.WR = 0
        self.budget_spent_t = 0
        self.bids_processed_in_current_timestep = 0
        self.state_repr.reset_step_fields(self)

    def _update_reward_cost(self, reward, cost, win):
        if win:
            self.budget_spent_t += cost
            self.wins_t += 1
            self.wins_e += 1
            self.total_wins += 1
            self.reward_t += reward
            self.rewards_e += reward
            self.total_rewards += reward
            self.cost_t += cost

    def _episode_done(self):
        return self.t_step >= max(1, self.episode_steps_total) or self.rem_budget <= 0

    def _record_step_history(self):
        self.step_memory.append([
            self.global_T, int(self.rem_budget), self.ctl_lambda,
            self.eps, self.dqn_action, self.dqn_agent.loss,
            self.rnet_r, self.reward_net.loss
        ])

    def _model_upd(self, eval_mode, done=False):
        next_state = self._get_state()
        a_beta = self.dqn_agent.act(next_state, eps=self.eps, eval_mode=eval_mode)

        self.ctl_lambda *= (1 + self.BETA[a_beta])

        if not eval_mode:
            sa = np.append(self.cur_state, self.BETA[self.dqn_action]).astype(np.float32)
            true_reward = float(self.reward_t)

            if self.state_repr.reward_net_order == "learn_first":
                self.reward_net.add(sa, np.asarray([true_reward], dtype=np.float32))
                self.reward_net.step()
                if len(self.reward_net.memory) > 32:
                    with torch.no_grad():
                        self.rnet_r = float(self.reward_net.act(sa).squeeze().cpu().item())
                else:
                    self.rnet_r = true_reward
            else:
                with torch.no_grad():
                    self.rnet_r = float(self.reward_net.act(sa).squeeze().cpu().item())
                self.reward_net.add(sa, np.asarray([true_reward], dtype=np.float32))
                self.reward_net.step()

            self.dqn_agent.step(
                self.cur_state,
                self.dqn_action,
                self.rnet_r,
                next_state,
                done=done,
            )

        self.cur_state = next_state
        self.dqn_action = a_beta

    def finalize_episode(self, eval_mode):
        if self.bids_processed_in_current_timestep <= 0:
            return

        self._update_step()
        self._model_upd(eval_mode, done=True)
        self._record_step_history()
        self._reset_step()

    def act(self, obs, eval_mode):
        if "balance" in obs and "initialBalance" in obs:
            self.sync_runtime_context(
                balance=obs["balance"],
                initial_budget=obs["initialBalance"],
                elapsed_time_ratio=obs.get("elapsedTimeRatio"),
                initial_budget_scale=obs.get("initialBudgetScale"),
            )
        elif self.state_repr.uses_campaign_meta:
            if 'elapsedTimeRatio' in obs:
                self.elapsed_time_ratio = float(np.clip(obs['elapsedTimeRatio'], 0.0, 1.0))
            if 'initialBudgetScale' in obs:
                self.initial_budget_scale = float(obs['initialBudgetScale'])

        if self.bids_processed_in_current_timestep >= self.bids_per_timestep:
            self._update_step()
            self._model_upd(eval_mode, done=self._episode_done())
            self._record_step_history()
            self._reset_step()

        self.imp_opps_t += 1
        self.bids_processed_in_current_timestep += 1

        bid = self.calc_bid(obs['ctr'])
        return bid

    def calc_bid(self, ctr_value):
        bid_amt = ctr_value / self.ctl_lambda

        curr_budget_left = self.rem_budget - self.budget_spent_t
        if bid_amt > curr_budget_left:
            bid_amt = curr_budget_left

        return max(0, bid_amt)
