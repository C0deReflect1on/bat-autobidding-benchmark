import os
import pandas as pd
import numpy as np
import torch
from .dqn import DQN
from .reward_net import RewardNet
from .model import set_seed


class RlBidAgent:

    @staticmethod
    def _scale_budget(budget):
        return float(np.log1p(max(float(budget), 0.0)) / 10.0)

    def _is_improved_exp(self):
        return self.exp_type in ("improved_drlb", "improved_drlb_eval")

    def _is_scaled_budget_exp(self):
        return self.exp_type in ("scaled_budget", "scaled_budget_eval")

    def _is_hybrid_exp(self):
        return self.exp_type in ("improved_hybrid_drlb", "improved_hybrid_drlb_eval")

    def _is_hybrid_smooth_exp(self):
        return self.exp_type in ("improved_hybrid_drlb_smooth", "improved_hybrid_drlb_smooth_eval")

    def _is_hypgrid_v2_exp(self):
        return self.exp_type in ("hypgrid_v2_drlb", "hypgrid_v2_drlb_eval")

    def _is_hypgrid_v3_exp(self):
        return self.exp_type in ("hypgrid_v3_drlb", "hypgrid_v3_drlb_eval")

    def _load_config(self, params):
        """
        Load DRLB runtime settings.
        Priority: explicit params -> config.cfg -> hardcoded defaults.
        """
        params = params or {}

        default_exp_type = "improved_drlb_eval"
        default_T = 48
        default_bids_per_timestep = 5000

        cfg_exp_type = None
        cfg_T = None
        cfg_bids_per_timestep = None

        config_path = params.get("config_path")
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config.cfg")

        if os.path.exists(config_path):
            import configparser
            cfg = configparser.ConfigParser(allow_no_value=True)
            cfg.read(config_path)
            if "experiment_type" in cfg and "type" in cfg["experiment_type"]:
                cfg_exp_type = str(cfg["experiment_type"]["type"])
                if cfg_exp_type in cfg:
                    cfg_T = cfg[cfg_exp_type].get("T")
                    cfg_bids_per_timestep = cfg[cfg_exp_type].get("bids_per_timestep")

        self.exp_type = str(params.get("exp_type", cfg_exp_type or default_exp_type))
        self.T = int(params.get("T", cfg_T or default_T))
        self.bids_per_timestep = int(
            params.get("bids_per_timestep", cfg_bids_per_timestep or default_bids_per_timestep)
        )
        self.dqn_gamma = float(params.get("dqn_gamma", 1.0))
        self.dqn_lr = float(params.get("dqn_lr", 1e-4))
        self.dqn_target_update_interval = int(params.get("dqn_target_update_interval", 100))
        self.dqn_soft_update_tau = float(params.get("dqn_soft_update_tau", 0.0))
        self.dqn_loss_type = str(params.get("dqn_loss_type", "mse"))
        dqn_grad_clip = params.get("dqn_grad_clip_norm")
        self.dqn_grad_clip_norm = None if dqn_grad_clip is None else float(dqn_grad_clip)
        dqn_reward_clip = params.get("dqn_reward_clip_value")
        self.dqn_reward_clip_value = None if dqn_reward_clip is None else float(dqn_reward_clip)
        self.reward_net_lr = float(params.get("reward_net_lr", 1e-3))
        self.reward_net_loss_type = str(params.get("reward_net_loss_type", "mse"))
        reward_net_grad_clip = params.get("reward_net_grad_clip_norm")
        self.reward_net_grad_clip_norm = (
            None if reward_net_grad_clip is None else float(reward_net_grad_clip)
        )
        reward_net_clip = params.get("reward_net_reward_clip_value")
        self.reward_net_reward_clip_value = (
            None if reward_net_clip is None else float(reward_net_clip)
        )
    
    def __init__(self, params=None):
        self._load_config(params)
        # Beta parameter adjusting the lambda parameter, that regulates the agent's bid amount
        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        # Starting value of epsilon in the adaptive eps-greedy policy
        self.eps = 0.9
        # Parameter controlling the annealing speed of epsilon
        self.anneal = 2e-5
        if self._is_improved_exp():
            # Original-style DRLB state adapted for BAT.
            self.dqn_agent = DQN(
                state_size=6,
                action_size=7,
                gamma=self.dqn_gamma,
                lr=self.dqn_lr,
                target_update_interval=self.dqn_target_update_interval,
            )
            self.reward_net = RewardNet(
                state_action_size=7,
                reward_size=1,
                lr=self.reward_net_lr,
            )
        elif self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp():
            self.dqn_agent = DQN(
                state_size=9,
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
                state_action_size=10,
                reward_size=1,
                lr=self.reward_net_lr,
                loss_type=self.reward_net_loss_type,
                grad_clip_norm=self.reward_net_grad_clip_norm,
                reward_clip_value=self.reward_net_reward_clip_value,
            )
        elif self._is_scaled_budget_exp():
            # DQN Network to learn Q function
            self.dqn_agent = DQN(
                state_size=7,
                action_size=7,
                gamma=self.dqn_gamma,
                lr=self.dqn_lr,
                target_update_interval=self.dqn_target_update_interval,
            )
            # Reward Network to learn the reward function
            self.reward_net = RewardNet(
                state_action_size=8,
                reward_size=1,
                lr=self.reward_net_lr,
            )
        elif self._is_hybrid_exp():
            self.dqn_agent = DQN(
                state_size=9,
                action_size=7,
                gamma=self.dqn_gamma,
                lr=self.dqn_lr,
                target_update_interval=self.dqn_target_update_interval,
            )
            self.reward_net = RewardNet(
                state_action_size=10,
                reward_size=1,
                lr=self.reward_net_lr,
            )
        else:
            self.dqn_agent = DQN(
                state_size=7,
                action_size=7,
                gamma=self.dqn_gamma,
                lr=self.dqn_lr,
                target_update_interval=self.dqn_target_update_interval,
            )
            self.reward_net = RewardNet(
                state_action_size=8,
                reward_size=1,
                lr=self.reward_net_lr,
            )
        
      
        self.dqn_action = 3
        self.ctl_lambda = 1.0/0.7  
        
        # Arrays saving the training history
        self.step_memory = []
        self.episode_memory = []
        
        # Params for tracking the progress
        self.global_T = 0 # Tracking the global time step
        self.episode_budgets = None
        self.budget = 10000 
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
        """
        Returns the state that will be used as input in the DQN
        Based on the original code, state consists of:
        """
        if self._is_improved_exp():
            return np.asarray([
                self.rem_budget_ratio,  # 1. Remaining budget ratio
                self.ROL_ratio,         # 2. Remaining regulation opportunities ratio
                self.BCR,               # 3. Budget consumption rate
                self.CPI,               # 4. Cost per impression between t-1 and t
                self.WR,                # 5. Auction win rate at state t
                self.rewards_prev_t_ratio  # 6. Observed reward ratio at timestep t-1
            ], dtype=np.float32)
        if self._is_scaled_budget_exp():
            return np.asarray([
                self.rem_budget_ratio,  # 1. the ratio of the remaining budget to total available budget at time-step t
                self.elapsed_time_ratio,  # 2. Campaign progress ratio within its own lifetime
                self.initial_budget_scale,  # 3. Log-scaled initial budget for cross-campaign generalization
                self.BCR,               # 4. Budget consumption rate
                self.CPI,               # 5. Cost per impression between t-1 and t
                self.WR,                # 6. Auction win rate at state t
                self.rewards_prev_t_ratio  # 7. Ratio of acquired/total clicks at timestep t-1
            ], dtype=np.float32)
        if self._is_hybrid_exp() or self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp():
            return np.asarray([
                self.rem_budget_ratio,   # 1. Remaining budget ratio
                self.elapsed_time_ratio, # 2. Campaign progress ratio
                self.initial_budget_scale,  # 3. Log-scaled initial budget
                self.t_step,             # 4. Episode-local step index
                self.ROL,                # 5. Remaining regulation opportunities
                self.BCR,                # 6. Budget consumption rate
                self.CPM,                # 7. Cost signal aligned across train/inference
                self.WR,                 # 8. Auction win rate
                self.rewards_prev_t,     # 9. Previous-step reward
            ], dtype=np.float32)
        else:
            return np.asarray([
                self.t_step,            # 1. Current time step (0-47)
                self.rem_budget,        # 2. the remaining budget at time-step t
                self.ROL,               # 3. The number of Lambda regulation opportunities left
                self.BCR,               # 4. Budget consumption rate
                self.CPM,               # 5. Cost per mille of impressions between t-1 and t
                self.WR,                # 6. Auction win rate at state t
                self.rewards_prev_t     # 7. Clicks acquired at timestep t-1
            ], dtype=np.float32)

    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        # Reset the count of time steps
        self.t_step = 0
        
        # Next episode -> reset step
        self._reset_step()
        
        # Устанавливаем фиксированный бюджет 10000 для каждого эпизода
        self.budget = 10000
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.initial_budget_scale = self._scale_budget(self.budget)
        self.elapsed_time_ratio = 0
        self.episode_steps_total = max(self.T, 1)
        self.budget_spent_t = 0
        self.budget_spent_e = 0
        
        self.ctl_lambda = 1.0/0.7
        self.dqn_action = 3

        self.ROL = self.T  # The number of Lambda regulation opportunities left
        self.ROL_ratio = 1
        
        self.cur_time_step = 0.0  # Start from timeStepIndex 0.0
        self.bids_processed_in_current_timestep = 0
        
        # Reset episode statistics
        self.wins_e = 0
        self.rewards_e = 0
        
        # Reset RewardNet for new episode
        self.reward_net.V = 0
        self.reward_net.S = []

    def configure_episode(self, budget, total_steps=None):
        self.budget = max(1.0, float(budget))
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.initial_budget_scale = self._scale_budget(self.budget)
        self.elapsed_time_ratio = 0
        self.episode_steps_total = max(1, int(total_steps or self.T))
        self.ROL = self.episode_steps_total
        self.ROL_ratio = 1

    def _update_step(self):
        """
        Function that is called after processing 5000 bids for current timestep
        and before transitioning into next timestep
        """
        self.global_T += 1
        self.t_step += 1
        
        # Update budget and statistics
        self.prev_budget = self.rem_budget
        self.rem_budget = max(self.prev_budget - self.budget_spent_t, 0)
        self.budget_spent_e += self.budget_spent_t
        self.rewards_prev_t = self.reward_t
        self.ROL = max(self.ROL - 1, 0)
        
        # Calculate metrics for state
        self.BCR = 0 if self.prev_budget == 0 else -((self.rem_budget - self.prev_budget) / self.prev_budget)
        
        if self._is_improved_exp():
            self.CPI = 0 if self.wins_t == 0 else (self.cost_t / self.wins_t) / 300
            self.rewards_prev_t_ratio = 1 if self.possible_clicks_t == 0 else self.reward_t / self.possible_clicks_t
            self.ROL_ratio = max(self.ROL, 0) / max(self.episode_steps_total, 1)
            self.rem_budget_ratio = max(self.rem_budget, 0) / max(self.budget, 1)
        elif self._is_scaled_budget_exp():
            self.CPI = 0 if self.wins_t == 0 else (self.cost_t / self.wins_t) / 300
            self.rewards_prev_t_ratio = 1 if self.possible_clicks_t == 0 else self.reward_t / self.possible_clicks_t
            self.ROL_ratio = max(self.ROL, 0) / max(self.episode_steps_total, 1)
            self.rem_budget_ratio = max(self.rem_budget, 0) / max(self.budget, 1)
        elif self._is_hybrid_exp() or self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp():
            self.CPM = 0 if self.wins_t == 0 else ((self.cost_t / self.wins_t) * 1000)
            self.ROL_ratio = max(self.ROL, 0) / max(self.episode_steps_total, 1)
            self.rem_budget_ratio = max(self.rem_budget, 0) / max(self.budget, 1)
        else:
            self.CPM = 0 if self.wins_t == 0 else ((self.cost_t / self.wins_t) * 1000)
        
        self.WR = self.wins_t / max(self.imp_opps_t, 1)  # Avoid division by zero
        
        # Adaptive eps-greedy policy
        self.eps = max(0.95 - self.anneal * self.global_T, 0.05)

    def _reset_step(self):
        """
        Function to call every time a new time step is entered.
        """
        self.possible_clicks_t = 0
        self.total_rewards_t = 0
        self.reward_t = 0
        self.cost_t = 0
        self.wins_t = 0
        self.imp_opps_t = 0
        self.BCR = 0
        if self._is_improved_exp() or self._is_scaled_budget_exp():
            self.CPI = 0
        elif self._is_hybrid_exp() or self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp():
            self.CPM = 0
        else:
            self.CPM = 0
        self.WR = 0
        self.budget_spent_t = 0
        self.bids_processed_in_current_timestep = 0
    
    def _update_reward_cost(self, bid, reward, potential_reward, cost, win):
        """
        Internal function to update reward and cost for each bid
        """
        self.possible_clicks_t += potential_reward
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
        """
        Update DQN and RewardNet models after processing 5000 bids in current timestep
        """
        

        next_state = self._get_state()
        a_beta = self.dqn_agent.act(next_state, eps=self.eps, eval_mode=eval_mode)

        

        self.ctl_lambda *= (1 + self.BETA[a_beta])
        
        if not eval_mode:
            sa = np.append(self.cur_state, self.BETA[self.dqn_action]).astype(np.float32)
            true_reward = float(self.reward_t)

            if self._is_scaled_budget_exp() or self._is_hybrid_exp() or self._is_hybrid_smooth_exp():
                self.reward_net.add(sa, np.asarray([true_reward], dtype=np.float32))
                self.reward_net.step()

                if len(self.reward_net.memory) > 32:
                    with torch.no_grad():
                        self.rnet_r = float(self.reward_net.act(sa).squeeze().cpu().item())
                else:
                    self.rnet_r = true_reward
            elif self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp():
                with torch.no_grad():
                    self.rnet_r = float(self.reward_net.act(sa).squeeze().cpu().item())
                self.reward_net.add(sa, np.asarray([true_reward], dtype=np.float32))
                self.reward_net.step()
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
      
        current_time_step = obs['timeStepIndex']
        if (self._is_scaled_budget_exp() or self._is_hybrid_exp() or self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp()) and 'elapsedTimeRatio' in obs:
            self.elapsed_time_ratio = float(np.clip(obs['elapsedTimeRatio'], 0.0, 1.0))
        if (self._is_scaled_budget_exp() or self._is_hybrid_exp() or self._is_hybrid_smooth_exp() or self._is_hypgrid_v2_exp() or self._is_hypgrid_v3_exp()) and 'initialBudgetScale' in obs:
            self.initial_budget_scale = float(obs['initialBudgetScale'])
        
        # Check if we're starting a new timestep (после 5000 аукционов)
        if self.bids_processed_in_current_timestep >= self.bids_per_timestep:
            # Process model update for previous timestep
            self._update_step()
            self._model_upd(eval_mode, done=self._episode_done())
            
            # Save history and reset for new timestep
            self._record_step_history()
            
            self._reset_step()
            self.cur_time_step = current_time_step

        # Process current bid
        self.imp_opps_t += 1
        self.bids_processed_in_current_timestep += 1
        
        # Calculate bid using current lambda
        bid = self.calc_bid(obs['ctr'])

        # For vanilla_drlb, collect data for greedy algorithm
        if self.exp_type == 'vanilla_drlb':
            self.greedy_memory.append([
                obs['ctr'], 
                obs['leastWinningCost'], 
                obs['ctr'] / max(obs['leastWinningCost'], 1)
            ])

        return bid

    def calc_bid(self, ctr_value):
        """
        Calculate bid amount using the formula: bid = ctr / lambda
        """
        bid_amt = ctr_value / self.ctl_lambda
        
        # Ensure we don't exceed remaining budget
        curr_budget_left = self.rem_budget - self.budget_spent_t
        if bid_amt > curr_budget_left:
            bid_amt = curr_budget_left

        return max(0, bid_amt)  # Ensure non-negative bid


# def main():
#     import importlib
#     AuctionEmulatorEnv = importlib.import_module("auction_emulator_env_alibaba").AuctionEmulatorEnv
#     env = AuctionEmulatorEnv(data_file='period-7.csv', nrows=20000000)
    
#     set_seed()
#     agent = RlBidAgent()

   
#     epochs = 10
#     episodes_per_epoch = 60  

#     os.makedirs('models', exist_ok=True)

#     for epoch in range(epochs):
#         print(f"Epoch: {epoch+1}")
        
#         for episode in range(episodes_per_epoch):
#             print(f"  Episode: {episode+1}")
#             obs, done = env.reset()
            
#             # Убираем episode_budgets - он не нужен в таком виде
#             agent._reset_episode()
#             agent.cur_time_step = obs['timeStepIndex']
#             agent.cur_state = agent._get_state()

#             while not done:
#                 bid = agent.act(obs, eval_mode=False)
#                 next_obs, cur_reward, potential_reward, cur_cost, win, done = env.step(bid)
#                 agent._update_reward_cost(bid, cur_reward, potential_reward, cur_cost, win)
#                 obs = next_obs
            
#             # Final update after episode ends
#             print(f"  Episode Result: Budget={int(agent.budget)}, Spend={int(agent.budget_spent_e)}, "
#                   f"Impressions={agent.wins_e}, Clicks={agent.rewards_e}")
#             agent.episode_memory.append([
#                 epoch + 1, episode + 1, agent.budget, int(agent.budget_spent_e), 
#                 agent.wins_e, agent.rewards_e
#             ])


#         if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
#             PATH = f'models/model_state_epoch_{epoch+1}_max.tar'
            
    
#             torch.save({
#                 'local_q_model': agent.dqn_agent.qnetwork_local.state_dict(),
#                 'target_q_model': agent.dqn_agent.qnetwork_target.state_dict(),
#                 'q_optimizer': agent.dqn_agent.optimizer.state_dict(),
#                 'rnet': agent.reward_net.reward_net.state_dict(),
#                 'rnet_optimizer': agent.reward_net.optimizer.state_dict(),
#                 'agent_config': {  # 🔥 СОХРАНЯЕМ КОНФИГУРАЦИЮ АГЕНТА
#                     'ctl_lambda': agent.ctl_lambda,
#                     'BETA': agent.BETA,
#                     'T': agent.T,
#                     'bids_per_timestep': agent.bids_per_timestep,
#                     'exp_type': agent.exp_type
#                 },
#                 'epoch': epoch + 1,
#                 'total_rewards': agent.total_rewards,
#                 'total_wins': agent.total_wins
#             }, PATH)
            
#             print(f"💾 Model saved: {PATH}")


#             lambda_path = f'models/lambda_epoch_{epoch+1}.pkl'
#             with open(lambda_path, 'wb') as f:
#                 import pickle
#                 pickle.dump({
#                     'ctl_lambda': agent.ctl_lambda,
#                     'epoch': epoch + 1,
#                     'performance': {
#                         'clicks': agent.rewards_e,
#                         'impressions': agent.wins_e,
#                         'spend': agent.budget_spent_e
#                     }
#                 }, f)
            
#             print(f"📊 Lambda saved: {agent.ctl_lambda:.4f}")

#         # 🔥 Saving history
#         if ((epoch + 1) % 10) == 0:  
#             # Memory files
#             with open(f'models/rnet_memory_{epoch+1}.txt', "wb") as f:
#                 pickle.dump(agent.dqn_agent.memory, f)
#             with open(f'models/rdqn_memory_{epoch+1}.txt', "wb") as f:
#                 pickle.dump(agent.reward_net.memory, f)

#             # CSV files
#             if agent.step_memory:
#                 pd.DataFrame(agent.step_memory).to_csv(f'models/step_history_{epoch+1}.csv', 
#                                                      header=None, index=False)
#                 agent.step_memory = []
#             if agent.episode_memory:
#                 pd.DataFrame(agent.episode_memory).to_csv(f'models/episode_history_{epoch+1}.csv', 
#                                                         header=None, index=False)
#                 agent.episode_memory = []

#         print("EPOCH ENDED")

   
#     final_path = 'models/model_state_final.tar'
#     torch.save({
#         'local_q_model': agent.dqn_agent.qnetwork_local.state_dict(),
#         'target_q_model': agent.dqn_agent.qnetwork_target.state_dict(),
#         'q_optimizer': agent.dqn_agent.optimizer.state_dict(),
#         'rnet': agent.reward_net.reward_net.state_dict(),
#         'rnet_optimizer': agent.reward_net.optimizer.state_dict(),
#         'agent_config': {
#             'ctl_lambda': agent.ctl_lambda,
#             'BETA': agent.BETA,
#             'T': agent.T,
#             'bids_per_timestep': agent.bids_per_timestep,
#             'exp_type': agent.exp_type
#         },
#         'epoch': epochs,
#         'total_rewards': agent.total_rewards,
#         'total_wins': agent.total_wins
#     }, final_path)
    
#     print(f"🎯 Final model saved: {final_path}")
#     print(f"📊 Final lambda persuaded: {agent.ctl_lambda:.4f}")

#     env.close()

# if __name__ == "__main__":
#     main()