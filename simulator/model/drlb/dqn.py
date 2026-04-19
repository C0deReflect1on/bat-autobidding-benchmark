# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np
import random

from .model import *
from .replay_buffer import QTransition, ReplayBuffer, collate_q_transitions

import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_BUFFER_SIZE = int(1e5)  # replay buffer size
DEFAULT_BATCH_SIZE = 32         # minibatch size
DEFAULT_GAMMA = 1.0             # discount factor
DEFAULT_LR = 1e-4               # learning rate
DEFAULT_C = 100                 # how often to update the network
DEFAULT_SOFT_UPDATE_TAU = 0.0   # 0 disables Polyak averaging

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DQN():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        buffer_size=DEFAULT_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        gamma=DEFAULT_GAMMA,
        lr=DEFAULT_LR,
        target_update_interval=DEFAULT_C,
        soft_update_tau=DEFAULT_SOFT_UPDATE_TAU,
        loss_type="mse",
        grad_clip_norm=None,
        reward_clip_value=None,
    ):
        """Initialize an Agent object.

            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.target_update_interval = int(target_update_interval)
        self.soft_update_tau = float(soft_update_tau)
        self.loss_type = str(loss_type)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.reward_clip_value = None if reward_clip_value is None else float(reward_clip_value)
        set_seed()

        # Q-Network
        self.qnetwork_local = Network(state_size, action_size).to(device)
        self.qnetwork_target = Network(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss() if self.loss_type == "smooth_l1" else nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=self.batch_size,
            seed=0,
            collate_fn=collate_q_transitions,
        )
        # Track time step for updating Q_target every C = 100 steps
        self.t_step = 0
        self.loss = 0
    
    def step(self, state, action, reward, next_state, done=False):
        if self.reward_clip_value is not None:
            reward = float(np.clip(reward, -self.reward_clip_value, self.reward_clip_value))
        # Save experience in replay memory
        self.memory.add(
            QTransition(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
            )
        )
        self.t_step += 1
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, eps, eval_mode):
        """Returns actions for given state as per current policy.

            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)[0] # [0] 'cause otherwise nested array
        self.qnetwork_local.train()

        if not eval_mode:
            # Epsilon-greedy action selection
            # Check if the Q-value distribution is unimodal, if so:
            if self.unimodal_check(action_values) == True:
                if random.random() <= eps:
                    # choose action randomly with prob epsilon
                    return random.choice(np.arange(self.action_size))
                else: # and a regular action with 1-eps
                    return np.argmax(action_values.cpu().data.numpy())
            # If not unimodal, increase epsilon, if it's small
            else:
                prob = max(eps, 0.5)
                if random.random() <= prob:
                    return random.choice(np.arange(self.action_size))
                else: # and with 1-p choose an action regularly 
                    return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).max(1, keepdim=True)[0].detach()
        y = rewards + gamma * q_targets_next * (1 - dones)

        # Get Q values from local model
        Q_local = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_local, y)
        ###print("DQN loss = {}".format(loss))
        # Grad descent
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.loss = loss.item()
        # Prefer Polyak averaging when tau > 0, otherwise keep periodic hard copies.
        if self.soft_update_tau > 0:
            self._soft_update(self.qnetwork_local, self.qnetwork_target, self.soft_update_tau)
        elif ((self.t_step + 1) % self.target_update_interval) == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(local_param.data)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def unimodal_check(self, action_values):
        """
        This function checks if the array of action-values is unimodal using
        some heuristic tests.

        Borrowed from https://github.com/ostigg/dqn-rtb/blob/master/e_greedy_policy.py

        :param action_values: predicted Q values for each action (sorted by default)
        :return: boolean variable describing whether the distribution of values
        in the action-value array is unimodal or "abnormal".
        """
        end = len(action_values)
        i = 1
        if (torch.max(action_values) == action_values[0]) or (torch.max(action_values) == action_values[-1]):
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            return i == end
        else:
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            return i == end
