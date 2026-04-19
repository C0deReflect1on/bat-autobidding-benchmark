# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np

from cachetools import LRUCache as LRU


from .model import *
from .replay_buffer import RTransition, ReplayBuffer, collate_reward_transitions

import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_BUFFER_SIZE = int(1e5)  # replay buffer size
DEFAULT_BATCH_SIZE = 32         # minibatch size
DEFAULT_LR = 1e-3               # learning rate

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class RewardNet():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_action_size,
        reward_size,
        buffer_size=DEFAULT_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        loss_type="mse",
        grad_clip_norm=None,
        reward_clip_value=None,
    ):
        """Initialize an RewardNet object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_action_size = state_action_size
        self.reward_size = reward_size
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.loss_type = str(loss_type)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.reward_clip_value = None if reward_clip_value is None else float(reward_clip_value)
        set_seed()

        # Reward-Network
        self.reward_net = Network(state_action_size, reward_size).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss() if self.loss_type == "smooth_l1" else nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=self.batch_size,
            seed=0,
            collate_fn=collate_reward_transitions,
        )
        # Reward dict - LRFU implementation not found, therefore just LRU
        self.M = LRU(self.buffer_size)
        self.S = []
        self.V = 0
        # Initialize loss for tracking the progress
        self.loss = 0

    def add(self, state_action, reward):
        # Save experience in replay memory
        if self.reward_clip_value is not None:
            reward = np.clip(reward, -self.reward_clip_value, self.reward_clip_value)
        self.memory.add(
            RTransition(
                state_action=state_action,
                reward=np.asarray(reward, dtype=np.float32),
            )
        )
    
    def add_to_M(self, sa, reward):
        # Add records to the reward dict
        self.M[sa] = reward
        if len(self.M) >= self.buffer_size:
            del self.M[self.M.peek_last_item()[0]] # discard LRU key

    def get_from_M(self, sa):
        # Retrieve items from M
        return(self.M.get(sa, 0))

    def step(self):
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state_action):
        """Returns actions for given state as per current policy.

            state (array_like): current state
        """
        sa = torch.from_numpy(state_action).float().unsqueeze(0).to(device)

        return(self.reward_net(sa))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

            experiences (Tuple[torch.Tensor]): tuple of (sa, r) tuples 
        """
        state_actions, rewards = experiences

        # Get expected Reward values
        R_pred = self.reward_net(state_actions)

        # Compute loss
        loss = self.criterion(R_pred, rewards)
        # Grad descent
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        # Keep track of the loss for the history
        self.loss = loss.item()
