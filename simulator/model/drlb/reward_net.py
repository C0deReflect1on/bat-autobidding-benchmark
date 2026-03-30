# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np
import os
import random
from collections import namedtuple, deque, defaultdict

from cachetools import LRUCache as LRU


from .model import *

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
        set_seed()

        # Reward-Network
        self.reward_net = Network(state_action_size, reward_size).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, self.batch_size, 0)
        # Reward dict - LRFU implementation not found, therefore just LRU
        self.M = LRU(self.buffer_size)
        self.S = []
        self.V = 0
        # Initialize loss for tracking the progress
        self.loss = 0

    def add(self, state_action, reward):
        # Save experience in replay memory
        self.memory.add(state_action, reward)
    
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
        self.optimizer.step()
        # Keep track of the loss for the history
        self.loss = loss.item()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_action", "reward"])
        random.seed(seed)
    
    def add(self, state_action, reward):
        """Add a new experience to memory."""
        e = self.experience(state_action, reward)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        state_actions = torch.from_numpy(np.vstack([e.state_action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        return (state_actions, rewards)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
