# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np
from functools import partial

from cachetools import LRUCache as LRU


from .model import *
from .replay_buffer import RTransition, ReplayBuffer, collate_reward_transitions
from .torch_device import resolve_training_device

import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_BUFFER_SIZE = int(1e5)  # replay buffer size
DEFAULT_BATCH_SIZE = 32         # minibatch size
DEFAULT_LR = 1e-3               # learning rate

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
        loss=None,
        scheduler_factory=None,
        grad_clip_norm=None,
        reward_clip_value=None,
        target_mode="monte_carlo_return",
        state_action_bucket_size=0.01,
        device=None,
    ):
        """Initialize an RewardNet object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        if device is None:
            self.device = resolve_training_device("cpu")
        elif isinstance(device, str):
            self.device = resolve_training_device(device)
        else:
            self.device = device
        self.state_action_size = state_action_size
        self.reward_size = reward_size
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.loss_type = loss_type
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.reward_clip_value = None if reward_clip_value is None else float(reward_clip_value)
        self.target_mode = target_mode
        self.state_action_bucket_size = state_action_bucket_size
        set_seed()

        # Reward-Network
        self.reward_net = Network(state_action_size, reward_size).to(self.device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr)
        self.criterion = loss if loss is not None else (
            nn.SmoothL1Loss() if self.loss_type == "smooth_l1" else nn.MSELoss()
        )
        self.scheduler = (
            scheduler_factory(self.optimizer)
            if scheduler_factory is not None
            else None
        )

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=self.batch_size,
            seed=0,
            collate_fn=partial(collate_reward_transitions, device=self.device),
        )
        # Reward dict - LRFU implementation not found, therefore just LRU
        self.M = LRU(self.buffer_size)
        self.S = []
        self.V = 0
        # Initialize loss for tracking the progress
        self.loss = 0

    def add_to_memory(self, state_action, reward):
        # Save experience in replay memory
        if self.reward_clip_value is not None:
            reward = np.clip(reward, -self.reward_clip_value, self.reward_clip_value)
        self.memory.add(
            RTransition(
                state_action=state_action,
                reward=np.asarray(reward, dtype=np.float32),
            )
        )

    def record_episode_step(self, state_action, immediate_reward):
        raw_state_action = np.asarray(state_action, dtype=np.float32).copy()
        reward = float(immediate_reward)
        self.S.append((raw_state_action, reward))
        self.V += reward

    def _state_action_key(self, state_action):
        state_action = np.asarray(state_action, dtype=np.float32)
        buckets = np.floor(state_action / self.state_action_bucket_size)
        return tuple(buckets.astype(np.int64).tolist())

    def flush_episode_targets(self):

        # update target rewards by strategy
        if self.target_mode == "monte_carlo_return":
            self.flush_monte_carlo_return()
        elif self.target_mode == "best_episode_return":
            self.flush_best_episode_return()

        # reset episode targets
        self.S = []
        self.V = 0

    def flush_monte_carlo_return(self):
        return_to_go = 0.0
        targets = []
        for state_action, immediate_reward in reversed(self.S):
            # in reverse order, to compute reward like r_T + r_{T-1} ... + r_{t}
            return_to_go += immediate_reward
            targets.append((state_action, return_to_go))

        for state_action, target in reversed(targets):
            self.add_to_memory(state_action, np.asarray([target], dtype=np.float32))

    def flush_best_episode_return(self):
        episode_return = self.V
        keyed_steps = [
            (state_action, self._state_action_key(state_action))
            for state_action, _ in self.S
        ]

        for _, key in keyed_steps:
            current_target = self.M.get(key, episode_return)
            self.M[key] = max(current_target, episode_return)

        for state_action, key in keyed_steps:
            self.add_to_memory(state_action, np.asarray([self.M[key]], dtype=np.float32))

    def step(self):
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state_action):
        """Returns actions for given state as per current policy.

            state (array_like): current state
        """
        sa = torch.from_numpy(state_action).float().unsqueeze(0).to(self.device)

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
        if self.scheduler is not None:
            self.scheduler.step()
        # Keep track of the loss for the history
        self.loss = loss.item()
