from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Callable, Generic, Sequence, TypeVar

import numpy as np
import torch


device = torch.device("cpu")

TTransition = TypeVar("TTransition")
TSample = TypeVar("TSample")


@dataclass(frozen=True)
class QTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class RTransition:
    state_action: np.ndarray
    reward: np.ndarray | float


class ReplayBuffer(Generic[TTransition, TSample]):
    """Fixed-size replay buffer with pluggable collation."""

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seed: int,
        collate_fn: Callable[[Sequence[TTransition]], TSample],
    ):
        self.memory: deque[TTransition] = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self._random = random.Random(seed)
        self._collate_fn = collate_fn

    def add(self, item: TTransition) -> None:
        self.memory.append(item)

    def sample(self) -> TSample:
        transitions = self._random.sample(self.memory, k=self.batch_size)
        return self._collate_fn(transitions)

    def __len__(self) -> int:
        return len(self.memory)


def collate_q_transitions(transitions: Sequence[QTransition]):
    states = torch.from_numpy(np.vstack([t.state for t in transitions])).float().to(device)
    actions = torch.from_numpy(np.vstack([t.action for t in transitions])).long().to(device)
    rewards = torch.from_numpy(np.vstack([t.reward for t in transitions])).float().to(device)
    next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions])).float().to(device)
    dones = torch.from_numpy(
        np.vstack([t.done for t in transitions]).astype(np.float32)
    ).float().to(device)
    return states, actions, rewards, next_states, dones


def collate_reward_transitions(transitions: Sequence[RTransition]):
    state_actions = torch.from_numpy(np.vstack([t.state_action for t in transitions])).float().to(device)
    rewards = torch.from_numpy(
        np.vstack([np.asarray(t.reward, dtype=np.float32) for t in transitions])
    ).float().to(device)
    return state_actions, rewards
