from __future__ import annotations

import random
from typing import Any

import numpy as np


SEED_OFFSETS = {
    "data_seed": 1000,
    "optuna_seed": 2000,
    "model_seed": 3000,
    "replay_buffer_seed": 4000,
    "train_seed": 5000,
    "eval_seed": 6000,
}


def derive_seed_map(master_seed: int) -> dict[str, int]:
    master_seed = int(master_seed)
    seed_map = {"master_seed": master_seed}
    for key, offset in SEED_OFFSETS.items():
        seed_map[key] = master_seed + offset
    return seed_map


def initialize_runtime_seeds(seed_map: dict[str, Any]) -> None:
    random.seed(int(seed_map["train_seed"]))
    np.random.seed(int(seed_map["train_seed"]))
    try:
        import torch

        torch.manual_seed(int(seed_map["model_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_map["model_seed"]))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            try:
                torch.mps.manual_seed(int(seed_map["model_seed"]))
            except AttributeError:
                pass
    except Exception:
        pass
