from __future__ import annotations

import torch


def resolve_training_device(preference: str = "cpu") -> torch.device:
    """Map a string preference to a concrete ``torch.device``.

    - ``auto``: CUDA if available, else Apple MPS if available, else CPU.
    - ``cpu`` / ``cuda`` / ``mps``: require or use that backend when applicable.
    """
    pref = preference.lower().strip()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch_device is 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("torch_device is 'mps' but MPS is not available (requires Apple Silicon PyTorch).")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "torch_device must be one of 'auto', 'cpu', 'cuda', 'mps'; "
        f"got {preference!r}"
    )
