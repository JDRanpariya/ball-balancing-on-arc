"""Cross-platform device resolution for PyTorch models.

Centralizes the cuda -> mps -> cpu fallback so callers don't reinvent it.
"""
from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick the best available accelerator.

    Resolution order:
      1. If the caller passed an explicit device string ("cuda", "mps",
         "cpu", or a torch device), honour it if available; else fall back.
      2. CUDA if available.
      3. Apple Metal (MPS) if available (for M-series Macs without nvidia).
      4. CPU.
    """
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Caller asked for something explicit
    name = str(requested).lower()
    if name in ("cuda", "cuda:0") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("mps", "mps:0") and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    # "cuda" requested but not available - try mps next, then cpu
    if name in ("cuda", "cuda:0"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
