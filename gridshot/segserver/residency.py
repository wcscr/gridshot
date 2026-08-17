"""Pure helpers for automatic Metal unified-memory residency."""

from __future__ import annotations


def optional_residency_mode(device: str) -> str:
    """Keep one optional model on Metal and preserve CUDA's multi-model behavior."""

    return "single" if device == "mps" else "multi"


def optional_model_idle_seconds(device: str) -> int:
    """Apply automatic idle eviction only to Metal unified memory."""

    return 300 if device == "mps" else 0
