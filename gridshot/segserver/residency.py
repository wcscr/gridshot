"""Pure helpers for bounded unified-memory residency.

Keep environment parsing and retained-size estimation independent of PyTorch so
the service can report configuration errors before loading any model runtime.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class ResidencyConfigurationError(ValueError):
    """A memory-residency environment setting is invalid."""


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_flag(
    name: str,
    *,
    default: bool,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read a strict boolean environment flag."""

    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ResidencyConfigurationError(
        f"invalid {name}={raw!r}; expected 1/0, true/false, yes/no, or on/off"
    )


def env_int(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Read an integer environment setting with explicit safe bounds."""

    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ResidencyConfigurationError(
            f"invalid {name}={raw!r}; expected an integer"
        ) from exc
    if not minimum <= value <= maximum:
        raise ResidencyConfigurationError(
            f"invalid {name}={raw!r}; expected {minimum}..{maximum}"
        )
    return value


def optional_residency_mode(
    device: str,
    *,
    value: str | None = None,
) -> str:
    """Resolve optional-model residency; Metal defaults to one optional lane."""

    raw = os.environ.get("GRIDSHOT_OPTIONAL_MODEL_RESIDENCY", "auto")
    normalized = (raw if value is None else value).strip().lower()
    if normalized == "auto":
        return "single" if device == "mps" else "multi"
    if normalized in {"single", "multi"}:
        return normalized
    raise ResidencyConfigurationError(
        "invalid GRIDSHOT_OPTIONAL_MODEL_RESIDENCY="
        f"{normalized!r}; expected auto, single, or multi"
    )


def retained_bytes(value: Any, _seen: set[int] | None = None) -> int:
    """Estimate bytes retained by tensors, arrays, images, and containers.

    This intentionally estimates the decoded PIL pixel buffer, not the source
    JPEG/PNG size: decoded images and accelerator tensors are what the embedding
    cache keeps alive.
    """

    if value is None or isinstance(value, (bool, int, float, complex)):
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)

    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    # Torch tensors expose both methods. Keep this duck-typed so importing this
    # module never initializes PyTorch or an accelerator backend.
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except (TypeError, ValueError, RuntimeError):
            pass

    # NumPy arrays expose an integer nbytes property.
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    # PIL images expose size=(width, height) and getbands().
    size = getattr(value, "size", None)
    getbands = getattr(value, "getbands", None)
    if (
        isinstance(size, tuple)
        and len(size) == 2
        and callable(getbands)
        and all(isinstance(part, int) for part in size)
    ):
        try:
            return int(size[0]) * int(size[1]) * max(1, len(getbands()))
        except (TypeError, ValueError):
            pass

    if isinstance(value, Mapping):
        return sum(retained_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(retained_bytes(item, seen) for item in value)
    return 0
