"""Accelerator selection and diagnostics for segmentation inference.

The inference API is intentionally backend-neutral.  This module keeps the
CUDA, Metal/MPS, and CPU policy in one place so model lanes cannot silently
choose different devices or precision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


ACCELERATOR_ENV = "GRIDSHOT_ACCELERATOR"
DTYPE_ENV = "GRIDSHOT_DTYPE"
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"

_ACCELERATORS = {"auto", "cuda", "mps", "cpu"}
_DTYPES = {"auto", "float32", "float16", "bfloat16"}


class AcceleratorConfigurationError(RuntimeError):
    """The requested inference accelerator or dtype cannot be used."""


@dataclass(frozen=True)
class AcceleratorPolicy:
    """Resolved device and precision for a model lane."""

    device: str
    dtype: Any
    dtype_name: str
    requested_accelerator: str
    requested_dtype: str


def _cuda_available(torch_module) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    probe = getattr(cuda, "is_available", None)
    return bool(probe and probe())


def _mps_backend(torch_module):
    return getattr(getattr(torch_module, "backends", None), "mps", None)


def _mps_built(torch_module) -> bool:
    probe = getattr(_mps_backend(torch_module), "is_built", None)
    return bool(probe and probe())


def _mps_available(torch_module) -> bool:
    probe = getattr(_mps_backend(torch_module), "is_available", None)
    return bool(probe and probe())


def _setting(value: str | None, env_name: str, default: str, allowed: set[str]) -> str:
    selected = (
        value if value is not None else os.environ.get(env_name, default)
    ).strip().lower()
    if selected not in allowed:
        choices = ", ".join(sorted(allowed))
        raise AcceleratorConfigurationError(
            f"invalid {env_name}={selected!r}; choose one of: {choices}"
        )
    return selected


def resolve_policy(
    torch_module,
    *,
    accelerator: str | None = None,
    dtype: str | None = None,
) -> AcceleratorPolicy:
    """Resolve an explicit, fail-closed inference policy.

    ``auto`` preserves the existing CUDA-first behavior while adding MPS ahead
    of the development-only CPU fallback.  An explicitly requested accelerator
    must be available; it never degrades silently to another device.
    """

    requested_accelerator = _setting(
        accelerator, ACCELERATOR_ENV, "auto", _ACCELERATORS
    )
    requested_dtype = _setting(dtype, DTYPE_ENV, "auto", _DTYPES)

    if requested_accelerator == "auto":
        if _cuda_available(torch_module):
            device = "cuda"
        elif _mps_available(torch_module):
            device = "mps"
        else:
            device = "cpu"
    elif requested_accelerator == "cuda":
        if not _cuda_available(torch_module):
            raise AcceleratorConfigurationError(
                "GRIDSHOT_ACCELERATOR=cuda requested, but CUDA is unavailable"
            )
        device = "cuda"
    elif requested_accelerator == "mps":
        if not _mps_built(torch_module):
            raise AcceleratorConfigurationError(
                "GRIDSHOT_ACCELERATOR=mps requested, but this PyTorch build has no MPS support"
            )
        if not _mps_available(torch_module):
            raise AcceleratorConfigurationError(
                "GRIDSHOT_ACCELERATOR=mps requested, but MPS is unavailable on this host"
            )
        device = "mps"
    else:
        device = "cpu"

    if requested_dtype == "auto":
        # Match the proven CUDA configuration.  Begin MPS in FP32 until each
        # model lane passes numerical and physical-output parity checks.
        dtype_name = "bfloat16" if device == "cuda" else "float32"
    else:
        dtype_name = requested_dtype

    return AcceleratorPolicy(
        device=device,
        dtype=getattr(torch_module, dtype_name),
        dtype_name=dtype_name,
        requested_accelerator=requested_accelerator,
        requested_dtype=requested_dtype,
    )


def _safe_counter(namespace, name: str) -> int | None:
    counter = getattr(namespace, name, None)
    if counter is None:
        return None
    try:
        return int(counter())
    except (RuntimeError, TypeError, ValueError):
        return None


def accelerator_report(torch_module=None) -> dict:
    """Return device configuration and MPS telemetry without loading a model."""

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as exc:  # pragma: no cover - exercised on minimal installs
            return {
                "status": "unavailable",
                "error": f"PyTorch unavailable: {str(exc)[:200]}",
                "requested_accelerator": os.environ.get(ACCELERATOR_ENV, "auto"),
                "requested_dtype": os.environ.get(DTYPE_ENV, "auto"),
            }

    report = {
        "status": "ok",
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "requested_accelerator": os.environ.get(ACCELERATOR_ENV, "auto").lower(),
        "requested_dtype": os.environ.get(DTYPE_ENV, "auto").lower(),
        "available": {
            "cuda": _cuda_available(torch_module),
            "mps_built": _mps_built(torch_module),
            "mps": _mps_available(torch_module),
            "cpu": True,
        },
        "mps_cpu_fallback_enabled": os.environ.get(MPS_FALLBACK_ENV, "0") == "1",
    }
    try:
        policy = resolve_policy(torch_module)
    except AcceleratorConfigurationError as exc:
        report.update({"status": "error", "selected": None, "error": str(exc)})
    else:
        report.update(
            {
                "selected": policy.device,
                "dtype": policy.dtype_name,
                "error": None,
            }
        )

    if report["available"]["mps"]:
        mps = getattr(torch_module, "mps", None)
        report["mps_memory"] = {
            "current_allocated_bytes": _safe_counter(mps, "current_allocated_memory"),
            "driver_allocated_bytes": _safe_counter(mps, "driver_allocated_memory"),
            "recommended_max_bytes": _safe_counter(mps, "recommended_max_memory"),
        }
    else:
        report["mps_memory"] = None
    return report
