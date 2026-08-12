from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gridshot.segserver.backend import (
    AcceleratorConfigurationError,
    accelerator_report,
    resolve_policy,
)


class _BackendFlag:
    def __init__(self, *, available: bool, built: bool | None = None):
        self.available = available
        self.built = available if built is None else built

    def is_available(self) -> bool:
        return self.available

    def is_built(self) -> bool:
        return self.built


class _MPSMemory:
    @staticmethod
    def current_allocated_memory() -> int:
        return 10

    @staticmethod
    def driver_allocated_memory() -> int:
        return 20

    @staticmethod
    def recommended_max_memory() -> int:
        return 30


def _torch(*, cuda: bool = False, mps: bool = False, mps_built: bool | None = None):
    return SimpleNamespace(
        __version__="test",
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=_BackendFlag(available=cuda),
        backends=SimpleNamespace(
            mps=_BackendFlag(available=mps, built=mps_built)
        ),
        mps=_MPSMemory(),
    )


class ResolvePolicyTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_keeps_existing_bfloat16_default(self):
        policy = resolve_policy(_torch(cuda=True, mps=True))
        self.assertEqual(policy.device, "cuda")
        self.assertEqual(policy.dtype_name, "bfloat16")
        self.assertEqual(policy.dtype, "bf16")

    def test_auto_selects_mps_before_cpu_and_starts_in_float32(self):
        policy = resolve_policy(_torch(mps=True))
        self.assertEqual(policy.device, "mps")
        self.assertEqual(policy.dtype_name, "float32")
        self.assertEqual(policy.dtype, "fp32")

    def test_explicit_mps_fails_closed_when_pytorch_lacks_mps(self):
        with self.assertRaisesRegex(
            AcceleratorConfigurationError, "PyTorch build has no MPS support"
        ):
            resolve_policy(
                _torch(mps=False, mps_built=False), accelerator="mps"
            )

    def test_explicit_mps_fails_closed_when_host_is_unavailable(self):
        with self.assertRaisesRegex(
            AcceleratorConfigurationError, "MPS is unavailable on this host"
        ):
            resolve_policy(
                _torch(mps=False, mps_built=True), accelerator="mps"
            )

    def test_explicit_dtype_is_preserved(self):
        policy = resolve_policy(_torch(mps=True), accelerator="mps", dtype="float16")
        self.assertEqual(policy.dtype_name, "float16")
        self.assertEqual(policy.dtype, "fp16")

    def test_invalid_setting_is_rejected(self):
        with self.assertRaisesRegex(AcceleratorConfigurationError, "invalid"):
            resolve_policy(_torch(), accelerator="metal")


class AcceleratorReportTests(unittest.TestCase):
    def test_report_includes_selection_fallback_state_and_mps_memory(self):
        with patch.dict(
            os.environ,
            {
                "GRIDSHOT_ACCELERATOR": "mps",
                "GRIDSHOT_DTYPE": "float32",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            },
            clear=False,
        ):
            report = accelerator_report(_torch(mps=True))

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["selected"], "mps")
        self.assertEqual(report["dtype"], "float32")
        self.assertTrue(report["mps_cpu_fallback_enabled"])
        self.assertEqual(
            report["mps_memory"],
            {
                "current_allocated_bytes": 10,
                "driver_allocated_bytes": 20,
                "recommended_max_bytes": 30,
            },
        )

    def test_report_surfaces_invalid_configuration_without_raising(self):
        with patch.dict(
            os.environ,
            {"GRIDSHOT_ACCELERATOR": "not-a-device"},
            clear=False,
        ):
            report = accelerator_report(_torch(mps=True))

        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["selected"])
        self.assertIn("invalid GRIDSHOT_ACCELERATOR", report["error"])


if __name__ == "__main__":
    unittest.main()
