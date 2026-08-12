from __future__ import annotations

import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image

from gridshot.segserver import main as segserver
from gridshot.segserver.residency import (
    ResidencyConfigurationError,
    env_flag,
    env_int,
    optional_residency_mode,
    retained_bytes,
)


class _FakeTensor:
    def __init__(self, size: int):
        self._size = size

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class ResidencyHelperTests(unittest.TestCase):
    def test_boolean_flags_are_strict(self):
        self.assertTrue(env_flag("FLAG", default=False, environ={"FLAG": "yes"}))
        self.assertFalse(env_flag("FLAG", default=True, environ={"FLAG": "off"}))
        with self.assertRaisesRegex(ResidencyConfigurationError, "invalid FLAG"):
            env_flag("FLAG", default=False, environ={"FLAG": "sometimes"})

    def test_integer_settings_are_bounded(self):
        self.assertEqual(
            env_int(
                "LIMIT",
                default=4,
                minimum=1,
                maximum=8,
                environ={"LIMIT": "7"},
            ),
            7,
        )
        with self.assertRaisesRegex(ResidencyConfigurationError, "expected 1..8"):
            env_int(
                "LIMIT",
                default=4,
                minimum=1,
                maximum=8,
                environ={"LIMIT": "9"},
            )

    def test_auto_residency_keeps_only_one_optional_model_on_mps(self):
        self.assertEqual(optional_residency_mode("mps", value="auto"), "single")
        self.assertEqual(optional_residency_mode("cuda", value="auto"), "multi")
        self.assertEqual(optional_residency_mode("mps", value="multi"), "multi")

    def test_retained_size_counts_tensors_arrays_and_decoded_images_once(self):
        tensor = _FakeTensor(1024)
        array = np.zeros((10, 20), dtype=np.float32)
        image = Image.new("RGB", (20, 10))
        value = {"tensor": tensor, "duplicate": tensor, "array": array, "image": image}
        self.assertEqual(retained_bytes(value), 1024 + 800 + 600)


class SegserverResidencyTests(unittest.TestCase):
    def setUp(self):
        self._state = dict(segserver._state)
        self._cache = OrderedDict(segserver._EMBED_CACHE)
        self._cache_bytes = segserver._EMBED_CACHE_BYTES
        self._cache_evictions = segserver._EMBED_CACHE_EVICTIONS
        segserver._state.clear()
        segserver._EMBED_CACHE.clear()
        segserver._EMBED_CACHE_BYTES = 0
        segserver._EMBED_CACHE_EVICTIONS = 0

    def tearDown(self):
        segserver._state.clear()
        segserver._state.update(self._state)
        segserver._EMBED_CACHE.clear()
        segserver._EMBED_CACHE.update(self._cache)
        segserver._EMBED_CACHE_BYTES = self._cache_bytes
        segserver._EMBED_CACHE_EVICTIONS = self._cache_evictions

    def test_byte_budget_evicts_the_oldest_embedding(self):
        with patch.dict(
            os.environ,
            {
                "GRIDSHOT_EMBED_CACHE_MAX_ITEMS": "8",
                "GRIDSHOT_EMBED_CACHE_MAX_MIB": "16",
            },
            clear=False,
        ):
            segserver._store_embedding(
                "old", _FakeTensor(10 * 1024 * 1024), Image.new("RGB", (1, 1))
            )
            segserver._store_embedding(
                "new", _FakeTensor(10 * 1024 * 1024), Image.new("RGB", (1, 1))
            )

        self.assertEqual(list(segserver._EMBED_CACHE), ["new"])
        self.assertEqual(segserver._EMBED_CACHE_EVICTIONS, 1)
        self.assertLessEqual(segserver._EMBED_CACHE_BYTES, 16 * 1024 * 1024)

    def test_optional_eviction_removes_all_lane_references_and_records_reason(self):
        segserver._state.update(
            {
                "concept_model": object(),
                "concept_processor": object(),
                "concept_device": "mps",
                "concept_dtype": "float32",
                "optional_last_used_monotonic": {"concept": 1.0},
            }
        )
        with (
            patch.object(segserver, "_synchronize_model_device") as synchronize,
            patch.object(segserver, "_release_model_memory") as release,
        ):
            evicted = segserver._evict_optional_lane("concept", "unit test")

        self.assertTrue(evicted)
        self.assertNotIn("concept_model", segserver._state)
        self.assertNotIn("concept_processor", segserver._state)
        self.assertEqual(
            segserver._state["optional_evicted"]["concept"]["reason"], "unit test"
        )
        self.assertEqual(segserver._state["optional_evictions_total"], 1)
        synchronize.assert_called_once_with("mps")
        release.assert_called_once_with("mps")

    def test_disabled_optional_lane_is_not_reported_as_a_load_error(self):
        with patch.dict(os.environ, {"GRIDSHOT_ENABLE_ROMA": "0"}, clear=False):
            lane = segserver._component_capability(
                key="matcher",
                error_key="matcher_error",
                model="roma-outdoor-0.1.2",
                revision=None,
                device_key="matcher_device",
                dtype_key="matcher_dtype",
                optional_lane="matcher",
                runtime_device="mps",
            )

        self.assertEqual(lane["status"], "disabled")
        self.assertFalse(lane["enabled"])
        self.assertIsNone(lane["error"])

    def test_optional_model_load_failure_is_a_service_unavailable_state(self):
        def fail_load():
            raise OSError("Cannot access gated repo; account is not in authorized list")

        with self.assertRaises(HTTPException) as raised:
            segserver._ensure_component("concept", fail_load, optional=True)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("gated Hugging Face model access denied", raised.exception.detail)
        self.assertNotIn("authorized list", raised.exception.detail)
        self.assertIn("hf auth login", segserver._state["concept_error"])

    def test_single_residency_evicts_the_other_optional_lane_on_admission(self):
        segserver._state.update(
            {
                "matcher": object(),
                "matcher_device": "mps",
                "matcher_dtype": "float32",
            }
        )
        with (
            patch.dict(
                os.environ,
                {
                    "GRIDSHOT_ENABLE_SAM3": "1",
                    "GRIDSHOT_OPTIONAL_MODEL_RESIDENCY": "single",
                },
                clear=False,
            ),
            patch.object(
                segserver,
                "_runtime_policy",
                return_value=SimpleNamespace(device="mps"),
            ),
            patch.object(segserver, "_synchronize_model_device"),
            patch.object(segserver, "_release_model_memory"),
        ):
            segserver._prepare_optional_lane("concept")

        self.assertNotIn("matcher", segserver._state)
        self.assertIn("concept", segserver._state["optional_last_used_at"])
        self.assertIn(
            "admitted concept",
            segserver._state["optional_evicted"]["matcher"]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
