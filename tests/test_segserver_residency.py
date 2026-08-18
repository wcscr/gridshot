from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from gridshot.segserver import main as segserver
from gridshot.segserver.residency import (
    optional_model_idle_seconds,
    optional_residency_mode,
)


class ResidencyHelperTests(unittest.TestCase):
    def test_residency_policy_is_selected_by_accelerator(self):
        self.assertEqual(optional_residency_mode("mps"), "single")
        self.assertEqual(optional_model_idle_seconds("mps"), 300)
        self.assertEqual(optional_residency_mode("cuda"), "multi")
        self.assertEqual(optional_model_idle_seconds("cuda"), 0)

    def test_checkpoint_pinning_is_metal_only(self):
        revision = "validated-revision"
        self.assertEqual(segserver._metal_revision("mps", revision), revision)
        self.assertIsNone(segserver._metal_revision("cuda", revision))
        self.assertIsNone(segserver._metal_revision("cpu", revision))


class SegserverResidencyTests(unittest.TestCase):
    def setUp(self):
        self._state = dict(segserver._state)
        segserver._state.clear()

    def tearDown(self):
        segserver._state.clear()
        segserver._state.update(self._state)

    def test_embedding_cache_matches_the_original_fixed_limit(self):
        self.assertEqual(segserver.EMBED_MAX, 8)

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

    def test_optional_model_load_failure_is_a_service_unavailable_state(self):
        def fail_load():
            raise OSError("Cannot access gated repo; account is not in authorized list")

        with self.assertRaises(HTTPException) as raised:
            segserver._ensure_component("concept", fail_load, optional=True)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("gated Hugging Face model access denied", raised.exception.detail)
        self.assertNotIn("authorized list", raised.exception.detail)
        self.assertIn("hf auth login", segserver._state["concept_error"])

    def test_required_model_load_failure_still_raises_the_original_error(self):
        def fail_load():
            raise OSError("disk full")

        with self.assertRaises(OSError):
            segserver._ensure_component("interactive", fail_load)

        self.assertEqual(segserver._state["interactive_error"], "disk full")

    def test_single_residency_evicts_the_other_optional_lane_on_admission(self):
        segserver._state.update(
            {
                "matcher": object(),
                "matcher_device": "mps",
                "matcher_dtype": "float32",
            }
        )
        with (
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

    def test_cuda_admission_preserves_other_optional_models(self):
        segserver._state.update(
            {
                "matcher": object(),
                "matcher_device": "cuda",
                "matcher_dtype": "bfloat16",
            }
        )
        with patch.object(
            segserver,
            "_runtime_policy",
            return_value=SimpleNamespace(device="cuda"),
        ):
            segserver._prepare_optional_lane("concept")

        self.assertIn("matcher", segserver._state)
        self.assertIn("concept", segserver._state["optional_last_used_at"])


if __name__ == "__main__":
    unittest.main()
