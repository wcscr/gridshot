"""Opt-in integration test for the pinned SAM 2 checkpoint on Apple MPS.

Run outside a restricted sandbox with:

    GRIDSHOT_RUN_MPS_TESTS=1 GRIDSHOT_ACCELERATOR=mps \
      python -m unittest tests.test_segserver_mps -v

The first run downloads the pinned checkpoint into ``HF_HOME``.
"""

from __future__ import annotations

import io
import json
import os
import unittest


@unittest.skipUnless(
    os.environ.get("GRIDSHOT_RUN_MPS_TESTS") == "1",
    "set GRIDSHOT_RUN_MPS_TESTS=1 to exercise the real Metal model path",
)
class SegserverMPSIntegrationTests(unittest.TestCase):
    def test_interactive_segmentation_endpoints_use_mps(self):
        import torch
        from fastapi.testclient import TestClient
        from PIL import Image, ImageDraw

        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable to this process")

        # Import after the environment gate: model family and revisions are
        # process-level configuration in the production service too.
        from gridshot.segserver import main as segserver

        image = Image.new("RGB", (640, 480), "white")
        ImageDraw.Draw(image).rectangle(
            (180, 100, 470, 390), fill=(40, 80, 160)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        with TestClient(segserver.app) as client:
            runtime = client.get("/capabilities").json()["runtime"]
            self.assertEqual(runtime["selected"], "mps")
            self.assertFalse(runtime["mps_cpu_fallback_enabled"])

            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200, ready.text)
            self.assertEqual(ready.json()["device"], "mps")
            self.assertEqual(ready.json()["dtype"], "float32")

            segmented = client.post(
                "/segment",
                files={"file": ("image.png", image_bytes, "image/png")},
                data={
                    "points": json.dumps([[325.0, 245.0]]),
                    "labels": json.dumps([1]),
                    "box": "null",
                },
            )
            self.assertEqual(segmented.status_code, 200, segmented.text)
            self.assertTrue(segmented.json()["mask"])

            embedded = client.post(
                "/embed",
                files={"file": ("image.png", image_bytes, "image/png")},
            )
            self.assertEqual(embedded.status_code, 200, embedded.text)

            decoded = client.post(
                "/decode",
                data={
                    "image_id": embedded.json()["image_id"],
                    "points": json.dumps([[325.0, 245.0]]),
                    "labels": json.dumps([1]),
                },
            )
            self.assertEqual(decoded.status_code, 200, decoded.text)
            self.assertTrue(decoded.json()["mask"])

            lane = client.get("/capabilities").json()["capabilities"][
                "interactive_segmentation"
            ]
            self.assertEqual(lane["status"], "ready")
            self.assertEqual(lane["device"], "mps")
            self.assertEqual(lane["dtype"], "float32")
            self.assertEqual(lane["revision"], segserver.MODEL_REVISION)
