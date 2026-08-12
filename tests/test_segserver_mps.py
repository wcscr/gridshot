"""Opt-in integration test for the pinned SAM 2 checkpoint on Apple MPS.

Run outside a restricted sandbox with:

    GRIDSHOT_RUN_MPS_TESTS=1 GRIDSHOT_ACCELERATOR=mps \
      python -m unittest tests.test_segserver_mps -v

The separately gated SAM 3 concept path can be exercised with:

    GRIDSHOT_RUN_MPS_SAM3_TESTS=1 GRIDSHOT_ACCELERATOR=mps \
      python -m unittest tests.test_segserver_mps -v

The SAM 3 operator surface can be smoke-tested without gated weights with:

    GRIDSHOT_RUN_MPS_SAM3_ARCH_TESTS=1 GRIDSHOT_ACCELERATOR=mps \
      python -m unittest tests.test_segserver_mps -v

The first run downloads the pinned checkpoint into ``HF_HUB_CACHE`` when set.
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

            capabilities = client.get("/capabilities").json()
            lane = capabilities["capabilities"]["interactive_segmentation"]
            self.assertEqual(lane["status"], "ready")
            self.assertEqual(lane["device"], "mps")
            self.assertEqual(lane["dtype"], "float32")
            self.assertEqual(lane["revision"], segserver.MODEL_REVISION)
            self.assertEqual(capabilities["residency"]["mode"], "single")
            self.assertEqual(capabilities["embedding_cache"]["entries"], 1)
            self.assertGreater(
                capabilities["embedding_cache"]["retained_bytes"], 0
            )


@unittest.skipUnless(
    os.environ.get("GRIDSHOT_RUN_MPS_SAM3_TESTS") == "1",
    "set GRIDSHOT_RUN_MPS_SAM3_TESTS=1 to exercise SAM 3 on Metal",
)
class SegserverSAM3MPSIntegrationTests(unittest.TestCase):
    def test_concept_segmentation_endpoint_uses_mps(self):
        import torch
        from fastapi.testclient import TestClient
        from PIL import Image, ImageDraw

        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable to this process")

        from gridshot.segserver import main as segserver

        image = Image.new("RGB", (640, 480), "white")
        ImageDraw.Draw(image).rectangle(
            (180, 100, 470, 390), fill=(40, 80, 160)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with TestClient(segserver.app) as client:
            runtime = client.get("/capabilities").json()["runtime"]
            self.assertEqual(runtime["selected"], "mps")
            self.assertFalse(runtime["mps_cpu_fallback_enabled"])

            response = client.post(
                "/segment_concept",
                files={"file": ("image.png", buffer.getvalue(), "image/png")},
                data={"prompt": "blue rectangle", "threshold": "0.2"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsInstance(response.json()["instances"], list)

            capabilities = client.get("/capabilities").json()
            lane = capabilities["capabilities"]["concept_segmentation"]
            self.assertEqual(lane["status"], "ready")
            self.assertEqual(lane["device"], "mps")
            self.assertEqual(lane["dtype"], "float32")
            self.assertEqual(lane["revision"], segserver.CONCEPT_MODEL_REVISION)
            self.assertEqual(
                capabilities["residency"]["loaded"], ["concept_segmentation"]
            )


@unittest.skipUnless(
    os.environ.get("GRIDSHOT_RUN_MPS_SAM3_ARCH_TESTS") == "1",
    "set GRIDSHOT_RUN_MPS_SAM3_ARCH_TESTS=1 for the SAM 3 Metal smoke test",
)
class SegserverSAM3ArchitectureMPSIntegrationTests(unittest.TestCase):
    def test_reduced_random_weight_model_runs_entire_forward_on_mps(self):
        import torch
        from transformers import Sam3Config, Sam3Model

        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable to this process")

        config = Sam3Config(
            vision_config={
                "backbone_config": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "image_size": 56,
                    "patch_size": 14,
                    "window_size": 4,
                    "global_attn_indexes": [1],
                    "pretrain_image_size": 56,
                },
                "fpn_hidden_size": 32,
                "scale_factors": [4.0, 2.0, 1.0, 0.5],
            },
            text_config={
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "projection_dim": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "max_position_embeddings": 16,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            geometry_encoder_config={
                "hidden_size": 32,
                "num_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "roi_size": 3,
            },
            detr_encoder_config={
                "hidden_size": 32,
                "num_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 64,
            },
            detr_decoder_config={
                "hidden_size": 32,
                "num_layers": 1,
                "num_queries": 8,
                "num_attention_heads": 4,
                "intermediate_size": 64,
            },
            mask_decoder_config={
                "hidden_size": 32,
                "num_upsampling_stages": 3,
                "num_attention_heads": 4,
            },
        )
        model = Sam3Model(config).eval().to("mps")
        with torch.inference_mode():
            output = model(
                pixel_values=torch.randn(1, 3, 56, 56, device="mps"),
                input_ids=torch.tensor([[1, 2, 3, 4]], device="mps"),
                attention_mask=torch.ones(1, 4, dtype=torch.long, device="mps"),
            )
        torch.mps.synchronize()

        self.assertEqual(output.pred_masks.device.type, "mps")
        self.assertEqual(tuple(output.pred_masks.shape), (1, 8, 16, 16))
        self.assertEqual(tuple(output.pred_boxes.shape), (1, 8, 4))
