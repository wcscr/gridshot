from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx
import numpy as np

from gridshot.seg import client


class OptionalCapabilityClientTests(unittest.TestCase):
    def test_concept_503_becomes_a_fallback_safe_exception(self):
        request = httpx.Request("POST", "http://segserver:8801/segment_concept")
        response = httpx.Response(
            503,
            request=request,
            json={"detail": "concept model unavailable: gated model access denied"},
        )
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)

        with (
            patch.object(httpx, "post", return_value=response),
            self.assertRaises(client.OptionalCapabilityUnavailable) as raised,
        ):
            client.segment_concept(pixels)

        self.assertIn("HTTP 503", str(raised.exception))
        self.assertIn("gated model access denied", str(raised.exception))

    def test_concept_transport_error_becomes_a_fallback_safe_exception(self):
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)

        with (
            patch.object(
                httpx, "post", side_effect=httpx.ConnectError("connection refused")
            ),
            self.assertRaises(client.OptionalCapabilityUnavailable) as raised,
        ):
            client.segment_concept(pixels)

        self.assertIn("connection refused", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
