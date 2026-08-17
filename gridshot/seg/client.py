"""Thin HTTP client for the GPU segserver."""

from __future__ import annotations

import base64
import io
import json
import os

import httpx
import numpy as np
from PIL import Image

DEFAULT_URL = "http://segserver:8801"


def server_url() -> str:
    return os.environ.get("GRIDSHOT_SEGSERVER_URL", DEFAULT_URL)


def _probe(path: str, timeout: float = 3.0) -> bool:
    try:
        response = httpx.get(f"{server_url()}{path}", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def liveness(timeout: float = 3.0) -> bool:
    return _probe("/live", timeout=timeout)


def readiness(timeout: float = 30.0) -> bool:
    return _probe("/ready", timeout=timeout)


def capabilities(timeout: float = 3.0) -> dict | None:
    try:
        response = httpx.get(f"{server_url()}/capabilities", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def available(timeout: float = 3.0) -> bool:
    """Compatibility alias for callers that need inference, not mere liveness."""
    return readiness(timeout=timeout)


def segment(
    pixels: np.ndarray,
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
    box: tuple[float, float, float, float] | None = None,
    timeout: float = 300.0,
) -> tuple[np.ndarray, float]:
    """Point- and/or box-prompted SAM mask.  Returns (mask 0/255, score)."""
    points = points or []
    labels = labels or [1] * len(points)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=92)
    r = httpx.post(
        f"{server_url()}/segment",
        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
        data={
            "points": json.dumps([[float(x), float(y)] for x, y in points]),
            "labels": json.dumps([int(v) for v in labels]),
            "box": json.dumps([float(v) for v in box] if box else None),
        },
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["mask"]))))
    return ((mask > 127) * 255).astype(np.uint8), float(payload["score"])


def embed(pixels: np.ndarray, timeout: float = 120.0) -> tuple[str, int, int]:
    """Compute+cache the SAM image embedding once. Returns (image_id, w, h)."""
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=92)
    r = httpx.post(
        f"{server_url()}/embed",
        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
        timeout=timeout,
    )
    r.raise_for_status()
    d = r.json()
    return d["image_id"], d["width"], d["height"]


def decode(
    image_id: str,
    points: list[tuple[float, float]],
    labels: list[int],
    box: list[float] | None = None,
    mask_poly: list[list[float]] | None = None,
    timeout: float = 30.0,
) -> tuple[np.ndarray, float]:
    """Per-click mask from the cached embedding (~ms). (mask 0/255, score).

    box ([x0,y0,x1,y1] px) is a strong whole-object prompt — far more reliable
    than points for thin parts (a screwdriver shaft). mask_poly (a pixel-space
    outline) seeds SAM's dense mask prompt to refine an existing shape.
    """
    data = {
        "image_id": image_id,
        "points": json.dumps([[float(x), float(y)] for x, y in points]),
        "labels": json.dumps([int(v) for v in labels]),
    }
    if box is not None and len(box) == 4:
        data["box"] = json.dumps([float(v) for v in box])
    if mask_poly and len(mask_poly) >= 3:
        data["mask_poly"] = json.dumps([[float(x), float(y)] for x, y in mask_poly])
    r = httpx.post(
        f"{server_url()}/decode",
        data=data,
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["mask"]))))
    return ((mask > 127) * 255).astype(np.uint8), float(payload["score"])


def match_dense(
    pixels_a: np.ndarray,
    mask_a: np.ndarray,
    pixels_b: np.ndarray,
    mask_b: np.ndarray,
    max_matches: int = 2000,
    timeout: float = 600.0,
) -> dict:
    """RoMa foreground correspondences from the GPU service."""
    buffers = []
    for array, fmt in (
        (pixels_a, "JPEG"),
        (mask_a, "PNG"),
        (pixels_b, "JPEG"),
        (mask_b, "PNG"),
    ):
        buf = io.BytesIO()
        save_options = {"quality": 95} if fmt == "JPEG" else {}
        Image.fromarray(array).save(buf, format=fmt, **save_options)
        buffers.append(buf.getvalue())
    response = httpx.post(
        f"{server_url()}/match",
        files={
            "file_a": ("a.jpg", buffers[0], "image/jpeg"),
            "mask_a": ("a-mask.png", buffers[1], "image/png"),
            "file_b": ("b.jpg", buffers[2], "image/jpeg"),
            "mask_b": ("b-mask.png", buffers[3], "image/png"),
        },
        data={"max_matches": str(int(max_matches))},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    payload["points_a"] = np.asarray(payload["points_a"], dtype=np.float32)
    payload["points_b"] = np.asarray(payload["points_b"], dtype=np.float32)
    payload["certainty"] = np.asarray(payload["certainty"], dtype=np.float32)
    return payload


def segment_concept(
    pixels: np.ndarray,
    prompt: str = "tool",
    threshold: float = 0.4,
    timeout: float = 600.0,
) -> list[tuple[np.ndarray, float]]:
    """Text-prompted instances via SAM 3's concept path, best score first."""
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=92)
    r = httpx.post(
        f"{server_url()}/segment_concept",
        files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
        data={"prompt": prompt, "threshold": str(threshold)},
        timeout=timeout,
    )
    r.raise_for_status()
    out = []
    for inst in r.json()["instances"]:
        mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(inst["mask"]))))
        out.append((((mask > 127) * 255).astype(np.uint8), float(inst["score"])))
    return out
