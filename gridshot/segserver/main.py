"""Accelerated segmentation service: SAM 2.1 point-prompted masks over HTTP.

Runs on CUDA in the Compose deployment and Metal/MPS in the native macOS
deployment. The M1.5 headless trace flow sends prompt points derived from
empty-mat differencing; the M3 web editor sends interactive clicks against
the same endpoint. Model weights download once into the configured HF cache.

API (multipart form):
  POST /segment
    file:   image (any PIL-readable format)
    points: JSON [[x, y], ...] pixel coordinates
    labels: JSON [1|0, ...]    1 = foreground, 0 = background
  → {"mask": <base64 PNG, 0/255>, "score": float}
  GET /live         → process liveness (never loads a model)
  GET /ready        → interactive-model readiness
  GET /capabilities → configured/loaded/error state for each model lane
  GET /health       → compatibility alias for /ready
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import threading
import time
import uuid as _uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image

from .backend import AcceleratorPolicy, accelerator_report, resolve_policy
from .residency import (
    optional_model_idle_seconds,
    optional_residency_mode,
)

# "sam2" → SAM 2.1 hiera-large; "sam3" → SAM 3's interactive tracker head.
# Pin the Metal validation artifacts without changing CUDA's upstream model loads.
MODEL_FAMILY = os.environ.get("GRIDSHOT_SAM", "sam2")
MODEL_IDS = {"sam2": "facebook/sam2.1-hiera-large", "sam3": "facebook/sam3"}
MODEL_REVISIONS = {
    "sam2": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "sam3": "3c879f39826c281e95690f02c7821c4de09afae7",
}
MODEL_ID = MODEL_IDS[MODEL_FAMILY]
MODEL_REVISION = (
    os.environ.get("GRIDSHOT_SAM_REVISION") or MODEL_REVISIONS[MODEL_FAMILY]
)
CONCEPT_MODEL_ID = "facebook/sam3"
CONCEPT_MODEL_REVISION = (
    os.environ.get("GRIDSHOT_SAM3_REVISION") or MODEL_REVISIONS["sam3"]
)

app = FastAPI(title="gridshot-segserver")

_state: dict = {}
_model_lock = threading.Lock()
_concept_lock = threading.Lock()
_matcher_lock = threading.Lock()
_inference_lock = threading.Lock()

_OPTIONAL_LANES = {
    "concept": {
        "capability": "concept_segmentation",
        "model_key": "concept_model",
        "device_key": "concept_device",
        "keys": (
            "concept_processor",
            "concept_model",
            "concept_device",
            "concept_dtype",
            "concept_revision",
        ),
    },
    "matcher": {
        "capability": "dense_matching",
        "model_key": "matcher",
        "device_key": "matcher_device",
        "keys": ("matcher", "matcher_device", "matcher_dtype"),
    },
}
_CAPABILITY_TO_OPTIONAL_LANE = {
    spec["capability"]: lane for lane, spec in _OPTIONAL_LANES.items()
}


def _queue_capacity() -> int:
    raw = os.environ.get("GRIDSHOT_INFERENCE_QUEUE_SIZE", "2")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


class InferenceAdmission:
    """Bound the one GPU worker plus requests allowed to wait for it."""

    def __init__(self, queue_capacity: int = 2):
        self.concurrency = 1
        self.queue_capacity = max(0, int(queue_capacity))
        self.capacity = self.concurrency + self.queue_capacity
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._stats_lock = threading.Lock()
        self._active = 0
        self._admitted = 0
        self._rejected = 0

    @contextmanager
    def admit(self, capability: str):
        if not self._slots.acquire(blocking=False):
            with self._stats_lock:
                self._rejected += 1
            raise HTTPException(
                status_code=429,
                detail=f"segmentation inference queue is full ({capability})",
                headers={"Retry-After": "2"},
            )
        with self._stats_lock:
            self._active += 1
            self._admitted += 1
        try:
            yield
        finally:
            with self._stats_lock:
                self._active -= 1
            self._slots.release()

    def stats(self) -> dict:
        with self._stats_lock:
            active = self._active
            return {
                "concurrency": self.concurrency,
                "queue_capacity": self.queue_capacity,
                "capacity": self.capacity,
                "active": active,
                "queued": max(0, active - self.concurrency),
                "admitted_total": self._admitted,
                "rejected_total": self._rejected,
            }


_inference_admission = InferenceAdmission(_queue_capacity())


def _runtime_policy() -> AcceleratorPolicy:
    """Resolve one shared accelerator policy for every model lane."""

    policy = _state.get("runtime_policy")
    if policy is None:
        import torch

        policy = resolve_policy(torch)
        _state["runtime_policy"] = policy
    return policy


def _metal_revision(device: str | None, revision: str) -> str | None:
    """Use validated Hub artifacts on Metal and preserve upstream elsewhere."""

    return revision if device == "mps" else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronize_model_device(device: str | None) -> None:
    """Finish queued Metal work before dropping a model's final references."""

    if device != "mps":
        return
    try:
        import torch

        torch.mps.synchronize()
    except (AttributeError, RuntimeError):
        pass


def _release_model_memory(device: str | None) -> None:
    """Return evicted model allocations to the Metal allocator when possible."""

    gc.collect()
    if device != "mps":
        return
    try:
        import torch

        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        # Eviction still removed every live model reference. Telemetry will show
        # whether the runtime retained allocator pages after this best effort.
        pass


def _touch_optional_lane(name: str) -> None:
    _state.setdefault("optional_last_used_monotonic", {})[name] = time.monotonic()
    _state.setdefault("optional_last_used_at", {})[name] = _utc_now()


def _evict_optional_lane(name: str, reason: str) -> bool:
    spec = _OPTIONAL_LANES[name]
    if spec["model_key"] not in _state:
        return False

    device = _state.get(spec["device_key"])
    _synchronize_model_device(device)
    for key in spec["keys"]:
        _state.pop(key, None)
    _state.setdefault("optional_last_used_monotonic", {}).pop(name, None)
    eviction = {"at": _utc_now(), "reason": reason}
    _state.setdefault("optional_evicted", {})[name] = eviction
    _state["optional_evictions_total"] = int(
        _state.get("optional_evictions_total", 0)
    ) + 1
    _release_model_memory(device)
    return True


def _evict_idle_optional_models(exclude: str | None = None) -> None:
    """Evict idle optional lanes while the global inference lock is held."""

    try:
        device = _runtime_policy().device
    except Exception:
        return
    idle_seconds = optional_model_idle_seconds(device)
    if idle_seconds <= 0:
        return

    now = time.monotonic()
    last_used = _state.get("optional_last_used_monotonic", {})
    for name, spec in _OPTIONAL_LANES.items():
        if name == exclude or spec["model_key"] not in _state:
            continue
        last = last_used.get(name)
        if last is not None and now - last >= idle_seconds:
            _evict_optional_lane(
                name,
                f"idle for at least {idle_seconds} seconds",
            )


def _prepare_optional_lane(name: str) -> None:
    """Validate, admit, and touch an optional model lane before loading it."""

    device = _runtime_policy().device
    if optional_residency_mode(device) == "single":
        for other in _OPTIONAL_LANES:
            if other != name:
                _evict_optional_lane(
                    other,
                    f"single optional-model residency admitted {name}",
                )
    _touch_optional_lane(name)


def bounded_inference(capability: str):
    """Run one accelerator operation while bounding requests waiting behind it."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with _inference_admission.admit(capability):
                with _inference_lock:
                    _evict_idle_optional_models(
                        exclude=_CAPABILITY_TO_OPTIONAL_LANE.get(capability)
                    )
                    return function(*args, **kwargs)

        return wrapped

    return decorate


def _load():
    if "model" in _state:
        return
    with _model_lock:
        if "model" in _state:
            return
        if MODEL_FAMILY == "sam3":
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            processor_cls, model_cls = Sam3TrackerProcessor, Sam3TrackerModel
        else:
            from transformers import Sam2Model, Sam2Processor

            processor_cls, model_cls = Sam2Processor, Sam2Model

        policy = _runtime_policy()
        device = policy.device
        revision = _metal_revision(device, MODEL_REVISION)
        revision_kwargs = {"revision": revision} if revision is not None else {}
        processor = processor_cls.from_pretrained(MODEL_ID, **revision_kwargs)
        model = model_cls.from_pretrained(
            MODEL_ID, dtype=policy.dtype, **revision_kwargs
        ).to(device)
        model.eval()
        _state.update(
            {
                "processor": processor,
                "model": model,
                "device": device,
                "dtype": policy.dtype_name,
                "revision": revision,
            }
        )


def _load_concept():
    """SAM 3's concept path (text prompt → all instances), loaded on demand,
    independent of the interactive model family."""
    _prepare_optional_lane("concept")
    if "concept_model" in _state:
        return
    with _concept_lock:
        if "concept_model" in _state:
            return
        from transformers import Sam3Model, Sam3Processor

        policy = _runtime_policy()
        device = policy.device
        revision = _metal_revision(device, CONCEPT_MODEL_REVISION)
        revision_kwargs = {"revision": revision} if revision is not None else {}
        processor = Sam3Processor.from_pretrained(
            CONCEPT_MODEL_ID, **revision_kwargs
        )
        model = Sam3Model.from_pretrained(
            CONCEPT_MODEL_ID,
            dtype=policy.dtype,
            **revision_kwargs,
        ).to(device)
        model.eval()
        _state.update(
            {
                "concept_processor": processor,
                "concept_model": model,
                "concept_device": device,
                "concept_dtype": policy.dtype_name,
                "concept_revision": revision,
            }
        )
        _state.setdefault("optional_evicted", {}).pop("concept", None)
        _touch_optional_lane("concept")


def _load_matcher():
    """Load RoMa lazily; segmentation does not pay its VRAM cost unless used."""
    _prepare_optional_lane("matcher")
    if "matcher" in _state:
        return
    with _matcher_lock:
        if "matcher" in _state:
            return
        from romatch import roma_outdoor

        policy = _runtime_policy()
        device = policy.device
        matcher = roma_outdoor(device=device, use_custom_corr=False)
        matcher.eval()
        # RoMa 0.1.2 only enables autocast for CUDA.  Report its actual
        # non-CUDA behavior instead of claiming the global requested dtype.
        matcher_dtype = policy.dtype_name if device == "cuda" else "float32"
        _state.update(
            {
                "matcher": matcher,
                "matcher_device": device,
                "matcher_dtype": matcher_dtype,
            }
        )
        _state.setdefault("optional_evicted", {}).pop("matcher", None)
        _touch_optional_lane("matcher")


def _model_load_error(exc: Exception) -> str:
    raw = " ".join(str(exc).split())
    lowered = raw.lower()
    if "gated repo" in lowered or "authorized list" in lowered:
        return (
            "gated Hugging Face model access denied; request model access and "
            "authenticate with `hf auth login` or an authorized HF_TOKEN"
        )
    return raw[:240]


def _ensure_component(name: str, loader, *, optional: bool = False) -> None:
    try:
        loader()
        _state.pop(f"{name}_error", None)
    except HTTPException:
        raise
    except Exception as exc:
        error = _model_load_error(exc)
        _state[f"{name}_error"] = error
        if optional:
            raise HTTPException(
                status_code=503,
                detail=f"{name} model unavailable: {error}",
            ) from exc
        raise


def _component_capability(
    *,
    key: str,
    error_key: str,
    model: str,
    revision: str | None,
    device_key: str,
    dtype_key: str,
    optional_lane: str | None = None,
) -> dict:
    error = _state.get(error_key)
    loaded = key in _state
    evicted = (
        _state.get("optional_evicted", {}).get(optional_lane)
        if optional_lane is not None
        else None
    )
    if loaded:
        status = "ready"
    elif error:
        status = "error"
    elif evicted:
        status = "evicted"
    else:
        status = "not_loaded"

    payload = {
        "status": status,
        "model": model,
        "revision": revision,
        "loaded": loaded,
        "device": _state.get(device_key),
        "dtype": _state.get(dtype_key),
        "error": error,
    }
    if optional_lane is not None:
        payload["last_used_at"] = _state.get("optional_last_used_at", {}).get(
            optional_lane
        )
        payload["last_eviction"] = evicted
    return payload


def _optional_residency_report(runtime_device: str | None) -> dict:
    if runtime_device is None:
        return {
            "mode": None,
            "idle_seconds": None,
            "loaded": [],
            "evictions_total": int(_state.get("optional_evictions_total", 0)),
        }
    return {
        "mode": optional_residency_mode(runtime_device),
        "idle_seconds": optional_model_idle_seconds(runtime_device),
        "loaded": [
            spec["capability"]
            for spec in _OPTIONAL_LANES.values()
            if spec["model_key"] in _state
        ],
        "evictions_total": int(_state.get("optional_evictions_total", 0)),
    }


def _capabilities_payload() -> dict:
    runtime = accelerator_report()
    runtime_device = runtime.get("selected")
    return {
        "status": "ok" if runtime["status"] == "ok" else "degraded",
        "runtime": runtime,
        "capabilities": {
            "interactive_segmentation": _component_capability(
                key="model",
                error_key="interactive_error",
                model=MODEL_ID,
                revision=_metal_revision(runtime_device, MODEL_REVISION),
                device_key="device",
                dtype_key="dtype",
            ),
            "concept_segmentation": _component_capability(
                key="concept_model",
                error_key="concept_error",
                model=CONCEPT_MODEL_ID,
                revision=_metal_revision(runtime_device, CONCEPT_MODEL_REVISION),
                device_key="concept_device",
                dtype_key="concept_dtype",
                optional_lane="concept",
            ),
            "dense_matching": _component_capability(
                key="matcher",
                error_key="matcher_error",
                model="roma-outdoor-0.1.2",
                revision=None,
                device_key="matcher_device",
                dtype_key="matcher_dtype",
                optional_lane="matcher",
            ),
        },
        "inference": _inference_admission.stats(),
        "residency": _optional_residency_report(runtime_device),
    }


@app.get("/live")
def live() -> dict:
    """Process liveness only. This endpoint never imports or loads a model."""
    return {"status": "alive"}


@app.get("/capabilities")
def capabilities() -> dict:
    """Report model-lane state without forcing optional models into VRAM."""
    return _capabilities_payload()


def _readiness_payload() -> tuple[dict, int]:
    if "model" not in _state:
        try:
            with _inference_admission.admit("readiness"):
                with _inference_lock:
                    _evict_idle_optional_models()
                    _ensure_component("interactive", _load)
        except Exception as exc:
            return (
                {
                    "status": "not_ready",
                    "model": MODEL_ID,
                    "revision": _metal_revision(
                        getattr(_state.get("runtime_policy"), "device", None),
                        MODEL_REVISION,
                    ),
                    "error": str(exc)[:240],
                },
                503,
            )
    return (
        {
            "status": "ready",
            "model": MODEL_ID,
            "revision": _state.get("revision"),
            "device": _state.get("device"),
            "dtype": _state.get("dtype"),
        },
        200,
    )


@app.get("/ready")
def ready(response: Response) -> dict:
    payload, response.status_code = _readiness_payload()
    return payload


@app.get("/health")
def health(response: Response) -> dict:
    """Compatibility alias; new probes should use /live or /ready explicitly."""
    payload, response.status_code = _readiness_payload()
    return payload


def _matcher_crop(image: Image.Image, mask: Image.Image):
    """Tight foreground crop with the repeated calibration mat neutralized."""
    import cv2

    if image.size != mask.size:
        raise HTTPException(status_code=422, detail="matcher image/mask dimensions differ")
    rgb = np.asarray(image.convert("RGB"))
    binary = np.asarray(mask.convert("L")) > 127
    ys, xs = np.nonzero(binary)
    if not xs.size:
        raise HTTPException(status_code=422, detail="empty matcher mask")
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    pad = max(12, int(round(0.10 * span)))
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])
    y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
    crop = rgb[y0:y1, x0:x1]
    crop_mask = binary[y0:y1, x0:x1]

    # A feathered edge avoids creating a synthetic hard contour while ensuring
    # that ChArUco markers cannot become the identity signal.
    alpha = cv2.GaussianBlur(crop_mask.astype(np.float32), (0, 0), sigmaX=2.0)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    neutral = np.full_like(crop, 127)
    masked = np.clip(crop * alpha + neutral * (1.0 - alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(masked), crop_mask, (x0, y0)


@app.post("/match")
@bounded_inference("dense_matching")
def match_dense(
    file_a: UploadFile = File(...),
    mask_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    mask_b: UploadFile = File(...),
    max_matches: int = Form(2000),
) -> dict:
    """RoMa correspondences restricted to independently detected foreground."""
    import cv2
    import time

    _ensure_component("matcher", _load_matcher)
    image_a = Image.open(file_a.file).convert("RGB")
    image_b = Image.open(file_b.file).convert("RGB")
    mask_image_a = Image.open(mask_a.file).convert("L")
    mask_image_b = Image.open(mask_b.file).convert("L")
    crop_a, foreground_a, offset_a = _matcher_crop(image_a, mask_image_a)
    crop_b, foreground_b, offset_b = _matcher_crop(image_b, mask_image_b)

    model = _state["matcher"]
    device = _state["matcher_device"]
    started = time.perf_counter()
    with _matcher_lock:
        warp, certainty = model.match(crop_a, crop_b, device=device)
        matches, match_certainty = model.sample(
            warp, certainty, num=max(8, min(int(max_matches), 10000))
        )
        points_a, points_b = model.to_pixel_coordinates(
            matches, crop_a.height, crop_a.width, crop_b.height, crop_b.width
        )
    points_a = points_a.detach().cpu().float().numpy()
    points_b = points_b.detach().cpu().float().numpy()
    match_certainty = match_certainty.detach().cpu().float().numpy()

    # Sampling is model-driven; enforce that both ends actually land on the
    # foreground cue (with a small tolerance for imperfect diff boundaries).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    allowed_a = cv2.dilate(foreground_a.astype(np.uint8), kernel) > 0
    allowed_b = cv2.dilate(foreground_b.astype(np.uint8), kernel) > 0
    ax = np.clip(np.rint(points_a[:, 0]).astype(int), 0, crop_a.width - 1)
    ay = np.clip(np.rint(points_a[:, 1]).astype(int), 0, crop_a.height - 1)
    bx = np.clip(np.rint(points_b[:, 0]).astype(int), 0, crop_b.width - 1)
    by = np.clip(np.rint(points_b[:, 1]).astype(int), 0, crop_b.height - 1)
    keep = allowed_a[ay, ax] & allowed_b[by, bx] & np.isfinite(match_certainty)
    points_a = points_a[keep] + np.asarray(offset_a, dtype=np.float32)
    points_b = points_b[keep] + np.asarray(offset_b, dtype=np.float32)
    match_certainty = match_certainty[keep]

    return {
        "model": "roma-outdoor-0.1.2",
        "device": device,
        "runtime_ms": round(1000.0 * (time.perf_counter() - started), 1),
        "sampled": int(len(matches)),
        "foreground_matches": int(len(points_a)),
        "points_a": points_a.tolist(),
        "points_b": points_b.tolist(),
        "certainty": match_certainty.tolist(),
    }


# ---------------------------------------------------------------------------
# interactive: embed an image once, then decode per click (~4ms) — the M3 editor

_EMBED_CACHE: "OrderedDict[str, dict]" = OrderedDict()
EMBED_MAX = 8


@app.post("/embed")
@bounded_inference("interactive_segmentation")
def embed(file: UploadFile = File(...)) -> dict:
    import torch

    _ensure_component("interactive", _load)
    image = Image.open(file.file).convert("RGB")
    inp = _state["processor"](images=image, return_tensors="pt").to(_state["device"])
    with torch.inference_mode():
        emb = _state["model"].get_image_embeddings(inp["pixel_values"])
    image_id = _uuid.uuid4().hex[:12]
    _EMBED_CACHE[image_id] = {"emb": emb, "image": image}
    while len(_EMBED_CACHE) > EMBED_MAX:
        _EMBED_CACHE.popitem(last=False)
    return {"image_id": image_id, "width": image.width, "height": image.height}


def _rasterize_poly(poly: list, image, output_size=None) -> np.ndarray | None:
    """Rasterize a photo-space polygon as a boolean mask."""
    from PIL import ImageDraw

    if not poly or len(poly) < 3:
        return None
    m = Image.new("L", image.size, 0)
    ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    if output_size is not None:
        m = m.resize(output_size, Image.Resampling.NEAREST)
    return np.asarray(m) > 127


def _mask_prior(poly: list, image, device, dtype) -> "object | None":
    """Build a deliberately soft dense prior so new clicks can still change it."""
    import torch

    arr = _rasterize_poly(poly, image, (256, 256))
    if arr is None:
        return None
    logits = (arr.astype(np.float32) * 2.0 - 1.0) * 2.0
    return torch.from_numpy(logits)[None, None].to(device=device, dtype=dtype)


def _select_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    points=None,
    labels=None,
    box=None,
    prior=None,
) -> tuple[int, dict]:
    """Rank SAM candidates by prompt compliance before model confidence."""
    binary = np.asarray(masks) > 0.5
    h, w = binary.shape[-2:]
    best_index = 0
    best_rank = None
    best_diag = {}

    for i, candidate in enumerate(binary):
        violations = 0
        for (x, y), label in zip(points or [], labels or []):
            ix, iy = int(round(x)), int(round(y))
            hit = 0 <= ix < w and 0 <= iy < h and bool(candidate[iy, ix])
            if hit != (int(label) == 1):
                violations += 1

        outside_fraction = 0.0
        if box is not None:
            x0, y0, x1, y1 = box
            x0, x1 = sorted((max(0, int(x0)), min(w, int(np.ceil(x1)))))
            y0, y1 = sorted((max(0, int(y0)), min(h, int(np.ceil(y1)))))
            area = int(candidate.sum())
            inside = int(candidate[y0:y1, x0:x1].sum())
            outside_fraction = (area - inside) / area if area else 1.0

        prior_iou = 0.0
        if prior is not None:
            union = int(np.logical_or(candidate, prior).sum())
            if union:
                prior_iou = int(np.logical_and(candidate, prior).sum()) / union

        confidence = float(scores[i])
        rank = (-violations, -outside_fraction, confidence + 0.15 * prior_iou)
        if best_rank is None or rank > best_rank:
            best_index = i
            best_rank = rank
            best_diag = {
                "candidate": i,
                "prompt_violations": violations,
                "outside_box_fraction": round(outside_fraction, 4),
                "prior_iou": round(prior_iou, 4),
            }

    return best_index, best_diag


@app.post("/decode")
@bounded_inference("interactive_segmentation")
def decode(
    image_id: str = Form(...),
    points: str = Form("[]"),
    labels: str = Form("[]"),
    box: str = Form(None),        # optional [x0,y0,x1,y1] px box prompt
    mask_poly: str = Form(None),  # optional px outline → continue from this shape
) -> dict:
    import torch

    ent = _EMBED_CACHE.get(image_id)
    if ent is None:
        raise HTTPException(status_code=404, detail="embedding expired — re-embed")
    _EMBED_CACHE.move_to_end(image_id)
    pts = json.loads(points)
    lbs = json.loads(labels)
    bx = json.loads(box) if box else None
    if not pts and bx is None:
        raise HTTPException(status_code=400, detail="no points or box")

    proc, model, device = _state["processor"], _state["model"], _state["device"]
    prior_poly = json.loads(mask_poly) if mask_poly else None
    input_masks = _mask_prior(
        prior_poly, ent["image"], device, _runtime_policy().dtype
    )
    prior = _rasterize_poly(prior_poly, ent["image"])
    proc_kwargs = {"images": ent["image"], "return_tensors": "pt"}
    if pts:
        proc_kwargs["input_points"] = [[pts]]
        proc_kwargs["input_labels"] = [[lbs]]
    if bx is not None:
        proc_kwargs["input_boxes"] = [[bx]]  # [image, box, [x0,y0,x1,y1]]
    inp = proc(**proc_kwargs).to(device)
    model_kwargs = {
        "image_embeddings": ent["emb"],
        "multimask_output": True,
        "input_masks": input_masks,
    }
    if "input_points" in inp:
        model_kwargs["input_points"] = inp["input_points"]
        model_kwargs["input_labels"] = inp["input_labels"]
    if "input_boxes" in inp:
        model_kwargs["input_boxes"] = inp["input_boxes"]
    with torch.inference_mode():
        out = model(**model_kwargs)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
    scores = out.iou_scores.cpu().float().numpy().ravel()
    masks = np.asarray(masks).reshape(-1, masks.shape[-2], masks.shape[-1])
    best, selection = _select_candidate(masks, scores, pts, lbs, bx, prior)
    mask = (masks[best] > 0.5).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return {
        "mask": base64.b64encode(buf.getvalue()).decode(),
        "score": float(scores[best]),
        "selection": selection,
    }


@app.post("/segment_concept")
@bounded_inference("concept_segmentation")
def segment_concept(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.5),
) -> dict:
    """Text-prompted instance segmentation: every '<prompt>' in the image."""
    import torch

    _ensure_component("concept", _load_concept, optional=True)
    image = Image.open(file.file).convert("RGB")
    processor = _state["concept_processor"]
    model = _state["concept_model"]
    device = _state["concept_device"]

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks_out = []
    for mask, score in zip(results["masks"], results["scores"]):
        arr = (np.asarray(mask.cpu()) > 0).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        masks_out.append(
            {"mask": base64.b64encode(buf.getvalue()).decode(), "score": float(score)}
        )
    masks_out.sort(key=lambda m: -m["score"])
    return {"instances": masks_out}


@app.post("/segment")
@bounded_inference("interactive_segmentation")
def segment(
    file: UploadFile = File(...),
    points: str = Form("[]"),
    labels: str = Form("[]"),
    box: str = Form("null"),
) -> dict:
    import torch

    _ensure_component("interactive", _load)
    image = Image.open(file.file).convert("RGB")
    pts = json.loads(points)
    lbs = json.loads(labels)
    box_xyxy = json.loads(box)

    processor = _state["processor"]
    model = _state["model"]
    device = _state["device"]

    kwargs: dict = {}
    if pts:
        kwargs["input_points"] = [[pts]]  # batch=1, one object, N points
        kwargs["input_labels"] = [[lbs]]
    if box_xyxy:
        kwargs["input_boxes"] = [[box_xyxy]]

    inputs = processor(
        images=image,
        return_tensors="pt",
        **kwargs,
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]  # (num_objects, num_masks, H, W) or (num_masks, H, W)
    scores = outputs.iou_scores.cpu().float().numpy().ravel()
    masks = np.asarray(masks.numpy() if hasattr(masks, "numpy") else masks)
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])

    best, selection = _select_candidate(masks, scores, pts, lbs, box_xyxy)
    mask = (masks[best] > 0.5).astype(np.uint8) * 255

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return {
        "mask": base64.b64encode(buf.getvalue()).decode(),
        "score": float(scores[best]),
        "selection": selection,
    }
