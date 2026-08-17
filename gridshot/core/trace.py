"""End-to-end M1 pipeline: photo → calibrated outline → gridfinity bin files.

Every stage's numbers land in the returned summary so the CLI can narrate
what happened and the bench (G1) can compare digital outlines to calipers.
The segmentation mask is injectable for tests and for future segserver use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import calibrate as calibrate_mod
from . import contour as contour_mod
from . import derive as derive_mod
from . import devices as devices_mod
from . import export as export_mod
from . import gridfinity as grid_mod
from . import ingest as ingest_mod
from . import mat as mat_mod
from . import parallax as parallax_mod
from . import readiness as readiness_mod
from .models import Calibration, MatProfile, Poly, PrinterProfile

MAX_TOOL_AREA_MM2 = 45_000.0  # bigger than any hand tool ⇒ mask grabbed the sheet


@dataclass
class TraceResult:
    calibration: Calibration
    raw_poly: Poly
    corrected_poly: Poly
    pocket_poly: Poly  # bin-centred frame (tool outline + clearance)
    grid: tuple[int, int]
    height_u: int
    pocket_depth_mm: float
    bin_style: grid_mod.BinStyle = "pocket"
    overall_height_mm: float = 0.0
    lip: bool = True
    tool_poly: Poly | None = None  # tool outline in the same bin frame as pocket
    thickness_mm: float = 0.0
    silhouette_height_mm: float = 0.0
    full_height_mm: float | None = None
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    readiness: readiness_mod.ReadinessReport | None = None
    thickness_source: str = "unknown"
    derivation_key: str = ""
    reconstruction: dict[str, float | str] | None = None
    reserved_cells: list[tuple[float, float]] = field(default_factory=list)
    available_cells: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CaptureArtifacts:
    """Replayable one-image capture state for batch matching benchmarks."""

    calibration: Calibration
    pixels: np.ndarray
    mask: np.ndarray
    raw_poly: Poly
    outline: Poly
    warnings: list[str] = field(default_factory=list)
    readiness: readiness_mod.ReadinessReport | None = None


@dataclass
class CapturedToolGeometry:
    """Accepted raw silhouette plus its physical footprint reconstruction."""

    calibration: Calibration
    raw_poly: Poly
    corrected_poly: Poly
    thickness_mm: float
    warnings: list[str] = field(default_factory=list)
    reconstruction: dict[str, float | str] | None = None


def _mask_area_mm2(mask: np.ndarray, calibration: Calibration) -> float:
    components = contour_mod.mask_to_polygons_px(mask)
    if not components:
        return 0.0
    p = contour_mod.polygon_px_to_mm(components[0][0], [], calibration)
    return contour_mod.to_shapely(p).area


def _prompt_mask(
    pixels, calibration, box_mm, pad_mm: float = 12.0
) -> tuple[np.ndarray | None, list[str]]:
    """Segment the tool in a second view from its known mat-mm footprint.

    The tool sits at the same place on the mat in both views, so its footprint
    (recovered in the first view) projects — through this view's own
    homography — onto where the tool appears here. That gives SAM a targeted
    box + centre point instead of a blind guess. Returns (None, warnings) if
    the segserver is offline so the caller can fall back.
    """
    import cv2

    from gridshot.seg import client as seg_client

    if not seg_client.available():
        return None, []

    x0, y0, x1, y1 = box_mm
    cx_mm, cy_mm = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    corners_mm = np.array(
        [
            [x0 - pad_mm, y0 - pad_mm],
            [x1 + pad_mm, y0 - pad_mm],
            [x1 + pad_mm, y1 + pad_mm],
            [x0 - pad_mm, y1 + pad_mm],
            [cx_mm, cy_mm],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    h, w = pixels.shape[:2]
    h_inv = np.linalg.inv(np.asarray(calibration.H_img_to_mm, dtype=np.float64))
    px = cv2.perspectiveTransform(corners_mm, h_inv).reshape(-1, 2)
    bx = np.clip(px[:4, 0], 0, w - 1)
    by = np.clip(px[:4, 1], 0, h - 1)
    box_px = (float(bx.min()), float(by.min()), float(bx.max()), float(by.max()))
    center_px = (
        float(np.clip(px[4, 0], 0, w - 1)),
        float(np.clip(px[4, 1], 0, h - 1)),
    )
    mask, score = seg_client.segment(
        pixels, points=[center_px], labels=[1], box=box_px
    )
    mask = ((mask > 127) * 255).astype("uint8")
    return mask, [
        f"second view segmented from the first view's tool location "
        f"(SAM {score:.2f})"
    ]


def _auto_mask(pixels, profile, calibration) -> tuple[np.ndarray, list[str]]:
    """Preferred: empty-mat diff → SAM prompts on the GPU segserver.

    SAM answers are validated by physical size; on a bad answer (e.g. it
    grabbed the whole sheet because a prompt sat on a shadow) each prompt is
    retried individually and the best plausibly-sized mask wins.
    Fallback: CPU saliency (unreliable on the mat pattern — warns loudly).
    """
    from gridshot.seg import client as seg_client

    from . import diffseg as diffseg_mod

    warnings: list[str] = []
    ref = mat_mod.load_reference(profile.mat_id)
    if ref is not None and seg_client.available():
        canonical = diffseg_mod.canonical_warp(pixels, calibration, profile.spec)
        binary = diffseg_mod.diff_mask(canonical, ref)
        binary = diffseg_mod.board_region_mask(binary, profile.spec)
        groups = diffseg_mod.prompt_points(binary)
        if groups:
            if len(groups) > 1:
                warnings.append(
                    f"{len(groups) - 1} other changed region(s) on the mat ignored — "
                    "M1 traces the largest only"
                )
            def to_img(pts_canon) -> np.ndarray:
                return diffseg_mod.canonical_to_image_px(
                    np.asarray(pts_canon, dtype=np.float64), calibration
                )

            peaks_img = to_img(groups[0])
            extremes_img = to_img(diffseg_mod.extreme_points(binary))

            box_canon = diffseg_mod.component_box(binary)
            corners = to_img(
                [
                    [box_canon[0], box_canon[1]],
                    [box_canon[2], box_canon[1]],
                    [box_canon[2], box_canon[3]],
                    [box_canon[0], box_canon[3]],
                ]
            )
            bx0, by0 = corners.min(axis=0)
            bx1, by1 = corners.max(axis=0)

            # crop to the region: SAM works at 1024² internally, so a crop
            # multiplies the tool's effective boundary resolution
            h, w = pixels.shape[:2]
            pad = 0.06 * max(bx1 - bx0, by1 - by0) + 40
            cx0, cy0 = int(max(bx0 - pad, 0)), int(max(by0 - pad, 0))
            cx1, cy1 = int(min(bx1 + pad, w)), int(min(by1 + pad, h))
            crop = np.ascontiguousarray(pixels[cy0:cy1, cx0:cx1])
            off = np.array([cx0, cy0], dtype=np.float64)
            box_crop = (bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0)

            attempts: list[dict] = [
                {"box": box_crop},
                {"box": box_crop, "points": [tuple(p) for p in extremes_img - off]},
                {"points": [tuple(p) for p in peaks_img - off]},
            ]
            attempts += [{"points": [tuple(p - off)]} for p in peaks_img]
            best: tuple[float, float, np.ndarray] | None = None
            validation_points = np.vstack([peaks_img, extremes_img])
            for kwargs in attempts:
                crop_mask, score = seg_client.segment(crop, **kwargs)
                mask = np.zeros((h, w), np.uint8)
                mask[cy0:cy1, cx0:cx1] = crop_mask
                area = _mask_area_mm2(mask, calibration)
                if contour_mod.MIN_COMPONENT_AREA_MM2 <= area <= MAX_TOOL_AREA_MM2:
                    # A high SAM self-score can still omit a thin shaft. Reward
                    # candidates that actually cover the independent diff-derived
                    # interior/extreme points instead of accepting the first mask
                    # whose total area merely looks plausible.
                    hits = []
                    for px, py in validation_points:
                        ix, iy = int(round(px)), int(round(py))
                        hits.append(
                            0 <= iy < h and 0 <= ix < w and bool(mask[iy, ix])
                        )
                    coverage = float(np.mean(hits)) if hits else 0.0
                    quality = float(score) + 0.35 * coverage
                    if best is None or quality > best[0]:
                        best = (quality, float(score), mask)
            if best is not None:
                message = f"SAM mask confidence {best[1]:.2f}"
                if best[1] < 0.8:
                    message += " — inspect the outline SVG"
                warnings.append(message)
                return best[2], warnings
            warnings.append(
                "SAM never returned a tool-sized mask from the diff prompts"
            )
        else:
            warnings.append(
                "nothing differs from the empty-mat reference — is the tool on the mat?"
            )

    if seg_client.available():
        # semantic locate → geometric refine: SAM 3 concept detection finds
        # the tool ("what"), SAM 2.1 crop+box sharpens the boundary ("where
        # exactly").  Works with no empty-mat reference at all.
        if ref is None:
            warnings.append(
                f"no empty-mat reference for '{profile.mat_id}' — using concept "
                "detection (a reference improves robustness: `gridshot mat reference`)"
            )
        result = _concept_locate_refine(pixels, calibration, seg_client)
        if result is not None:
            mask, score = result
            message = f"concept-path mask confidence {score:.2f}"
            if score < 0.8:
                message += " — inspect the outline SVG"
            warnings.append(message)
            return mask, warnings
        warnings.append("concept detection found no tool-sized object")
    else:
        warnings.append("segserver unreachable — `docker compose up -d segserver`")

    warnings.append("falling back to CPU saliency — unreliable on the mat pattern")
    from . import segment as segment_mod

    return segment_mod.mask_for_image(pixels), warnings


def _concept_locate_refine(
    pixels: np.ndarray, calibration: Calibration, seg_client, prompt: str = "tool"
) -> tuple[np.ndarray, float] | None:
    """SAM 3 concept instances, best size-valid one refined by SAM 2.1 crop+box."""
    h, w = pixels.shape[:2]
    for concept_mask, concept_score in seg_client.segment_concept(pixels, prompt=prompt)[:3]:
        area = _mask_area_mm2(concept_mask, calibration)
        if not (contour_mod.MIN_COMPONENT_AREA_MM2 <= area <= MAX_TOOL_AREA_MM2):
            continue
        ys, xs = np.nonzero(concept_mask > 127)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 0.06 * max(bx1 - bx0, by1 - by0) + 40
        cx0, cy0 = int(max(bx0 - pad, 0)), int(max(by0 - pad, 0))
        cx1, cy1 = int(min(bx1 + pad, w)), int(min(by1 + pad, h))
        crop = np.ascontiguousarray(pixels[cy0:cy1, cx0:cx1])
        refined, score = seg_client.segment(
            crop, box=(bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0)
        )
        full = np.zeros((h, w), np.uint8)
        full[cy0:cy1, cx0:cx1] = refined
        refined_area = _mask_area_mm2(full, calibration)
        if (
            contour_mod.MIN_COMPONENT_AREA_MM2 <= refined_area <= MAX_TOOL_AREA_MM2
            and _refinement_preserves_concept(concept_mask, full)
        ):
            return full, score
        return concept_mask, concept_score  # refine fragmented/drifted; concept is valid
    return None


def _refinement_preserves_concept(
    concept_mask: np.ndarray,
    refined_mask: np.ndarray,
    min_recall: float = 0.65,
    min_iou: float = 0.45,
) -> bool:
    """Reject a crisp SAM refinement that is only one fragment of the tool.

    Concept masks are coarser but usually cover the complete semantic object.
    A box-only refinement can score well while selecting just a handle, jaw, or
    blade. Area limits alone cannot detect that failure, so require the refined
    mask to retain most of the concept object and overlap it substantially.
    """
    concept = concept_mask > 127
    refined = refined_mask > 127
    concept_area = int(concept.sum())
    refined_area = int(refined.sum())
    if concept_area == 0 or refined_area == 0:
        return False
    intersection = int(np.logical_and(concept, refined).sum())
    union = concept_area + refined_area - intersection
    recall = intersection / concept_area
    iou = intersection / union if union else 0.0
    return recall >= min_recall and iou >= min_iou


def _pick_tool(
    mask: np.ndarray, calibration: Calibration
) -> tuple[Poly, list[str]]:
    warnings: list[str] = []
    components = contour_mod.mask_to_polygons_px(mask)
    candidates: list[Poly] = []
    for exterior, holes in components:
        p = contour_mod.polygon_px_to_mm(exterior, holes, calibration)
        area = contour_mod.to_shapely(p).area
        if area < contour_mod.MIN_COMPONENT_AREA_MM2:
            continue
        if area > MAX_TOOL_AREA_MM2:
            warnings.append(
                f"skipped a {area / 100:.0f}cm² region — segmentation likely grabbed "
                "the sheet, not the tool"
            )
            continue
        candidates.append(p)
    if not candidates:
        raise contour_mod.NoToolFoundError(
            "no tool-sized region in the segmentation mask"
        )
    candidates.sort(key=lambda p: -contour_mod.to_shapely(p).area)
    if len(candidates) > 1:
        warnings.append(
            f"{len(candidates) - 1} other object(s) in frame ignored — "
            "M1 traces the largest only"
        )
    return contour_mod.clean(candidates[0]), warnings


def capture_artifacts(
    photo: Path,
    profile: MatProfile,
    smooth_mm: float | None,
    mask: np.ndarray | None,
    prompt_box_mm: tuple[float, float, float, float] | None = None,
) -> CaptureArtifacts:
    """Photo -> calibration, rectified pixels, accepted mask, and outlines.

    smooth_mm=None self-calibrates the smoothing radius from the measured
    boundary-noise amplitude of THIS image (≈3× the 75th-pct jitter, clamped),
    so the filter tracks the data instead of a fixed constant.

    prompt_box_mm is the tool's known mat-mm footprint from another calibrated
    view (same mat, same tool position). When given and no explicit mask is
    supplied, the tool is segmented by projecting that box into this image and
    prompting SAM there — far more reliable than a blind auto-segment, which on
    a second view often locks onto tape or the mat border."""
    src = ingest_mod.load(photo)
    prepared = devices_mod.prepare_image(src)
    device = prepared.profile
    calibration = calibrate_mod.calibrate_image(
        prepared.pixels,
        profile,
        K=prepared.K,
        dist=None,
        exif=src.exif,
        device_profile_id=device.device_id if device else None,
        device_profile_revision=device.revision if device else None,
        capture_signature=prepared.signature,
        intrinsics_source="profile" if device else None,
    )
    pixels = prepared.pixels
    warnings = [*calibration.warnings, *prepared.warnings]
    if device is None:
        warnings.append(
            "no device profile for this camera — run `gridshot calib intrinsics` "
            "for distortion-corrected traces"
        )
    if mask is None and prompt_box_mm is not None:
        mask, seg_warnings = _prompt_mask(pixels, calibration, prompt_box_mm)
        warnings += seg_warnings
        if mask is None:  # segserver offline → fall back to blind auto-segment
            mask, seg_warnings = _auto_mask(pixels, profile, calibration)
            warnings += seg_warnings
    if mask is None:
        mask, seg_warnings = _auto_mask(pixels, profile, calibration)
        warnings += seg_warnings
    raw_poly, pick_warnings = _pick_tool(mask, calibration)
    warnings += pick_warnings
    if smooth_mm is None:
        from . import quality as quality_mod

        outline, cleanup = quality_mod.bounded_cleanup(raw_poly)
        if cleanup["available"]:
            warnings.append(
                "auto smoothing radius "
                f"{cleanup['radius_mm']:.2f}mm (noise {cleanup['noise_mm']:.2f}mm, "
                f"max shift {cleanup['max_shift_mm']:.2f}mm / "
                f"{cleanup['max_shift_cap_mm']:.2f}mm cap)"
            )
        else:
            warnings.append(f"auto smoothing skipped: {cleanup['reason']}")
    else:
        # A numeric radius is an explicit caller override. Interactive editors
        # pass zero because their accepted raw/cleaned candidate is supplied as
        # authoritative geometry downstream.
        outline = (
            raw_poly
            if smooth_mm <= 0
            else contour_mod.straighten(contour_mod.smooth(raw_poly, smooth_mm))
        )
    readiness = readiness_mod.evaluate(
        calibration=calibration,
        warnings=warnings,
        outline=outline,
        require_thickness=False,
    )
    return CaptureArtifacts(
        calibration=calibration,
        pixels=pixels,
        mask=((mask > 127) * 255).astype(np.uint8),
        raw_poly=raw_poly,
        outline=outline,
        warnings=warnings,
        readiness=readiness,
    )


def _capture(
    photo: Path,
    profile: MatProfile,
    smooth_mm: float | None,
    mask: np.ndarray | None,
    prompt_box_mm: tuple[float, float, float, float] | None = None,
) -> tuple[Calibration, Poly, list[str]]:
    """Compatibility geometry view over :func:`capture_artifacts`."""
    captured = capture_artifacts(photo, profile, smooth_mm, mask, prompt_box_mm)
    return captured.calibration, captured.outline, captured.warnings


def capture_tool_geometry(
    photo: Path,
    profile: MatProfile,
    thickness_mm: float | None = None,
    photo2: Path | None = None,
    smooth_mm: float | None = 0.6,
    mask: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    outline_override: Poly | None = None,
    outline_override_warnings: list[str] | None = None,
) -> CapturedToolGeometry:
    """Resolve one accepted tool mask and, when needed, two-view thickness.

    This is the geometry-only portion of :func:`run`: it deliberately stops
    before clearance, bin construction, and file export so an edited selection
    can be saved to the library without first generating a bin.
    """
    first_smooth_mm = 0.0 if outline_override is not None else smooth_mm
    calibration, smoothed, warnings = _capture(
        photo, profile, first_smooth_mm, mask
    )
    if outline_override is not None:
        smoothed = outline_override
        warnings += list(outline_override_warnings or [])
    reconstruction: dict[str, float | str] | None = None
    corrected: Poly | None = None

    if thickness_mm is None:
        if photo2 is None:
            raise ValueError(
                "pass --thickness, or a second photo for automatic thickness"
            )
        if (
            calibration.device_profile_id is None
            or calibration.intrinsics_source not in {None, "profile"}
        ):
            raise ValueError(
                "automatic thickness requires a calibrated device profile; "
                "calibrate this capture setup or enter measured thickness"
            )
        # Steer photo2's segmentation to the SAME tool using its footprint from
        # photo1 (the tool doesn't move between shots), so photo2 doesn't blindly
        # auto-segment and lock onto tape/mat — the top cause of a failed solve.
        ex = np.asarray(smoothed.exterior, dtype=np.float64)
        box_mm = (
            float(ex[:, 0].min()), float(ex[:, 1].min()),
            float(ex[:, 0].max()), float(ex[:, 1].max()),
        )
        secondary_smooth_mm = (
            None if outline_override is not None else smooth_mm
        )
        cal2, smoothed2, w2 = _capture(
            photo2, profile, secondary_smooth_mm, mask2, prompt_box_mm=box_mm
        )
        if (
            cal2.device_profile_id is None
            or cal2.intrinsics_source not in {None, "profile"}
        ):
            raise ValueError(
                "automatic thickness requires calibrated device profiles for "
                "both views; calibrate the second capture setup or enter "
                "measured thickness"
            )
        warnings += w2

        d_nadir = math.hypot(
            calibration.nadir_xy_mm[0] - cal2.nadir_xy_mm[0],
            calibration.nadir_xy_mm[1] - cal2.nadir_xy_mm[1],
        )
        d_h = abs(calibration.camera_height_mm - cal2.camera_height_mm)
        if d_nadir < 30 and d_h < 40:
            raise ValueError(
                f"the two photos are from nearly the same camera position "
                f"(nadir Δ{d_nadir:.0f}mm, height Δ{d_h:.0f}mm), so thickness "
                "can't be triangulated. Move the camera 15–25cm to the side "
                "between the two shots (keep the whole mat in frame), or enter "
                "the tool's widest-outline height as the thickness."
            )
        thickness_mm, residual = parallax_mod.solve_thickness(
            smoothed, calibration, smoothed2, cal2
        )
        ceiling = parallax_mod.thickness_ceiling(calibration, cal2)
        if thickness_mm >= ceiling - 1.0:
            raise ValueError(
                f"two-photo thickness solve hit its {ceiling:.0f}mm ceiling "
                f"(outline agreement poor, residual {residual:.0f}mm²). The two "
                "photos likely don't show the same tool in the same spot — "
                "reshoot both angles without moving the tool, or enter the "
                "tool's widest-outline height as the thickness."
            )
        warnings.append(
            f"auto thickness: {thickness_mm:.1f}mm from two views "
            f"(outline agreement residual {residual:.1f}mm²)"
        )
        try:
            local = parallax_mod.reconstruct_footprint(
                smoothed,
                calibration,
                smoothed2,
                cal2,
                scalar_height_mm=thickness_mm,
                scalar_residual_mm2=residual,
            )
            corrected = local.polygon
            reconstruction = local.diagnostics()
            warnings.append(
                "local footprint: two-view silhouette reconstruction "
                f"{local.reconstructed_major_extent_mm:.2f} × "
                f"{local.reconstructed_minor_extent_mm:.2f}mm "
                f"(boundary p95 {local.boundary_p95_error_mm:.2f}mm)"
            )
        except parallax_mod.LocalReconstructionError as exc:
            warnings.append(
                "local footprint fallback: using one-height parallax because "
                f"{exc}"
            )

    if corrected is None:
        corrected = parallax_mod.correct_polygon(
            smoothed, calibration, float(thickness_mm)
        )
    return CapturedToolGeometry(
        calibration=calibration,
        raw_poly=smoothed,
        corrected_poly=corrected,
        thickness_mm=float(thickness_mm),
        warnings=warnings,
        reconstruction=reconstruction,
    )


def capture_tool_outline(
    photo: Path,
    profile: MatProfile,
    thickness_mm: float | None = None,
    photo2: Path | None = None,
    smooth_mm: float = 0.6,
    mask: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
) -> tuple[Calibration, Poly, float, list[str]]:
    """Compatibility wrapper returning the accepted plane-mapped outline."""
    captured = capture_tool_geometry(
        photo,
        profile,
        thickness_mm=thickness_mm,
        photo2=photo2,
        smooth_mm=smooth_mm,
        mask=mask,
        mask2=mask2,
    )
    return (
        captured.calibration,
        captured.raw_poly,
        captured.thickness_mm,
        captured.warnings,
    )


def run(
    photo: Path,
    thickness_mm: float | None = None,
    photo2: Path | None = None,
    clearance_mm: float = 1.0,
    smooth_mm: float = 0.6,
    bin_style: grid_mod.BinStyle = "pocket",
    pocket_depth_mm: float | None = None,
    full_height_mm: float | None = None,
    height_u: int | None = None,
    overall_height_mm: float | None = None,
    lip: bool = True,
    finger_hole: bool = False,
    mat_id: str | None = None,
    out_dir: Path = Path("out"),
    stem: str | None = None,
    mask: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    round_tool: bool = False,
    corrected_override: Poly | None = None,
    outline_override: Poly | None = None,
    outline_override_warnings: list[str] | None = None,
    reconstruction_override: dict[str, float | str] | None = None,
) -> TraceResult:
    if mat_id is None:
        verified = [p for p in mat_mod.list_profiles() if p.verified]
        if len(verified) != 1:
            raise RuntimeError(
                f"{len(verified)} verified mats registered — pass --mat explicitly"
            )
        profile: MatProfile = verified[0]
    else:
        profile = mat_mod.load_profile(mat_id)

    thickness_source = "automatic" if thickness_mm is None else "manual"
    captured = capture_tool_geometry(
        photo,
        profile,
        thickness_mm=thickness_mm,
        photo2=photo2,
        smooth_mm=smooth_mm,
        mask=mask,
        mask2=mask2,
        outline_override=outline_override,
        outline_override_warnings=outline_override_warnings,
    )
    corrected = corrected_override or captured.corrected_poly
    warnings = list(captured.warnings)
    reconstruction = captured.reconstruction
    if corrected_override is not None:
        warnings.append(
            "physical cutout override: using the manually edited physical "
            "footprint without reapplying parallax"
        )
        reconstruction = reconstruction_override
    readiness = readiness_mod.evaluate(
        calibration=captured.calibration,
        warnings=warnings,
        outline=corrected,
        thickness_mm=captured.thickness_mm,
        thickness_source=thickness_source,
    )
    if readiness.blocked:
        raise ValueError(f"not ready: {readiness_mod.blocking_message(readiness)}")
    return finalize_bin(
        captured.raw_poly,
        captured.calibration,
        captured.thickness_mm,
        clearance_mm=clearance_mm,
        pocket_depth_mm=pocket_depth_mm,
        full_height_mm=full_height_mm,
        height_u=height_u,
        overall_height_mm=overall_height_mm,
        lip=lip,
        finger_hole=finger_hole,
        out_dir=out_dir,
        stem=stem or f"{photo.stem}-bin",
        warnings=warnings,
        corrected_override=corrected,
        reconstruction=reconstruction,
        round_tool=round_tool,
        bin_style=bin_style,
        readiness=readiness, thickness_source=thickness_source,
    )


def finalize_bin(
    smoothed: Poly,
    calibration: Calibration | None,
    thickness_mm: float,
    clearance_mm: float = 1.0,
    bin_style: grid_mod.BinStyle = "pocket",
    pocket_depth_mm: float | None = None,
    full_height_mm: float | None = None,
    height_u: int | None = None,
    overall_height_mm: float | None = None,
    lip: bool = True,
    finger_hole: bool = False,
    out_dir: Path = Path("out"),
    stem: str = "bin",
    warnings: list[str] | None = None,
    pre_corrected: bool = False,
    corrected_override: Poly | None = None,
    reconstruction: dict[str, float | str] | None = None,
    round_tool: bool = False,
    readiness: readiness_mod.ReadinessReport | None = None,
    thickness_source: str = "unknown",
    printer_profile: PrinterProfile | None = None,
) -> TraceResult:
    """Tool outline (mat-mm) + thickness → generated gridfinity bin + exports.

    The bin-building tail shared by single-tool `run` and saved-library
    generation: parallax is applied from thickness, then clearance, printer
    compensation, CAD-frame flip, alignment, finger hole, sizing, and export.
    `smoothed` is the plane-mapped outline; `calibration` supplies the nadir and
    height the parallax correction uses. `pre_corrected=True` takes the outline
    as already parallax-corrected (calibration unused) — for regenerating a bin
    from a saved library outline.
    """
    warnings = warnings if warnings is not None else []
    initial_readiness = readiness or readiness_mod.evaluate(
        calibration=calibration,
        warnings=warnings,
        outline=smoothed,
        thickness_mm=thickness_mm,
        thickness_source=thickness_source,
        require_calibration=not pre_corrected,
    )
    if initial_readiness.blocked:
        raise ValueError(
            f"not ready: {readiness_mod.blocking_message(initial_readiness)}"
        )
    corrected = (
        corrected_override
        if corrected_override is not None
        else smoothed
        if pre_corrected
        else parallax_mod.correct_polygon(smoothed, calibration, thickness_mm)
    )
    from . import bench as bench_mod

    printer = printer_profile or bench_mod.load_profile() or bench_mod.default_profile()
    spec = derive_mod.derive_bin_spec(
        derive_mod.ToolGeometry(
            outline=corrected,
            silhouette_height_mm=thickness_mm,
            full_height_mm=full_height_mm,
        ),
        derive_mod.BinSettings(
            clearance_mm=clearance_mm,
            bin_style=bin_style,
            pocket_depth_mm=pocket_depth_mm,
            height_u=height_u,
            overall_height_mm=overall_height_mm,
            lip=lip,
            finger_hole=finger_hole,
            round_tool=round_tool,
        ),
        printer,
    )
    warnings.extend(spec.warnings)
    pocket = spec.pocket_poly
    tool_bin = spec.tool_poly
    gx, gy = spec.grid
    height_u = spec.height_u
    depth = spec.pocket_depth_mm
    fingers = spec.finger_holes

    solid = grid_mod.bin_solid(
        gx,
        gy,
        height_u,
        pocket=pocket,
        pocket_depth=depth,
        finger_holes=fingers,
        lip=spec.lip,
        style=spec.bin_style,
    )
    mesh = grid_mod.to_trimesh(solid)
    if not mesh.is_watertight:
        raise RuntimeError("generated mesh is not watertight — geometry bug, not printable")

    retention_label = (
        "pocket" if spec.bin_style == "pocket" else f"{spec.bin_style} recess"
    )
    svg = export_mod.debug_svg(
        [
            ("raw outline (plane-mapped)", "#888888", smoothed),
            ("parallax-corrected", "#d62728", corrected),
            (
                f"{retention_label} (+{clearance_mm}mm clearance)",
                "#1f77b4",
                spec.compensated_poly,
            ),
        ]
    )
    layout = export_mod.layout_svg(
        gx, gy, height_u, pocket, tool_bin, fingers, clearance_mm
    )
    files = export_mod.write_all(out_dir, stem, mesh, svg=svg, layout=layout)

    runtime_readiness = readiness_mod.evaluate(
        calibration=calibration,
        warnings=warnings,
        outline=smoothed,
        thickness_mm=thickness_mm,
        thickness_source=thickness_source,
        require_calibration=not pre_corrected,
    )
    final_readiness = readiness_mod.combine(initial_readiness, runtime_readiness)

    return TraceResult(
        calibration=calibration,
        raw_poly=smoothed,
        corrected_poly=corrected,
        pocket_poly=pocket,
        grid=(gx, gy),
        height_u=height_u,
        pocket_depth_mm=depth,
        bin_style=spec.bin_style,
        overall_height_mm=spec.overall_height_mm,
        lip=spec.lip,
        tool_poly=tool_bin,
        thickness_mm=thickness_mm,
        silhouette_height_mm=thickness_mm,
        full_height_mm=full_height_mm,
        files=files,
        warnings=warnings,
        readiness=final_readiness,
        thickness_source=thickness_source,
        derivation_key=spec.derivation_key,
        reconstruction=reconstruction,
        reserved_cells=spec.reserved_cells,
        available_cells=spec.available_cells,
    )
