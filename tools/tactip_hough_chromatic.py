#!/usr/bin/env python3
"""Reusable Hough-boundary and chromatic TacTip marker preprocessing.

Marker centres and radii are measured from circular image boundaries.  A
blue-yellow opponent score rejects circular gold/glass highlights without
using the brightest pixel.  A frame is accepted only when the requested
number of markers remains separated before and after 256px conversion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import cv2
import numpy as np


EXPECTED_MARKERS = 331
REFERENCE_MIN_DIMENSION_PX = 960.0


class MarkerDetectionError(RuntimeError):
    """Raised when a frame cannot produce a validated marker binary."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class HoughChromaticConfig:
    expected_markers: int = EXPECTED_MARKERS
    reference_min_dimension_px: float = REFERENCE_MIN_DIMENSION_PX
    hough_dp: float = 1.0
    minimum_centre_distance_px: float = 14.0
    canny_high_threshold: float = 80.0
    hough_vote_threshold: float = 13.0
    hough_vote_search_radius: int = 3
    minimum_radius_px: float = 4.0
    maximum_radius_px: float = 11.0
    minimum_blue_yellow_score: float = 20.0
    render_radius_scale: float = 0.90
    render_radius_offset_px: float = 0.0
    roi_padding_px: float = 24.0
    output_size: int = 256
    downsample_threshold: int = 80

    def validate(self) -> None:
        if self.expected_markers <= 0:
            raise ValueError("expected_markers must be positive")
        if self.reference_min_dimension_px <= 0.0:
            raise ValueError("reference_min_dimension_px must be positive")
        if self.minimum_centre_distance_px <= 0.0:
            raise ValueError("minimum_centre_distance_px must be positive")
        if self.minimum_radius_px <= 0.0 or self.maximum_radius_px <= self.minimum_radius_px:
            raise ValueError("marker radius limits are invalid")
        if not 0.3 <= self.render_radius_scale <= 1.2:
            raise ValueError("render_radius_scale must be between 0.3 and 1.2")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if not 0 <= self.downsample_threshold <= 255:
            raise ValueError("downsample_threshold must be in [0, 255]")


def config_dict(config: HoughChromaticConfig) -> dict[str, Any]:
    return asdict(config)


def blue_yellow_opponent(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image.astype(np.float32))
    return 2.0 * blue - green - red


def component_metrics(binary: np.ndarray) -> dict[str, Any]:
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if count > 1 else np.asarray([])
    return {
        "components": int(count - 1),
        "unique_values": [int(value) for value in np.unique(binary)],
        "area_min": float(np.min(areas)) if len(areas) else 0.0,
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "area_max": float(np.max(areas)) if len(areas) else 0.0,
    }


def _scaled_geometry(image: np.ndarray, config: HoughChromaticConfig) -> dict[str, Any]:
    height, width = image.shape[:2]
    scale = float(min(height, width)) / float(config.reference_min_dimension_px)
    blur_size = max(3, int(round(5.0 * scale)))
    if blur_size % 2 == 0:
        blur_size += 1
    return {
        "scale": scale,
        "blur_size": blur_size,
        "minimum_centre_distance_px": max(6.0, config.minimum_centre_distance_px * scale),
        "minimum_radius_px": max(2, int(round(config.minimum_radius_px * scale))),
        "maximum_radius_px": max(4, int(round(config.maximum_radius_px * scale))),
        "roi_padding_px": max(4, int(round(config.roi_padding_px * scale))),
    }


def _colour_score(opponent: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = opponent.shape
    core_radius = max(3.0, float(radius) * 0.55)
    x0 = max(0, int(np.floor(x - core_radius)))
    x1 = min(width, int(np.ceil(x + core_radius)) + 1)
    y0 = max(0, int(np.floor(y - core_radius)))
    y1 = min(height, int(np.ceil(y + core_radius)) + 1)
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
    core = (x0 + xx - float(x)) ** 2 + (y0 + yy - float(y)) ** 2 <= core_radius**2
    return float(np.median(opponent[y0:y1, x0:x1][core]))


def _vote_thresholds(config: HoughChromaticConfig) -> list[float]:
    base = float(config.hough_vote_threshold)
    values = [base]
    for delta in range(1, int(config.hough_vote_search_radius) + 1):
        values.extend((base - float(delta), base + float(delta)))
    return [value for value in values if value > 0.0]


def _detect_for_vote(
    gray: np.ndarray,
    opponent: np.ndarray,
    config: HoughChromaticConfig,
    geometry: dict[str, Any],
    vote_threshold: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]], int]:
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        float(config.hough_dp),
        float(geometry["minimum_centre_distance_px"]),
        param1=float(config.canny_high_threshold),
        param2=float(vote_threshold),
        minRadius=int(geometry["minimum_radius_px"]),
        maxRadius=int(geometry["maximum_radius_px"]),
    )
    accepted: list[dict[str, float]] = []
    rejected: list[dict[str, float]] = []
    if circles is None:
        return accepted, rejected, 0
    for x, y, radius in circles[0]:
        score = _colour_score(opponent, float(x), float(y), float(radius))
        record = {
            "x_px": float(x),
            "y_px": float(y),
            "hough_radius_px": float(radius),
            "blue_yellow_score": score,
        }
        target = accepted if score > float(config.minimum_blue_yellow_score) else rejected
        target.append(record)
    accepted.sort(key=lambda record: (record["y_px"], record["x_px"]))
    rejected.sort(key=lambda record: (record["y_px"], record["x_px"]))
    return accepted, rejected, int(circles.shape[1])


def detect_marker_circles(
    image: np.ndarray,
    config: HoughChromaticConfig,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    config.validate()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected one BGR colour image")
    geometry = _scaled_geometry(image, config)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (int(geometry["blur_size"]), int(geometry["blur_size"])),
        max(0.5, float(geometry["scale"])),
    )
    opponent = blue_yellow_opponent(image)
    trials: list[dict[str, Any]] = []
    best: tuple[list[dict[str, float]], list[dict[str, float]], float] | None = None
    for vote_threshold in _vote_thresholds(config):
        accepted, rejected, total = _detect_for_vote(
            gray, opponent, config, geometry, vote_threshold
        )
        trials.append(
            {
                "vote_threshold": vote_threshold,
                "total_circles": total,
                "accepted_markers": len(accepted),
                "rejected_glare": len(rejected),
            }
        )
        if best is None or abs(len(accepted) - config.expected_markers) < abs(
            len(best[0]) - config.expected_markers
        ):
            best = accepted, rejected, vote_threshold
        if len(accepted) == config.expected_markers:
            best = accepted, rejected, vote_threshold
            break
    assert best is not None
    accepted, rejected, selected_vote = best
    diagnostics = {
        "scaled_geometry": geometry,
        "selected_vote_threshold": selected_vote,
        "trials": trials,
        "accepted_markers": len(accepted),
        "rejected_glare": len(rejected),
    }
    if len(accepted) != config.expected_markers:
        raise MarkerDetectionError(
            "Detected {} markers; expected {}".format(len(accepted), config.expected_markers),
            diagnostics,
        )
    return accepted, rejected, diagnostics


def _strong_glare_fallback_configs(
    config: HoughChromaticConfig,
) -> list[tuple[str, HoughChromaticConfig]]:
    """Return marker-sized circle searches used only after primary failure.

    At 640x480 the 331-pin sensor markers occupy roughly four to seven image
    pixels in radius.  Strong glass glare can create smaller Hough circles or
    suppress shallow marker edges.  This fallback raises the lower radius
    bound and widens the upper bound, but keeps the caller's marker count and
    chromatic glare criterion intact.  It is deliberately geometry-only: no
    saved marker positions or image template are consulted.
    """
    proposals = [
        (
            "wide_4_to_7px",
            replace(
                config,
                hough_vote_threshold=10.0,
                minimum_radius_px=7.0,
                maximum_radius_px=14.0,
                canny_high_threshold=80.0,
            ),
        ),
        (
            "tight_3_to_6px",
            replace(
                config,
                hough_vote_threshold=10.0,
                minimum_radius_px=6.0,
                maximum_radius_px=12.0,
                canny_high_threshold=80.0,
            ),
        ),
        (
            "wide_high_canny",
            replace(
                config,
                hough_vote_threshold=10.0,
                minimum_radius_px=7.0,
                maximum_radius_px=16.0,
                canny_high_threshold=95.0,
            ),
        ),
        (
            "wide_3_to_8px",
            replace(
                config,
                # Some 640x480 frames enlarge otherwise valid marker rims by
                # roughly one pixel.  This remains a boundary/radius search;
                # it does not infer missing locations from a saved layout.
                hough_vote_threshold=11.0,
                hough_vote_search_radius=0,
                minimum_radius_px=6.0,
                maximum_radius_px=16.0,
                canny_high_threshold=80.0,
            ),
        ),
    ]
    unique: list[tuple[str, HoughChromaticConfig]] = []
    seen: set[HoughChromaticConfig] = set()
    for name, proposal in proposals:
        if proposal != config and proposal not in seen:
            unique.append((name, proposal))
            seen.add(proposal)
    return unique


def _detect_marker_circles_with_glare_fallback(
    image: np.ndarray,
    config: HoughChromaticConfig,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any], HoughChromaticConfig]:
    """Run the normal detector, then a deterministic geometry fallback.

    The fallback is only eligible when the normal configuration does not
    yield the exact required count.  Frames that already pass retain their
    original detector settings and outputs.
    """
    try:
        accepted, rejected, diagnostics = detect_marker_circles(image, config)
        diagnostics["detector_mode"] = "primary"
        return accepted, rejected, diagnostics, config
    except MarkerDetectionError as primary_error:
        fallback_attempts: list[dict[str, Any]] = []
        for name, fallback in _strong_glare_fallback_configs(config):
            try:
                accepted, rejected, diagnostics = detect_marker_circles(image, fallback)
            except MarkerDetectionError as fallback_error:
                fallback_attempts.append(
                    {
                        "name": name,
                        "configuration": config_dict(fallback),
                        "diagnostics": fallback_error.diagnostics,
                    }
                )
                continue
            diagnostics["detector_mode"] = "strong_glare_geometry_fallback:{}".format(name)
            diagnostics["primary_diagnostics"] = primary_error.diagnostics
            diagnostics["primary_configuration"] = config_dict(config)
            diagnostics["prior_glare_fallback_attempts"] = fallback_attempts
            return accepted, rejected, diagnostics, fallback
        primary_diagnostics = dict(primary_error.diagnostics)
        primary_diagnostics["detector_mode"] = "primary_and_glare_fallback_failed"
        primary_diagnostics["glare_fallback_attempts"] = fallback_attempts
        raise MarkerDetectionError(
            "Primary and strong-glare Hough searches did not produce the required marker count",
            primary_diagnostics,
        ) from primary_error


def _crop_bounds(points: np.ndarray, shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    height, width = shape
    left = max(0, int(np.floor(np.min(points[:, 0]))) - padding)
    right = min(width, int(np.ceil(np.max(points[:, 0]))) + padding + 1)
    top = max(0, int(np.floor(np.min(points[:, 1]))) - padding)
    bottom = min(height, int(np.ceil(np.max(points[:, 1]))) + padding + 1)
    return left, top, right, bottom


def _draw_overlay(
    image: np.ndarray,
    accepted: list[dict[str, float]],
    rejected: list[dict[str, float]],
) -> np.ndarray:
    overlay = image.copy()
    for record in rejected:
        centre = (int(round(record["x_px"])), int(round(record["y_px"])))
        cv2.circle(
            overlay,
            centre,
            int(round(record["hough_radius_px"])),
            (40, 40, 245),
            2,
            cv2.LINE_AA,
        )
    for record in accepted:
        centre = (int(round(record["x_px"])), int(round(record["y_px"])))
        cv2.circle(
            overlay,
            centre,
            int(round(record["hough_radius_px"])),
            (30, 245, 70),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _render_marker_binary(
    accepted: list[dict[str, float]],
    image_shape: tuple[int, ...],
    config: HoughChromaticConfig,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    """Draw only the measured marker circles in native camera coordinates."""
    binary = np.zeros(image_shape[:2], dtype=np.uint8)
    render_radii: list[float] = []
    for marker in accepted:
        radius = max(
            2.0,
            marker["hough_radius_px"] * config.render_radius_scale
            + config.render_radius_offset_px * float(geometry["scale"]),
        )
        render_radii.append(radius)
        cv2.circle(
            binary,
            (int(round(marker["x_px"])), int(round(marker["y_px"]))),
            int(round(radius)),
            255,
            -1,
            cv2.LINE_8,
        )
    return binary, render_radii


def process_frame(
    image: np.ndarray,
    config: HoughChromaticConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, float]]]:
    config = config or HoughChromaticConfig()
    accepted, rejected, diagnostics, active_config = _detect_marker_circles_with_glare_fallback(
        image, config
    )
    geometry = diagnostics["scaled_geometry"]
    points = np.asarray([[record["x_px"], record["y_px"]] for record in accepted])
    left, top, right, bottom = _crop_bounds(
        points, image.shape[:2], int(geometry["roi_padding_px"])
    )
    requested_scale = float(active_config.render_radius_scale)
    candidate_scales: list[float] = []
    for scale in (requested_scale, 0.85, 0.80, 0.75, 0.70, 0.65):
        if scale <= requested_scale and scale not in candidate_scales:
            candidate_scales.append(scale)
    render_attempts: list[dict[str, Any]] = []
    selected: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[float],
        dict[str, Any],
        dict[str, Any],
        HoughChromaticConfig,
    ] | None = None
    for scale in candidate_scales:
        trial_config = replace(active_config, render_radius_scale=scale)
        binary, render_radii = _render_marker_binary(accepted, image.shape, trial_config, geometry)
        binary_roi = binary[top:bottom, left:right]
        soft_output = cv2.resize(
            binary_roi,
            (trial_config.output_size, trial_config.output_size),
            interpolation=cv2.INTER_AREA,
        )
        binary_output = np.where(
            soft_output >= int(trial_config.downsample_threshold), 255, 0
        ).astype(np.uint8)
        source_metrics = component_metrics(binary_roi)
        output_metrics = component_metrics(binary_output)
        render_attempts.append(
            {
                "render_radius_scale": scale,
                "source_components": source_metrics["components"],
                "output_components": output_metrics["components"],
            }
        )
        if (
            source_metrics["components"] == trial_config.expected_markers
            and output_metrics["components"] == trial_config.expected_markers
        ):
            selected = (
                binary,
                binary_roi,
                binary_output,
                render_radii,
                source_metrics,
                output_metrics,
                trial_config,
            )
            break
    if selected is None:
        diagnostics.update(
            {
                "crop_bounds_xyxy": [left, top, right, bottom],
                "render_radius_scale_attempts": render_attempts,
            }
        )
        raise MarkerDetectionError(
            "Marker circles merged or disappeared during binary rendering", diagnostics
        )
    (
        binary,
        binary_roi,
        binary_output,
        render_radii,
        source_metrics,
        output_metrics,
        active_config,
    ) = selected
    diagnostics.update(
        {
            "effective_configuration": config_dict(active_config),
            "crop_bounds_xyxy": [left, top, right, bottom],
            "render_radius_scale_attempts": render_attempts,
            "source_metrics": source_metrics,
            "output_metrics": output_metrics,
        }
    )
    overlay = _draw_overlay(image, accepted, rejected)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    opponent = blue_yellow_opponent(image)
    opponent_u8 = cv2.normalize(opponent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for marker_id, (record, render_radius) in enumerate(
        zip(accepted, render_radii), start=1
    ):
        record["marker_id"] = marker_id
        record["render_radius_px"] = float(render_radius)
    record = {
        "marker_count": len(accepted),
        "rejected_glare_count": len(rejected),
        "selected_vote_threshold": diagnostics["selected_vote_threshold"],
        "detector_mode": diagnostics["detector_mode"],
        "effective_configuration": diagnostics["effective_configuration"],
        "crop_bounds_xyxy": diagnostics["crop_bounds_xyxy"],
        "source_components": source_metrics["components"],
        "output_components": output_metrics["components"],
        "binary_values": output_metrics["unique_values"],
        "diagnostics": diagnostics,
    }
    images = {
        # Native camera coordinates stay fixed between frames.  The runtime
        # contact detector consumes this map for optical flow, while the
        # cropped images below remain the model-facing outputs.
        "contact_binary": binary,
        "raw_roi": image[top:bottom, left:right],
        "gray_roi": gray[top:bottom, left:right],
        "blue_yellow_score_roi": opponent_u8[top:bottom, left:right],
        "overlay_roi": overlay[top:bottom, left:right],
        "binary_roi": binary_roi,
        "model_input_256": binary_output,
    }
    return record, images, accepted
