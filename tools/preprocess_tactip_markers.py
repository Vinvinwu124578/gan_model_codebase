#!/usr/bin/env python3
"""Suppress TacTip illumination rings and prepare marker-focused grayscale images.

The reference frame is used once to estimate the marker-lattice center, marker
extent, and usable inner radius.  Each frame first segments high-gray outer
pixels with K-means, retains only the large outer connected clusters as the
illumination ring, and suppresses them.  A shared circular crop is then applied
to every image, so marker coordinates stay in a common image frame.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One image or a directory of images.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.png", help="Glob used when --input is a directory.")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="Optional image used to estimate one fixed crop. Defaults to the first input image.",
    )
    parser.add_argument(
        "--center",
        default="",
        help="Optional fixed optical center as x,y pixels. Omitting it enables marker-lattice estimation.",
    )
    parser.add_argument(
        "--crop-radius-px",
        type=float,
        default=0.0,
        help="Optional fixed inner crop radius. Omitting it uses --crop-policy on the reference frame.",
    )
    parser.add_argument(
        "--crop-policy",
        choices=("ring-aware", "marker-padding"),
        default="ring-aware",
        help=(
            "Automatic crop selection when --crop-radius-px is omitted. ring-aware finds the illumination-ring "
            "inner edge from gray clusters; marker-padding uses only the outer marker lattice."
        ),
    )
    parser.add_argument("--search-radius-ratio", type=float, default=0.42)
    parser.add_argument(
        "--crop-padding-px",
        type=float,
        default=8.0,
        help="Minimum margin outside the outer marker lattice used by automatic crop selection.",
    )
    parser.add_argument("--min-marker-area", type=int, default=5)
    parser.add_argument("--max-marker-area", type=int, default=180)
    parser.add_argument("--marker-percentile", type=float, default=96.0)
    parser.add_argument("--background-sigma", type=float, default=5.0)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.5)
    parser.add_argument(
        "--mask-mode",
        choices=("circle", "marker-hull"),
        default="circle",
        help=(
            "circle keeps the full inner disk. marker-hull uses the reference marker lattice's convex hull plus "
            "a fixed margin, which removes outer illumination without shrinking around individual frames."
        ),
    )
    parser.add_argument(
        "--marker-hull-margin-px",
        type=int,
        default=16,
        help="Fixed safety margin outside the reference marker hull when --mask-mode marker-hull.",
    )
    parser.add_argument(
        "--ring-suppression",
        choices=("none", "gray-clusters"),
        default="gray-clusters",
        help=(
            "How to remove the outer illumination before cropping. gray-clusters groups gray values in the "
            "outer annulus and only suppresses its large high-light components."
        ),
    )
    parser.add_argument(
        "--ring-clusters",
        type=int,
        default=4,
        help="Number of gray-value clusters used by --ring-suppression gray-clusters.",
    )
    parser.add_argument(
        "--ring-marker-guard-px",
        type=float,
        default=8.0,
        help="Protected margin outside the outer reference marker radius before illumination-ring detection starts.",
    )
    parser.add_argument(
        "--ring-min-component-area",
        type=int,
        default=100,
        help="Minimum high-gray outer connected-component area classified as illumination ring.",
    )
    parser.add_argument(
        "--ring-dilate-px",
        type=int,
        default=5,
        help="Pixels used to expand detected illumination clusters before inpainting.",
    )
    parser.add_argument(
        "--ring-inpaint-radius",
        type=float,
        default=5.0,
        help="OpenCV inpainting radius for detected outer illumination pixels.",
    )
    parser.add_argument(
        "--ring-crop-percentile",
        type=float,
        default=5.0,
        help="Inner radial percentile of the detected illumination mask used by --crop-policy ring-aware.",
    )
    parser.add_argument(
        "--ring-crop-safety-px",
        type=float,
        default=2.0,
        help="Pixels kept inside the detected illumination ring by --crop-policy ring-aware.",
    )
    parser.add_argument(
        "--outer-rim-treatment",
        choices=("none", "cosine-fade", "gray-cap", "roi-mask", "flat-field"),
        default="none",
        help="Optional final-stage suppression of residual outer-rim illumination without changing crop geometry.",
    )
    parser.add_argument(
        "--model-input-mask",
        choices=("none", "marker-neighborhood", "tracked-marker-components", "outer-ring-protected"),
        default="outer-ring-protected",
        help=(
            "Support used only for the downstream model image. outer-ring-protected keeps the full "
            "inner response and protects the coherent outer marker lattice while removing only the external ring. "
            "tracked-marker-components follows the largest coherent lattice, while marker-neighborhood "
            "uses fixed reference safety disks."
        ),
    )
    parser.add_argument(
        "--model-marker-neighborhood-radius-px",
        type=float,
        default=20.0,
        help=(
            "Safety-disk radius around each reference marker for --model-input-mask marker-neighborhood. "
            "The default preserves marker motion while excluding the exterior illumination ring."
        ),
    )
    parser.add_argument(
        "--model-component-neighbor-min-px",
        type=float,
        default=6.0,
        help="Minimum marker-center separation used to build the tracked marker-lattice graph.",
    )
    parser.add_argument(
        "--model-component-neighbor-max-px",
        type=float,
        default=30.0,
        help="Maximum marker-center separation used to build the tracked marker-lattice graph.",
    )
    parser.add_argument(
        "--model-component-dilate-px",
        type=int,
        default=6,
        help="Safety dilation around the tracked marker components in each final model input.",
    )
    parser.add_argument(
        "--model-outer-ring-start-margin-px",
        type=float,
        default=3.5,
        help=(
            "For --model-input-mask outer-ring-protected, begin suppressing non-marker response this "
            "many pixels beyond the reference outer marker radius."
        ),
    )
    parser.add_argument(
        "--model-outer-marker-protection-dilate-px",
        type=int,
        default=6,
        help="Safety dilation around every detected marker preserved in the outer-ring suppression zone.",
    )
    parser.add_argument(
        "--outer-rim-fade-start-margin-px",
        type=float,
        default=17.0,
        help="Start fading this many pixels beyond the reference outer marker radius.",
    )
    parser.add_argument(
        "--outer-rim-fade-end-margin-px",
        type=float,
        default=1.0,
        help="Finish fading this many pixels inside the circular crop boundary.",
    )
    parser.add_argument(
        "--outer-rim-cap-percentile",
        type=float,
        default=90.0,
        help="Gray percentile in the clean inner background band used as the outer-rim brightness cap.",
    )
    parser.add_argument(
        "--outer-rim-roi-margin-px",
        type=float,
        default=30.0,
        help="Extra valid radius beyond the outer marker lattice for --outer-rim-treatment roi-mask.",
    )
    parser.add_argument(
        "--flat-field-sigma",
        type=float,
        default=20.0,
        help="Gaussian sigma for the batch-median illumination template used by --outer-rim-treatment flat-field.",
    )
    parser.add_argument(
        "--flat-field-level",
        type=float,
        default=82.0,
        help="Target gray level after subtracting the flat-field illumination template.",
    )
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--preview-limit", type=int, default=40)
    return parser.parse_args()


def parse_center(text: str) -> np.ndarray | None:
    if not text.strip():
        return None
    values = [float(value) for value in text.replace(",", " ").split() if value]
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError("--center must contain two finite values: x,y")
    return np.asarray(values, dtype=np.float64)


def image_paths(input_path: Path, pattern: str) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    if not paths:
        raise FileNotFoundError("No images matched {} in {}".format(pattern, path))
    return paths


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not read {}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def disk_mask(shape: tuple[int, int], center: np.ndarray, radius: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2 <= float(radius) ** 2


def marker_response(gray: np.ndarray, background_sigma: float) -> np.ndarray:
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
    return cv2.subtract(gray, background)


def marker_centers(
    gray: np.ndarray,
    search_center: np.ndarray,
    search_radius: float,
    percentile: float,
    min_area: int,
    max_area: int,
    background_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = marker_response(gray, background_sigma)
    search_mask = disk_mask(gray.shape, search_center, search_radius)
    values = response[search_mask]
    if not len(values):
        raise ValueError("The marker search area is empty")
    threshold = float(np.percentile(values, percentile))
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(response >= threshold) & search_mask] = 255
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    accepted: list[np.ndarray] = []
    accepted_mask = np.zeros_like(binary)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(1, min(width, height))
        if min_area <= area <= max_area and aspect <= 3.5:
            accepted.append(centers[index])
            accepted_mask[_labels == index] = 255
    points = np.asarray(accepted, dtype=np.float64).reshape(-1, 2)
    return points, response, accepted_mask


def robust_lattice(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 12:
        raise ValueError("Only {} marker candidates were found; expected at least 12.".format(len(points)))
    center = np.median(points, axis=0)
    for _ in range(3):
        distances = np.linalg.norm(points - center[None, :], axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        limit = max(float(np.quantile(distances, 0.98)), median + 3.5 * max(mad, 1.0))
        retained = points[distances <= limit]
        if len(retained) < 12:
            break
        points = retained
        center = np.median(points, axis=0)
    return center, points


def crop_square(image: np.ndarray, center: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    diameter = max(2, int(radius) * 2)
    left = int(round(float(center[0]))) - int(radius)
    top = int(round(float(center[1]))) - int(radius)
    right = left + diameter
    bottom = top + diameter
    source_height, source_width = image.shape[:2]
    cropped = np.zeros((diameter, diameter), dtype=image.dtype)
    sx0, sy0 = max(left, 0), max(top, 0)
    sx1, sy1 = min(right, source_width), min(bottom, source_height)
    dx0, dy0 = sx0 - left, sy0 - top
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    cropped[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    local_center = np.asarray([radius, radius], dtype=np.float64)
    mask = disk_mask(cropped.shape, local_center, radius - 1.0)
    return cropped, mask


def processing_mask(
    center: np.ndarray,
    radius: int,
    lattice_points: np.ndarray,
    mode: str,
    hull_margin_px: int,
) -> np.ndarray:
    """Build one fixed mask in crop coordinates from the reference marker lattice."""
    diameter = int(radius) * 2
    circle = disk_mask((diameter, diameter), np.asarray([radius, radius], dtype=np.float64), radius - 1.0)
    if mode == "circle":
        return circle
    if len(lattice_points) < 3:
        raise ValueError("marker-hull masking requires at least three reference marker points")
    left = int(round(float(center[0]))) - int(radius)
    top = int(round(float(center[1]))) - int(radius)
    local = np.round(lattice_points - np.asarray([left, top], dtype=np.float64)).astype(np.int32)
    hull = cv2.convexHull(local.reshape(-1, 1, 2))
    support = np.zeros((diameter, diameter), dtype=np.uint8)
    cv2.fillConvexPoly(support, hull, 255)
    margin = max(0, int(hull_margin_px))
    if margin:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        support = cv2.dilate(support, kernel)
    return (support > 0) & circle


def model_input_mask(
    center: np.ndarray,
    radius: int,
    lattice_points: np.ndarray,
    support_mask: np.ndarray,
    mode: str,
    neighborhood_radius_px: float,
) -> np.ndarray:
    """Build a fixed ring-free model ROI from reference marker neighborhoods.

    The optical ring is outside the marker lattice. Keeping a generous disk
    around each reference marker is safer than trying to synthesize that ring's
    background: marker motion remains inside the disks while light outside the
    tactile marker field is exactly zero in the model input.
    """
    if mode in ("none", "tracked-marker-components", "outer-ring-protected"):
        return support_mask.copy()
    if len(lattice_points) < 1:
        raise ValueError("marker-neighborhood model masking requires at least one reference marker")
    if neighborhood_radius_px <= 0.0:
        raise ValueError("--model-marker-neighborhood-radius-px must be positive")

    diameter = int(radius) * 2
    left = int(round(float(center[0]))) - int(radius)
    top = int(round(float(center[1]))) - int(radius)
    local = np.round(lattice_points - np.asarray([left, top], dtype=np.float64)).astype(np.int32)
    support = np.zeros((diameter, diameter), dtype=np.uint8)
    disk_radius = max(1, int(round(float(neighborhood_radius_px))))
    for point in local:
        cv2.circle(support, tuple(point), disk_radius, 255, -1, cv2.LINE_AA)
    return (support > 0) & support_mask


def largest_marker_lattice(
    accepted: np.ndarray,
    min_neighbor_px: float,
    max_neighbor_px: float,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Keep the largest spatially coherent component of the marker lattice.

    The illumination ring can leave small bright blobs after suppression. Real
    TacTip markers instead form one dense, locally connected lattice, even under
    contact deformation. Selecting that largest component retains moving outer
    markers while discarding isolated ring fragments.
    """
    count, labels, _stats, centers = cv2.connectedComponentsWithStats(accepted, connectivity=8)
    component_count = count - 1
    if component_count <= 0:
        return accepted.copy(), {
            "method": "tracked-marker-components fallback; no components",
            "accepted_components": 0,
            "selected_components": 0,
        }
    if component_count == 1:
        return accepted.copy(), {
            "method": "tracked-marker-components",
            "accepted_components": 1,
            "selected_components": 1,
        }

    points = centers[1:]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    adjacency = (distances >= float(min_neighbor_px)) & (distances <= float(max_neighbor_px))
    visited = np.zeros(component_count, dtype=bool)
    groups: list[list[int]] = []
    for start in range(component_count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        group: list[int] = []
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in np.flatnonzero(adjacency[index] & ~visited):
                visited[int(neighbor)] = True
                stack.append(int(neighbor))
        groups.append(group)
    selected_indices = max(groups, key=len)
    selected = np.zeros_like(accepted)
    for index in selected_indices:
        selected[labels == index + 1] = 255
    return selected, {
        "method": "tracked-marker-components",
        "accepted_components": int(component_count),
        "selected_components": int(len(selected_indices)),
        "discarded_components": int(component_count - len(selected_indices)),
        "neighbor_min_px": float(min_neighbor_px),
        "neighbor_max_px": float(max_neighbor_px),
    }


def model_roi_for_frame(
    accepted: np.ndarray,
    fixed_model_mask: np.ndarray,
    usable_mask: np.ndarray,
    marker_outer_radius: float,
    radius: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Build the final model ROI after this frame's marker components are known."""
    if args.model_input_mask == "outer-ring-protected":
        yy, xx = np.indices(accepted.shape)
        radial = np.hypot(xx - float(radius), yy - float(radius))
        start_radius = min(
            float(radius) - 1.0,
            float(marker_outer_radius) + float(args.model_outer_ring_start_margin_px),
        )
        protected_markers, lattice_info = largest_marker_lattice(
            accepted,
            float(args.model_component_neighbor_min_px),
            float(args.model_component_neighbor_max_px),
        )
        dilation = max(0, int(args.model_outer_marker_protection_dilate_px))
        if dilation:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
            protected_markers = cv2.dilate(protected_markers, kernel)
        support = (radial < start_radius) | (protected_markers > 0)
        return support & usable_mask, {
            "method": "outer-ring-lattice-protected",
            "outer_ring_start_radius_px": float(start_radius),
            "marker_protection_dilate_px": int(dilation),
            "protected_marker_pixels": int(np.count_nonzero(protected_markers)),
            "lattice_selected_components": int(lattice_info.get("selected_components", 0)),
            "lattice_discarded_components": int(lattice_info.get("discarded_components", 0)),
        }
    if args.model_input_mask != "tracked-marker-components":
        return fixed_model_mask & usable_mask, {
            "method": str(args.model_input_mask),
            "selected_components": None,
        }
    selected, info = largest_marker_lattice(
        accepted,
        float(args.model_component_neighbor_min_px),
        float(args.model_component_neighbor_max_px),
    )
    dilation = max(0, int(args.model_component_dilate_px))
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        selected = cv2.dilate(selected, kernel)
    info["dilate_px"] = int(dilation)
    return (selected > 0) & usable_mask, info


def suppress_outer_illumination(
    gray: np.ndarray,
    center: np.ndarray,
    marker_outer_radius: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Identify the external light ring from gray-value clusters and remove it.

    Bright marker dots and the illumination ring have similar gray values, so
    intensity alone is not enough.  The reference marker lattice defines a
    protected inner radius; only large bright components outside that radius
    are eligible to be classified as the ring.
    """
    if args.ring_suppression == "none":
        return gray.copy(), np.zeros_like(gray, dtype=np.uint8), {
            "method": "none",
            "ring_pixels": 0,
        }

    height, width = gray.shape
    yy, xx = np.indices(gray.shape)
    radial = np.hypot(xx - float(center[0]), yy - float(center[1]))
    start_radius = float(marker_outer_radius) + float(args.ring_marker_guard_px)
    # The TacTip optical ring lies inside the camera's shorter image dimension.
    end_radius = min(float(min(height, width)) * 0.49, float(np.max(radial)))
    annulus = (radial >= start_radius) & (radial <= end_radius)
    values = gray[annulus].reshape(-1, 1).astype(np.float32)
    if len(values) < 64:
        raise ValueError("The outer annulus is too small for illumination clustering")

    sample_count = min(len(values), 60000)
    if sample_count < len(values):
        sample_indices = np.linspace(0, len(values) - 1, sample_count, dtype=np.int64)
        sample = values[sample_indices]
    else:
        sample = values
    cluster_count = max(2, min(int(args.ring_clusters), len(sample)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    # Keep the preprocessing identical across reruns of the same tactile batch.
    cv2.setRNGSeed(0)
    _compactness, _labels, centers = cv2.kmeans(
        sample,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    sorted_centers = np.sort(centers.reshape(-1).astype(np.float64))
    threshold = float((sorted_centers[-1] + sorted_centers[-2]) / 2.0)

    bright = np.zeros_like(gray, dtype=np.uint8)
    bright[annulus & (gray.astype(np.float32) >= threshold)] = 255
    component_count, labels, stats, _centers = cv2.connectedComponentsWithStats(bright, connectivity=8)
    ring = np.zeros_like(bright)
    accepted_components = 0
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < int(args.ring_min_component_area):
            continue
        component_radial = radial[labels == index]
        if len(component_radial) and float(np.median(component_radial)) >= start_radius:
            ring[labels == index] = 255
            accepted_components += 1

    dilation = max(0, int(args.ring_dilate_px))
    if dilation and np.any(ring):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        ring = cv2.dilate(ring, kernel)
        ring[~annulus] = 0
    if np.any(ring):
        suppressed = cv2.inpaint(gray, ring, float(args.ring_inpaint_radius), cv2.INPAINT_TELEA)
    else:
        suppressed = gray.copy()
    return suppressed, ring, {
        "method": "gray-clusters",
        "gray_cluster_centers": sorted_centers.astype(float).tolist(),
        "gray_threshold": threshold,
        "start_radius_px": start_radius,
        "end_radius_px": end_radius,
        "accepted_components": accepted_components,
        "ring_pixels": int(np.count_nonzero(ring)),
    }


def normalize_in_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    if not len(values):
        return np.zeros_like(image)
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low + 1.0e-6:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
    scaled[~mask] = 0
    return scaled


def outer_rim_weight(
    shape: tuple[int, int],
    radius: int,
    marker_outer_radius: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Return a smooth outer-rim attenuation weight in crop coordinates.

    The fade begins outside the reference marker lattice, so it keeps the
    marker positions and image scale intact while preventing residual white
    illumination from becoming an edge feature after CLAHE.
    """
    if args.outer_rim_treatment in ("none", "gray-cap", "flat-field"):
        return np.ones(shape, dtype=np.float32), {"method": "none"}
    yy, xx = np.indices(shape)
    radial = np.hypot(xx - float(radius), yy - float(radius))
    if args.outer_rim_treatment == "roi-mask":
        valid_radius = min(
            float(radius) - 1.0,
            float(marker_outer_radius) + float(args.outer_rim_roi_margin_px),
        )
        return (radial <= valid_radius).astype(np.float32), {
            "method": "roi-mask",
            "valid_radius_px": float(valid_radius),
        }
    start = min(
        float(radius) - 2.0,
        float(marker_outer_radius) + float(args.outer_rim_fade_start_margin_px),
    )
    end = max(start + 1.0, float(radius) - float(args.outer_rim_fade_end_margin_px))
    progress = np.clip((radial - start) / (end - start), 0.0, 1.0)
    weight = 0.5 * (1.0 + np.cos(np.pi * progress))
    weight[radial >= end] = 0.0
    return weight.astype(np.float32), {
        "method": "cosine-fade",
        "start_radius_px": float(start),
        "end_radius_px": float(end),
    }


def cap_outer_rim_gray(
    gray: np.ndarray,
    radius: int,
    marker_outer_radius: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Cap only overly bright outer-rim pixels using nearby sensor background.

    This avoids a hard black annulus. The reference band is outside the marker
    lattice but inside the illumination ring, then its robust gray percentile
    becomes the maximum allowed brightness for the external rim.
    """
    if args.outer_rim_treatment != "gray-cap":
        return gray, {"method": "none"}
    yy, xx = np.indices(gray.shape)
    radial = np.hypot(xx - float(radius), yy - float(radius))
    band_start = float(marker_outer_radius) + 8.0
    band_end = float(marker_outer_radius) + 16.0
    reference_band = (radial >= band_start) & (radial < band_end)
    values = gray[reference_band]
    if not len(values):
        return gray, {"method": "gray-cap fallback; no reference band"}
    cap = float(np.percentile(values, float(args.outer_rim_cap_percentile)))
    start = float(marker_outer_radius) + float(args.outer_rim_fade_start_margin_px)
    target = radial >= start
    corrected = gray.copy()
    corrected[target] = np.minimum(corrected[target], int(round(cap)))
    return corrected, {
        "method": "gray-cap",
        "cap_gray": cap,
        "start_radius_px": start,
        "reference_band_start_px": band_start,
        "reference_band_end_px": band_end,
    }


def build_flat_field_template(
    paths: list[Path],
    center: np.ndarray,
    radius: int,
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Estimate the fixed optical illumination from the batch median.

    The physical illumination ring is stationary across samples, while tactile
    contacts vary. Taking the median then a broad Gaussian blur keeps the
    low-frequency light field but removes the marker lattice before correction.
    """
    if args.outer_rim_treatment != "flat-field":
        return None
    crops: list[np.ndarray] = []
    for path in paths:
        gray = read_gray(path)
        cropped, _mask = crop_square(gray, center, radius)
        crops.append(cropped.astype(np.float32))
    median = np.median(np.stack(crops, axis=0), axis=0).astype(np.float32)
    sigma = float(args.flat_field_sigma)
    return cv2.GaussianBlur(median, (0, 0), sigmaX=sigma, sigmaY=sigma)


def process_color_image(
    color: np.ndarray,
    center: np.ndarray,
    radius: int,
    mask: np.ndarray,
    model_mask: np.ndarray,
    marker_outer_radius: float,
    flat_field: np.ndarray | None,
    args: argparse.Namespace,
    source: str = "",
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the fixed TacTip marker preprocessing on one in-memory BGR frame."""
    if color is None or not isinstance(color, np.ndarray) or color.size == 0:
        raise ValueError("TacTip frame is empty")
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    elif color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("TacTip frame must be grayscale or BGR")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    suppressed, ring_mask, ring_info = suppress_outer_illumination(gray, center, marker_outer_radius, args)
    crop_source = gray if flat_field is not None else suppressed
    cropped, _circle_mask = crop_square(crop_source, center, radius)
    if mask.shape != cropped.shape:
        raise ValueError("Fixed processing mask shape does not match the image crop")
    if model_mask.shape != cropped.shape:
        raise ValueError("Fixed model-input mask shape does not match the image crop")
    flat_field_info: dict[str, float | str] = {"method": "none"}
    if flat_field is not None:
        if flat_field.shape != cropped.shape:
            raise ValueError("Flat-field template shape does not match the image crop")
        cropped = np.clip(
            cropped.astype(np.float32) - flat_field + float(args.flat_field_level),
            0.0,
            255.0,
        ).astype(np.uint8)
        flat_field_info = {
            "method": "batch-median subtraction",
            "sigma": float(args.flat_field_sigma),
            "level": float(args.flat_field_level),
        }
    rim_weight, rim_info = outer_rim_weight(cropped.shape, radius, marker_outer_radius, args)
    usable_mask = mask & (rim_weight > 1.0e-3)
    clahe = cv2.createCLAHE(clipLimit=float(args.clahe_clip_limit), tileGridSize=(8, 8))
    gray_clahe = clahe.apply(cropped)
    gray_clahe, rim_cap_info = cap_outer_rim_gray(gray_clahe, radius, marker_outer_radius, args)
    gray_clahe = np.rint(gray_clahe.astype(np.float32) * rim_weight).astype(np.uint8)
    gray_clahe[~usable_mask] = 0
    response = marker_response(gray_clahe, float(args.background_sigma))
    response = normalize_in_mask(response, usable_mask)
    response = np.rint(response.astype(np.float32) * rim_weight).astype(np.uint8)
    values = response[usable_mask]
    threshold = float(np.percentile(values, float(args.marker_percentile)))
    binary = np.zeros_like(response)
    binary[(response >= threshold) & usable_mask] = 255
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    accepted = np.zeros_like(binary)
    marker_points: list[np.ndarray] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(1, min(width, height))
        if int(args.min_marker_area) <= area <= int(args.max_marker_area) and aspect <= 3.5:
            accepted[labels == index] = 255
            marker_points.append(centers[index])
    resized = cv2.resize(response, (int(args.output_size), int(args.output_size)), interpolation=cv2.INTER_AREA)
    active_model_mask, model_info = model_roi_for_frame(
        accepted,
        model_mask,
        usable_mask,
        marker_outer_radius,
        radius,
        args,
    )
    model_input = response.copy()
    model_input[~active_model_mask] = 0
    model_input_256 = cv2.resize(
        model_input,
        (int(args.output_size), int(args.output_size)),
        interpolation=cv2.INTER_AREA,
    )
    overlay = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    overlay = np.rint(overlay.astype(np.float32) * rim_weight[:, :, None]).astype(np.uint8)
    overlay[~usable_mask] = 0
    for point in marker_points:
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 4, (41, 220, 116), 1, cv2.LINE_AA)
    contours, _hierarchy = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (71, 148, 240), 1, cv2.LINE_AA)
    ring_overlay = color.copy()
    ring_pixels = ring_mask > 0
    if np.any(ring_pixels):
        red = np.zeros_like(ring_overlay)
        red[:, :, 2] = 255
        ring_overlay[ring_pixels] = cv2.addWeighted(
            ring_overlay[ring_pixels], 0.35, red[ring_pixels], 0.65, 0.0
        )
    return (
        {
            "source": str(source),
            "marker_count": len(marker_points),
            "marker_threshold": threshold,
            "illumination_ring": ring_info,
            "outer_rim_treatment": {**rim_info, **rim_cap_info},
            "flat_field": flat_field_info,
            "model_input_nonzero_fraction": float(np.mean(model_input > 0)),
            "model_roi": model_info,
        },
        {
            "gray": gray_clahe,
            "response": response,
            "marker_mask": accepted,
            "response_256": resized,
            "model_input": model_input,
            "model_input_256": model_input_256,
            "model_roi": active_model_mask.astype(np.uint8) * 255,
            "overlay": overlay,
            "ring_mask": ring_mask,
            "ring_suppressed": suppressed,
            "ring_overlay": ring_overlay,
            "outer_rim_weight": np.rint(rim_weight * 255.0).astype(np.uint8),
        },
    )


def process_image(
    path: Path,
    center: np.ndarray,
    radius: int,
    mask: np.ndarray,
    model_mask: np.ndarray,
    marker_outer_radius: float,
    flat_field: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load an image from disk then process it through :func:`process_color_image`."""
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("OpenCV could not read {}".format(path))
    return process_color_image(
        color,
        center,
        radius,
        mask,
        model_mask,
        marker_outer_radius,
        flat_field,
        args,
        source=str(path),
    )


def save_preview(output_dir: Path, rows: list[dict[str, object]], limit: int) -> Path:
    cards: list[str] = []
    for row in rows[: max(1, int(limit))]:
        name = html.escape(str(row["name"]))
        marker_count = int(row["marker_count"])
        threshold = float(row["marker_threshold"])
        cards.append(
            """
<article class=\"card\"><h2>{}</h2><p>{} marker candidates, response threshold {:.1f}; outer-ring pixels {}</p>
<div class=\"grid\">
<figure><figcaption>Detected outer illumination clusters (red)</figcaption><img src=\"ring_overlay/{}\"></figure>
<figure><figcaption>Gray image after ring suppression</figcaption><img src=\"ring_suppressed/{}\"></figure>
<figure><figcaption>Final inner grayscale crop</figcaption><img src=\"gray/{}\"></figure>
<figure><figcaption>Marker-enhanced grayscale</figcaption><img src=\"marker_response/{}\"></figure>
<figure><figcaption>Marker mask</figcaption><img src=\"marker_mask/{}\"></figure>
<figure><figcaption>Tracked marker region</figcaption><img src=\"model_roi/{}\"></figure>
<figure><figcaption>Ring-free model input</figcaption><img src=\"model_input/{}\"></figure>
<figure><figcaption>Detection overlay</figcaption><img src=\"overlay/{}\"></figure>
</div></article>""".format(
                name,
                marker_count,
                threshold,
                int(row["illumination_ring"]["ring_pixels"]),
                name,
                name,
                name,
                name,
                name,
                name,
                name,
                name,
            )
        )
    page = """<!doctype html><html><head><meta charset=\"utf-8\"><title>TacTip marker preprocessing</title>
<style>body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232;position:sticky;top:0}h1{margin:0;font-size:22px}header p{margin:8px 0 0;color:#b7c5d6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;padding:16px}.card{background:#182232;border:1px solid #33475f;border-radius:7px;padding:14px}h2{font-size:15px;margin:0}p{color:#b7c5d6;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}figure{margin:0;background:#0d141e;padding:8px}figcaption{font-size:12px;color:#cbd7e5;margin-bottom:6px}img{width:100%;display:block;image-rendering:pixelated;background:#000}</style></head>
<body><header><h1>TacTip marker preprocessing</h1><p>High-gray outer pixels are clustered and spatially filtered before one shared inner crop. The red overlay shows only the illumination ring selected for removal. The default ring-free model input keeps the full inner response and removes only exterior radial-ring response outside the coherent outer marker lattice.</p></header><main>""" + "\n".join(cards) + """</main></body></html>"""
    path = output_dir / "preview.html"
    path.write_text(page, encoding="utf-8")
    return path


def save_collection(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    """Write a compact all-sample before/after gallery for batch review."""
    cards: list[str] = []
    for row in rows:
        name = html.escape(str(row["name"]))
        source = html.escape(os.path.relpath(str(row["source"]), start=str(output_dir)))
        marker_count = int(row["marker_count"])
        cards.append(
            """
<article class="card"><h2>{}</h2><p>{} detected markers</p>
<div class="images">
<figure><figcaption>Raw capture</figcaption><img src="{}"></figure>
<figure><figcaption>Ring-suppressed grayscale</figcaption><img src="ring_suppressed/{}"></figure>
<figure><figcaption>Final 256x256 model input</figcaption><img src="model_input_256/{}"></figure>
</div></article>""".format(name, marker_count, source, name, name)
        )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>TacTip all-sample preprocessing collection</title>
<style>body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232;position:sticky;top:0;z-index:1}h1{margin:0;font-size:22px}header p{margin:8px 0 0;color:#b7c5d6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:16px;padding:16px}.card{background:#182232;border:1px solid #33475f;border-radius:7px;padding:12px}h2{font-size:15px;margin:0}p{color:#b7c5d6;font-size:12px;margin:6px 0 10px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0;background:#0d141e;padding:6px}figcaption{font-size:11px;color:#cbd7e5;margin-bottom:5px;min-height:25px}img{width:100%;display:block;image-rendering:pixelated;background:#000}</style></head>
<body><header><h1>TacTip all-sample preprocessing collection</h1><p>Every capture uses the same optical center and crop. Compare the raw image, illumination-ring-suppressed grayscale image, and the final marker-preserving 256x256 model input.</p></header><main>""" + "\n".join(cards) + """</main></body></html>"""
    path = output_dir / "all_samples_collection.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if not 0.1 <= float(args.search_radius_ratio) <= 0.9:
        raise ValueError("--search-radius-ratio must be in [0.1, 0.9]")
    if not 50.0 <= float(args.marker_percentile) < 100.0:
        raise ValueError("--marker-percentile must be in [50, 100)")
    if args.output_size <= 0:
        raise ValueError("--output-size must be positive")
    if int(args.ring_clusters) < 2:
        raise ValueError("--ring-clusters must be at least 2")
    if int(args.ring_min_component_area) <= 0:
        raise ValueError("--ring-min-component-area must be positive")
    if float(args.ring_inpaint_radius) <= 0.0:
        raise ValueError("--ring-inpaint-radius must be positive")
    if not 0.0 < float(args.ring_crop_percentile) < 100.0:
        raise ValueError("--ring-crop-percentile must be in (0, 100)")
    if float(args.flat_field_sigma) <= 0.0:
        raise ValueError("--flat-field-sigma must be positive")
    if float(args.model_marker_neighborhood_radius_px) <= 0.0:
        raise ValueError("--model-marker-neighborhood-radius-px must be positive")
    if float(args.model_component_neighbor_min_px) <= 0.0:
        raise ValueError("--model-component-neighbor-min-px must be positive")
    if float(args.model_component_neighbor_max_px) <= float(args.model_component_neighbor_min_px):
        raise ValueError("--model-component-neighbor-max-px must exceed --model-component-neighbor-min-px")
    if int(args.model_component_dilate_px) < 0:
        raise ValueError("--model-component-dilate-px must be non-negative")
    if float(args.model_outer_ring_start_margin_px) < 0.0:
        raise ValueError("--model-outer-ring-start-margin-px must be non-negative")
    if int(args.model_outer_marker_protection_dilate_px) < 0:
        raise ValueError("--model-outer-marker-protection-dilate-px must be non-negative")
    paths = image_paths(args.input, args.glob)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("{} already contains files; choose a new output directory.".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        "gray",
        "marker_response",
        "marker_mask",
        "marker_response_256",
        "model_input",
        "model_input_256",
        "model_roi",
        "overlay",
        "ring_mask",
        "ring_suppressed",
        "ring_overlay",
        "outer_rim_weight",
    ):
        (output_dir / directory).mkdir()

    reference = args.reference_image.expanduser().resolve() if args.reference_image else paths[0]
    reference_gray = read_gray(reference)
    initial_center = parse_center(args.center)
    if initial_center is None:
        initial_center = np.asarray([reference_gray.shape[1] / 2.0, reference_gray.shape[0] / 2.0])
    initial_radius = min(reference_gray.shape) * float(args.search_radius_ratio)
    candidates, _response, _mask = marker_centers(
        reference_gray,
        initial_center,
        initial_radius,
        float(args.marker_percentile),
        int(args.min_marker_area),
        int(args.max_marker_area),
        float(args.background_sigma),
    )
    if parse_center(args.center) is None:
        center, lattice_points = robust_lattice(candidates)
    else:
        center, lattice_points = initial_center, candidates
    distances = np.linalg.norm(lattice_points - center[None, :], axis=1)
    marker_outer_radius = float(np.quantile(distances, 0.99))
    minimum_radius = marker_outer_radius + float(args.crop_padding_px)
    crop_selection: dict[str, object] = {
        "policy": "fixed" if args.crop_radius_px > 0 else str(args.crop_policy),
        "minimum_radius_px": minimum_radius,
    }
    if args.crop_radius_px > 0:
        auto_radius = float(args.crop_radius_px)
        crop_selection["source"] = "--crop-radius-px"
    elif args.crop_policy == "ring-aware" and args.ring_suppression == "gray-clusters":
        _reference_suppressed, reference_ring_mask, reference_ring_info = suppress_outer_illumination(
            reference_gray,
            center,
            marker_outer_radius,
            args,
        )
        ring_y, ring_x = np.nonzero(reference_ring_mask)
        if len(ring_x):
            ring_radii = np.hypot(ring_x - float(center[0]), ring_y - float(center[1]))
            ring_inner_radius = float(np.percentile(ring_radii, float(args.ring_crop_percentile)))
            auto_radius = max(minimum_radius, ring_inner_radius - float(args.ring_crop_safety_px))
            crop_selection.update(
                {
                    "source": "gray-cluster illumination-ring inner edge",
                    "ring_inner_percentile": float(args.ring_crop_percentile),
                    "ring_inner_radius_px": ring_inner_radius,
                    "ring_crop_safety_px": float(args.ring_crop_safety_px),
                    "reference_illumination_ring": reference_ring_info,
                }
            )
        else:
            auto_radius = minimum_radius
            crop_selection["source"] = "marker-padding fallback; no ring cluster found"
    else:
        auto_radius = minimum_radius
        crop_selection["source"] = "marker-padding"
    radius = int(round(auto_radius))
    radius = max(32, min(radius, int(min(reference_gray.shape) * 0.49)))
    crop_selection["selected_radius_px"] = radius
    fixed_mask = processing_mask(
        center,
        radius,
        lattice_points,
        str(args.mask_mode),
        int(args.marker_hull_margin_px),
    )
    cv2.imwrite(str(output_dir / "support_mask.png"), fixed_mask.astype(np.uint8) * 255)
    fixed_model_mask = model_input_mask(
        center,
        radius,
        lattice_points,
        fixed_mask,
        str(args.model_input_mask),
        float(args.model_marker_neighborhood_radius_px),
    )
    cv2.imwrite(str(output_dir / "model_input_reference_support.png"), fixed_model_mask.astype(np.uint8) * 255)
    flat_field = build_flat_field_template(paths, center, radius, args)
    if flat_field is not None:
        cv2.imwrite(str(output_dir / "illumination_template.png"), np.rint(flat_field).astype(np.uint8))
    print(
        "[INFO] fixed {} mask center=({:.2f}, {:.2f}) radius={}px from {} reference markers ({})".format(
            args.mask_mode, center[0], center[1], radius, len(lattice_points), crop_selection["source"]
        )
    )

    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        record, images = process_image(
            path,
            center,
            radius,
            fixed_mask,
            fixed_model_mask,
            marker_outer_radius,
            flat_field,
            args,
        )
        name = path.name
        cv2.imwrite(str(output_dir / "gray" / name), images["gray"])
        cv2.imwrite(str(output_dir / "marker_response" / name), images["response"])
        cv2.imwrite(str(output_dir / "marker_mask" / name), images["marker_mask"])
        cv2.imwrite(str(output_dir / "marker_response_256" / name), images["response_256"])
        cv2.imwrite(str(output_dir / "model_input" / name), images["model_input"])
        cv2.imwrite(str(output_dir / "model_input_256" / name), images["model_input_256"])
        cv2.imwrite(str(output_dir / "model_roi" / name), images["model_roi"])
        cv2.imwrite(str(output_dir / "overlay" / name), images["overlay"])
        cv2.imwrite(str(output_dir / "ring_mask" / name), images["ring_mask"])
        cv2.imwrite(str(output_dir / "ring_suppressed" / name), images["ring_suppressed"])
        cv2.imwrite(str(output_dir / "ring_overlay" / name), images["ring_overlay"])
        cv2.imwrite(str(output_dir / "outer_rim_weight" / name), images["outer_rim_weight"])
        record["name"] = name
        rows.append(record)
        print("[INFO] {}/{} {}: {} markers".format(index, len(paths), name, record["marker_count"]))

    preview = save_preview(output_dir, rows, int(args.preview_limit))
    collection = save_collection(output_dir, rows)
    summary = {
        "schema": "tactip_marker_preprocess.v1",
        "input": str(args.input.expanduser().resolve()),
        "reference_image": str(reference),
        "image_count": len(paths),
        "fixed_crop_center_px": center.astype(float).tolist(),
        "fixed_crop_radius_px": radius,
        "reference_marker_outer_radius_px": marker_outer_radius,
        "crop_selection": crop_selection,
        "mask_mode": str(args.mask_mode),
        "marker_hull_margin_px": int(args.marker_hull_margin_px),
        "support_mask": str(output_dir / "support_mask.png"),
        "model_input_reference_support": str(output_dir / "model_input_reference_support.png"),
        "model_roi_dir": str(output_dir / "model_roi"),
        "illumination_template": str(output_dir / "illumination_template.png") if flat_field is not None else None,
        "output_size": int(args.output_size),
        "configuration": {
            "marker_percentile": float(args.marker_percentile),
            "background_sigma": float(args.background_sigma),
            "clahe_clip_limit": float(args.clahe_clip_limit),
            "min_marker_area": int(args.min_marker_area),
            "max_marker_area": int(args.max_marker_area),
            "ring_suppression": str(args.ring_suppression),
            "ring_clusters": int(args.ring_clusters),
            "ring_marker_guard_px": float(args.ring_marker_guard_px),
            "ring_min_component_area": int(args.ring_min_component_area),
            "ring_dilate_px": int(args.ring_dilate_px),
            "ring_inpaint_radius": float(args.ring_inpaint_radius),
            "ring_crop_percentile": float(args.ring_crop_percentile),
            "ring_crop_safety_px": float(args.ring_crop_safety_px),
            "outer_rim_treatment": str(args.outer_rim_treatment),
            "model_input_mask": str(args.model_input_mask),
            "model_marker_neighborhood_radius_px": float(args.model_marker_neighborhood_radius_px),
            "model_component_neighbor_min_px": float(args.model_component_neighbor_min_px),
            "model_component_neighbor_max_px": float(args.model_component_neighbor_max_px),
            "model_component_dilate_px": int(args.model_component_dilate_px),
            "model_outer_ring_start_margin_px": float(args.model_outer_ring_start_margin_px),
            "model_outer_marker_protection_dilate_px": int(args.model_outer_marker_protection_dilate_px),
            "outer_rim_fade_start_margin_px": float(args.outer_rim_fade_start_margin_px),
            "outer_rim_fade_end_margin_px": float(args.outer_rim_fade_end_margin_px),
            "outer_rim_cap_percentile": float(args.outer_rim_cap_percentile),
            "outer_rim_roi_margin_px": float(args.outer_rim_roi_margin_px),
            "flat_field_sigma": float(args.flat_field_sigma),
            "flat_field_level": float(args.flat_field_level),
        },
        "images": rows,
        "preview": str(preview),
        "collection": str(collection),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[DONE] preview: {}".format(preview))
    print("[DONE] collection: {}".format(collection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
