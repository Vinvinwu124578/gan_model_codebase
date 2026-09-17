#!/usr/bin/env python3
"""Use official VSP marker centres and Hough only for measured gap filling.

The detector is intentionally auditable.  VSP measurements are the primary
output and are never shifted toward Hough centres.  Hough starts at a strict
vote threshold and relaxes only enough to provide candidates when VSP detects
fewer than the known 331 sensor markers.  Chromatic, same-frame spacing and
radial gates decide which measured candidates fill the gaps.  No marker
position is synthesized from a template or a brightness maximum.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tactip_hough_chromatic import (
    HoughChromaticConfig,
    _detect_for_vote,
    _scaled_geometry,
    blue_yellow_opponent,
)


@dataclass(frozen=True)
class HybridConfig:
    expected_markers: int = 331
    match_radius_px: float = 6.0
    supplement_hough_max_vote_threshold: int = 13
    supplement_hough_min_vote_threshold: int = 6
    supplement_min_blue_yellow_score: float = 25.0
    shell_min_spacing_ratio: float = 0.60
    shell_max_spacing_ratio: float = 1.55
    severe_duplicate_ratio: float = 0.48
    supplement_min_support: int = 3
    radial_margin_spacing_ratio: float = 0.35
    maximum_radial_excess_ratio: float = 0.10
    binary_marker_radius_px: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditably combine official VSP and Hough TacTip detections."
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vsp-root", type=Path, default=Path("/Users/vincent/Downloads/vsp")
    )
    return parser.parse_args()


def make_official_vsp_detector(vsp_root: Path) -> Any:
    detector_source = vsp_root / "vsp" / "detector.py"
    if not detector_source.is_file():
        raise FileNotFoundError(detector_source)
    sys.path.insert(0, str(vsp_root))
    from vsp.detector import CvBlobDetector  # type: ignore[import-not-found]

    # Copied verbatim from vsp/examples/processor_test.py.
    return CvBlobDetector(
        min_threshold=31.23,
        max_threshold=207.05,
        filter_by_color=True,
        blob_color=255,
        filter_by_area=True,
        min_area=17.05,
        max_area=135.46,
        filter_by_circularity=True,
        min_circularity=0.62,
        filter_by_inertia=True,
        min_inertia_ratio=0.27,
        filter_by_convexity=True,
        min_convexity=0.60,
    )


def detect_hough_supplement_candidates(
    image: np.ndarray,
    config: HybridConfig,
    vote_threshold: float | None = None,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    """Run one chromatically gated Hough pass for missing-point candidates."""
    selected_vote = (
        float(config.supplement_hough_max_vote_threshold)
        if vote_threshold is None
        else float(vote_threshold)
    )
    hough_config = HoughChromaticConfig(
        expected_markers=config.expected_markers,
        minimum_blue_yellow_score=config.supplement_min_blue_yellow_score,
    )
    geometry = _scaled_geometry(image, hough_config)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (int(geometry["blur_size"]), int(geometry["blur_size"])),
        max(0.5, float(geometry["scale"])),
    )
    accepted, rejected, total = _detect_for_vote(
        gray,
        blue_yellow_opponent(image),
        hough_config,
        geometry,
        selected_vote,
    )
    return accepted, rejected, {
        "vote_threshold": selected_vote,
        "total_circles": total,
        "chromatic_candidates": len(accepted),
        "rejected_glare": len(rejected),
        "minimum_blue_yellow_score": config.supplement_min_blue_yellow_score,
        "scaled_geometry": geometry,
    }


def one_to_one_matches(
    vsp_points: np.ndarray,
    hough_points: np.ndarray,
    match_radius_px: float,
) -> list[tuple[int, int, float]]:
    """Return minimum-distance matches while allowing either side to be unmatched."""
    if len(vsp_points) == 0 or len(hough_points) == 0:
        return []
    distances = np.linalg.norm(
        vsp_points[:, None, :] - hough_points[None, :, :], axis=2
    )
    dummy_cost = float(match_radius_px) + 0.01
    costs = np.full(
        (len(vsp_points), len(hough_points) + len(vsp_points)),
        dummy_cost,
        dtype=float,
    )
    costs[:, : len(hough_points)] = np.where(
        distances <= match_radius_px, distances, 1.0e6
    )
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(vsp_index), int(hough_index), float(distances[vsp_index, hough_index]))
        for vsp_index, hough_index in zip(rows, columns)
        if hough_index < len(hough_points)
        and distances[vsp_index, hough_index] <= match_radius_px
    ]


def estimate_lattice_spacing(points: np.ndarray) -> float:
    if len(points) < 3:
        raise ValueError("At least three points are needed to estimate spacing")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(np.min(distances, axis=1)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Could not estimate a finite marker spacing")
    return spacing


def deduplicate_vsp_measurements(
    vsp_points: np.ndarray,
    vsp_sizes: np.ndarray,
    config: HybridConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Merge only unmistakably duplicated VSP responses to one marker blob."""
    points = np.asarray(vsp_points, dtype=float).reshape(-1, 2)
    sizes = np.asarray(vsp_sizes, dtype=float).reshape(-1)
    if len(points) != len(sizes):
        raise ValueError("VSP point and size counts differ")
    if len(points) < 3:
        raise ValueError("VSP supplied too few points to estimate the marker lattice")

    raw_spacing_px = estimate_lattice_spacing(points)
    duplicate_distance_px = config.severe_duplicate_ratio * raw_spacing_px
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    adjacency = (distances < duplicate_distance_px) & (distances > 0.0)
    visited: set[int] = set()
    groups: list[list[int]] = []
    for start in range(len(points)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in np.flatnonzero(adjacency[current]):
                neighbour_index = int(neighbour)
                if neighbour_index not in visited:
                    visited.add(neighbour_index)
                    stack.append(neighbour_index)
        groups.append(sorted(component))

    clean_points: list[np.ndarray] = []
    clean_sizes: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for group in groups:
        group_points = points[group]
        group_sizes = sizes[group]
        # The centre remains entirely VSP-derived; Hough never moves it.
        weights = np.maximum(group_sizes, 1.0)
        centre = np.average(group_points, axis=0, weights=weights)
        clean_points.append(centre)
        clean_sizes.append(float(np.max(group_sizes)))
        if len(group) > 1:
            diagnostics.append(
                {
                    "raw_vsp_indices": group,
                    "raw_points_xy": group_points.tolist(),
                    "merged_point_xy": centre.tolist(),
                    "maximum_internal_distance_px": float(
                        np.max(
                            np.linalg.norm(
                                group_points[:, None, :] - group_points[None, :, :],
                                axis=2,
                            )
                        )
                    ),
                }
            )
    return (
        np.asarray(clean_points, dtype=float),
        np.asarray(clean_sizes, dtype=float),
        diagnostics,
        raw_spacing_px,
    )


def supplement_candidate_metrics(
    point: np.ndarray,
    selected_points: np.ndarray,
    spacing_px: float,
    radial_centre: np.ndarray,
    radial_reference_px: float,
    config: HybridConfig,
) -> dict[str, Any]:
    """Score a measured Hough candidate by same-frame lattice geometry only."""
    distances = np.sort(np.linalg.norm(selected_points - point, axis=1))
    nearest = float(distances[0]) if len(distances) else float("inf")
    shell = distances[
        (distances >= config.shell_min_spacing_ratio * spacing_px)
        & (distances <= config.shell_max_spacing_ratio * spacing_px)
    ]
    support = int(len(shell))
    spacing_error = (
        float(np.mean(np.abs(shell[: min(4, support)] - spacing_px)) / spacing_px)
        if support
        else 2.0
    )
    radial_distance = float(np.linalg.norm(point - radial_centre))
    allowed_radius = (
        radial_reference_px + config.radial_margin_spacing_ratio * spacing_px
    )
    radial_excess = max(0.0, (radial_distance - allowed_radius) / spacing_px)
    too_close = nearest < config.shell_min_spacing_ratio * spacing_px
    isolated = nearest > config.shell_max_spacing_ratio * spacing_px
    score = (
        float(min(support, 4))
        - spacing_error
        - 4.0 * float(too_close)
        - 3.0 * float(isolated)
        - 2.0 * radial_excess
    )
    return {
        "nearest_px": nearest,
        "support_count": support,
        "spacing_error_ratio": spacing_error,
        "radial_distance_px": radial_distance,
        "radial_excess_spacing_ratio": radial_excess,
        "too_close": bool(too_close),
        "isolated": bool(isolated),
        "score": score,
    }


def fuse_detections(
    vsp_points: np.ndarray,
    vsp_sizes: np.ndarray,
    hough_markers: list[dict[str, float]],
    config: HybridConfig | None = None,
) -> dict[str, Any]:
    """Keep VSP centres and use Hough solely to fill measured missing points."""
    config = config or HybridConfig()
    hough_points = np.asarray(
        [[item["x_px"], item["y_px"]] for item in hough_markers], dtype=float
    ).reshape(-1, 2)
    vsp_points = np.asarray(vsp_points, dtype=float).reshape(-1, 2)
    vsp_sizes = np.asarray(vsp_sizes, dtype=float).reshape(-1)
    clean_vsp, clean_sizes, duplicate_groups, raw_spacing_px = (
        deduplicate_vsp_measurements(vsp_points, vsp_sizes, config)
    )
    if len(clean_vsp) > config.expected_markers:
        raise ValueError(
            f"VSP supplied {len(clean_vsp)} distinct points, above the known "
            f"{config.expected_markers}; output was not fabricated."
        )
    if not len(hough_points) and len(clean_vsp) < config.expected_markers:
        raise ValueError("Hough supplied no measured candidates for VSP gap filling")

    matches = one_to_one_matches(
        clean_vsp, hough_points, config.match_radius_px
    )
    matched_hough = {hough_index for _, hough_index, _ in matches}
    unmatched_hough_indices = [
        index for index in range(len(hough_points)) if index not in matched_hough
    ]
    spacing_px = estimate_lattice_spacing(clean_vsp)
    radial_centre = np.median(clean_vsp, axis=0)
    radial_reference_px = float(
        np.percentile(np.linalg.norm(clean_vsp - radial_centre, axis=1), 99.0)
    )
    selected_points = [point.copy() for point in clean_vsp]
    remaining = set(unmatched_hough_indices)
    required_supplements = config.expected_markers - len(clean_vsp)
    selected_supplements: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    while len(selected_supplements) < required_supplements and remaining:
        current_points = np.asarray(selected_points, dtype=float).reshape(-1, 2)
        options: list[dict[str, Any]] = []
        for hough_index in sorted(remaining):
            metrics = supplement_candidate_metrics(
                hough_points[hough_index],
                current_points,
                spacing_px,
                radial_centre,
                radial_reference_px,
                config,
            )
            options.append({"hough_index": hough_index, **metrics})
        best = max(
            options,
            key=lambda item: (
                item["score"],
                item["support_count"],
                -item["spacing_error_ratio"],
                -item["hough_index"],
            ),
        )
        remaining.remove(int(best["hough_index"]))
        reliable = bool(
            not best["too_close"]
            and not best["isolated"]
            and best["support_count"] >= config.supplement_min_support
            and best["radial_excess_spacing_ratio"]
            <= config.maximum_radial_excess_ratio
        )
        record = {**best, "accepted": reliable}
        if reliable:
            selected_supplements.append(record)
            selected_points.append(hough_points[int(best["hough_index"])].copy())
        else:
            rejected_candidates.append(record)

    if len(selected_supplements) != required_supplements:
        raise ValueError(
            f"VSP found {len(clean_vsp)} distinct markers, but only "
            f"{len(selected_supplements)}/{required_supplements} credible measured "
            "Hough supplements were available; output was not fabricated."
        )

    final_markers: list[dict[str, Any]] = [
        {
            "x_px": float(point[0]),
            "y_px": float(point[1]),
            "radius_px": float(clean_sizes[index] / 2.0),
            "source": "vsp_primary",
            "vsp_primary_index": int(index),
        }
        for index, point in enumerate(clean_vsp)
    ]
    for supplement in selected_supplements:
        hough_index = int(supplement["hough_index"])
        marker = hough_markers[hough_index]
        final_markers.append(
            {
                "x_px": float(marker["x_px"]),
                "y_px": float(marker["y_px"]),
                "radius_px": float(marker["hough_radius_px"]),
                "source": "hough_supplement",
                "hough_index": hough_index,
                "selection_metrics": {
                    key: value
                    for key, value in supplement.items()
                    if key not in {"hough_index", "accepted"}
                },
            }
        )

    return {
        "raw_vsp_count": int(len(vsp_points)),
        "clean_vsp_count": int(len(clean_vsp)),
        "raw_vsp_spacing_px": raw_spacing_px,
        "lattice_spacing_px": spacing_px,
        "radial_centre_xy": radial_centre.tolist(),
        "radial_reference_px": radial_reference_px,
        "vsp_duplicate_groups": duplicate_groups,
        "matches": [
            {
                "vsp_index": i,
                "hough_index": j,
                "distance_px": distance,
            }
            for i, j, distance in matches
        ],
        "unmatched_hough_indices": unmatched_hough_indices,
        "required_supplements": int(required_supplements),
        "selected_supplements": selected_supplements,
        "rejected_candidates": rejected_candidates,
        "remaining_unselected_hough_indices": sorted(remaining),
        "final_markers": final_markers,
    }


def strict_hough_gap_fill(
    image: np.ndarray,
    vsp_points: np.ndarray,
    vsp_sizes: np.ndarray,
    config: HybridConfig,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, Any],
    dict[str, Any],
]:
    """Use the highest Hough vote threshold that can credibly fill VSP gaps."""
    if (
        config.supplement_hough_max_vote_threshold
        < config.supplement_hough_min_vote_threshold
    ):
        raise ValueError("Hough maximum vote threshold is below its minimum")
    trials: list[dict[str, Any]] = []
    final_error = "no threshold was evaluated"
    for vote_threshold in range(
        config.supplement_hough_max_vote_threshold,
        config.supplement_hough_min_vote_threshold - 1,
        -1,
    ):
        markers, rejected, detector_record = detect_hough_supplement_candidates(
            image, config, float(vote_threshold)
        )
        trial = {
            "vote_threshold": vote_threshold,
            "chromatic_candidates": len(markers),
            "rejected_glare": len(rejected),
        }
        try:
            fused = fuse_detections(vsp_points, vsp_sizes, markers, config)
        except ValueError as error:
            final_error = str(error)
            trial["accepted"] = False
            trial["reason"] = final_error
            trials.append(trial)
            continue
        trial["accepted"] = True
        trial["hough_supplements"] = len(fused["selected_supplements"])
        trials.append(trial)
        detector_record["selection_policy"] = (
            "highest vote threshold that supplies all strictly validated VSP gaps"
        )
        detector_record["threshold_trials"] = trials
        return markers, rejected, detector_record, fused
    raise ValueError(
        "Strict Hough could not supply every measured VSP gap down to vote "
        f"threshold {config.supplement_hough_min_vote_threshold}: {final_error}"
    )


def _heading(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 40), (13, 16, 21), -1)
    cv2.putText(
        output,
        text,
        (9, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _draw_circle(
    image: np.ndarray,
    point: tuple[float, float] | np.ndarray,
    colour: tuple[int, int, int],
    radius: int = 5,
    thickness: int = 1,
) -> None:
    cv2.circle(
        image,
        (int(round(float(point[0]))), int(round(float(point[1])))),
        radius,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def render_comparison(
    raw: np.ndarray,
    vsp_points: np.ndarray,
    hough_markers: list[dict[str, float]],
    fused: dict[str, Any],
) -> np.ndarray:
    raw_panel = _heading(raw, "Raw saved frame")
    vsp_panel = _heading(
        raw,
        f"Official VSP raw: {len(vsp_points)} | primary: {fused['clean_vsp_count']}",
    )
    for point in vsp_points:
        _draw_circle(vsp_panel, point, (60, 230, 80), radius=5, thickness=1)
    hough_panel = _heading(
        raw, f"Strict Hough candidates: {len(hough_markers)} measured"
    )
    for marker in hough_markers:
        _draw_circle(
            hough_panel,
            (marker["x_px"], marker["y_px"]),
            (0, 215, 255),
            radius=max(3, int(round(marker["hough_radius_px"]))),
            thickness=1,
        )
    final_markers = fused["final_markers"]
    hybrid_panel = _heading(
        raw,
        f"VSP-primary final: {len(final_markers)} | Hough fills: "
        f"{len(fused['selected_supplements'])}",
    )
    colours = {
        "vsp_primary": (60, 230, 80),
        "hough_supplement": (245, 80, 245),
    }
    for marker in final_markers:
        _draw_circle(
            hybrid_panel,
            (marker["x_px"], marker["y_px"]),
            colours[marker["source"]],
            radius=max(3, min(8, int(round(marker["radius_px"])))),
            thickness=2 if marker["source"] == "hough_supplement" else 1,
        )
    return np.vstack((np.hstack((raw_panel, vsp_panel)), np.hstack((hough_panel, hybrid_panel))))


def _component_count(binary: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(binary, connectivity=8)
    return int(count - 1)


def render_binary_outputs(
    final_markers: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    expected_markers: int,
    marker_radius_px: int,
) -> dict[str, Any]:
    """Render all measured centres with identical brightness and radius."""
    points = np.asarray(
        [[marker["x_px"], marker["y_px"]] for marker in final_markers], dtype=float
    )
    if len(points) != expected_markers:
        raise ValueError(
            f"Cannot render {len(points)} points as an expected-{expected_markers} mask"
        )
    height, width = image_shape[:2]
    padding = max(8, int(round(min(height, width) * 0.035)))
    left = max(0, int(np.floor(points[:, 0].min())) - padding)
    right = min(width, int(np.ceil(points[:, 0].max())) + padding + 1)
    top = max(0, int(np.floor(points[:, 1].min())) - padding)
    bottom = min(height, int(np.ceil(points[:, 1].max())) + padding + 1)

    binary_native = np.zeros((height, width), dtype=np.uint8)
    for marker in final_markers:
        cv2.circle(
            binary_native,
            (int(round(marker["x_px"])), int(round(marker["y_px"]))),
            int(marker_radius_px),
            255,
            -1,
            cv2.LINE_8,
        )
    binary_crop = binary_native[top:bottom, left:right]
    crop_height, crop_width = binary_crop.shape
    square_side = max(crop_height, crop_width)
    square_crop = np.zeros((square_side, square_side), dtype=np.uint8)
    square_left = (square_side - crop_width) // 2
    square_top = (square_side - crop_height) // 2
    square_crop[
        square_top : square_top + crop_height,
        square_left : square_left + crop_width,
    ] = binary_crop
    soft_256 = cv2.resize(square_crop, (256, 256), interpolation=cv2.INTER_AREA)
    binary_256 = np.where(soft_256 >= 80, 255, 0).astype(np.uint8)
    native_components = _component_count(binary_native)
    model_components = _component_count(binary_256)
    if native_components != expected_markers or model_components != expected_markers:
        raise ValueError(
            "Fixed-radius marker circles merged or disappeared during binary rendering: "
            f"native={native_components}, model={model_components}, "
            f"expected={expected_markers}, radius={marker_radius_px}"
        )
    return {
        "binary_native": binary_native,
        "binary_crop": binary_crop,
        "binary_square_crop": square_crop,
        "binary_256": binary_256,
        "fixed_marker_radius_px": int(marker_radius_px),
        "native_components": native_components,
        "model_components": model_components,
        "crop_bounds_xyxy": [left, top, right, bottom],
        "square_padding_ltrb": [
            square_left,
            square_top,
            square_side - crop_width - square_left,
            square_side - crop_height - square_top,
        ],
    }


def render_binary_comparison(
    raw: np.ndarray,
    detector_comparison: np.ndarray,
    binary_native: np.ndarray,
    binary_256: np.ndarray,
) -> np.ndarray:
    height, width = raw.shape[:2]
    hybrid_panel = detector_comparison[height : 2 * height, width : 2 * width]
    native_panel = cv2.cvtColor(binary_native, cv2.COLOR_GRAY2BGR)
    model_side = min(height, width)
    model_square = cv2.cvtColor(
        cv2.resize(
            binary_256,
            (model_side, model_side),
            interpolation=cv2.INTER_NEAREST,
        ),
        cv2.COLOR_GRAY2BGR,
    )
    model_panel = np.zeros((height, width, 3), dtype=np.uint8)
    model_left = (width - model_side) // 2
    model_panel[:, model_left : model_left + model_side] = model_square
    return np.vstack(
        (
            np.hstack((_heading(raw, "Raw saved frame"), hybrid_panel)),
            np.hstack(
                (
                    _heading(native_panel, "VSP-primary binary, native 640 x 480"),
                    _heading(model_panel, "Cropped model input, 256 x 256"),
                )
            ),
        )
    )


def depth_from_name(path: Path) -> float | None:
    match = re.search(r"_capture_([0-9]+(?:\.[0-9]+)?)mm", path.stem)
    return float(match.group(1)) if match else None


def build_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = []
    for row in rows:
        supplements = row["hybrid"]["selected_supplements"]
        cards.append(
            "<article>"
            f"<h3>{row['index']:02d}. {html.escape(row['frame'])}</h3>"
            f"<p>Depth {row['depth_mm']:.2f} mm | VSP raw {row['vsp_raw_count']} | "
            f"VSP primary {row['vsp_primary_count']} | Hough candidates "
            f"{row['hough_candidate_count']} | Hough fills {len(supplements)} | "
            f"final <b>{row['hybrid_count']}</b></p>"
            f"<a href=\"{html.escape(row['overlay'])}\"><img loading=\"lazy\" "
            f"src=\"{html.escape(row['overlay'])}\"></a>"
            f"<a href=\"{html.escape(row['binary_preview'])}\"><img loading=\"lazy\" "
            f"src=\"{html.escape(row['binary_preview'])}\"></a>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VSP-primary marker detection with Hough gap filling</title>
<style>
body {{ margin:0; background:#0c1118; color:#eef3f8; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1400px,96vw); margin:auto; padding:24px 0 60px; }}
h1 {{ margin:0; }} p {{ color:#b8c4d1; line-height:1.5; }} code {{ color:#8fc8ff; }}
.legend span {{ margin-right:18px; }} .green {{color:#3ce650}} .yellow {{color:#00dcff}} .magenta {{color:#f550f5}}
article {{ background:#141c27; border:1px solid #2c3949; border-radius:6px; padding:12px; margin:18px 0; }}
article h3 {{ margin:0; }} img {{ width:100%; height:auto; display:block; margin-top:10px; }}
</style></head><body><main>
<h1>Official VSP primary + measured Hough gap filling</h1>
<p>VSP owns every marker it detects. Its centres are not averaged with or shifted toward Hough. Hough starts at a strict vote threshold of {summary['config']['supplement_hough_max_vote_threshold']} and is lowered only as far as {summary['config']['supplement_hough_min_vote_threshold']} when a frame still has measured VSP gaps. Candidates must also pass chromatic, lattice-neighbour, duplicate-distance and radial-boundary gates.</p>
<p class="legend"><span class="green">Green: VSP primary</span><span class="yellow">Yellow: all measured Hough candidates</span><span class="magenta">Magenta: selected Hough supplements</span></p>
<p>Frames: {summary['frames']}; final count range: {summary['hybrid_count_min']}-{summary['hybrid_count_max']}; exact 331: {summary['exact_331_frames']}/{summary['frames']}; Hough supplements: {summary['total_hough_supplements']} across {summary['frames_with_hough_supplements']} frames; selected Hough vote range: {summary['selected_hough_vote_threshold_range'][0]:g}-{summary['selected_hough_vote_threshold_range'][1]:g}; synthetic/inferred points: 0.</p>
<p>Binary validation: native exact 331 components in {summary['native_binary_exact_331_frames']}/{summary['frames']} frames; 256 x 256 exact 331 components in {summary['model_binary_exact_331_frames']}/{summary['frames']} frames. All markers use value 255 and the same fixed {summary['config']['binary_marker_radius_px']} px radius. The crop is padded to a square before resizing, so it is not stretched.</p>
<p>VSP commit: <code>{summary['vsp_commit']}</code>.</p>
<a href="hybrid_overview.png"><img src="hybrid_overview.png" style="width:100%"></a>
{''.join(cards)}
</main></body></html>"""


def _make_overview(rows: list[dict[str, Any]], output_dir: Path) -> None:
    targets = (1.1, 2.1, 3.1, 4.3, 5.3, 5.9)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in targets:
        eligible = [row for row in rows if row["frame"] not in used]
        row = min(eligible, key=lambda item: abs(item["depth_mm"] - target))
        chosen.append(row)
        used.add(row["frame"])
    tiles = []
    for row in chosen:
        image = cv2.imread(str(output_dir / row["overlay"]), cv2.IMREAD_COLOR)
        tile = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    overview = np.vstack(
        (np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:]))
    )
    cv2.imwrite(str(output_dir / "hybrid_overview.png"), overview)


def main() -> int:
    args = parse_args()
    config = HybridConfig()
    vsp_root = args.vsp_root.resolve()
    detector = make_official_vsp_detector(vsp_root)
    output_dir = args.output_dir.resolve()
    overlay_dir = output_dir / "overlays"
    binary_native_dir = output_dir / "binary_native_640x480"
    binary_model_dir = output_dir / "binary_model_input_256"
    binary_preview_dir = output_dir / "binary_previews"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    binary_native_dir.mkdir(parents=True, exist_ok=True)
    binary_model_dir.mkdir(parents=True, exist_ok=True)
    binary_preview_dir.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[str, Path]] = []
    for run_dir_arg in args.run_dir:
        run_dir = run_dir_arg.resolve()
        frames.extend(
            (run_dir.name, frame)
            for frame in sorted((run_dir / "frames").glob("*_capture_*.png"))
        )
    if not frames:
        raise FileNotFoundError("No *_capture_*.png frames found")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, (run_name, frame_path) in enumerate(frames, start=1):
        raw = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if raw is None:
            failures.append({"frame": str(frame_path), "error": "cv2.imread failed"})
            continue
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        vsp_points = np.asarray([item.point for item in keypoints], dtype=float).reshape(-1, 2)
        vsp_sizes = np.asarray([item.size for item in keypoints], dtype=float)
        try:
            hough_markers, rejected_hough, hough_record, fused = (
                strict_hough_gap_fill(raw, vsp_points, vsp_sizes, config)
            )
        except ValueError as error:
            failures.append({"frame": str(frame_path), "error": str(error)})
            continue

        comparison = render_comparison(raw, vsp_points, hough_markers, fused)
        binary_outputs = render_binary_outputs(
            fused["final_markers"],
            raw.shape,
            config.expected_markers,
            config.binary_marker_radius_px,
        )
        relative_overlay = Path("overlays") / f"{run_name}__{frame_path.stem}.jpg"
        relative_binary_native = (
            Path("binary_native_640x480") / f"{run_name}__{frame_path.stem}.png"
        )
        relative_binary_model = (
            Path("binary_model_input_256") / f"{run_name}__{frame_path.stem}.png"
        )
        relative_binary_preview = (
            Path("binary_previews") / f"{run_name}__{frame_path.stem}.jpg"
        )
        cv2.imwrite(
            str(output_dir / relative_overlay),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        cv2.imwrite(
            str(output_dir / relative_binary_native), binary_outputs["binary_native"]
        )
        cv2.imwrite(
            str(output_dir / relative_binary_model), binary_outputs["binary_256"]
        )
        binary_comparison = render_binary_comparison(
            raw,
            comparison,
            binary_outputs["binary_native"],
            binary_outputs["binary_256"],
        )
        cv2.imwrite(
            str(output_dir / relative_binary_preview),
            binary_comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        depth_mm = depth_from_name(frame_path)
        if depth_mm is None:
            depth_mm = float("nan")
        rows.append(
            {
                "index": index,
                "run": run_name,
                "frame": frame_path.name,
                "source": str(frame_path),
                "depth_mm": depth_mm,
                "vsp_raw_count": int(len(vsp_points)),
                "vsp_primary_count": int(fused["clean_vsp_count"]),
                "hough_candidate_count": int(len(hough_markers)),
                "hough_rejected_glare_count": int(len(rejected_hough)),
                "hough_vote_threshold": float(hough_record["vote_threshold"]),
                "hybrid_count": int(len(fused["final_markers"])),
                "hough_candidate_detector": hough_record,
                "vsp_points_xy": vsp_points.tolist(),
                "hough_markers": hough_markers,
                "hybrid": fused,
                "overlay": str(relative_overlay),
                "binary_preview": str(relative_binary_preview),
                "binary_native": str(relative_binary_native),
                "binary_model_input_256": str(relative_binary_model),
                "binary": {
                    key: value
                    for key, value in binary_outputs.items()
                    if not isinstance(value, np.ndarray)
                },
            }
        )

    if not rows:
        raise RuntimeError(f"Every frame failed: {failures}")
    hybrid_counts = np.asarray([row["hybrid_count"] for row in rows], dtype=int)
    supplement_counts = np.asarray(
        [len(row["hybrid"]["selected_supplements"]) for row in rows], dtype=int
    )
    vsp_raw_counts = np.asarray([row["vsp_raw_count"] for row in rows], dtype=int)
    vsp_primary_counts = np.asarray(
        [row["vsp_primary_count"] for row in rows], dtype=int
    )
    hough_candidate_counts = np.asarray(
        [row["hough_candidate_count"] for row in rows], dtype=int
    )
    hough_vote_thresholds = np.asarray(
        [row["hough_vote_threshold"] for row in rows], dtype=float
    )
    native_component_counts = np.asarray(
        [row["binary"]["native_components"] for row in rows], dtype=int
    )
    model_component_counts = np.asarray(
        [row["binary"]["model_components"] for row in rows], dtype=int
    )
    commit = subprocess.check_output(
        ["git", "-C", str(vsp_root), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "schema": "tactip_vsp_primary_hough_supplement.v2",
        "strategy": "official VSP primary; Hough only fills measured missing markers",
        "frames": len(rows),
        "failed_frames": len(failures),
        "expected_markers": config.expected_markers,
        "hybrid_count_min": int(hybrid_counts.min()),
        "hybrid_count_max": int(hybrid_counts.max()),
        "exact_331_frames": int(np.count_nonzero(hybrid_counts == config.expected_markers)),
        "vsp_raw_count_range": [int(vsp_raw_counts.min()), int(vsp_raw_counts.max())],
        "vsp_primary_count_range": [
            int(vsp_primary_counts.min()),
            int(vsp_primary_counts.max()),
        ],
        "hough_candidate_count_range": [
            int(hough_candidate_counts.min()),
            int(hough_candidate_counts.max()),
        ],
        "selected_hough_vote_threshold_range": [
            float(hough_vote_thresholds.min()),
            float(hough_vote_thresholds.max()),
        ],
        "selected_hough_vote_threshold_counts": {
            str(int(value)): int(np.count_nonzero(hough_vote_thresholds == value))
            for value in np.unique(hough_vote_thresholds)
        },
        "total_hough_supplements": int(supplement_counts.sum()),
        "frames_with_hough_supplements": int(np.count_nonzero(supplement_counts)),
        "native_binary_exact_331_frames": int(
            np.count_nonzero(native_component_counts == config.expected_markers)
        ),
        "model_binary_exact_331_frames": int(
            np.count_nonzero(model_component_counts == config.expected_markers)
        ),
        "synthetic_or_inferred_points": 0,
        "config": asdict(config),
        "vsp_implementation": "unmodified vsp.detector.CvBlobDetector",
        "vsp_parameters_source": "vsp/examples/processor_test.py",
        "vsp_commit": commit,
        "run_dirs": [str(path.resolve()) for path in args.run_dir],
    }
    payload = {"summary": summary, "failures": failures, "frames": rows}
    (output_dir / "hybrid_detections.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _make_overview(rows, output_dir)
    report = output_dir / "hybrid_results.html"
    report.write_text(build_html(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(report)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
