#!/usr/bin/env python3
"""Batch John Lloyd-style marker-density KDE over audited TacTip detections.

The calculation intentionally follows ``tactile_image_processing/kernel_density.py``:

    K = 1 / (2*pi*h) * exp(-d^2 / (2*h^2))
    rho = mean(K, marker_axis) / normalization
    delta = rho_contact - rho_undeformed

The Gaussian prefactor is kept verbatim, even though a conventionally normalized
2-D Gaussian would contain ``h**2`` in the denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CAPTURE_SUFFIX = re.compile(r"_capture_[0-9]+(?:\.[0-9]+)?mm$")


@dataclass(frozen=True)
class KdeConfig:
    bbox_xyxy: tuple[float, float, float, float] = (130.0, 54.0, 550.0, 474.0)
    grid_size: int = 200
    kernel_width_px: float = 15.0
    normalization: float = 5.0e-5
    display_mask_radius_grid_px: float = 100.0
    reference_match_radius_px: float = 6.0
    expected_markers: int = 331
    paper_vlim: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate marker-density difference maps for audited TacTip frames."
    )
    parser.add_argument("--detections-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(130.0, 54.0, 550.0, 474.0),
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
    )
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--kernel-width", type=float, default=15.0)
    parser.add_argument("--normalization", type=float, default=5.0e-5)
    parser.add_argument("--mask-radius", type=float, default=100.0)
    parser.add_argument("--reference-match-radius", type=float, default=6.0)
    parser.add_argument(
        "--reference-step",
        type=Path,
        help=(
            "Use the 331 camera-facing marker-tip centres from this STEP file as "
            "the undeformed reference. Paired baseline images are then used only "
            "to calibrate the 3-D CAD-to-pixel projection."
        ),
    )
    return parser.parse_args()


def marker_points_from_records(records: list[dict[str, Any]]) -> np.ndarray:
    points = np.asarray(
        [[float(item["x_px"]), float(item["y_px"])] for item in records],
        dtype=np.float64,
    )
    return points.reshape(-1, 2)


def binary_component_centroids(path: Path, expected: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.where(image > 0, 255, 0).astype(np.uint8), connectivity=8
    )
    points = centroids[1:].astype(np.float64)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(points) != expected:
        raise ValueError(
            f"{path} contains {len(points)} foreground components; expected {expected}"
        )
    # Stable ordering makes the selected anchor and diagnostics deterministic.
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order]


def paired_baseline_path(frame_row: dict[str, Any]) -> Path:
    source = Path(frame_row["source"]).expanduser().resolve()
    run_dir = source.parent.parent
    sample_id = CAPTURE_SUFFIX.sub("", source.stem)
    path = run_dir / "tactip_preprocessed" / "contact_binary" / f"{sample_id}_baseline.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_robust_reference(
    baseline_paths: list[Path], config: KdeConfig
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Match measured baseline points to one anchor and take per-marker medians."""
    point_sets = [
        binary_component_centroids(path, config.expected_markers)
        for path in baseline_paths
    ]
    anchor = point_sets[0]
    observations: list[list[np.ndarray]] = [[point] for point in anchor]
    baseline_records: list[dict[str, Any]] = [
        {
            "path": str(baseline_paths[0]),
            "matched": config.expected_markers,
            "unmatched_anchor": 0,
            "unmatched_frame": 0,
            "maximum_accepted_distance_px": 0.0,
        }
    ]
    for path, points in zip(baseline_paths[1:], point_sets[1:]):
        distances = np.linalg.norm(anchor[:, None, :] - points[None, :, :], axis=2)
        anchor_indices, point_indices = linear_sum_assignment(distances)
        accepted = distances[anchor_indices, point_indices] <= config.reference_match_radius_px
        accepted_anchor = anchor_indices[accepted]
        accepted_points = point_indices[accepted]
        for anchor_index, point_index in zip(accepted_anchor, accepted_points):
            observations[int(anchor_index)].append(points[int(point_index)])
        accepted_distances = distances[accepted_anchor, accepted_points]
        baseline_records.append(
            {
                "path": str(path),
                "matched": int(np.count_nonzero(accepted)),
                "unmatched_anchor": int(config.expected_markers - np.count_nonzero(accepted)),
                "unmatched_frame": int(config.expected_markers - np.count_nonzero(accepted)),
                "maximum_accepted_distance_px": (
                    float(np.max(accepted_distances)) if len(accepted_distances) else None
                ),
            }
        )
    reference = np.asarray(
        [np.median(np.asarray(items), axis=0) for items in observations],
        dtype=np.float64,
    )
    support = np.asarray([len(items) for items in observations], dtype=np.int32)
    return reference, support, baseline_records


def extract_step_marker_tips(step_path: Path, expected: int) -> np.ndarray:
    """Return camera-facing marker-tip centres in assembled STEP coordinates.

    The 331 pins in the supplied TacTip STEP each contain two planar caps and a
    toroidal side/fillet.  The smaller cap has radius 0.5 mm and faces inward,
    toward the camera.  Selecting by topology and then by cap area avoids using
    either the skin/housing solids or the pin centre of mass.
    """
    try:
        import cadquery as cq
    except ImportError as error:
        raise RuntimeError(
            "STEP reference mode requires CadQuery (for example: "
            "uv run --with cadquery --with opencv-python --with scipy "
            "--with matplotlib python tools/batch_marker_density_kde.py ...)"
        ) from error

    shape = cq.importers.importStep(str(step_path)).val()
    marker_solids = [
        solid
        for solid in shape.Solids()
        if tuple(sorted(face.geomType() for face in solid.Faces()))
        == ("PLANE", "PLANE", "TORUS")
    ]
    if len(marker_solids) != expected:
        raise ValueError(
            f"{step_path} contains {len(marker_solids)} pin-shaped solids; "
            f"expected {expected}"
        )

    points: list[tuple[float, float, float]] = []
    for solid in marker_solids:
        planar_faces = [face for face in solid.Faces() if face.geomType() == "PLANE"]
        visible_tip = min(planar_faces, key=lambda face: face.Area())
        expected_tip_area = np.pi * 0.5**2
        if not np.isclose(visible_tip.Area(), expected_tip_area, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"Unexpected camera-facing marker cap area {visible_tip.Area():.12g} "
                f"in {step_path}"
            )
        points.append(tuple(float(value) for value in visible_tip.Center().toTuple()))
    return np.asarray(points, dtype=np.float64)


def concentric_ring_groups(points: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Partition this 331-marker design into its 1, 6, ..., 60 point rings."""
    counts = [1] + [6 * ring for ring in range(1, 11)]
    if len(points) != sum(counts):
        raise ValueError(f"Concentric-ring grouping expects {sum(counts)} points")
    centre_index = int(np.argmin(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
    radius = np.linalg.norm(points - points[centre_index], axis=1)
    order = np.argsort(radius)
    groups: list[np.ndarray] = []
    start = 0
    for count in counts:
        groups.append(order[start : start + count])
        start += count
    return groups, radius


def associate_step_tips_to_pixels(
    cad_xyz: np.ndarray,
    calibration_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Associate unlabeled CAD tips to measured pixels without changing CAD geometry."""
    cad_xz = cad_xyz[:, [0, 2]]
    cad_groups, cad_radius = concentric_ring_groups(cad_xz)
    pixel_groups, pixel_radius = concentric_ring_groups(calibration_points)
    cad_centre = cad_xz[cad_groups[0][0]]
    pixel_centre = calibration_points[pixel_groups[0][0]]
    scale = float(
        np.median(
            [
                np.mean(pixel_radius[pixel_group]) / np.mean(cad_radius[cad_group])
                for cad_group, pixel_group in zip(cad_groups[1:], pixel_groups[1:])
            ]
        )
    )

    best: tuple[float, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    # The design has sixfold symmetry, so [0, 60) degrees plus reflection spans
    # every distinguishable unlabeled correspondence.
    for reflection in (1, -1):
        oriented = cad_xz.copy()
        oriented[:, 1] *= reflection
        for angle_degrees in np.linspace(0.0, 60.0, 61, endpoint=False):
            angle = np.deg2rad(angle_degrees)
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float64,
            )
            predicted = (oriented - cad_centre) @ rotation.T * scale + pixel_centre
            assignment = np.empty(len(cad_xyz), dtype=np.int32)
            for cad_group, pixel_group in zip(cad_groups, pixel_groups):
                distance = np.linalg.norm(
                    predicted[cad_group, None, :]
                    - calibration_points[pixel_group][None, :, :],
                    axis=2,
                )
                cad_local, pixel_local = linear_sum_assignment(distance)
                assignment[cad_group[cad_local]] = pixel_group[pixel_local]
            design = np.column_stack((oriented, np.ones(len(oriented))))
            affine = np.linalg.lstsq(
                design, calibration_points[assignment], rcond=None
            )[0]
            affine_prediction = design @ affine
            error = np.linalg.norm(
                affine_prediction - calibration_points[assignment], axis=1
            )
            score = float(np.sqrt(np.mean(error**2)))
            candidate = (
                score,
                reflection,
                float(angle_degrees),
                oriented,
                assignment,
                affine,
            )
            if best is None or score < best[0]:
                best = candidate

    assert best is not None
    score, reflection, angle_degrees, oriented, assignment, affine = best
    # One assignment/refit pass removes sensitivity to the coarse angle grid.
    design = np.column_stack((oriented, np.ones(len(oriented))))
    affine_prediction = design @ affine
    for cad_group, pixel_group in zip(cad_groups, pixel_groups):
        distance = np.linalg.norm(
            affine_prediction[cad_group, None, :]
            - calibration_points[pixel_group][None, :, :],
            axis=2,
        )
        cad_local, pixel_local = linear_sum_assignment(distance)
        assignment[cad_group[cad_local]] = pixel_group[pixel_local]
    affine = np.linalg.lstsq(design, calibration_points[assignment], rcond=None)[0]
    affine_error = np.linalg.norm(design @ affine - calibration_points[assignment], axis=1)
    metadata = {
        "initial_scale_px_per_mm": scale,
        "orientation_search_degrees": angle_degrees,
        "reflection": int(reflection),
        "affine_association_rms_px": float(np.sqrt(np.mean(affine_error**2))),
    }
    return assignment, metadata


def project_points(points_xyz: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    projected = homogeneous @ camera_matrix.T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1.0e-9):
        raise ValueError("Degenerate CAD-to-pixel projective camera matrix")
    return projected[:, :2] / denominator[:, None]


def fit_projective_camera(
    points_xyz: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an uncalibrated pinhole matrix with DLT, then refine reprojection error."""
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    equations: list[np.ndarray] = []
    for point, (u, v) in zip(homogeneous, points_xy):
        equations.append(np.concatenate((point, np.zeros(4), -u * point)))
        equations.append(np.concatenate((np.zeros(4), point, -v * point)))
    _u, _singular, vh = np.linalg.svd(np.asarray(equations, dtype=np.float64))
    initial = vh[-1].reshape(3, 4)
    initial /= initial[2, 3]

    parameters = np.concatenate((initial[0], initial[1], initial[2, :3]))

    def unpack(values: np.ndarray) -> np.ndarray:
        return np.vstack((values[:4], values[4:8], np.append(values[8:11], 1.0)))

    def residual(values: np.ndarray) -> np.ndarray:
        return (project_points(points_xyz, unpack(values)) - points_xy).ravel()

    optimized = least_squares(
        residual,
        parameters,
        method="lm",
        max_nfev=10000,
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
    )
    camera_matrix = unpack(optimized.x)
    projected = project_points(points_xyz, camera_matrix)
    error = np.linalg.norm(projected - points_xy, axis=1)
    return camera_matrix, projected, error


def make_grid(config: KdeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, top, right, bottom = config.bbox_xyxy
    x = np.linspace(left, right, config.grid_size)
    y = np.linspace(top, bottom, config.grid_size)
    xx, yy = np.meshgrid(x, y)
    return x, y, np.column_stack((xx.ravel(), yy.ravel()))


def marker_density(points: np.ndarray, grid: np.ndarray, config: KdeConfig) -> np.ndarray:
    squared_distance = cdist(grid, points[:, :2], metric="sqeuclidean")
    kernel = (1.0 / (2.0 * np.pi * config.kernel_width_px)) * np.exp(
        -squared_distance / (2.0 * config.kernel_width_px**2)
    )
    density = np.mean(kernel, axis=1) / config.normalization
    return density.reshape(config.grid_size, config.grid_size)


def circular_display_mask(config: KdeConfig) -> np.ndarray:
    yy, xx = np.ogrid[: config.grid_size, : config.grid_size]
    centre = (config.grid_size - 1) / 2.0
    return (xx - centre) ** 2 + (yy - centre) ** 2 <= config.display_mask_radius_grid_px**2


def short_label(row: dict[str, Any]) -> str:
    sample_id = CAPTURE_SUFFIX.sub("", Path(row["frame"]).stem)
    match = re.search(r"(g\d+_d\d+)$", sample_id)
    return match.group(1) if match else sample_id


def save_frame_panel(
    row: dict[str, Any],
    points: np.ndarray,
    delta: np.ndarray,
    mask: np.ndarray,
    config: KdeConfig,
    output_path: Path,
) -> None:
    raw = cv2.imread(str(Path(row["source"])), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(row["source"])
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    left, top, right, bottom = (int(round(value)) for value in config.bbox_xyxy)
    cropped = raw[top:bottom, left:right]
    local_points = points - np.asarray([left, top], dtype=np.float64)
    shown = np.where(mask, delta, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), constrained_layout=True)
    axes[0].imshow(cropped)
    axes[0].scatter(
        local_points[:, 0],
        local_points[:, 1],
        s=8,
        facecolors="none",
        edgecolors="#25ff64",
        linewidths=0.55,
    )
    axes[0].set_title(f"Detected markers ({len(points)})")
    axes[0].axis("off")
    image = axes[1].imshow(
        shown,
        cmap="jet",
        vmin=-config.paper_vlim,
        vmax=config.paper_vlim,
        origin="upper",
    )
    axes[1].set_title("Marker-density change")
    axes[1].axis("off")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\Delta\rho / (5\times10^{-5})$")
    fig.suptitle(
        f"{short_label(row)}  |  contact depth {float(row['depth_mm']):.2f} mm",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def save_overview(
    results: list[dict[str, Any]],
    mask: np.ndarray,
    output_path: Path,
    vlim: float,
    title: str,
) -> None:
    rows, columns = 6, 6
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15), constrained_layout=True)
    plotted = None
    for axis, result in zip(axes.ravel(), results):
        plotted = axis.imshow(
            np.where(mask, result["delta"], np.nan),
            cmap="jet",
            vmin=-vlim,
            vmax=vlim,
            origin="upper",
        )
        axis.set_title(
            f"{result['label']}\n{result['depth_mm']:.2f} mm",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.ravel()[len(results) :]:
        axis.axis("off")
    if plotted is not None:
        colorbar = fig.colorbar(plotted, ax=axes, shrink=0.72, pad=0.012)
        colorbar.set_label(r"$\Delta\rho / (5\times10^{-5})$")
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def save_reference_diagnostics(
    anchor_path: Path,
    reference: np.ndarray,
    support: np.ndarray,
    config: KdeConfig,
    output_path: Path,
) -> None:
    binary = cv2.imread(str(anchor_path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(anchor_path)
    left, top, right, bottom = (int(round(value)) for value in config.bbox_xyxy)
    local = reference - np.asarray([left, top], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].imshow(binary[top:bottom, left:right], cmap="gray", vmin=0, vmax=255)
    points = axes[0].scatter(
        local[:, 0], local[:, 1], c=support, cmap="viridis", s=12, vmin=1, vmax=len(support)
    )
    axes[0].set_title("Robust undeformed reference")
    axes[0].axis("off")
    fig.colorbar(points, ax=axes[0], fraction=0.046, pad=0.04, label="baseline support")
    bins = np.arange(int(support.min()), int(support.max()) + 2) - 0.5
    axes[1].hist(support, bins=bins, color="#4472c4", edgecolor="white")
    axes[1].set_xlabel("Matched baseline observations per marker")
    axes[1].set_ylabel("Markers")
    axes[1].set_title(f"Support range: {int(support.min())}-{int(support.max())}")
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def save_step_reference_diagnostics(
    calibration_image_path: Path,
    reference: np.ndarray,
    reprojection_error: np.ndarray,
    calibration_points: np.ndarray,
    config: KdeConfig,
    output_path: Path,
) -> None:
    """Show the CAD-derived reference and its one-time pixel-registration residuals."""
    binary = cv2.imread(str(calibration_image_path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(calibration_image_path)
    left, top, right, bottom = (int(round(value)) for value in config.bbox_xyxy)
    offset = np.asarray([left, top], dtype=np.float64)
    reference_local = reference - offset
    measured_local = calibration_points - offset
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].imshow(binary[top:bottom, left:right], cmap="gray", vmin=0, vmax=255)
    axes[0].scatter(
        measured_local[:, 0],
        measured_local[:, 1],
        s=9,
        facecolors="none",
        edgecolors="#00d7ff",
        linewidths=0.55,
        label="baseline calibration target",
    )
    plotted = axes[0].scatter(
        reference_local[:, 0],
        reference_local[:, 1],
        c=reprojection_error,
        cmap="magma",
        s=10,
        vmin=0.0,
    )
    axes[0].set_title("331 STEP marker tips projected to pixels")
    axes[0].axis("off")
    figure.colorbar(
        plotted,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="CAD projection residual (px)",
    )
    axes[1].hist(reprojection_error, bins=24, color="#8e44ad", edgecolor="white")
    axes[1].axvline(
        float(np.sqrt(np.mean(reprojection_error**2))),
        color="#d1495b",
        linewidth=1.7,
        label="RMS",
    )
    axes[1].set_xlabel("CAD-to-pixel reprojection residual (px)")
    axes[1].set_ylabel("Markers")
    axes[1].set_title(
        "RMS {:.3f} px; p95 {:.3f} px".format(
            float(np.sqrt(np.mean(reprojection_error**2))),
            float(np.quantile(reprojection_error, 0.95)),
        )
    )
    axes[1].legend()
    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)


def save_depth_response(metrics: list[dict[str, Any]], output_path: Path) -> float:
    depth = np.asarray([row["depth_mm"] for row in metrics], dtype=np.float64)
    rms = np.asarray([row["rms_delta_masked"] for row in metrics], dtype=np.float64)
    correlation = float(np.corrcoef(depth, rms)[0, 1])
    slope, intercept = np.polyfit(depth, rms, 1)
    xline = np.linspace(float(depth.min()), float(depth.max()), 200)
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.scatter(depth, rms, color="#285f9e", s=42, alpha=0.85)
    axis.plot(xline, slope * xline + intercept, color="#d1495b", linewidth=1.8)
    axis.set_xlabel("Recorded contact depth (mm)")
    axis.set_ylabel(r"RMS marker-density change $\Delta\rho/(5\times10^{-5})$")
    axis.set_title(f"KDE response versus depth (Pearson r = {correlation:.3f})")
    axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    return correlation


def write_report(
    output_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    cards = []
    for result in results:
        panel = html.escape(result["panel"])
        cards.append(
            "<article><h3>{index:02d}. {label} &mdash; {depth:.2f} mm</h3>"
            '<a href="{panel}"><img loading="lazy" src="{panel}"></a>'
            "<p>RMS {rms:.4f}; range [{minimum:.4f}, {maximum:.4f}]; "
            "paper-scale saturation {saturation:.2%}.</p></article>".format(
                index=result["index"],
                label=html.escape(result["label"]),
                depth=result["depth_mm"],
                panel=panel,
                rms=result["rms_delta_masked"],
                minimum=result["minimum_delta_masked"],
                maximum=result["maximum_delta_masked"],
                saturation=result["paper_scale_saturation_fraction"],
            )
        )
    payload = html.escape(json.dumps(summary, indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TacTip marker-density KDE</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f4f6f8;color:#17202a}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin-bottom:4px}}
.hero{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .hero img{{width:100%;background:white;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px;margin-top:24px}}
article{{background:white;padding:14px;border-radius:9px;box-shadow:0 1px 5px #0002}} article img{{width:100%}}
pre{{background:#101820;color:#e8f1f5;padding:16px;overflow:auto;border-radius:8px}}
</style></head><body><main>
<h1>TacTip marker-density KDE: 34 contact frames</h1>
<p>John Lloyd-style Gaussian KDE, current marker density minus the configured undeformed reference.</p>
<div class="hero"><a href="overview_34_paper_scale.png"><img src="overview_34_paper_scale.png"></a>
<a href="depth_response.png"><img src="depth_response.png"></a></div>
<h2>Run summary</h2><pre>{payload}</pre>
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = KdeConfig(
        bbox_xyxy=tuple(float(value) for value in args.bbox),
        grid_size=int(args.grid_size),
        kernel_width_px=float(args.kernel_width),
        normalization=float(args.normalization),
        display_mask_radius_grid_px=float(args.mask_radius),
        reference_match_radius_px=float(args.reference_match_radius),
    )
    if config.grid_size <= 1 or config.kernel_width_px <= 0 or config.normalization <= 0:
        raise ValueError("grid size, kernel width and normalization must be positive")

    detections_path = args.detections_json.expanduser().resolve()
    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    rows = payload["frames"]
    if len(rows) != 34:
        raise ValueError(f"Expected the audited 34-frame set; found {len(rows)}")

    output_dir = args.output_dir.expanduser().resolve()
    panel_dir = output_dir / "per_frame"
    array_dir = output_dir / "arrays"
    panel_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    baseline_paths = [paired_baseline_path(row) for row in rows]
    calibration_points, calibration_support, baseline_records = build_robust_reference(
        baseline_paths, config
    )
    if int(calibration_support.min()) < len(rows) // 2:
        raise ValueError(
            "Baseline calibration target has inadequate support: minimum "
            f"{calibration_support.min()}"
        )

    cad_xyz: np.ndarray | None = None
    camera_matrix: np.ndarray | None = None
    reprojection_error: np.ndarray | None = None
    calibration_target: np.ndarray | None = None
    if args.reference_step is not None:
        step_path = args.reference_step.expanduser().resolve()
        if not step_path.is_file():
            raise FileNotFoundError(step_path)
        cad_xyz = extract_step_marker_tips(step_path, config.expected_markers)
        assignment, association_metadata = associate_step_tips_to_pixels(
            cad_xyz, calibration_points
        )
        calibration_target = calibration_points[assignment]
        camera_matrix, reference_points, reprojection_error = fit_projective_camera(
            cad_xyz, calibration_target
        )
        reference_support = calibration_support[assignment]

        nearest_distance = cdist(calibration_target, calibration_target)
        np.fill_diagonal(nearest_distance, np.inf)
        minimum_marker_spacing = float(np.min(nearest_distance))
        rms_error = float(np.sqrt(np.mean(reprojection_error**2)))
        p95_error = float(np.quantile(reprojection_error, 0.95))
        maximum_error = float(np.max(reprojection_error))
        if (
            rms_error >= 2.0
            or p95_error >= 3.0
            or maximum_error >= 0.5 * minimum_marker_spacing
        ):
            raise ValueError(
                "STEP-to-pixel registration failed acceptance thresholds: "
                f"RMS={rms_error:.3f}px, p95={p95_error:.3f}px, "
                f"max={maximum_error:.3f}px, min spacing={minimum_marker_spacing:.3f}px"
            )

        sphere_design = np.column_stack((2.0 * cad_xyz, np.ones(len(cad_xyz))))
        sphere_rhs = np.sum(cad_xyz**2, axis=1)
        sphere_solution = np.linalg.lstsq(sphere_design, sphere_rhs, rcond=None)[0]
        sphere_centre = sphere_solution[:3]
        sphere_radius = float(
            np.sqrt(sphere_solution[3] + np.dot(sphere_centre, sphere_centre))
        )
        sphere_residual = np.abs(
            np.linalg.norm(cad_xyz - sphere_centre, axis=1) - sphere_radius
        )
        cad_groups, cad_radius = concentric_ring_groups(cad_xyz[:, [0, 2]])
        step_digest = hashlib.sha256(step_path.read_bytes()).hexdigest()
        reference_record = {
            "method": (
                "331 camera-facing marker-tip centres extracted from STEP and "
                "projected by a fitted 3-D pinhole camera matrix"
            ),
            "geometry_source": "STEP marker-tip centres only",
            "step_file": str(step_path),
            "step_sha256": step_digest,
            "step_size_bytes": int(step_path.stat().st_size),
            "step_units": "mm",
            "marker_selection": (
                "solids with PLANE+PLANE+TORUS topology; centre of the smaller "
                "planar cap (radius 0.5 mm, facing inward/camera)"
            ),
            "marker_count": int(len(reference_points)),
            "cad_axis_convention": "x,z are lateral; y is the dome/optical axis",
            "cad_xyz_minimum_mm": np.min(cad_xyz, axis=0).tolist(),
            "cad_xyz_maximum_mm": np.max(cad_xyz, axis=0).tolist(),
            "cad_ring_counts": [int(len(group)) for group in cad_groups],
            "cad_ring_radii_mm": [
                float(np.mean(cad_radius[group])) for group in cad_groups
            ],
            "cad_sphere_centre_xyz_mm": sphere_centre.tolist(),
            "cad_sphere_radius_mm": sphere_radius,
            "cad_sphere_maximum_residual_mm": float(np.max(sphere_residual)),
            "pixel_registration": {
                "role_of_baselines": (
                    "calibrate only the CAD-mm to native-camera-pixel projection; "
                    "baseline marker coordinates are not the KDE reference points"
                ),
                "calibration_images": len(baseline_paths),
                "calibration_target_method": (
                    "per-marker median after one-to-one matching of measured "
                    "no-contact baselines"
                ),
                "camera_model": "3x4 uncalibrated pinhole/projective matrix",
                "camera_matrix": camera_matrix.tolist(),
                "rms_reprojection_error_px": rms_error,
                "median_reprojection_error_px": float(np.median(reprojection_error)),
                "p95_reprojection_error_px": p95_error,
                "maximum_reprojection_error_px": maximum_error,
                "minimum_calibration_marker_spacing_px": minimum_marker_spacing,
                "acceptance_thresholds": {
                    "rms_px_less_than": 2.0,
                    "p95_px_less_than": 3.0,
                    "maximum_error_less_than_fraction_of_minimum_spacing": 0.5,
                },
                "association": association_metadata,
                "d6_symmetry_note": (
                    "Unlabeled geometry is equivalent under rotations by 60 degrees "
                    "and reflection; these alternatives produce the same KDE point set."
                ),
            },
            "points_cad_xyz_mm": cad_xyz.tolist(),
            "points_xy": reference_points.tolist(),
            "cad_to_calibration_point_index": assignment.tolist(),
            "calibration_support_per_marker": reference_support.tolist(),
            "calibration_reprojection_error_px": reprojection_error.tolist(),
            "baseline_calibration_matching": baseline_records,
        }
    else:
        reference_points = calibration_points
        reference_support = calibration_support
        reference_record = {
            "method": (
                "per-marker median after one-to-one matching of 34 measured "
                "no-contact baselines"
            ),
            "anchor": str(baseline_paths[0]),
            "marker_count": int(len(reference_points)),
            "support_minimum": int(reference_support.min()),
            "support_maximum": int(reference_support.max()),
            "points_xy": reference_points.tolist(),
            "support": reference_support.tolist(),
            "baseline_matching": baseline_records,
        }

    x_grid, y_grid, grid = make_grid(config)
    reference_density = marker_density(reference_points, grid, config)
    mask = circular_display_mask(config)
    results: list[dict[str, Any]] = []
    all_masked_values: list[np.ndarray] = []

    for index, row in enumerate(rows, start=1):
        points = marker_points_from_records(row["hybrid"]["final_markers"])
        if len(points) != config.expected_markers:
            raise ValueError(f"{row['frame']} has {len(points)} markers")
        density = marker_density(points, grid, config)
        delta = density - reference_density
        masked = delta[mask]
        all_masked_values.append(masked)
        label = short_label(row)
        output_stem = f"{index:02d}_{label}_{float(row['depth_mm']):.2f}mm"
        panel_relative = Path("per_frame") / f"{output_stem}.png"
        array_relative = Path("arrays") / f"{output_stem}.npz"
        save_frame_panel(
            row,
            points,
            delta,
            mask,
            config,
            output_dir / panel_relative,
        )
        array_payload: dict[str, np.ndarray] = {
            "delta": delta.astype(np.float32),
            "density_current": density.astype(np.float32),
            "density_reference": reference_density.astype(np.float32),
            "points_current_xy": points.astype(np.float32),
            "points_reference_xy": reference_points.astype(np.float32),
            "reference_support": reference_support,
            "x_grid": x_grid.astype(np.float32),
            "y_grid": y_grid.astype(np.float32),
            "display_mask": mask,
        }
        if cad_xyz is not None and camera_matrix is not None:
            array_payload["points_reference_cad_xyz_mm"] = cad_xyz.astype(np.float32)
            array_payload["cad_to_pixel_camera_matrix"] = camera_matrix.astype(np.float64)
        np.savez_compressed(output_dir / array_relative, **array_payload)
        result = {
            "index": index,
            "label": label,
            "run": row["run"],
            "frame": row["frame"],
            "source": row["source"],
            "baseline": str(baseline_paths[index - 1]),
            "depth_mm": float(row["depth_mm"]),
            "marker_count": int(len(points)),
            "minimum_delta_masked": float(np.min(masked)),
            "maximum_delta_masked": float(np.max(masked)),
            "rms_delta_masked": float(np.sqrt(np.mean(masked**2))),
            "mean_absolute_delta_masked": float(np.mean(np.abs(masked))),
            "positive_mass_masked": float(np.sum(np.maximum(masked, 0.0))),
            "negative_mass_masked": float(np.sum(np.minimum(masked, 0.0))),
            "paper_scale_saturation_fraction": float(
                np.mean(np.abs(masked) > config.paper_vlim)
            ),
            "panel": str(panel_relative),
            "array": str(array_relative),
            "delta": delta,
        }
        results.append(result)

    pooled = np.concatenate(all_masked_values)
    maximum_absolute = float(np.max(np.abs(pooled)))
    full_vlim = math.ceil(maximum_absolute * 10.0) / 10.0
    save_overview(
        results,
        mask,
        output_dir / "overview_34_paper_scale.png",
        config.paper_vlim,
        "34 TacTip contacts — paper scale [-1, 1]",
    )
    save_overview(
        results,
        mask,
        output_dir / "overview_34_full_scale.png",
        full_vlim,
        f"34 TacTip contacts — unclipped shared scale [-{full_vlim:g}, {full_vlim:g}]",
    )
    if cad_xyz is not None:
        assert reprojection_error is not None and calibration_target is not None
        save_step_reference_diagnostics(
            baseline_paths[0],
            reference_points,
            reprojection_error,
            calibration_target,
            config,
            output_dir / "reference_diagnostics.png",
        )
    else:
        save_reference_diagnostics(
            baseline_paths[0],
            reference_points,
            reference_support,
            config,
            output_dir / "reference_diagnostics.png",
        )

    metric_rows = [
        {key: value for key, value in result.items() if key != "delta"}
        for result in results
    ]
    correlation = save_depth_response(metric_rows, output_dir / "depth_response.png")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    (output_dir / "reference.json").write_text(
        json.dumps(reference_record, indent=2), encoding="utf-8"
    )
    excluded_reference_details = {
        "points_xy",
        "points_cad_xyz_mm",
        "cad_to_calibration_point_index",
        "calibration_support_per_marker",
        "calibration_reprojection_error_px",
        "baseline_calibration_matching",
        "support",
        "baseline_matching",
    }
    summary = {
        "schema": "tactip_marker_density_kde.v2",
        "frames": len(results),
        "detections_json": str(detections_path),
        "detector_summary": payload.get("summary", {}),
        "kde": asdict(config),
        "reference": {
            key: value
            for key, value in reference_record.items()
            if key not in excluded_reference_details
        },
        "pooled_masked_delta": {
            "minimum": float(np.min(pooled)),
            "maximum": float(np.max(pooled)),
            "absolute_p99": float(np.quantile(np.abs(pooled), 0.99)),
            "paper_scale_saturation_fraction": float(
                np.mean(np.abs(pooled) > config.paper_vlim)
            ),
            "unclipped_overview_vlim": full_vlim,
        },
        "depth_vs_rms_pearson_r": correlation,
        "formula_note": (
            "Uses the source implementation's 1/(2*pi*h) prefactor, not the "
            "conventional normalized 2-D Gaussian 1/(2*pi*h^2)."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(output_dir, results, summary)
    print(json.dumps(summary, indent=2))
    print(output_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
