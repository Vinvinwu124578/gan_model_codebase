#!/usr/bin/env python3
"""Generate a four-tile, 400 mm TacTip GAN geometry-coverage board.

This is the large-area companion to ``design_tactile_gan_coverage_board.py``.
It preserves all sixteen controlled geometry classes, but expands every
sampling tile from 50 x 50 mm to 70 x 70 mm.  The full 400 x 400 mm board is
split into four 200 x 200 mm STL tiles so that each part fits a Bambu Lab A1.
Every M6 mounting-hole coordinate lies on one common 50 mm grid after a
single 25 mm assembly offset, so the four printed parts can be bolted to a
50 mm-pitch base plate without an adapter pattern.

The exported CSV uses one assembly-centred board coordinate frame.  Real and
simulation collection should always use that frame, even though the physical
print consists of four separately mounted parts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from design_tactile_gan_coverage_board import (
    M6_CLEARANCE_DIAMETER_MM,
    REGIONS,
    cell_support,
    feature_height,
    height_field_mesh,
    local_seed_offsets,
    vertical_cylinder,
)


DEFAULT_OUTPUT_DIR = Path("outputs/tactile_gan_coverage_board_v5_modular_70mm_mountgrid50")
HIGH_CONTRAST_OUTPUT_DIR = Path("outputs/tactile_gan_coverage_board_v5_highcontrast_70mm_mountgrid50")
DEEP_CONTACT_OUTPUT_DIR = Path("outputs/tactile_gan_coverage_board_v5_highprotrusion_deepcontact_70mm_mountgrid50")
BOARD_SIZE_MM = 400.0
TILE_SIZE_MM = 200.0
CELL_SIZE_MM = 70.0
CELL_GAP_MM = 4.0
BASE_THICKNESS_MM = 6.0
GRID_RESOLUTION_MM = 0.5
MOUNT_GRID_PITCH_MM = 50.0
MOUNT_GRID_PHASE_MM = 25.0
TILE_CENTRES_MM = (-100.0, 100.0)
LOCAL_CELL_CENTRES_MM = (-37.0, 37.0)
# The 150 mm tile-local pitch leaves each hole just outside the 70 mm tactile
# cells.  With 200 mm tiles it also leaves a 50 mm gap across each tile seam.
LOCAL_TILE_HOLE_CENTRES_MM = ((-75.0, -75.0), (-75.0, 75.0), (75.0, -75.0), (75.0, 75.0))
BOARD_FILE_STEM = "tactile_gan_coverage_board_{}mm".format(int(BOARD_SIZE_MM))
HEIGHT_PROFILES = ("standard", "high-contrast", "deep-contact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--height-profile",
        choices=HEIGHT_PROFILES,
        default="standard",
        help="standard reproduces v2; high-contrast and deep-contact increase relief without changing XY geometry.",
    )
    parser.add_argument(
        "--grid-resolution-mm",
        type=float,
        default=GRID_RESOLUTION_MM,
        help="Surface mesh pitch. Keep 0.5 mm to retain the small-feature regions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        default_dirs = {
            "standard": DEFAULT_OUTPUT_DIR,
            "high-contrast": HIGH_CONTRAST_OUTPUT_DIR,
            "deep-contact": DEEP_CONTACT_OUTPUT_DIR,
        }
        args.output_dir = default_dirs[args.height_profile]
    args.output_dir = args.output_dir.expanduser().resolve()
    if not math.isfinite(float(args.grid_resolution_mm)) or float(args.grid_resolution_mm) <= 0.0:
        parser.error("--grid-resolution-mm must be positive and finite")
    if float(args.grid_resolution_mm) > 1.0:
        parser.error("--grid-resolution-mm above 1 mm is too coarse for R14-R16")
    divisions = TILE_SIZE_MM / float(args.grid_resolution_mm)
    if not math.isclose(divisions, round(divisions), abs_tol=1e-6):
        parser.error("--grid-resolution-mm must divide {} mm exactly".format(int(TILE_SIZE_MM)))
    return args


def high_contrast_feature_height(stimulus: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return a high-contrast counterpart of the v2 tactile stimuli.

    Broad geometry is increased to roughly 6 mm relief, while the small-scale
    regions remain intentionally lower so their frequency, not just amplitude,
    remains the source of tactile variation.
    """
    if stimulus == "flat_reference":
        return np.full_like(u, 10.0, dtype=float)
    if stimulus == "flat_step_2mm":
        return np.where(u < 0.0, 8.0, 12.0)
    if stimulus == "sharp_edge_x":
        return 10.0 + np.maximum(0.0, 6.0 - np.abs(v))
    if stimulus == "sharp_edge_y":
        return 10.0 + np.maximum(0.0, 6.0 - np.abs(u))
    if stimulus == "sharp_edge_diagonal":
        normal = (u - v) / math.sqrt(2.0)
        return 10.0 + np.maximum(0.0, 6.0 - np.abs(normal))
    if stimulus == "rounded_ridge":
        return 10.0 + 5.5 * np.exp(-0.5 * (u / 4.5) ** 2)
    if stimulus == "convex_cylinder":
        cap = np.sqrt(np.clip(1.0 - (u / 26.0) ** 2, 0.0, 1.0))
        return 7.0 + 6.0 * cap
    if stimulus == "concave_cylinder":
        cup = np.sqrt(np.clip(1.0 - (u / 26.0) ** 2, 0.0, 1.0))
        return 13.0 - 6.0 * cup
    if stimulus == "convex_sphere":
        cap = np.sqrt(np.clip(1.0 - (u * u + v * v) / 26.0**2, 0.0, 1.0))
        return 7.0 + 6.0 * cap
    if stimulus == "concave_sphere":
        cup = np.sqrt(np.clip(1.0 - (u * u + v * v) / 26.0**2, 0.0, 1.0))
        return 13.0 - 6.0 * cup
    if stimulus == "saddle":
        return np.clip(10.2 + 2.4 * ((u / 20.0) ** 2 - (v / 20.0) ** 2), 6.2, 14.2)
    if stimulus == "double_bump_12mm":
        return np.minimum(
            14.8,
            8.6
            + 6.0 * np.exp(-((u + 6.0) ** 2 + v * v) / (2.0 * 3.8**2))
            + 6.0 * np.exp(-((u - 6.0) ** 2 + v * v) / (2.0 * 3.8**2)),
        )
    if stimulus == "triple_bump_9mm":
        return np.minimum(
            14.8,
            8.4
            + 5.5 * np.exp(-((u + 4.5) ** 2 + (v + 3.0) ** 2) / (2.0 * 3.2**2))
            + 5.5 * np.exp(-((u - 4.5) ** 2 + (v + 3.0) ** 2) / (2.0 * 3.2**2))
            + 5.5 * np.exp(-(u * u + (v - 4.8) ** 2) / (2.0 * 3.2**2)),
        )
    if stimulus == "fine_ridges_3mm":
        return 9.0 + 1.8 * (0.5 + 0.5 * np.cos(2.0 * math.pi * u / 3.0))
    if stimulus == "micro_dot_lattice_5mm":
        result = np.full_like(u, 8.5, dtype=float)
        for x in np.arange(-10.0, 10.1, 5.0):
            for y in np.arange(-10.0, 10.1, 5.0):
                result += 3.0 * np.exp(-((u - x) ** 2 + (v - y) ** 2) / (2.0 * 1.2**2))
        return np.minimum(result, 11.7)
    if stimulus == "micro_checker_4mm":
        row = np.floor((u + 18.0) / 4.0).astype(int)
        column = np.floor((v + 18.0) / 4.0).astype(int)
        return 8.8 + 1.2 * ((row + column) % 2)
    raise ValueError("Unsupported stimulus {!r}".format(stimulus))


def deep_contact_feature_height(stimulus: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the high-protrusion profile for approximately 2-8 mm indentation work.

    Eight millimetres of broad relief is intentionally reserved for large
    structures.  Fine features stay substantially lower so they remain
    printable and continue to represent small-scale contact patterns.
    """
    if stimulus == "flat_reference":
        return np.full_like(u, 10.0, dtype=float)
    if stimulus == "flat_step_2mm":
        return np.where(u < 0.0, 7.0, 13.0)
    if stimulus == "sharp_edge_x":
        return 10.0 + np.maximum(0.0, 8.0 - np.abs(v))
    if stimulus == "sharp_edge_y":
        return 10.0 + np.maximum(0.0, 8.0 - np.abs(u))
    if stimulus == "sharp_edge_diagonal":
        normal = (u - v) / math.sqrt(2.0)
        return 10.0 + np.maximum(0.0, 8.0 - np.abs(normal))
    if stimulus == "rounded_ridge":
        return 10.0 + 8.0 * np.exp(-0.5 * (u / 6.0) ** 2)
    if stimulus == "convex_cylinder":
        cap = np.sqrt(np.clip(1.0 - (u / 30.0) ** 2, 0.0, 1.0))
        return 6.8 + 8.0 * cap
    if stimulus == "concave_cylinder":
        cup = np.sqrt(np.clip(1.0 - (u / 30.0) ** 2, 0.0, 1.0))
        return 14.8 - 8.0 * cup
    if stimulus == "convex_sphere":
        cap = np.sqrt(np.clip(1.0 - (u * u + v * v) / 30.0**2, 0.0, 1.0))
        return 6.8 + 8.0 * cap
    if stimulus == "concave_sphere":
        cup = np.sqrt(np.clip(1.0 - (u * u + v * v) / 30.0**2, 0.0, 1.0))
        return 14.8 - 8.0 * cup
    if stimulus == "saddle":
        return np.clip(11.0 + 3.4 * ((u / 22.0) ** 2 - (v / 22.0) ** 2), 6.0, 18.0)
    if stimulus == "double_bump_12mm":
        return np.minimum(
            17.5,
            8.3
            + 8.0 * np.exp(-((u + 6.0) ** 2 + v * v) / (2.0 * 3.8**2))
            + 8.0 * np.exp(-((u - 6.0) ** 2 + v * v) / (2.0 * 3.8**2)),
        )
    if stimulus == "triple_bump_9mm":
        return np.minimum(
            17.0,
            8.0
            + 7.5 * np.exp(-((u + 4.5) ** 2 + (v + 3.0) ** 2) / (2.0 * 3.8**2))
            + 7.5 * np.exp(-((u - 4.5) ** 2 + (v + 3.0) ** 2) / (2.0 * 3.8**2))
            + 7.5 * np.exp(-(u * u + (v - 4.8) ** 2) / (2.0 * 3.8**2)),
        )
    if stimulus == "fine_ridges_3mm":
        return 8.8 + 2.5 * (0.5 + 0.5 * np.cos(2.0 * math.pi * u / 3.0))
    if stimulus == "micro_dot_lattice_5mm":
        result = np.full_like(u, 8.2, dtype=float)
        for x in np.arange(-10.0, 10.1, 5.0):
            for y in np.arange(-10.0, 10.1, 5.0):
                result += 4.0 * np.exp(-((u - x) ** 2 + (v - y) ** 2) / (2.0 * 1.3**2))
        return np.minimum(result, 12.4)
    if stimulus == "micro_checker_4mm":
        row = np.floor((u + 18.0) / 4.0).astype(int)
        column = np.floor((v + 18.0) / 4.0).astype(int)
        return 8.5 + 1.6 * ((row + column) % 2)
    raise ValueError("Unsupported stimulus {!r}".format(stimulus))


def feature_height_for_profile(height_profile: str, stimulus: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if height_profile == "standard":
        return feature_height(stimulus, u, v)
    if height_profile == "high-contrast":
        return high_contrast_feature_height(stimulus, u, v)
    if height_profile == "deep-contact":
        return deep_contact_feature_height(stimulus, u, v)
    raise ValueError("Unsupported height profile {!r}".format(height_profile))


def surface_height_for_profile(
    height_profile: str,
    stimulus: str,
    u: float,
    v: float,
) -> float:
    local_u = np.asarray([u], dtype=float)
    local_v = np.asarray([v], dtype=float)
    support = float(cell_support(local_u, local_v, CELL_SIZE_MM)[0])
    feature = float(feature_height_for_profile(height_profile, stimulus, local_u, local_v)[0])
    return float(BASE_THICKNESS_MM + support * (feature - BASE_THICKNESS_MM))


def height_profile_spec(height_profile: str) -> dict[str, object]:
    if height_profile == "standard":
        return {
            "name": "standard",
            "description": "Original v2 conservative relief profile.",
        }
    if height_profile == "high-contrast":
        return {
            "name": "high-contrast",
            "description": "High-relief v3 profile for stronger TacTip marker displacement.",
            "flat_step_mm": 4.0,
            "sharp_edge_peak_above_flat_mm": 6.0,
            "rounded_ridge_peak_above_flat_mm": 5.5,
            "broad_curvature_peak_to_rim_mm": 6.0,
            "saddle_total_range_mm": 8.0,
            "fine_ridge_peak_to_trough_mm": 1.8,
            "micro_dot_peak_above_floor_mm": 3.0,
            "micro_checker_step_mm": 1.2,
        }
    if height_profile == "deep-contact":
        return {
            "name": "deep-contact",
            "description": "High-protrusion profile for controlled 2-8 mm TacTip indentation experiments.",
            "flat_step_mm": 6.0,
            "sharp_edge_peak_above_flat_mm": 8.0,
            "rounded_ridge_peak_above_flat_mm": 8.0,
            "broad_curvature_peak_to_rim_mm": 8.0,
            "saddle_total_range_mm": 12.0,
            "fine_ridge_peak_to_trough_mm": 2.5,
            "micro_dot_peak_above_floor_mm": 4.0,
            "micro_checker_step_mm": 1.6,
            "recommended_large_indentation_band_mm": [2.0, 8.0],
        }
    raise ValueError("Unsupported height profile {!r}".format(height_profile))


def display_stimulus(height_profile: str, stimulus: str) -> str:
    if height_profile == "high-contrast" and stimulus == "flat_step_2mm":
        return "flat_step_4mm"
    if height_profile == "deep-contact" and stimulus == "flat_step_2mm":
        return "flat_step_6mm"
    return stimulus


def region_description_for_profile(height_profile: str, region: object) -> str:
    if height_profile == "standard":
        return str(region.description)
    high_contrast_descriptions = {
        "flat_reference": "Broad planar reference patch for pressure, lighting, and pose baselines.",
        "flat_step_2mm": "Two planar levels separated by a 4 mm step; samples include pre-, on-, and post-step contacts.",
        "sharp_edge_x": "A 6 mm high sharp roof edge running along local X.",
        "sharp_edge_y": "A 6 mm high sharp roof edge running along local Y.",
        "sharp_edge_diagonal": "A 6 mm high sharp roof edge at 45 degrees to the board axes.",
        "rounded_ridge": "A smooth 5.5 mm high rounded ridge with a 4.5 mm lateral scale.",
        "convex_cylinder": "A broad convex cylindrical cap with 6 mm peak-to-rim relief.",
        "concave_cylinder": "A broad concave cylindrical cup with 6 mm rim-to-center relief.",
        "convex_sphere": "A convex spherical cap with 6 mm peak-to-rim relief.",
        "concave_sphere": "A concave spherical cup with 6 mm rim-to-center relief.",
        "saddle": "A saddle with opposite signed principal curvatures and an 8 mm total range.",
        "double_bump_12mm": "Two rounded protrusions with 12 mm centre spacing and approximately 6 mm peak relief.",
        "triple_bump_9mm": "Three rounded protrusions at 9 mm pitch and approximately 5.5 mm peak relief.",
        "fine_ridges_3mm": "Parallel 3 mm-pitch ridges with 1.8 mm peak-to-trough amplitude.",
        "micro_dot_lattice_5mm": "A 5 mm-pitch lattice of approximately 2.4 mm diameter rounded dots with 3 mm peak relief.",
        "micro_checker_4mm": "A 4 mm checker pattern with 1.2 mm steps for small-scale contrast.",
    }
    deep_contact_descriptions = {
        "flat_reference": "Broad planar reference patch for controlled 2-8 mm indentation, lighting, and pose baselines.",
        "flat_step_2mm": "Two planar levels separated by a 6 mm step for deep-contact crossing samples.",
        "sharp_edge_x": "An 8 mm high sharp roof edge running along local X for large-indentation edge contact.",
        "sharp_edge_y": "An 8 mm high sharp roof edge running along local Y for large-indentation edge contact.",
        "sharp_edge_diagonal": "An 8 mm high sharp roof edge at 45 degrees to the board axes.",
        "rounded_ridge": "A smooth 8 mm high rounded ridge with a 6 mm lateral scale.",
        "convex_cylinder": "A broad convex cylindrical cap with 8 mm peak-to-rim relief.",
        "concave_cylinder": "A broad concave cylindrical cup with 8 mm rim-to-center relief.",
        "convex_sphere": "A convex spherical cap with 8 mm peak-to-rim relief.",
        "concave_sphere": "A concave spherical cup with 8 mm rim-to-center relief.",
        "saddle": "A saddle with opposite signed principal curvatures and a 12 mm total range.",
        "double_bump_12mm": "Two rounded protrusions with 12 mm centre spacing and approximately 8 mm peak relief.",
        "triple_bump_9mm": "Three rounded protrusions at 9 mm pitch and approximately 7.5 mm peak relief.",
        "fine_ridges_3mm": "Parallel 3 mm-pitch ridges with 2.5 mm peak-to-trough amplitude.",
        "micro_dot_lattice_5mm": "A 5 mm-pitch lattice of approximately 2.6 mm diameter rounded dots with 4 mm peak relief.",
        "micro_checker_4mm": "A 4 mm checker pattern with 1.6 mm steps for high-depth small-scale contrast.",
    }
    if height_profile == "high-contrast":
        return high_contrast_descriptions[str(region.stimulus)]
    if height_profile == "deep-contact":
        return deep_contact_descriptions[str(region.stimulus)]
    raise ValueError("Unsupported height profile {!r}".format(height_profile))


def recommended_depth_range_for_profile(height_profile: str, region: object) -> tuple[float, float]:
    if height_profile != "deep-contact":
        return tuple(float(value) for value in region.depth_range_mm)
    large_indentation_ranges = {
        "flat": (1.0, 8.0),
        "edge": (0.75, 6.0),
        "curvature": (1.0, 8.0),
        "multi_touch": (0.75, 6.5),
        "small_feature": (0.5, 4.0),
    }
    return large_indentation_ranges[str(region.category)]


def global_region_centres() -> dict[str, tuple[float, float]]:
    """Return all 16 centres in the assembled board frame.

    Each print tile contains two 70 mm cells with a 4 mm internal moat.  The
    200 mm tile envelope leaves a 56 mm cross-shaped central low area, which
    separates tiles and makes their registration/fastening accessible.
    """
    west, east = TILE_CENTRES_MM
    inner_west, inner_east = LOCAL_CELL_CENTRES_MM
    values = (west + inner_west, west + inner_east, east + inner_west, east + inner_east)
    row_values = tuple(reversed(values))
    return {
        region.region_id: (float(values[region.column]), float(row_values[region.row])) for region in REGIONS
    }


def tile_definitions() -> tuple[dict[str, object], ...]:
    west, east = TILE_CENTRES_MM
    return (
        {"tile_id": "tile_nw", "center_xy_mm": (west, east), "rows": (0, 1), "columns": (0, 1)},
        {"tile_id": "tile_ne", "center_xy_mm": (east, east), "rows": (0, 1), "columns": (2, 3)},
        {"tile_id": "tile_sw", "center_xy_mm": (west, west), "rows": (2, 3), "columns": (0, 1)},
        {"tile_id": "tile_se", "center_xy_mm": (east, west), "rows": (2, 3), "columns": (2, 3)},
    )


def axes_for_tile(center_xy_mm: tuple[float, float], resolution_mm: float) -> tuple[np.ndarray, np.ndarray]:
    divisions = int(round(TILE_SIZE_MM / resolution_mm))
    half = TILE_SIZE_MM / 2.0
    return (
        np.linspace(center_xy_mm[0] - half, center_xy_mm[0] + half, divisions + 1, dtype=float),
        np.linspace(center_xy_mm[1] - half, center_xy_mm[1] + half, divisions + 1, dtype=float),
    )


def build_tile_heights(
    xs: np.ndarray,
    ys: np.ndarray,
    regions: tuple[object, ...],
    centres: dict[str, tuple[float, float]],
    height_profile: str,
) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    heights = np.full_like(x_grid, BASE_THICKNESS_MM, dtype=float)
    for region in regions:
        center_x, center_y = centres[region.region_id]
        u = x_grid - center_x
        v = y_grid - center_y
        in_cell = (np.abs(u) <= CELL_SIZE_MM / 2.0) & (np.abs(v) <= CELL_SIZE_MM / 2.0)
        if not bool(np.any(in_cell)):
            continue
        support = cell_support(u[in_cell], v[in_cell], CELL_SIZE_MM)
        top = feature_height_for_profile(height_profile, region.stimulus, u[in_cell], v[in_cell])
        heights[in_cell] = BASE_THICKNESS_MM + support * (top - BASE_THICKNESS_MM)
    return heights


def drill_tile_holes(
    mesh: trimesh.Trimesh,
    tile_center_xy_mm: tuple[float, float],
    top_height_mm: float,
) -> tuple[trimesh.Trimesh, list[list[float]]]:
    global_centres = [
        [tile_center_xy_mm[0] + x, tile_center_xy_mm[1] + y] for x, y in LOCAL_TILE_HOLE_CENTRES_MM
    ]
    for centre in global_centres:
        for coordinate in centre:
            grid_units = (coordinate - MOUNT_GRID_PHASE_MM) / MOUNT_GRID_PITCH_MM
            if not math.isclose(grid_units, round(grid_units), abs_tol=1e-9):
                raise RuntimeError("M6 mounting-hole coordinate is not on the 50 mm assembly grid: {}".format(centre))
    cutters = [
        vertical_cylinder(M6_CLEARANCE_DIAMETER_MM / 2.0, top_height_mm + 4.0, (x, y, top_height_mm / 2.0))
        for x, y in global_centres
    ]
    result = trimesh.boolean.difference([mesh, *cutters], engine="manifold")
    if result is None:
        raise RuntimeError("Manifold could not create M6 holes for tile at {}".format(tile_center_xy_mm))
    if isinstance(result, list):
        result = trimesh.util.concatenate(result)
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("Unexpected boolean result while drilling M6 holes")
    result.remove_unreferenced_vertices()
    result.fix_normals()
    if not result.is_watertight:
        raise RuntimeError("Drilled tile at {} is not watertight".format(tile_center_xy_mm))
    return result, global_centres


def write_sampling_sites(
    path: Path,
    centres: dict[str, tuple[float, float]],
    height_profile: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        center_x, center_y = centres[region.region_id]
        depth_min, depth_max = recommended_depth_range_for_profile(height_profile, region)
        for index, (u, v) in enumerate(local_seed_offsets(region.stimulus), start=1):
            rows.append(
                {
                    "site_id": "{}_S{:02d}".format(region.region_id, index),
                    "region_id": region.region_id,
                    "category": region.category,
                    "stimulus": region.stimulus,
                    "stimulus_display": display_stimulus(height_profile, region.stimulus),
                    "height_profile": height_profile,
                    "board_x_mm": round(center_x + u, 4),
                    "board_y_mm": round(center_y + v, 4),
                    "expected_surface_z_mm": round(
                        surface_height_for_profile(height_profile, region.stimulus, u, v), 4
                    ),
                    "recommended_depth_min_mm": depth_min,
                    "recommended_depth_max_mm": depth_max,
                    "sampling_jitter_radius_mm": 1.5 if region.category in ("edge", "small_feature") else 4.0,
                    "notes": region_description_for_profile(height_profile, region),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_assembly_urdf(path: Path, tile_records: list[dict[str, object]]) -> None:
    links: list[str] = [
        "  <link name=\"board_frame\"><inertial><mass value=\"0.001\"/><inertia ixx=\"1e-6\" ixy=\"0\" ixz=\"0\" iyy=\"1e-6\" iyz=\"0\" izz=\"1e-6\"/></inertial></link>"
    ]
    for record in tile_records:
        tile_id = str(record["tile_id"])
        filename = str(record["stl"])
        x, y = (float(value) / 1000.0 for value in record["center_board_xy_mm"])
        links.extend(
            (
                "  <link name=\"{}\">".format(tile_id),
                "    <inertial><origin xyz=\"0 0 0.006\"/><mass value=\"0.75\"/><inertia ixx=\"0.004\" ixy=\"0\" ixz=\"0\" iyy=\"0.004\" iyz=\"0\" izz=\"0.008\"/></inertial>",
                "    <visual><geometry><mesh filename=\"{}\" scale=\"0.001 0.001 0.001\"/></geometry><material name=\"stimulus_blue\"><color rgba=\"0.16 0.42 0.72 1\"/></material></visual>".format(filename),
                "    <collision><geometry><mesh filename=\"{}\" scale=\"0.001 0.001 0.001\"/></geometry></collision>".format(filename),
                "  </link>",
                "  <joint name=\"board_frame_to_{}\" type=\"fixed\">".format(tile_id),
                "    <parent link=\"board_frame\"/><child link=\"{}\"/><origin xyz=\"{:.6f} {:.6f} 0\" rpy=\"0 0 0\"/>".format(tile_id, x, y),
                "  </joint>",
            )
        )
    path.write_text("<?xml version=\"1.0\"?>\n<robot name=\"{}\">\n{}\n</robot>\n".format(BOARD_FILE_STEM, "\n".join(links)), encoding="utf-8")


def render_preview(
    path: Path,
    centres: dict[str, tuple[float, float]],
    resolution_mm: float,
    all_tile_records: list[dict[str, object]],
    height_profile: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    divisions = int(round(BOARD_SIZE_MM / resolution_mm))
    xs = np.linspace(-BOARD_SIZE_MM / 2.0, BOARD_SIZE_MM / 2.0, divisions + 1, dtype=float)
    ys = np.linspace(-BOARD_SIZE_MM / 2.0, BOARD_SIZE_MM / 2.0, divisions + 1, dtype=float)
    heights = build_tile_heights(xs, ys, tuple(REGIONS), centres, height_profile)
    figure = plt.figure(figsize=(16, 7.5), dpi=200)
    top_axis = figure.add_subplot(1, 2, 1)
    image = top_axis.imshow(
        heights,
        origin="lower",
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        cmap="viridis",
        vmin=float(np.min(heights)),
        vmax=float(np.max(heights)),
        interpolation="nearest",
    )
    for record in all_tile_records:
        x, y = record["center_board_xy_mm"]
        top_axis.add_patch(
            plt.Rectangle((x - TILE_SIZE_MM / 2.0, y - TILE_SIZE_MM / 2.0), TILE_SIZE_MM, TILE_SIZE_MM, fill=False, edgecolor="#ffcf38", linewidth=1.2)
        )
        for hx, hy in record["m6_hole_centres_board_xy_mm"]:
            top_axis.add_patch(plt.Circle((hx, hy), M6_CLEARANCE_DIAMETER_MM / 2.0, fill=False, edgecolor="#ffcf38", linewidth=0.85))
    for region in REGIONS:
        x, y = centres[region.region_id]
        label = "{}\n{}".format(region.region_id, display_stimulus(height_profile, region.stimulus).replace("_", " "))
        text = top_axis.text(x, y, label, ha="center", va="center", color="white", fontsize=7.2, weight="bold")
        text.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground="#111111"), path_effects.Normal()])
        top_axis.add_patch(
            plt.Rectangle((x - CELL_SIZE_MM / 2.0, y - CELL_SIZE_MM / 2.0), CELL_SIZE_MM, CELL_SIZE_MM, fill=False, edgecolor="white", linewidth=0.6)
        )
    top_axis.set_title("Assembly-top board frame; yellow: printed tile and M6 boundaries")
    top_axis.set_xlabel("board X (mm)")
    top_axis.set_ylabel("board Y (mm)")
    figure.colorbar(image, ax=top_axis, fraction=0.046, pad=0.04, label="surface height (mm)")

    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    step = max(1, int(round(2.0 / resolution_mm)))
    x_grid, y_grid = np.meshgrid(xs[::step], ys[::step], indexing="xy")
    axis_3d.plot_surface(x_grid, y_grid, heights[::step, ::step], cmap="viridis", linewidth=0.0, antialiased=True, shade=True)
    axis_3d.set_xlim(-BOARD_SIZE_MM / 2.0, BOARD_SIZE_MM / 2.0)
    axis_3d.set_ylim(-BOARD_SIZE_MM / 2.0, BOARD_SIZE_MM / 2.0)
    axis_3d.set_zlim(0.0, float(np.max(heights)) + 1.0)
    axis_3d.set_box_aspect((1.0, 1.0, 0.11))
    axis_3d.view_init(elev=38, azim=-52)
    axis_3d.set_title("Assembled geometry; print all four tiles with features upward")
    axis_3d.set_xlabel("X (mm)")
    axis_3d.set_ylabel("Y (mm)")
    axis_3d.set_zlabel("Z (mm)")
    variants = {
        "standard": "v2 standard",
        "high-contrast": "v3 high contrast",
        "deep-contact": "v5 high protrusion",
    }
    variant = variants[height_profile]
    figure.suptitle(
        "TacTip GAN geometry-coverage board {} | 4 printable 200 mm tiles | 16 expanded 70 mm regions | M6 grid: 50 mm compatible".format(variant),
        y=0.98,
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_readme(path: Path, metadata: dict[str, object]) -> None:
    profile = str(metadata["height_profile"])
    profile_notes = {
        "standard": "This v2 set uses the original conservative relief profile.",
        "high-contrast": "This v3 set uses the high-contrast profile: broad edge and curvature relief is approximately 6 mm, while fine detail remains 1.2-3 mm.",
        "deep-contact": "This v5 high-protrusion set is designed for controlled 2-8 mm TacTip indentation: broad edge and curvature relief is approximately 8 mm, while fine detail remains 1.6-4 mm.",
    }
    profile_note = profile_notes[profile]
    path.write_text(
        """# Modular 400 mm TacTip GAN geometry-coverage board

This set enlarges each controlled tactile region to 70 x 70 mm while retaining
all sixteen geometry classes.  It consists of four 200 x 200 mm print tiles.
Do not rotate any tile while assembling it: the shared coordinate system in
`sampling_sites.csv` assumes the tile locations shown in `preview.png`.

{}

## Assembly

1. Print all four STL files flat, with the tactile geometry facing upward.
2. Lay them out as `tile_nw tile_ne` above `tile_sw tile_se`.
3. Use the four 6.6 mm M6 holes in each tile to fix all tiles to one flat
   breadboard/base plate.  The hole centres form one 50 mm-compatible grid:
   every X/Y centre-to-centre distance is an integer multiple of 50 mm.  The
   board-frame grid is phase-shifted by 25 mm, so align any one hole to the
   base plate and every other hole will align automatically.
4. The assembled outside size is 400 x 400 mm.  Its origin is at the centre
   of the four tiles, not at any individual tile origin.

## Collection

The 144 CSV seed contacts are in the assembly-centred board frame.  Detect
contact locally from tactile-image change at every seed; nominal Z is only a
simulation/reference value.  Each region has 512 balanced pair slots, for
8192 total pairs.  Keep the exact generated pose list for the corresponding
simulation render.  Edge and small-feature regions intentionally use lower
depth ranges than broad planar/curved regions.

## Files

- Four STL tiles are printable on a Bambu A1.
- `{}.urdf` loads the four tiles together in
  PyBullet, using metres in the URDF and millimetres in the STL/CSV.
- `manifest.json` is the authoritative tile placement and region mapping.
""".format(profile_note, BOARD_FILE_STEM),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [
        output_dir / "{}.urdf".format(BOARD_FILE_STEM),
        output_dir / "{}_sampling_sites.csv".format(BOARD_FILE_STEM),
        output_dir / "{}_manifest.json".format(BOARD_FILE_STEM),
        output_dir / "{}_preview.png".format(BOARD_FILE_STEM),
        output_dir / "README.md",
    ]
    expected_paths.extend(output_dir / "{}_{}.stl".format(BOARD_FILE_STEM, item["tile_id"]) for item in tile_definitions())
    if not args.overwrite:
        existing = next((path for path in expected_paths if path.exists()), None)
        if existing is not None:
            raise FileExistsError("{} already exists. Use --overwrite or another --output-dir.".format(existing))

    centres = global_region_centres()
    tile_records: list[dict[str, object]] = []
    maximum_height = BASE_THICKNESS_MM
    for definition in tile_definitions():
        tile_id = str(definition["tile_id"])
        center = tuple(float(value) for value in definition["center_xy_mm"])
        rows = tuple(int(value) for value in definition["rows"])
        columns = tuple(int(value) for value in definition["columns"])
        tile_regions = tuple(region for region in REGIONS if region.row in rows and region.column in columns)
        xs, ys = axes_for_tile(center, float(args.grid_resolution_mm))
        heights = build_tile_heights(xs, ys, tile_regions, centres, args.height_profile)
        maximum_height = max(maximum_height, float(np.max(heights)))
        mesh = height_field_mesh(xs, ys, heights)
        mesh, hole_centres = drill_tile_holes(mesh, center, float(np.max(heights)))
        mesh.apply_translation((-center[0], -center[1], 0.0))
        stl_name = "{}_{}.stl".format(BOARD_FILE_STEM, tile_id)
        mesh.export(output_dir / stl_name)
        tile_records.append(
            {
                "tile_id": tile_id,
                "stl": stl_name,
                "center_board_xy_mm": list(center),
                "m6_hole_centres_board_xy_mm": hole_centres,
                "region_ids": [region.region_id for region in tile_regions],
                "local_stl_bounds_mm": [[float(value) for value in row] for row in mesh.bounds],
                "face_count": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
            }
        )

    sites_path = output_dir / "{}_sampling_sites.csv".format(BOARD_FILE_STEM)
    sample_sites = write_sampling_sites(sites_path, centres, args.height_profile)
    urdf_path = output_dir / "{}.urdf".format(BOARD_FILE_STEM)
    write_assembly_urdf(urdf_path, tile_records)
    preview_path = output_dir / "{}_preview.png".format(BOARD_FILE_STEM)
    render_preview(preview_path, centres, float(args.grid_resolution_mm), tile_records, args.height_profile)

    category_counts: dict[str, int] = {}
    for region in REGIONS:
        category_counts[region.category] = category_counts.get(region.category, 0) + 1
    metadata: dict[str, object] = {
        "schema": {
            "standard": "tactile_gan_coverage_board.modular.v5.mountgrid50",
            "high-contrast": "tactile_gan_coverage_board.modular.v5.highcontrast.mountgrid50",
            "deep-contact": "tactile_gan_coverage_board.modular.v5.deepcontact.mountgrid50",
        }[args.height_profile],
        "units": "mm",
        "height_profile": args.height_profile,
        "height_profile_spec": height_profile_spec(args.height_profile),
        "assembled_board_size_mm": [BOARD_SIZE_MM, BOARD_SIZE_MM],
        "tile_size_mm": [TILE_SIZE_MM, TILE_SIZE_MM],
        "tile_count": len(tile_records),
        "cell_size_mm": CELL_SIZE_MM,
        "cell_gap_mm": CELL_GAP_MM,
        "central_tile_seam_width_mm": 26.0,
        "base_thickness_mm": BASE_THICKNESS_MM,
        "surface_height_range_mm": [BASE_THICKNESS_MM, maximum_height],
        "grid_resolution_mm": float(args.grid_resolution_mm),
        "m6_hole_diameter_mm": M6_CLEARANCE_DIAMETER_MM,
        "m6_mount_grid_pitch_mm": MOUNT_GRID_PITCH_MM,
        "m6_mount_grid_phase_mm": MOUNT_GRID_PHASE_MM,
        "nominal_tactip_compliant_diameter_mm": 40.0,
        "safe_central_feature_zone_mm": 60.0,
        "region_category_counts": category_counts,
        "regions": [
            asdict(region)
            | {
                "stimulus_display": display_stimulus(args.height_profile, region.stimulus),
                "description": region_description_for_profile(args.height_profile, region),
                "center_board_xy_mm": list(centres[region.region_id]),
            }
            for region in REGIONS
        ],
        "tiles": tile_records,
        "sampling_site_count": len(sample_sites),
        "samples_per_region": 512,
        "total_target_pairs": 512 * len(REGIONS),
        "urdf": urdf_path.name,
        "sampling_sites": sites_path.name,
        "preview": preview_path.name,
    }
    manifest_path = output_dir / "{}_manifest.json".format(BOARD_FILE_STEM)
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", metadata)

    print("Output directory: {}".format(output_dir))
    for record in tile_records:
        print("{}: {} faces, watertight={}".format(record["tile_id"], record["face_count"], record["watertight"]))
    print("Preview: {}".format(preview_path))
    print("Sampling sites: {} ({})".format(sites_path, len(sample_sites)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
