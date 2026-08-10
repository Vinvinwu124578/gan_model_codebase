#!/usr/bin/env python3
"""Generate a printable geometry-coverage board for real-to-sim tactile GAN data.

The board is deliberately a *training stimulus*, rather than a realistic object:
it exposes the TacTip to a balanced mixture of broad planar contact, sharp and
rounded edges, convex/concave/saddle curvature, multi-feature contacts, and
small printable detail.  It also exports the exact STL, URDF, region manifest,
and seed contacts needed to reproduce the same geometry in simulation.

The default 240 x 240 mm plate fits a Bambu Lab A1 build plate.  It contains
sixteen 50 x 50 mm tiles separated by low isolation moats.  The TacTip has an
approximately 40 mm compliant diameter, so each feature is kept inside a
36 mm central zone and the accompanying CSV restricts sampling to a safe
central window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh


DEFAULT_OUTPUT_DIR = Path("outputs/tactile_gan_coverage_board_v1")
DEFAULT_BOARD_SIZE_MM = 240.0
DEFAULT_BASE_THICKNESS_MM = 6.0
DEFAULT_CELL_SIZE_MM = 50.0
DEFAULT_CELL_GAP_MM = 4.0
DEFAULT_GRID_RESOLUTION_MM = 0.5
M6_CLEARANCE_DIAMETER_MM = 6.6
M6_CORNER_CENTRES_MM = ((-112.0, -112.0), (-112.0, 112.0), (112.0, -112.0), (112.0, 112.0))


@dataclass(frozen=True)
class Region:
    region_id: str
    category: str
    stimulus: str
    row: int
    column: int
    description: str
    depth_range_mm: tuple[float, float]
    samples_per_region: int


REGIONS = (
    Region(
        "R01",
        "flat",
        "flat_reference",
        0,
        0,
        "Broad planar reference patch for pressure, lighting, and pose baselines.",
        (0.75, 5.0),
        512,
    ),
    Region(
        "R02",
        "flat",
        "flat_step_2mm",
        0,
        1,
        "Two planar levels separated by a 2 mm step; samples include pre-, on-, and post-step contacts.",
        (0.75, 4.0),
        512,
    ),
    Region(
        "R03",
        "edge",
        "sharp_edge_x",
        0,
        2,
        "A 90 degree roof edge running along local X.",
        (0.5, 3.5),
        512,
    ),
    Region(
        "R04",
        "edge",
        "sharp_edge_y",
        0,
        3,
        "A 90 degree roof edge running along local Y.",
        (0.5, 3.5),
        512,
    ),
    Region(
        "R05",
        "edge",
        "sharp_edge_diagonal",
        1,
        0,
        "A 90 degree roof edge at 45 degrees to the board axes.",
        (0.5, 3.5),
        512,
    ),
    Region(
        "R06",
        "edge",
        "rounded_ridge",
        1,
        1,
        "A smooth rounded ridge with a 3.5 mm lateral scale.",
        (0.5, 4.0),
        512,
    ),
    Region(
        "R07",
        "curvature",
        "convex_cylinder",
        1,
        2,
        "A broad convex cylindrical cap; curvature varies along local X only.",
        (0.75, 4.5),
        512,
    ),
    Region(
        "R08",
        "curvature",
        "concave_cylinder",
        1,
        3,
        "A broad concave cylindrical cup; curvature varies along local X only.",
        (0.75, 4.0),
        512,
    ),
    Region(
        "R09",
        "curvature",
        "convex_sphere",
        2,
        0,
        "A convex spherical cap, giving two-axis positive curvature.",
        (0.75, 4.5),
        512,
    ),
    Region(
        "R10",
        "curvature",
        "concave_sphere",
        2,
        1,
        "A concave spherical cup, giving two-axis negative curvature.",
        (0.75, 4.0),
        512,
    ),
    Region(
        "R11",
        "curvature",
        "saddle",
        2,
        2,
        "A saddle with opposite signed principal curvatures.",
        (0.75, 4.0),
        512,
    ),
    Region(
        "R12",
        "multi_touch",
        "double_bump_12mm",
        2,
        3,
        "Two rounded protrusions with 12 mm centre spacing for dual-contact imagery.",
        (0.5, 3.5),
        512,
    ),
    Region(
        "R13",
        "multi_touch",
        "triple_bump_9mm",
        3,
        0,
        "Three rounded protrusions at 9 mm pitch for multi-point contact states.",
        (0.5, 3.0),
        512,
    ),
    Region(
        "R14",
        "small_feature",
        "fine_ridges_3mm",
        3,
        1,
        "Parallel 3 mm-pitch ridges with 1 mm amplitude.",
        (0.5, 3.0),
        512,
    ),
    Region(
        "R15",
        "small_feature",
        "micro_dot_lattice_5mm",
        3,
        2,
        "A 5 mm-pitch lattice of approximately 2.4 mm diameter rounded dots.",
        (0.5, 3.0),
        512,
    ),
    Region(
        "R16",
        "small_feature",
        "micro_checker_4mm",
        3,
        3,
        "A 4 mm checker pattern with 0.8 mm steps for small-scale contrast.",
        (0.5, 3.0),
        512,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--board-size-mm", type=float, default=DEFAULT_BOARD_SIZE_MM)
    parser.add_argument("--base-thickness-mm", type=float, default=DEFAULT_BASE_THICKNESS_MM)
    parser.add_argument("--cell-size-mm", type=float, default=DEFAULT_CELL_SIZE_MM)
    parser.add_argument("--cell-gap-mm", type=float, default=DEFAULT_CELL_GAP_MM)
    parser.add_argument(
        "--grid-resolution-mm",
        type=float,
        default=DEFAULT_GRID_RESOLUTION_MM,
        help="Height-field mesh spacing; 0.5 mm preserves the small printed stimuli.",
    )
    parser.add_argument("--no-m6-mount-holes", action="store_true", help="Export a solid plate without four corner M6 holes.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing generated files in --output-dir.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    for name in ("board_size_mm", "base_thickness_mm", "cell_size_mm", "cell_gap_mm", "grid_resolution_mm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error("--{} must be positive and finite".format(name.replace("_", "-")))
    if float(args.board_size_mm) > 250.0:
        parser.error("--board-size-mm must stay at or below 250 mm for a Bambu A1")
    if float(args.grid_resolution_mm) > 1.0:
        parser.error("--grid-resolution-mm above 1 mm is too coarse for the small-feature regions")
    return args


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def cell_centres(board_size: float, cell_size: float, cell_gap: float) -> tuple[np.ndarray, np.ndarray]:
    span = 4.0 * cell_size + 3.0 * cell_gap
    if span >= board_size - 12.0:
        raise ValueError("Cells leave less than 6 mm of outer border; reduce --cell-size-mm or --cell-gap-mm")
    start = -span / 2.0 + cell_size / 2.0
    centres = start + np.arange(4, dtype=float) * (cell_size + cell_gap)
    return centres, centres[::-1]


def cell_support(u: np.ndarray, v: np.ndarray, cell_size: float) -> np.ndarray:
    """Fade each raised tile into its low moat without changing the centre geometry."""
    half = cell_size / 2.0
    core_half = half - 5.0
    q = np.maximum(np.abs(u), np.abs(v))
    return smoothstep((half - q) / (half - core_half))


def gaussian(u: np.ndarray, v: np.ndarray, x: float, y: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-((u - x) ** 2 + (v - y) ** 2) / (2.0 * sigma * sigma))


def feature_height(stimulus: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the exposed top height in mm within one tile's local coordinates."""
    flat = np.full_like(u, 10.0, dtype=float)
    if stimulus == "flat_reference":
        return flat
    if stimulus == "flat_step_2mm":
        return np.where(u < 0.0, 9.0, 11.0)
    if stimulus == "sharp_edge_x":
        return 10.0 + np.maximum(0.0, 3.8 - np.abs(v))
    if stimulus == "sharp_edge_y":
        return 10.0 + np.maximum(0.0, 3.8 - np.abs(u))
    if stimulus == "sharp_edge_diagonal":
        normal = (u - v) / math.sqrt(2.0)
        return 10.0 + np.maximum(0.0, 3.8 - np.abs(normal))
    if stimulus == "rounded_ridge":
        return 10.0 + 3.4 * np.exp(-0.5 * (u / 3.5) ** 2)
    if stimulus == "convex_cylinder":
        cap = np.sqrt(np.clip(1.0 - (u / 22.0) ** 2, 0.0, 1.0))
        return 7.4 + 3.7 * cap
    if stimulus == "concave_cylinder":
        cup = np.sqrt(np.clip(1.0 - (u / 22.0) ** 2, 0.0, 1.0))
        return 11.1 - 3.7 * cup
    if stimulus == "convex_sphere":
        cap = np.sqrt(np.clip(1.0 - (u * u + v * v) / 22.0**2, 0.0, 1.0))
        return 7.4 + 3.8 * cap
    if stimulus == "concave_sphere":
        cup = np.sqrt(np.clip(1.0 - (u * u + v * v) / 22.0**2, 0.0, 1.0))
        return 11.2 - 3.8 * cup
    if stimulus == "saddle":
        return np.clip(9.3 + 1.8 * ((u / 18.0) ** 2 - (v / 18.0) ** 2), 6.4, 12.2)
    if stimulus == "double_bump_12mm":
        return np.minimum(13.4, 8.6 + gaussian(u, v, -6.0, 0.0, 3.6, 4.0) + gaussian(u, v, 6.0, 0.0, 3.6, 4.0))
    if stimulus == "triple_bump_9mm":
        return np.minimum(
            13.4,
            8.4
            + gaussian(u, v, -4.5, -3.0, 3.0, 4.0)
            + gaussian(u, v, 4.5, -3.0, 3.0, 4.0)
            + gaussian(u, v, 0.0, 4.8, 3.0, 4.0),
        )
    if stimulus == "fine_ridges_3mm":
        return 9.2 + 1.0 * (0.5 + 0.5 * np.cos(2.0 * math.pi * u / 3.0))
    if stimulus == "micro_dot_lattice_5mm":
        result = np.full_like(u, 8.8, dtype=float)
        for x in np.arange(-10.0, 10.1, 5.0):
            for y in np.arange(-10.0, 10.1, 5.0):
                result += gaussian(u, v, float(x), float(y), 1.15, 2.3)
        return np.minimum(result, 11.4)
    if stimulus == "micro_checker_4mm":
        row = np.floor((u + 18.0) / 4.0).astype(int)
        column = np.floor((v + 18.0) / 4.0).astype(int)
        return 9.0 + 0.8 * ((row + column) % 2)
    raise ValueError("Unsupported stimulus {!r}".format(stimulus))


def build_heights(
    board_size: float,
    base_thickness: float,
    cell_size: float,
    cell_gap: float,
    grid_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    divisions = int(round(board_size / grid_resolution))
    if not math.isclose(divisions * grid_resolution, board_size, abs_tol=1e-6):
        raise ValueError("--board-size-mm must be an integer multiple of --grid-resolution-mm")
    xs = np.linspace(-board_size / 2.0, board_size / 2.0, divisions + 1, dtype=float)
    ys = np.linspace(-board_size / 2.0, board_size / 2.0, divisions + 1, dtype=float)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    heights = np.full_like(x_grid, base_thickness, dtype=float)
    x_centres, y_centres = cell_centres(board_size, cell_size, cell_gap)
    centres: dict[str, tuple[float, float]] = {}

    for region in REGIONS:
        centre = (float(x_centres[region.column]), float(y_centres[region.row]))
        centres[region.region_id] = centre
        u = x_grid - centre[0]
        v = y_grid - centre[1]
        in_tile = (np.abs(u) <= cell_size / 2.0) & (np.abs(v) <= cell_size / 2.0)
        if not bool(np.any(in_tile)):
            raise RuntimeError("Grid does not cover {}".format(region.region_id))
        top = feature_height(region.stimulus, u[in_tile], v[in_tile])
        support = cell_support(u[in_tile], v[in_tile], cell_size)
        heights[in_tile] = base_thickness + support * (top - base_thickness)
    return xs, ys, heights, centres


def height_field_mesh(xs: np.ndarray, ys: np.ndarray, heights: np.ndarray) -> trimesh.Trimesh:
    """Create a watertight solid below a sampled height field."""
    count_x = len(xs)
    count_y = len(ys)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    top = np.column_stack((x_grid.ravel(), y_grid.ravel(), heights.ravel()))
    bottom = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros_like(heights).ravel()))
    vertices = np.vstack((top, bottom))
    lower = count_x * count_y
    faces: list[tuple[int, int, int]] = []

    for row in range(count_y - 1):
        offset = row * count_x
        next_offset = (row + 1) * count_x
        for column in range(count_x - 1):
            a = offset + column
            b = a + 1
            c = next_offset + column + 1
            d = next_offset + column
            faces.extend(((a, b, c), (a, c, d), (lower + a, lower + c, lower + b), (lower + a, lower + d, lower + c)))

    # Close the outer perimeter.  The height field itself is continuous across
    # the low moats, so it needs no interior side walls.
    for column in range(count_x - 1):
        a, b = column, column + 1
        faces.extend(((lower + a, b, a), (lower + a, lower + b, b)))
        a = (count_y - 1) * count_x + column
        b = a + 1
        faces.extend(((lower + a, a, b), (lower + a, b, lower + b)))
    for row in range(count_y - 1):
        a = row * count_x
        b = (row + 1) * count_x
        faces.extend(((lower + a, a, b), (lower + a, b, lower + b)))
        a = row * count_x + count_x - 1
        b = (row + 1) * count_x + count_x - 1
        faces.extend(((lower + a, b, a), (lower + a, lower + b, b)))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise RuntimeError("Height-field board is not watertight")
    return mesh


def vertical_cylinder(radius: float, height: float, centre: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=96)
    mesh.apply_translation(centre)
    return mesh


def drill_mount_holes(mesh: trimesh.Trimesh, top_height: float) -> trimesh.Trimesh:
    cutters = [
        vertical_cylinder(M6_CLEARANCE_DIAMETER_MM / 2.0, top_height + 4.0, (x, y, top_height / 2.0))
        for x, y in M6_CORNER_CENTRES_MM
    ]
    result = trimesh.boolean.difference([mesh, *cutters], engine="manifold")
    if result is None:
        raise RuntimeError("Manifold could not create the four M6 mounting holes")
    if isinstance(result, list):
        result = trimesh.util.concatenate(result)
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("Unexpected M6-hole boolean output")
    result.remove_unreferenced_vertices()
    result.fix_normals()
    if not result.is_watertight:
        raise RuntimeError("M6-hole board is not watertight")
    return result


def local_seed_offsets(stimulus: str) -> list[tuple[float, float]]:
    if stimulus == "sharp_edge_x":
        return [(u, v) for u in (-12.0, 0.0, 12.0) for v in (-1.5, 0.0, 1.5)]
    if stimulus in ("sharp_edge_y", "rounded_ridge"):
        return [(u, v) for u in (-1.5, 0.0, 1.5) for v in (-12.0, 0.0, 12.0)]
    if stimulus == "sharp_edge_diagonal":
        result: list[tuple[float, float]] = []
        for tangent in (-12.0, 0.0, 12.0):
            for normal in (-1.5, 0.0, 1.5):
                result.append(((tangent + normal) / math.sqrt(2.0), (tangent - normal) / math.sqrt(2.0)))
        return result
    if stimulus in ("double_bump_12mm", "triple_bump_9mm", "micro_dot_lattice_5mm", "micro_checker_4mm"):
        return [(u, v) for u in (-5.0, 0.0, 5.0) for v in (-5.0, 0.0, 5.0)]
    return [(u, v) for u in (-7.0, 0.0, 7.0) for v in (-7.0, 0.0, 7.0)]


def surface_height_at(stimulus: str, u: float, v: float, cell_size: float, base_thickness: float) -> float:
    local_u = np.asarray([u], dtype=float)
    local_v = np.asarray([v], dtype=float)
    support = float(cell_support(local_u, local_v, cell_size)[0])
    feature = float(feature_height(stimulus, local_u, local_v)[0])
    return float(base_thickness + support * (feature - base_thickness))


def write_sampling_sites(
    path: Path,
    centres: dict[str, tuple[float, float]],
    cell_size: float,
    base_thickness: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for region in REGIONS:
        center_x, center_y = centres[region.region_id]
        for index, (u, v) in enumerate(local_seed_offsets(region.stimulus), start=1):
            records.append(
                {
                    "site_id": "{}_S{:02d}".format(region.region_id, index),
                    "region_id": region.region_id,
                    "category": region.category,
                    "stimulus": region.stimulus,
                    "board_x_mm": round(center_x + u, 4),
                    "board_y_mm": round(center_y + v, 4),
                    "expected_surface_z_mm": round(surface_height_at(region.stimulus, u, v, cell_size, base_thickness), 4),
                    "recommended_depth_min_mm": region.depth_range_mm[0],
                    "recommended_depth_max_mm": region.depth_range_mm[1],
                    "sampling_jitter_radius_mm": 1.5 if region.category in ("edge", "small_feature") else 3.0,
                    "notes": region.description,
                }
            )
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


def write_urdf(path: Path, mesh_name: str, top_height: float) -> None:
    mass = 3.0
    inertia = mass * (0.240**2 + 0.240**2) / 12.0
    path.write_text(
        """<?xml version=\"1.0\"?>
<robot name=\"tactile_gan_coverage_board\">
  <link name=\"board\">
    <inertial>
      <origin xyz=\"0 0 {com_z:.6f}\" rpy=\"0 0 0\"/>
      <mass value=\"{mass:.3f}\"/>
      <inertia ixx=\"{inertia:.6f}\" ixy=\"0\" ixz=\"0\" iyy=\"{inertia:.6f}\" iyz=\"0\" izz=\"{inertia:.6f}\"/>
    </inertial>
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><mesh filename=\"{mesh_name}\" scale=\"0.001 0.001 0.001\"/></geometry>
      <material name=\"stimulus_blue\"><color rgba=\"0.16 0.42 0.72 1\"/></material>
    </visual>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><mesh filename=\"{mesh_name}\" scale=\"0.001 0.001 0.001\"/></geometry>
    </collision>
  </link>
</robot>
""".format(com_z=top_height / 2000.0, mass=mass, inertia=inertia, mesh_name=mesh_name),
        encoding="utf-8",
    )


def render_preview(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    centres: dict[str, tuple[float, float]],
    cell_size: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 7), dpi=200)
    axis_top = figure.add_subplot(1, 2, 1)
    image = axis_top.imshow(
        heights,
        origin="lower",
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        cmap="viridis",
        vmin=float(np.min(heights)),
        vmax=float(np.max(heights)),
        interpolation="nearest",
    )
    for region in REGIONS:
        x, y = centres[region.region_id]
        label = "{}\n{}".format(region.region_id, region.stimulus.replace("_", " "))
        text = axis_top.text(x, y, label, ha="center", va="center", color="white", fontsize=7.2, weight="bold")
        text.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground="#111111"), path_effects.Normal()])
        axis_top.add_patch(
            plt.Rectangle((x - cell_size / 2.0, y - cell_size / 2.0), cell_size, cell_size, fill=False, edgecolor="white", linewidth=0.65)
        )
    for x, y in M6_CORNER_CENTRES_MM:
        axis_top.add_patch(plt.Circle((x, y), M6_CLEARANCE_DIAMETER_MM / 2.0, fill=False, edgecolor="#ffcf38", linewidth=1.2))
    axis_top.set_title("Topography and labelled sampling regions")
    axis_top.set_xlabel("board X (mm)")
    axis_top.set_ylabel("board Y (mm)")
    figure.colorbar(image, ax=axis_top, fraction=0.046, pad=0.04, label="surface height (mm)")

    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    step = max(1, int(round(2.0 / float(xs[1] - xs[0]))))
    x_grid, y_grid = np.meshgrid(xs[::step], ys[::step], indexing="xy")
    axis_3d.plot_surface(x_grid, y_grid, heights[::step, ::step], cmap="viridis", linewidth=0.0, antialiased=True, shade=True)
    axis_3d.set_xlim(float(xs[0]), float(xs[-1]))
    axis_3d.set_ylim(float(ys[0]), float(ys[-1]))
    axis_3d.set_zlim(0.0, float(np.max(heights)) + 1.0)
    axis_3d.set_box_aspect((1.0, 1.0, 0.15))
    axis_3d.view_init(elev=37, azim=-52)
    axis_3d.set_title("Isometric print orientation: features face upward")
    axis_3d.set_xlabel("X (mm)")
    axis_3d.set_ylabel("Y (mm)")
    axis_3d.set_zlabel("Z (mm)")
    figure.suptitle("TacTip GAN Geometry-Coverage Board | 16 balanced real-to-sim stimulus regions", y=0.98, fontsize=14, weight="bold")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_readme(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        """# TacTip GAN geometry-coverage board

This board is designed to broaden a real-to-sim tactile image translation
dataset.  It is not a reconstruction benchmark object: each of the 16 zones is
an intentionally controlled local-contact stimulus.

## Physical design

- Overall size: {board_size_mm:.1f} x {board_size_mm:.1f} mm.
- Base thickness: {base_thickness_mm:.1f} mm.
- Four corner mounting holes: {m6_hole_diameter_mm:.1f} mm clearance for M6,
  centred at (+/-112, +/-112) mm.
- Print flat on the build plate with the labelled contact geometry facing up.
- Recommended first print: 0.20 mm layers, 4 walls, 25 percent gyroid infill,
  rigid PLA/PETG, no supports on the tactile surface.

## Dataset protocol

Use `sampling_sites.csv` as the common real/simulation reference.  Treat every
site as a local visual-contact search target rather than trusting nominal Z:
the printed surface and TacTip skin both introduce millimetre-scale variation.
Each region has {samples_per_region} balanced pair slots, giving
{total_target_pairs} pair slots for one full training pass.  Randomize within
the CSV jitter radius, vary depth within the per-region safety range, and keep
the exact same sampled pose list for the simulation render.

Do not let the four M6 mounting holes enter the sample plan.  The low 4 mm
moats are deliberate isolation gaps, not tactile regions.

## Simulation

`tactile_gan_coverage_board.urdf` references the exported STL in metres and
can be loaded into PyBullet.  The CSV/JSON are the authoritative definitions
of region IDs and seed contacts; they should be copied with any paired dataset.
""".format(**metadata),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "tactile_gan_coverage_board_{}mm".format(int(round(args.board_size_mm)))
    paths = {
        "stl": output_dir / "{}.stl".format(stem),
        "urdf": output_dir / "{}.urdf".format(stem),
        "preview": output_dir / "{}_preview.png".format(stem),
        "sites": output_dir / "{}_sampling_sites.csv".format(stem),
        "manifest": output_dir / "{}_manifest.json".format(stem),
        "readme": output_dir / "README.md",
    }
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Output already exists: {}. Use --overwrite or a new --output-dir.".format(existing[0]))

    xs, ys, heights, centres = build_heights(
        float(args.board_size_mm),
        float(args.base_thickness_mm),
        float(args.cell_size_mm),
        float(args.cell_gap_mm),
        float(args.grid_resolution_mm),
    )
    mesh = height_field_mesh(xs, ys, heights)
    if not args.no_m6_mount_holes:
        mesh = drill_mount_holes(mesh, float(np.max(heights)))
    mesh.export(paths["stl"])
    sample_records = write_sampling_sites(paths["sites"], centres, float(args.cell_size_mm), float(args.base_thickness_mm))
    write_urdf(paths["urdf"], paths["stl"].name, float(np.max(heights)))
    render_preview(paths["preview"], xs, ys, heights, centres, float(args.cell_size_mm))

    category_counts: dict[str, int] = {}
    for region in REGIONS:
        category_counts[region.category] = category_counts.get(region.category, 0) + 1
    metadata: dict[str, object] = {
        "schema": "tactile_gan_coverage_board.v1",
        "units": "mm",
        "board_size_mm": float(args.board_size_mm),
        "base_thickness_mm": float(args.base_thickness_mm),
        "cell_size_mm": float(args.cell_size_mm),
        "cell_gap_mm": float(args.cell_gap_mm),
        "grid_resolution_mm": float(args.grid_resolution_mm),
        "surface_height_range_mm": [float(np.min(heights)), float(np.max(heights))],
        "m6_hole_diameter_mm": None if args.no_m6_mount_holes else M6_CLEARANCE_DIAMETER_MM,
        "m6_hole_centres_mm": [] if args.no_m6_mount_holes else [list(value) for value in M6_CORNER_CENTRES_MM],
        "nominal_tactip_compliant_diameter_mm": 40.0,
        "safe_central_feature_zone_mm": 36.0,
        "region_category_counts": category_counts,
        "regions": [asdict(region) | {"center_board_xy_mm": list(centres[region.region_id])} for region in REGIONS],
        "sampling_site_count": len(sample_records),
        "samples_per_region": REGIONS[0].samples_per_region,
        "total_target_pairs": sum(region.samples_per_region for region in REGIONS),
        "stl": paths["stl"].name,
        "urdf": paths["urdf"].name,
        "sampling_sites": paths["sites"].name,
        "preview": paths["preview"].name,
    }
    paths["manifest"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(paths["readme"], metadata)

    print("STL: {}".format(paths["stl"]))
    print("Preview: {}".format(paths["preview"]))
    print("Sampling sites: {} ({} seed contacts)".format(paths["sites"], len(sample_records)))
    print("Manifest: {}".format(paths["manifest"]))
    print("Mesh: {} vertices, {} faces, watertight={}".format(len(mesh.vertices), len(mesh.faces), mesh.is_watertight))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
