#!/usr/bin/env python3
"""Build verified TacTip docks for both printed v4 coverage-board hole patterns.

The first v4 board release has corner holes at +/-78 mm (156 mm pitch).  The
later mount-pitch150 release has them at +/-75 mm (150 mm pitch).  Their outer
tile edges are identical, so using a 150 mm dock on a 156 mm physical tile
misses each south mounting hole by 3 mm in X and Y.

This generator exports three one-piece printable docks:

* exact_150: closed 6.6 mm clearance holes for the +/-75 mm tile;
* exact_156: closed 6.6 mm clearance holes for the +/-78 mm tile;
* dual_150_156: short diagonal capsule slots that accept either pattern.

The exact versions are the preferred rigid fixtures.  The dual version is a
diagnostic/transition fixture when the already printed board version is not
known.  A continuous C-shaped board-edge clamp fixes the tile edge before
either pair of screws is tightened, so the dock does not hang from two screws
with an unsupported gap in the middle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

from design_v4_150mm_tactip_calibration_dock import (
    BOARD_BASE_TOP_Z_MM,
    TAC_TIP_APEX_BELOW_SHOULDER_MM,
    TAC_TIP_FLANGE_DIAMETER_MM,
    TAC_TIP_SOFT_TIP_THROUGH_DIAMETER_MM,
    annulus,
    box,
    clean,
    cylinder,
    difference,
    triangular_prism_x,
    union,
)


DEFAULT_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
TILE_SIZE_MM = 170.0
BOARD_THICKNESS_MM = 6.0
CLIP_VERTICAL_CLEARANCE_MM = 0.4
M6_DOCK_CLEARANCE_DIAMETER_MM = 6.6
GUIDE_OUTER_RADIUS_MM = 30.0
GUIDE_CLEARANCE_PER_SIDE_MM = 0.65
GUIDE_INNER_DIAMETER_MM = TAC_TIP_FLANGE_DIAMETER_MM + 2.0 * GUIDE_CLEARANCE_PER_SIDE_MM
SOFT_TIP_THROUGH_DIAMETER_MM = TAC_TIP_SOFT_TIP_THROUGH_DIAMETER_MM
PATTERNS = {
    "exact_150": ((-75.0, -75.0), (75.0, -75.0)),
    "exact_156": ((-78.0, -78.0), (78.0, -78.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to tactip_calibration_dock_verified_pitch_v2 under --board-dir.",
    )
    parser.add_argument("--tile-stl", type=Path, help="Tile STL used in the 3D preview.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    args.board_dir = args.board_dir.expanduser().resolve()
    if not args.board_dir.is_dir():
        parser.error("--board-dir does not exist: {}".format(args.board_dir))
    if args.output_dir is None:
        args.output_dir = args.board_dir / "tactip_calibration_dock_verified_pitch_v2"
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.tile_stl is None:
        args.tile_stl = args.board_dir / "tactile_gan_coverage_board_340mm_tile_nw.stl"
    args.tile_stl = args.tile_stl.expanduser().resolve()
    if not args.tile_stl.is_file():
        parser.error("--tile-stl does not exist: {}".format(args.tile_stl))
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_mesh(result: object, label: str) -> trimesh.Trimesh:
    if result is None:
        raise RuntimeError("{} returned no mesh".format(label))
    if isinstance(result, list):
        result = trimesh.util.concatenate(result)
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("{} returned {} rather than a mesh".format(label, type(result).__name__))
    return clean(result)


def oriented_box_xy(length_mm: float, width_mm: float, height_mm: float, start: np.ndarray, end: np.ndarray, z_mm: float) -> trimesh.Trimesh:
    """Return an XY-aligned rectangular part between two capsule endpoints."""
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        raise ValueError("Capsule endpoints must be different")
    angle = math.atan2(float(vector[1]), float(vector[0]))
    transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
    transform[:3, 3] = np.asarray(((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, z_mm), dtype=float)
    mesh = trimesh.creation.box(extents=(length_mm, width_mm, height_mm), transform=transform)
    return clean(mesh)


def capsule_slot(start: tuple[float, float], end: tuple[float, float], diameter_mm: float, height_mm: float, z_mm: float) -> trimesh.Trimesh:
    """Closed rounded slot with a centreline from start to end."""
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    return union(
        [
            cylinder(float(diameter_mm) / 2.0, height_mm, (a[0], a[1], z_mm)),
            cylinder(float(diameter_mm) / 2.0, height_mm, (b[0], b[1], z_mm)),
            oriented_box_xy(float(np.linalg.norm(b - a)), float(diameter_mm), height_mm, a, b, z_mm),
        ]
    )


def build_frame() -> tuple[list[trimesh.Trimesh], dict[str, Any]]:
    """Build material shared by all three hole variants."""
    ring_centre_xy = np.asarray((0.0, -139.0), dtype=float)
    reference_pad_centre = np.asarray((0.0, -110.0, 8.75), dtype=float)
    reference_pad_top_z = 12.0
    seat_top_z = reference_pad_top_z + TAC_TIP_APEX_BELOW_SHOULDER_MM

    # A full-width C-shaped clamp replaces the earlier disconnected pair of
    # tabs.  The board slides in from +Y until its south edge touches the
    # vertical rail.  It then rests on the lower lip while the upper lip and
    # two M6 screws lock the planar datum.  The 6.4 mm internal height accepts
    # a nominal 6.0 mm printed board without a loose visible gap.
    clip_top_bottom_z = BOARD_THICKNESS_MM + CLIP_VERTICAL_CLEARANCE_MM
    clip_top_thickness_z = 2.2
    clip_lower_thickness_z = 2.0
    clip_overlap_y = 0.35
    edge_y = -TILE_SIZE_MM / 2.0
    top_lip_north_y = -72.5
    lip_south_y = edge_y - clip_overlap_y
    lip_length_y = top_lip_north_y - lip_south_y
    lip_centre_y = (top_lip_north_y + lip_south_y) / 2.0
    upper_clip_lip = box(
        (TILE_SIZE_MM, lip_length_y, clip_top_thickness_z),
        (0.0, lip_centre_y, clip_top_bottom_z + clip_top_thickness_z / 2.0),
    )
    lower_clip_lip = box(
        (TILE_SIZE_MM, lip_length_y, clip_lower_thickness_z),
        (0.0, lip_centre_y, -clip_lower_thickness_z / 2.0),
    )
    edge_locator_bridge = box((TILE_SIZE_MM, 7.0, clip_top_bottom_z + clip_top_thickness_z), (0.0, -88.5, (clip_top_bottom_z + clip_top_thickness_z) / 2.0 - clip_lower_thickness_z))

    # The lower lip establishes the new floor at Z=-2 mm, so the two feet and
    # load-path members use the same bottom plane.  Board-local Z=0 remains
    # unchanged in the sampler transform; the seated TCP is always taught
    # after installation.
    floor_z = -clip_lower_thickness_z
    feet = [box((14.0, 96.0, 6.0), (sign * 27.0, -139.0, floor_z + 3.0)) for sign in (-1.0, 1.0)]
    crossbar = box((64.0, 10.0, 6.0), (0.0, -128.0, floor_z + 3.0))
    reference_spine = box((16.0, 44.0, 6.0), (0.0, -111.0, floor_z + 3.0))
    reference_pad = box((52.0, 40.0, 6.5), reference_pad_centre)

    gusset_vertices = ((-96.0, 4.0), (-182.0, 4.0), (float(ring_centre_xy[1]), seat_top_z + 2.0))
    gussets = [triangular_prism_x(sign * 31.0, 6.0, gusset_vertices) for sign in (-1.0, 1.0)]
    fusion_lugs = [box((8.0, 16.0, 16.0), (sign * 31.0, ring_centre_xy[1], 49.0)) for sign in (-1.0, 1.0)]
    seat = annulus(SOFT_TIP_THROUGH_DIAMETER_MM / 2.0, GUIDE_OUTER_RADIUS_MM, 3.0, (ring_centre_xy[0], ring_centre_xy[1], seat_top_z - 1.5))
    guide = annulus(GUIDE_INNER_DIAMETER_MM / 2.0, GUIDE_OUTER_RADIUS_MM, 10.0, (ring_centre_xy[0], ring_centre_xy[1], seat_top_z + 5.0))

    return [
        upper_clip_lip,
        lower_clip_lip,
        edge_locator_bridge,
        *feet,
        crossbar,
        reference_spine,
        reference_pad,
        *gussets,
        *fusion_lugs,
        seat,
        guide,
    ], {
        "ring_centre_xy_mm": ring_centre_xy.tolist(),
        "reference_pad_centre_local_mm": reference_pad_centre.tolist(),
        "reference_pad_top_z_mm": reference_pad_top_z,
        "seat_top_z_mm": seat_top_z,
        "clip_internal_height_mm": clip_top_bottom_z,
        "clip_bottom_z_mm": floor_z,
    }


def pattern_cutters(pattern: str) -> tuple[list[trimesh.Trimesh], dict[str, Any]]:
    if pattern in PATTERNS:
        holes = PATTERNS[pattern]
        return [cylinder(M6_DOCK_CLEARANCE_DIAMETER_MM / 2.0, 24.0, (x, y, 8.0)) for x, y in holes], {
            "type": "closed_round_holes",
            "physical_board_hole_centres_tile_local_xy_mm": [list(point) for point in holes],
            "mount_hole_diameter_mm": M6_DOCK_CLEARANCE_DIAMETER_MM,
            "mounting_instruction": "Place the south tile edge against the dock rail, then insert and tighten two M6 screws through the closed clearance holes.",
        }
    if pattern != "dual_150_156":
        raise ValueError(pattern)

    # The 150 and 156 pattern pairs differ by 3 mm outward and 3 mm rearward
    # at each end.  Each diagonal capsule contains both true screw axes.
    cutters = [
        capsule_slot(PATTERNS["exact_150"][index], PATTERNS["exact_156"][index], M6_DOCK_CLEARANCE_DIAMETER_MM, 24.0, 8.0)
        for index in range(2)
    ]
    return cutters, {
        "type": "dual_pattern_diagonal_capsule_slots",
        "supported_board_hole_patterns_tile_local_xy_mm": {
            "v4_original_156mm": [list(point) for point in PATTERNS["exact_156"]],
            "v4_mountpitch150": [list(point) for point in PATTERNS["exact_150"]],
        },
        "slot_diameter_mm": M6_DOCK_CLEARANCE_DIAMETER_MM,
        "mounting_instruction": "Hold the south tile edge against the dock rail. Align both screws at the matching 150 mm or 156 mm ends of the two slots, then tighten both M6 screws fully.",
    }


def build_variant(pattern: str) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    solids, shared = build_frame()
    cutters, mounting = pattern_cutters(pattern)
    dock = difference(union(solids), cutters)
    if not dock.is_watertight:
        raise RuntimeError("{} dock is not watertight".format(pattern))
    components = dock.split(only_watertight=False)
    if len(components) != 1:
        raise RuntimeError("{} dock has {} disconnected components".format(pattern, len(components)))
    info = {
        "schema": "tactile_gan_coverage_board.v4.verified_pitch_dock.v2",
        "variant": pattern,
        "units": "mm",
        "tile_frame": {
            "origin": "centre of one 170 x 170 mm tile at its bottom plane",
            "x_positive": "tile local +X",
            "y_positive": "from the south dock edge into the tactile board",
            "z_positive": "physical up",
        },
        "board_interface": {
            "tile_size_mm": TILE_SIZE_MM,
            "board_base_top_z_mm": BOARD_BASE_TOP_Z_MM,
            "south_tile_edge_y_mm": -TILE_SIZE_MM / 2.0,
            "edge_locator_face_y_mm": -TILE_SIZE_MM / 2.0,
            "board_thickness_mm": BOARD_THICKNESS_MM,
            "clip_internal_height_mm": shared["clip_internal_height_mm"],
            "clip_clearance_mm": CLIP_VERTICAL_CLEARANCE_MM,
            "clip_bottom_z_mm": shared["clip_bottom_z_mm"],
            "dock_mount_clearance_diameter_mm": M6_DOCK_CLEARANCE_DIAMETER_MM,
            "note": "The continuous C-shaped clamp bears against the full south tile edge, its lower lip supports the board underside, and its upper lip bears on the board top. The printed board may keep nominal 6.0 mm holes; the dock deliberately uses 6.6 mm clearance so the M6 screw passes through it and locks in the board/base thread.",
            **mounting,
        },
        "tactip_reference": {
            "tool": 2,
            "tool_tcp_assumption": "Tool(2) origin is the TacTip compliant contact apex.",
            "rigid_flange_diameter_mm": TAC_TIP_FLANGE_DIAMETER_MM,
            "guide_inner_diameter_mm": GUIDE_INNER_DIAMETER_MM,
            "guide_outer_diameter_mm": GUIDE_OUTER_RADIUS_MM * 2.0,
            "soft_tip_through_diameter_mm": SOFT_TIP_THROUGH_DIAMETER_MM,
            "seat_top_z_mm": shared["seat_top_z_mm"],
            "apex_below_shoulder_mm": TAC_TIP_APEX_BELOW_SHOULDER_MM,
            "nominal_seated_tool_tcp_local_mm": [0.0, -139.0, shared["reference_pad_top_z_mm"]],
            "cable_slot_direction": "tile local -Y",
            "cable_slot_width_mm": 8.0,
        },
        "reference_pad": {
            "centre_local_mm": shared["reference_pad_centre_local_mm"],
            "top_z_mm": shared["reference_pad_top_z_mm"],
            "usable_flat_size_mm": [50.0, 38.0],
            "purpose": "A flat visual-contact reference after the dock is mechanically mated to the tile.",
        },
        "sampler_transform": {
            "mapping": "p_base = p_seated_tcp + R_tile_to_base @ (p_tile - p_seated_tool_tcp_local)",
            "note": "Teach a fresh Tool(2) seated TCP after installing a new dock variant. Do not reuse a profile from the old lightweight dock.",
        },
        "print": {
            "material": "PETG preferred; PLA+ acceptable for a calibration-only dock.",
            "orientation": "Print with both feet on the plate and locating ring upward.",
            "settings": "0.28 mm layers, 4 walls, 20% gyroid infill.",
            "supports": "Normally none; add inside-guide support only if the slicer flags a bridge.",
        },
    }
    return dock, info


def render_alignment(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    figure, axis = plt.subplots(figsize=(10.5, 8.8), dpi=200)
    axis.add_patch(Rectangle((-85.0, -85.0), 170.0, 170.0, facecolor="#2e6f9e", edgecolor="#153650", linewidth=2.0, alpha=0.9))
    axis.text(0.0, 10.0, "170 x 170 mm v4 tile", color="white", ha="center", va="center", fontsize=12, weight="bold")
    axis.plot((-85.0, 85.0), (-85.0, -85.0), color="#0c2437", linewidth=3.0)
    axis.text(0.0, -81.5, "physical south tile edge / dock locating rail", color="#fff5d8", ha="center", va="bottom", fontsize=8.5, weight="bold")

    colors = {"exact_150": "#f1c40f", "exact_156": "#f45b69"}
    labels = {"exact_150": "150 mm board holes (+/-75 mm)", "exact_156": "156 mm board holes (+/-78 mm)"}
    for name, points in PATTERNS.items():
        for point in points:
            axis.add_patch(Circle(point, 3.0, facecolor=colors[name], edgecolor="#151515", linewidth=0.9, zorder=5))
        axis.plot([point[0] for point in points], [point[1] for point in points], color=colors[name], linewidth=1.1, alpha=0.8, label=labels[name])

    for index in range(2):
        a = np.asarray(PATTERNS["exact_150"][index])
        b = np.asarray(PATTERNS["exact_156"][index])
        axis.plot((a[0], b[0]), (a[1], b[1]), color="#462c8c", linewidth=8.0, alpha=0.66, solid_capstyle="round", label="v2 dual-pitch capsule slot" if index == 0 else None)
        axis.plot((a[0], b[0]), (a[1], b[1]), color="#f7f3ff", linewidth=3.6, alpha=0.95, solid_capstyle="round")

    axis.add_patch(Rectangle((-85.0, -92.0), 170.0, 7.0, facecolor="#e27024", edgecolor="#6b2d04", linewidth=1.3, zorder=4))
    axis.add_patch(Circle((0.0, -139.0), GUIDE_OUTER_RADIUS_MM, facecolor="#e27024", edgecolor="#6b2d04", linewidth=1.4, zorder=3))
    axis.add_patch(Circle((0.0, -139.0), GUIDE_INNER_DIAMETER_MM / 2.0, facecolor="white", edgecolor="#6b2d04", linewidth=1.0, zorder=4))
    axis.add_patch(Circle((0.0, -139.0), SOFT_TIP_THROUGH_DIAMETER_MM / 2.0, facecolor="#b9e4f1", edgecolor="#6b2d04", linewidth=0.8, zorder=5))
    axis.text(0.0, -139.0, "TacTip\nseat", ha="center", va="center", fontsize=8.5, weight="bold")

    axis.annotate("150 mm", xy=(-75.0, -104.0), xytext=(75.0, -104.0), ha="center", va="bottom", arrowprops={"arrowstyle": "<->", "color": "#f1c40f"}, color="#8f7200", weight="bold")
    axis.annotate("156 mm", xy=(-78.0, -113.0), xytext=(78.0, -113.0), ha="center", va="bottom", arrowprops={"arrowstyle": "<->", "color": "#f45b69"}, color="#aa1024", weight="bold")
    axis.add_patch(FancyArrowPatch((0.0, 90.0), (0.0, 115.0), arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color="#163d5a"))
    axis.text(4.0, 107.0, "tile +Y", color="#163d5a", fontsize=10, weight="bold")
    axis.text(0.0, -210.0, "The orange rail is a continuous C-shaped board-edge clamp: it supports the board from below, stops its edge, and bears from above.\nUse the yellow 150-mm version only with the 150-mm board; use the red 156-mm version only with the original board.\nThe purple dual-pitch slot version is the only variant that fits both.", ha="center", va="center", fontsize=8.8)
    axis.set_aspect("equal")
    axis.set_xlim(-108.0, 108.0)
    axis.set_ylim(-220.0, 125.0)
    axis.set_xlabel("tile local X (mm)")
    axis.set_ylabel("tile local Y (mm)")
    axis.set_title("Verified v4 board / TacTip dock hole alignment", weight="bold")
    axis.legend(loc="upper right", fontsize=8.5)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def render_clip_section(path: Path) -> None:
    """Render the board-edge C-clamp section at an X location away from holes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    edge_y = -TILE_SIZE_MM / 2.0
    clip_top_bottom_z = BOARD_THICKNESS_MM + CLIP_VERTICAL_CLEARANCE_MM
    clip_top_thickness_z = 2.2
    clip_lower_thickness_z = 2.0
    lip_south_y = edge_y - 0.35
    top_lip_north_y = -72.5
    figure, axis = plt.subplots(figsize=(10.0, 5.0), dpi=200)
    axis.add_patch(Rectangle((edge_y, 0.0), 50.0, BOARD_THICKNESS_MM, facecolor="#2e6f9e", edgecolor="#153650", linewidth=1.7, label="v4 tile base"))
    axis.add_patch(Rectangle((edge_y - 7.0, -clip_lower_thickness_z), 7.0, clip_top_bottom_z + clip_top_thickness_z + clip_lower_thickness_z, facecolor="#e27024", edgecolor="#6b2d04", linewidth=1.5, label="dock edge stop"))
    axis.add_patch(Rectangle((lip_south_y, -clip_lower_thickness_z), top_lip_north_y - lip_south_y, clip_lower_thickness_z, facecolor="#e27024", edgecolor="#6b2d04", linewidth=1.5, label="lower support lip"))
    axis.add_patch(Rectangle((lip_south_y, clip_top_bottom_z), top_lip_north_y - lip_south_y, clip_top_thickness_z, facecolor="#e27024", edgecolor="#6b2d04", linewidth=1.5, label="upper clamping lip"))
    axis.annotate("board slides in from +Y until its edge meets this stop", xy=(edge_y, 3.0), xytext=(-132.0, 12.5), arrowprops={"arrowstyle": "->", "color": "#1d3557"}, color="#1d3557", fontsize=9.5, weight="bold")
    axis.annotate("lower lip carries board underside", xy=(-78.0, -0.2), xytext=(-132.0, -7.0), arrowprops={"arrowstyle": "->", "color": "#1d3557"}, color="#1d3557", fontsize=9.5, weight="bold")
    axis.annotate("0.4 mm assembly clearance", xy=(-75.0, 6.2), xytext=(-50.0, 13.0), arrowprops={"arrowstyle": "->", "color": "#7a3e00"}, color="#7a3e00", fontsize=9.5, weight="bold")
    axis.annotate("upper lip prevents lift", xy=(-76.0, clip_top_bottom_z + 1.0), xytext=(-50.0, 18.0), arrowprops={"arrowstyle": "->", "color": "#1d3557"}, color="#1d3557", fontsize=9.5, weight="bold")
    axis.text(-108.0, 23.5, "side section: tile local Y-Z", fontsize=13, weight="bold")
    axis.set_aspect("equal")
    axis.set_xlim(-145.0, -20.0)
    axis.set_ylim(-10.0, 27.0)
    axis.set_xlabel("tile local Y (mm)")
    axis.set_ylabel("Z (mm)")
    axis.legend(loc="upper right", fontsize=8.5)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def render_preview(path: Path, tile_path: Path, dock: trimesh.Trimesh) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tile = trimesh.load_mesh(tile_path, force="mesh", process=False)
    if not isinstance(tile, trimesh.Trimesh):
        raise RuntimeError("Could not load tile preview mesh")
    tip = trimesh.creation.uv_sphere(radius=20.0, count=[40, 28])
    tip.apply_translation((0.0, -139.0, 32.0))
    figure = plt.figure(figsize=(12.0, 5.5), dpi=180)
    items = ((tile, "#2f6f9e", 0.82), (dock, "#e27024", 1.0), (tip, "#42b6ce", 0.42))
    vertices = np.vstack([mesh.vertices for mesh, _colour, _alpha in items])
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
    centre = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1.0)
    for index, (elev, azim, title) in enumerate(((32.0, -62.0, "Assembly"), (83.0, -90.0, "Top: hole pattern")), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        for mesh, colour, alpha in items:
            faces = mesh.faces
            if len(faces) > 12000:
                faces = faces[np.linspace(0, len(faces) - 1, 12000, dtype=int)]
            axis.add_collection3d(Poly3DCollection(mesh.vertices[faces], facecolor=colour, edgecolor="none", alpha=alpha))
        axis.set_xlim(centre[0] - radius, centre[0] + radius)
        axis.set_ylim(centre[1] - radius, centre[1] + radius)
        axis.set_zlim(0.0, centre[2] + radius * 0.55)
        axis.set_box_aspect((1.2, 1.25, 0.42))
        axis.view_init(elev=elev, azim=azim)
        axis.set_axis_off()
        axis.set_title(title)
    figure.suptitle("Blue: v4 tile | Orange: verified dual-pitch dock | Cyan: seated TacTip apex", weight="bold")
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def write_readme(path: Path, files: dict[str, Path]) -> None:
    path.write_text(
        "# Verified TacTip Calibration Dock v2\n\n"
        "The earlier 150 mm dock cannot mount to the original v4 board, whose south pair is 156 mm apart. "
        "This directory contains independently verified alternatives.\n\n"
        "## Choose one STL\n\n"
        "- `v4_tactip_calibration_dock_v2_exact_150.stl`: use only when the two south board-hole centres measure **150 mm**.\n"
        "- `v4_tactip_calibration_dock_v2_exact_156.stl`: use only when they measure **156 mm**.\n"
        "- `v4_tactip_calibration_dock_v2_dual_150_156.stl`: fits either pattern through two short diagonal slots. "
        "Use it only when the printed board version is unknown; the exact version is more rigid.\n\n"
        "## Mechanical check before collecting\n\n"
        "1. Put the tile with its tactile surface facing up.\n"
        "2. Put the dock on the south edge: the straight rear rail must touch the tile's straight edge.\n"
        "3. Slide the board into the continuous C-shaped rail: its underside sits on the lower lip and its south edge reaches the hard stop.\n"
        "4. Insert both M6 screws without force. Both must drop through the dock and into the board together.\n"
        "5. Tighten both screws. The dock must not translate or yaw by hand.\n"
        "6. Only then seat the TacTip and record a **new** Tool(2) fixture profile.\n\n"
        "The board's 6.0 mm holes remain unchanged. The dock's 6.6 mm holes are deliberate M6 clearance, not an error.\n\n"
        "Key files:\n\n"
        + "\n".join("- `{}`".format(path.name) for path in files.values())
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("exact_150", "exact_156", "dual_150_156")
    outputs = {name: args.output_dir / "v4_tactip_calibration_dock_v2_{}.stl".format(name) for name in names}
    metadata_paths = {name: args.output_dir / "v4_tactip_calibration_dock_v2_{}_design.json".format(name) for name in names}
    layout_path = args.output_dir / "v4_tactip_calibration_dock_v2_hole_alignment.png"
    section_path = args.output_dir / "v4_tactip_calibration_dock_v2_clip_section.png"
    preview_path = args.output_dir / "v4_tactip_calibration_dock_v2_dual_preview.png"
    existing = [path for path in (*outputs.values(), *metadata_paths.values(), layout_path, section_path, preview_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("{} already exists; use --overwrite to replace generated dock files".format(existing[0]))

    dual_dock: trimesh.Trimesh | None = None
    for name in names:
        dock, info = build_variant(name)
        stl_path = outputs[name]
        dock.export(stl_path)
        verify = trimesh.load_mesh(stl_path, force="mesh", process=True)
        if not isinstance(verify, trimesh.Trimesh) or not verify.is_watertight:
            raise RuntimeError("{} export is not a watertight mesh".format(name))
        info.update(
            {
                "stl": str(stl_path.resolve()),
                "stl_sha256": sha256_file(stl_path),
                "extent_mm": [float(value) for value in verify.extents],
                "volume_mm3": float(abs(verify.volume)),
                "watertight": bool(verify.is_watertight),
                "components": int(len(verify.split(only_watertight=False))),
                "tile_preview_stl": str(args.tile_stl.resolve()),
                "tile_preview_stl_sha256": sha256_file(args.tile_stl),
            }
        )
        metadata_paths[name].write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print("Wrote {} ({:.1f} mm^3)".format(stl_path, abs(verify.volume)), flush=True)
        if name == "dual_150_156":
            dual_dock = dock

    if not args.no_preview:
        render_alignment(layout_path)
        render_clip_section(section_path)
        if dual_dock is None:
            raise RuntimeError("Dual dock was not generated")
        render_preview(preview_path, args.tile_stl, dual_dock)
    write_readme(args.output_dir / "README.md", outputs)
    print("Wrote {}".format(layout_path))
    print("Wrote {}".format(section_path))
    print("Wrote {}".format(preview_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
