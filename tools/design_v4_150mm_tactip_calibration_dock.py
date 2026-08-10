#!/usr/bin/env python3
"""Create a lightweight TacTip calibration dock for the v4 150 mm-pitch tile.

The dock is a one-piece printable add-on for one 170 x 170 mm v4 tactile
coverage-board tile.  It uses the two southern M6 holes at tile-local
``(-75, -75)`` and ``(75, -75)`` mm, whose centre distance is exactly 150 mm.
The tactile board itself must be secured to a flat table through its four M6
holes; the dock then shares its southern pair of screws and transfers vertical
TacTip seating load through two low feet and two solid triangular gussets.

This is deliberately a skeletal fixture, rather than a full base plate: the
rigid TacTip flange ring, a 50 x 40 mm reference pad, two narrow feet, and the
two load paths are retained, while large non-functional slabs are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh


DEFAULT_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
# Match the board's requested nominal 6.0 mm holes exactly. The U-slots are
# therefore a snug printed M6 fit rather than a 6.6 mm clearance fit.
M6_CLEARANCE_MM = 6.0
BOARD_TILE_SIZE_MM = 170.0
BOARD_BASE_TOP_Z_MM = 6.0
TAC_TIP_FLANGE_DIAMETER_MM = 49.5
TAC_TIP_GUIDE_INNER_DIAMETER_MM = 50.8
TAC_TIP_SOFT_TIP_THROUGH_DIAMETER_MM = 44.0
TAC_TIP_APEX_BELOW_SHOULDER_MM = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a tactip_calibration_dock_lightweight_v1 directory in --board-dir.",
    )
    parser.add_argument(
        "--tile-stl",
        type=Path,
        help="Optional v4 tile STL used only in the assembly preview.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--cable-slot-width-mm", type=float, default=8.0)
    parser.add_argument("--guide-clearance-per-side-mm", type=float, default=0.65)
    args = parser.parse_args()
    args.board_dir = args.board_dir.expanduser().resolve()
    if not args.board_dir.is_dir():
        parser.error("--board-dir does not exist: {}".format(args.board_dir))
    if args.output_dir is None:
        args.output_dir = args.board_dir / "tactip_calibration_dock_lightweight_v1"
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.tile_stl is None:
        args.tile_stl = args.board_dir / "tactile_gan_coverage_board_340mm_tile_nw.stl"
    args.tile_stl = args.tile_stl.expanduser().resolve()
    if not args.tile_stl.is_file():
        parser.error("--tile-stl does not exist: {}".format(args.tile_stl))
    if not 0.0 <= float(args.cable_slot_width_mm) <= 16.0:
        parser.error("--cable-slot-width-mm must be in [0, 16]")
    if not 0.2 <= float(args.guide_clearance_per_side_mm) <= 1.5:
        parser.error("--guide-clearance-per-side-mm must be in [0.2, 1.5]")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def as_mesh(result: object, operation: str) -> trimesh.Trimesh:
    if result is None:
        raise RuntimeError("Boolean {} returned no mesh".format(operation))
    if isinstance(result, list):
        result = trimesh.util.concatenate(result)
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("Unexpected {} result: {}".format(operation, type(result).__name__))
    return clean(result)


def union(meshes: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    return as_mesh(trimesh.boolean.union(list(meshes), engine="manifold"), "union")


def difference(base: trimesh.Trimesh, cutters: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    return as_mesh(trimesh.boolean.difference([base, *list(cutters)], engine="manifold"), "difference")


def box(extents: Iterable[float], centre: Iterable[float]) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(tuple(centre), dtype=float)
    return clean(trimesh.creation.box(extents=np.asarray(tuple(extents), dtype=float), transform=transform))


def cylinder(radius_mm: float, height_mm: float, centre: Iterable[float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=float(radius_mm), height=float(height_mm), sections=96)
    mesh.apply_translation(np.asarray(tuple(centre), dtype=float))
    return clean(mesh)


def annulus(inner_radius_mm: float, outer_radius_mm: float, height_mm: float, centre: Iterable[float]) -> trimesh.Trimesh:
    return difference(
        cylinder(outer_radius_mm, height_mm, centre),
        [cylinder(inner_radius_mm, height_mm + 0.8, centre)],
    )


def triangular_prism_x(
    centre_x_mm: float,
    width_x_mm: float,
    yz_vertices: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> trimesh.Trimesh:
    """Create a solid triangular gusset extruded along the tile-local X axis."""
    half = float(width_x_mm) / 2.0
    vertices: list[tuple[float, float, float]] = []
    for x in (float(centre_x_mm) - half, float(centre_x_mm) + half):
        vertices.extend((x, y, z) for y, z in yz_vertices)
    faces = np.asarray(
        (
            (0, 2, 1),
            (3, 4, 5),
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (2, 0, 3),
            (2, 3, 5),
        ),
        dtype=np.int64,
    )
    return clean(trimesh.Trimesh(vertices=np.asarray(vertices), faces=faces, process=True))


def load_and_validate_manifest(board_dir: Path) -> dict[str, Any]:
    manifests = sorted(board_dir.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Expected exactly one manifest under {}, found {}".format(board_dir, len(manifests)))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if list(manifest.get("tile_size_mm", ())) != [170.0, 170.0]:
        raise ValueError("This dock is only for the 170 mm v4 board tiles")
    if not np.isclose(float(manifest.get("m6_tile_hole_pitch_mm", 0.0)), 150.0):
        raise ValueError("This dock requires the v4 150 mm-pitch board variant")
    return manifest


def build_dock(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    _ = manifest
    board_holes = ((-75.0, -75.0), (75.0, -75.0))
    ring_centre_xy = np.asarray((0.0, -139.0), dtype=float)
    reference_pad_centre = np.asarray((0.0, -110.0, 8.75), dtype=float)
    reference_pad_top_z = 12.0
    seat_top_z = reference_pad_top_z + TAC_TIP_APEX_BELOW_SHOULDER_MM
    seated_tcp_local = np.asarray((ring_centre_xy[0], ring_centre_xy[1], reference_pad_top_z), dtype=float)
    guide_inner_radius = TAC_TIP_FLANGE_DIAMETER_MM / 2.0 + float(args.guide_clearance_per_side_mm)
    guide_outer_radius = 30.0
    soft_tip_radius = TAC_TIP_SOFT_TIP_THROUGH_DIAMETER_MM / 2.0
    if guide_outer_radius <= guide_inner_radius:
        raise ValueError("Guide wall is non-positive; reduce --guide-clearance-per-side-mm")

    # Do not span the raised tactile cells with a full rail. Each mounting tab
    # occupies only a quiet corner around one M6 hole. Its north edge ends at
    # Y=-72.5 mm, where the board is still at the 6 mm low-border height. A
    # 0.30 mm vertical clearance avoids rubbing against the printed board
    # surface. The M6 screw and washer clamp the tab without needing a plastic
    # bearing face on the board. The circular M6 cut opens through that edge,
    # making a slide-on U-slot:
    # loosen the two board screws, slide the dock in from local -Y, retighten.
    mount_tabs = [box((18.0, 13.5, 4.0), (sign * 75.0, -79.25, 8.3)) for sign in (-1.0, 1.0)]
    outboard_bridge = box((170.0, 7.0, 10.0), (0.0, -88.5, 5.0))

    # Two floor-contact feet retain structural stiffness without a heavy full
    # plate. The outboard bridge is wholly outside the tile's y=-85 mm edge
    # and overlaps both the mounting tabs and the feet.
    feet = [box((14.0, 96.0, 6.0), (sign * 27.0, -139.0, 3.0)) for sign in (-1.0, 1.0)]

    # A small central spine and crossbar support a documented flat reference
    # pad. It is clear of both the tile and the TacTip locating ring.
    crossbar = box((64.0, 10.0, 6.0), (0.0, -128.0, 3.0))
    reference_spine = box((16.0, 44.0, 6.0), (0.0, -111.0, 3.0))
    reference_pad = box((52.0, 40.0, 6.5), reference_pad_centre)

    # Full solid side gussets meet only the guide's outer wall. Their inner
    # edges start at |X|=28 mm, beyond the 25.4 mm guide radius, so nothing
    # protrudes into the 50.8 mm TacTip locating bore.
    gusset_vertices = ((-96.0, 4.0), (-182.0, 4.0), (float(ring_centre_xy[1]), seat_top_z + 2.0))
    gussets = [triangular_prism_x(sign * 31.0, 6.0, gusset_vertices) for sign in (-1.0, 1.0)]
    fusion_lugs = [box((8.0, 16.0, 16.0), (sign * 31.0, ring_centre_xy[1], 49.0)) for sign in (-1.0, 1.0)]

    # The lower seat touches only the rigid flange. The 44 mm through opening
    # remains free for the compliant TacTip skin. A straight top guide is less
    # material than a funnel and sufficient for robot placement after teaching.
    seat = annulus(soft_tip_radius, guide_outer_radius, 3.0, (ring_centre_xy[0], ring_centre_xy[1], seat_top_z - 1.5))
    guide = annulus(guide_inner_radius, guide_outer_radius, 10.0, (ring_centre_xy[0], ring_centre_xy[1], seat_top_z + 5.0))

    dock = union(
        [
            *mount_tabs,
            outboard_bridge,
            *feet,
            crossbar,
            reference_spine,
            reference_pad,
            *gussets,
            *fusion_lugs,
            seat,
            guide,
        ]
    )
    cutters = [cylinder(M6_CLEARANCE_MM / 2.0, 24.0, (x, y, 8.0)) for x, y in board_holes]
    if float(args.cable_slot_width_mm) > 0.0:
        # A radial outboard (-Y) slot gives the cable a repeatable exit path,
        # while both side gussets continue to retain the ring rigidly.
        slot = box(
            (float(args.cable_slot_width_mm), guide_outer_radius + 10.0, 24.0),
            (0.0, ring_centre_xy[1] - guide_outer_radius / 2.0 - 5.0, seat_top_z + 6.0),
        )
        cutters.append(slot)
    dock = difference(dock, cutters)
    if not dock.is_watertight:
        raise RuntimeError("Generated lightweight dock is not watertight")
    components = dock.split(only_watertight=False)
    if len(components) != 1:
        raise RuntimeError("Generated lightweight dock has {} disconnected components".format(len(components)))

    info: dict[str, Any] = {
        "schema": "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v1",
        "units": "mm",
        "tile_frame": {
            "origin": "centre of one printed 170 x 170 mm v4 tile at its bottom plane",
            "x_positive": "tile local +X",
            "y_positive": "from the south dock edge into the tactile board",
            "z_positive": "physical up",
        },
        "board_interface": {
            "tile_size_mm": BOARD_TILE_SIZE_MM,
            "board_base_top_z_mm": BOARD_BASE_TOP_Z_MM,
            "shared_board_m6_holes_local_xy_mm": [list(point) for point in board_holes],
            "shared_board_hole_pitch_mm": 150.0,
            "m6_clearance_diameter_mm": M6_CLEARANCE_MM,
            "dock_tab_clearance_above_board_mm": 0.30,
            "mounting_note": (
                "Secure all four board-tile holes to a flat table/base first. Then use two longer M6 screws through "
                "the two dock U-slots and the south pair at (-75,-75) and (75,-75) mm. The dock feet sit on the same "
                "table; the open side of each U-slot must face tile local +Y."
            ),
        },
        "tactip_reference": {
            "tool": 2,
            "tool_tcp_assumption": "Tool(2) origin is the TacTip compliant contact apex.",
            "rigid_flange_diameter_mm": TAC_TIP_FLANGE_DIAMETER_MM,
            "guide_inner_diameter_mm": guide_inner_radius * 2.0,
            "guide_outer_diameter_mm": guide_outer_radius * 2.0,
            "soft_tip_through_diameter_mm": TAC_TIP_SOFT_TIP_THROUGH_DIAMETER_MM,
            "seat_top_z_mm": seat_top_z,
            "apex_below_shoulder_mm": TAC_TIP_APEX_BELOW_SHOULDER_MM,
            "nominal_seated_tool_tcp_local_mm": seated_tcp_local.tolist(),
            "cable_slot_direction": "tile local -Y",
            "cable_slot_width_mm": float(args.cable_slot_width_mm),
        },
        "reference_pad": {
            "centre_local_mm": reference_pad_centre.tolist(),
            "top_z_mm": reference_pad_top_z,
            "usable_flat_size_mm": [50.0, 38.0],
            "purpose": "Optional visual-contact / image-change check before executing board sampling.",
        },
        "sampler_transform": {
            "mapping": "p_base = p_seated_tcp + R_tile_to_base @ (p_tile - p_seated_tool_tcp_local)",
            "note": "Teach the seated Tool(2) TCP once after the fixture is bolted down. Keep Joint 6 and the cable exit orientation unchanged across the run.",
        },
        "print": {
            "part_count": 1,
            "material": "PETG preferred; PLA+ is acceptable for calibration-only use.",
            "orientation": "Print with both narrow feet on the build plate and the locating ring upward.",
            "settings": "0.28 mm layers, 4 walls, 18-22% gyroid infill.",
            "supports": "Normally no support. Enable support only for the inside of the locating ring if the slicer flags a bridge.",
            "lightweight_design": "Two 14 mm feet, two 10 mm solid gussets, and open central space replace a full base plate.",
        },
        "workflow": [
            "Bolt the 170 mm v4 tile flat to the table through all four of its M6 holes.",
            "Loosen the two southern screws, slide the dock U-slots in from tile local -Y, then retighten the M6 screws.",
            "Seat the TacTip rigid flange in the ring, with the cable in the outboard slot.",
            "Record the Tool(2) TCP while seated. This becomes the repeatable rest datum for that board setup.",
            "Move vertically out of the dock before moving over a tactile cell; use image-change contact detection at every sampling site.",
        ],
    }
    return dock, info


def preview_triangles(mesh: trimesh.Trimesh, max_faces: int = 22000) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh.faces) <= max_faces:
        return mesh.triangles, mesh.face_normals
    indices = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=int)
    return mesh.triangles[indices], mesh.face_normals[indices]


def shaded_colours(normals: np.ndarray, colour: tuple[float, float, float], alpha: float) -> np.ndarray:
    light = np.asarray((-0.45, -0.55, 0.70), dtype=float)
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.maximum(0.0, normals @ light)
    base = np.asarray(colour, dtype=float)
    return np.column_stack((base[0] * shade, base[1] * shade, base[2] * shade, np.full_like(shade, alpha)))


def render_preview(path: Path, tile_path: Path, dock: trimesh.Trimesh, info: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tile = trimesh.load_mesh(tile_path, force="mesh", process=False)
    if not isinstance(tile, trimesh.Trimesh):
        raise RuntimeError("Could not load tile preview mesh")
    seated = np.asarray(info["tactip_reference"]["nominal_seated_tool_tcp_local_mm"], dtype=float)
    tactip_tip = trimesh.creation.uv_sphere(radius=20.0, count=[40, 28])
    tactip_tip.apply_translation(seated + np.asarray((0.0, 0.0, 20.0)))
    all_vertices = np.vstack((tile.vertices, dock.vertices, tactip_tip.vertices))
    lower, upper = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
    midpoint = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1.0)
    figure = plt.figure(figsize=(17, 6), dpi=180)
    views = (
        (31.0, -58.0, "Dock on the south (-Y) tile edge"),
        (78.0, -90.0, "Top view: 150 mm shared M6 pair"),
        (16.0, 22.0, "Open frame and solid triangular load paths"),
    )
    renderables = (
        (tile, (0.10, 0.34, 0.58), 0.88),
        (dock, (0.95, 0.39, 0.06), 1.0),
        (tactip_tip, (0.08, 0.65, 0.82), 0.45),
    )
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        for mesh, colour, alpha in renderables:
            triangles, normals = preview_triangles(mesh)
            axis.add_collection3d(Poly3DCollection(triangles, facecolors=shaded_colours(normals, colour, alpha), edgecolor="none"))
        axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
        axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
        axis.set_zlim(0.0, midpoint[2] + radius * 0.55)
        axis.set_box_aspect((1.18, 1.18, 0.42))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(title)
    figure.suptitle("Blue: v4 170 mm tile | Orange: lightweight dock | Cyan: nominal seated TacTip tip", y=0.98, fontsize=13, weight="bold")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_installation_layout(path: Path) -> None:
    """Draw a readable top-view installation plan independent of mesh density."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

    figure, axis = plt.subplots(figsize=(8.2, 9.0), dpi=180)
    axis.add_patch(Rectangle((-85.0, -85.0), 170.0, 170.0, facecolor="#2c6c99", edgecolor="#12334b", linewidth=1.8))
    axis.add_patch(Rectangle((-70.0, -70.0), 140.0, 140.0, facecolor="#78a7c5", alpha=0.55, edgecolor="none"))
    axis.text(0.0, 6.0, "v4 170 x 170 mm tile\nraised tactile regions", ha="center", va="center", color="white", fontsize=11, weight="bold")

    for x in (-75.0, 75.0):
        for y in (-75.0, 75.0):
            axis.add_patch(Circle((x, y), M6_CLEARANCE_MM / 2.0, facecolor="white", edgecolor="#17354e", linewidth=1.0))
    for x in (-75.0, 75.0):
        axis.add_patch(Rectangle((x - 9.0, -86.0), 18.0, 13.5, facecolor="#e86d12", edgecolor="#733108", linewidth=1.0))
    axis.add_patch(Rectangle((-85.0, -92.0), 170.0, 7.0, facecolor="#e86d12", edgecolor="#733108", linewidth=1.0))
    for x in (-27.0, 27.0):
        axis.add_patch(Rectangle((x - 7.0, -187.0), 14.0, 96.0, facecolor="#e86d12", edgecolor="#733108", linewidth=1.0))
        axis.add_patch(Polygon(((x - 5.0, -96.0), (x + 5.0, -96.0), (x + 5.0, -182.0), (x - 5.0, -182.0)), closed=True, facecolor="#f3a340", edgecolor="#733108", linewidth=0.8))
    axis.add_patch(Rectangle((-32.0, -133.0), 64.0, 10.0, facecolor="#e86d12", edgecolor="#733108", linewidth=1.0))
    axis.add_patch(Rectangle((-8.0, -133.0), 16.0, 44.0, facecolor="#e86d12", edgecolor="#733108", linewidth=1.0))
    axis.add_patch(Rectangle((-26.0, -130.0), 52.0, 40.0, facecolor="#f5bc67", edgecolor="#733108", linewidth=1.0))
    axis.add_patch(Circle((0.0, -139.0), 30.0, facecolor="#e86d12", edgecolor="#733108", linewidth=1.2))
    axis.add_patch(Circle((0.0, -139.0), 25.4, facecolor="white", edgecolor="#733108", linewidth=1.0))
    axis.add_patch(Circle((0.0, -139.0), 22.0, facecolor="#bde4ee", edgecolor="#733108", linewidth=1.0))

    axis.annotate("150 mm", xy=(-75.0, -102.0), xytext=(75.0, -102.0), ha="center", va="bottom", fontsize=11, weight="bold", arrowprops={"arrowstyle": "<->", "color": "#1a1a1a"})
    axis.plot((-75.0, -75.0), (-92.0, -100.0), color="#1a1a1a", linewidth=0.8)
    axis.plot((75.0, 75.0), (-92.0, -100.0), color="#1a1a1a", linewidth=0.8)
    axis.text(0.0, -205.0, "Orange: lightweight dock | White circles: M6 holes | Cyan: TacTip opening", ha="center", va="center", fontsize=9)
    axis.add_patch(FancyArrowPatch((0.0, 92.0), (0.0, 120.0), arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color="#17354e"))
    axis.text(4.0, 111.0, "tile +Y / tactile regions", color="#17354e", va="center", fontsize=10, weight="bold")
    axis.text(0.0, -139.0, "TacTip\nguide", ha="center", va="center", fontsize=9, weight="bold")
    axis.text(0.0, -110.0, "reference\npad", ha="center", va="center", fontsize=8, weight="bold")
    axis.set_aspect("equal")
    axis.set_xlim(-105.0, 105.0)
    axis.set_ylim(-220.0, 125.0)
    axis.set_xlabel("tile local X (mm)")
    axis.set_ylabel("tile local Y (mm)")
    axis.set_title("v4 150 mm-pitch board and lightweight TacTip dock installation")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_readme(path: Path, info: dict[str, Any]) -> None:
    hole_pitch = info["board_interface"]["shared_board_hole_pitch_mm"]
    path.write_text(
        "# Lightweight TacTip calibration dock for v4 150 mm board\n\n"
        "This one-piece dock is for the non-destructive v4 mounting-hole variant. It shares the two south-edge M6 "
        "tile holes at local `(-75,-75)` and `(75,-75)` mm, so the mounting pitch is exactly `{} mm`. "
        "The board must remain bolted flat through all four of its own M6 holes; the dock's two feet rest on that same table.\n\n"
        "## Installation\n\n"
        "1. Print the dock with its two feet down.\n"
        "2. Bolt a v4 170 mm tile to the table. Keep the dock at tile-local `-Y`.\n"
        "3. Loosen the south pair of board screws, slide the dock U-slots in from local `-Y`, and retighten the two M6 screws.\n"
        "4. Seat the rigid TacTip flange in the 50.8 mm guide. The compliant tip remains clear through the 44 mm opening.\n"
        "5. Record Tool(2) while seated. This is the repeatable rest datum for automatic collection.\n\n"
        "The `*_design.json` file records the complete local geometry, including the nominal seated Tool(2) TCP and reference-pad height.\n".format(hole_pitch),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifest = load_and_validate_manifest(args.board_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = args.output_dir / "v4_150mm_tactip_calibration_dock_lightweight.stl"
    design_path = args.output_dir / "v4_150mm_tactip_calibration_dock_lightweight_design.json"
    preview_path = args.output_dir / "v4_150mm_tactip_calibration_dock_lightweight_preview.png"
    layout_path = args.output_dir / "v4_150mm_tactip_calibration_dock_installation_layout.png"
    readme_path = args.output_dir / "README.md"
    existing = [path for path in (stl_path, design_path, preview_path, layout_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("{} already exists; use --overwrite to replace generated dock files".format(existing[0]))
    dock, info = build_dock(args, manifest)
    dock.export(stl_path)
    verify = trimesh.load_mesh(stl_path, force="mesh", process=True)
    if not isinstance(verify, trimesh.Trimesh) or not verify.is_watertight:
        raise RuntimeError("Exported dock did not remain watertight")
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
    if not args.no_preview:
        render_preview(preview_path, args.tile_stl, dock, info)
        info["preview"] = str(preview_path.resolve())
        render_installation_layout(layout_path)
        info["installation_layout"] = str(layout_path.resolve())
    design_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_readme(readme_path, info)
    print("Wrote {}".format(stl_path))
    print("Wrote {}".format(design_path))
    if not args.no_preview:
        print("Wrote {}".format(preview_path))
        print("Wrote {}".format(layout_path))
    print("Volume: {:.1f} mm^3; extent: {} mm".format(abs(verify.volume), [round(float(value), 2) for value in verify.extents]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
