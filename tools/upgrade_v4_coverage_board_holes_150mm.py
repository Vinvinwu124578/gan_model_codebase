#!/usr/bin/env python3
"""Create a v4 four-tile coverage board with a 150 mm M6 hole pitch.

The original v4 tiles are preserved untouched.  This tool fills their
156 mm-pitch corner holes and drills an exact 150 x 150 mm M6 pattern at
tile-local coordinates (+/-75, +/-75) mm.  Tactile geometry, sampling CSV and
assembly frame remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle, Rectangle


SOURCE_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm")
OUTPUT_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
# The requested nominal M6 hole diameter. This is intentionally not a
# clearance fit: threads are expected to form in the printed plastic.
M6_CLEARANCE_MM = 6.0
SOURCE_V4_HOLE_DIAMETER_MM = 6.6
TARGET_LOCAL_HOLES_MM = ((-75.0, -75.0), (-75.0, 75.0), (75.0, -75.0), (75.0, 75.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.source_dir.is_dir():
        parser.error("--source-dir does not exist: {}".format(args.source_dir))
    return args


def as_mesh(result: object, label: str) -> trimesh.Trimesh:
    if result is None:
        raise RuntimeError("{} returned no mesh".format(label))
    if isinstance(result, list):
        result = trimesh.util.concatenate(result)
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("{} returned {} rather than a mesh".format(label, type(result).__name__))
    result.remove_unreferenced_vertices()
    result.fix_normals()
    if not result.is_watertight:
        raise RuntimeError("{} produced a non-watertight mesh".format(label))
    return result


def cylinder(radius_mm: float, height_mm: float, z_centre_mm: float, xy: tuple[float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius_mm, height=height_mm, sections=96)
    mesh.apply_translation((float(xy[0]), float(xy[1]), float(z_centre_mm)))
    return mesh


def fill_and_redrill(
    source: trimesh.Trimesh,
    old_holes: Iterable[tuple[float, float]],
    new_holes: Iterable[tuple[float, float]],
) -> trimesh.Trimesh:
    # The existing holes sit in the low 6 mm outer perimeter.  These inserts
    # overlap the former hole wall by 0.15 mm and rise just 0.05 mm above its
    # surface, avoiding an under-side protrusion while making the CSG union
    # robust and leaving the tactile cells untouched.
    fills = [cylinder(SOURCE_V4_HOLE_DIAMETER_MM / 2.0 + 0.15, 6.05, 3.025, point) for point in old_holes]
    filled = as_mesh(trimesh.boolean.union([source, *fills], engine="manifold"), "fill old M6 holes")
    cutters = [cylinder(M6_CLEARANCE_MM / 2.0, 30.0, 7.0, point) for point in new_holes]
    return as_mesh(trimesh.boolean.difference([filled, *cutters], engine="manifold"), "drill 150 mm M6 holes")


def tile_holes_in_board_frame(tile: dict[str, Any]) -> list[list[float]]:
    centre = np.asarray(tile["center_board_xy_mm"], dtype=float)
    return [(centre + np.asarray(point, dtype=float)).tolist() for point in TARGET_LOCAL_HOLES_MM]


def render_layout(path: Path, manifest: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 8.2), dpi=180)
    colors = {"tile_nw": "#3e83bd", "tile_ne": "#377dad", "tile_sw": "#326f9d", "tile_se": "#2d668f"}
    for tile in manifest["tiles"]:
        centre_x, centre_y = (float(value) for value in tile["center_board_xy_mm"])
        size = float(manifest["tile_size_mm"][0])
        axis.add_patch(
            Rectangle((centre_x - size / 2.0, centre_y - size / 2.0), size, size, facecolor=colors[str(tile["tile_id"])], edgecolor="#152536", linewidth=1.5)
        )
        for x, y in tile["m6_hole_centres_board_xy_mm"]:
            axis.add_patch(Circle((float(x), float(y)), M6_CLEARANCE_MM / 2.0, facecolor="#f5d747", edgecolor="#101820", linewidth=0.7))
        axis.text(centre_x, centre_y, str(tile["tile_id"]), color="white", ha="center", va="center", fontsize=11, weight="bold")
    axis.set_aspect("equal")
    axis.set_xlim(-185, 185)
    axis.set_ylim(-185, 185)
    axis.set_xlabel("assembled board X (mm)")
    axis.set_ylabel("assembled board Y (mm)")
    axis.set_title("v4 four-tile board: 150 mm M6 pitch on every 170 mm tile")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def write_readme(path: Path) -> None:
    path.write_text(
        "# v4 70 mm Coverage Board: 150 mm M6 Pitch\n\n"
        "This is a non-destructive mounting-hole variant of the original v4 deep-contact board. "
        "Each of the four 170 x 170 mm tiles keeps the same tactile geometry, STL-local frame, "
        "URDF frame and sampling CSV as v4. Only its four M6 clearance holes changed.\n\n"
        "## Hole pattern\n\n"
        "Every tile uses local hole centres `(-75,-75)`, `(-75,75)`, `(75,-75)`, `(75,75)` mm. "
        "The horizontal and vertical centre-to-centre pitch is therefore exactly `150 mm`. "
        "The nominal M6 hole diameter is `6.0 mm`.\n\n"
        "The calibration dock for this variant attaches to one edge pair at local Y = -75 mm.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifest_path = next(iter(sorted(args.source_dir.glob("*_manifest.json"))), None)
    if manifest_path is None:
        raise FileNotFoundError("No v4 manifest found under {}".format(args.source_dir))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if list(source_manifest.get("tile_size_mm", ())) != [170.0, 170.0]:
        raise ValueError("This converter expects the 170 mm v4 tile set")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} already exists; use --overwrite".format(args.output_dir))
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = dict(source_manifest)
    manifest["schema"] = "tactile_gan_coverage_board.modular.v4.deepcontact.mountpitch150"
    manifest["m6_tile_hole_pitch_mm"] = 150.0
    manifest["m6_hole_diameter_mm"] = M6_CLEARANCE_MM
    manifest["mounting_variant"] = "Existing 156 mm tile holes filled and redrilled at local +/-75 mm; tactile geometry unchanged."
    converted_tiles: list[dict[str, Any]] = []
    for source_tile in source_manifest["tiles"]:
        tile = dict(source_tile)
        input_path = args.source_dir / str(tile["stl"])
        source_mesh = trimesh.load_mesh(input_path, force="mesh", process=True)
        if not isinstance(source_mesh, trimesh.Trimesh) or not source_mesh.is_watertight:
            raise RuntimeError("Could not load watertight input tile {}".format(input_path))
        centre = np.asarray(tile["center_board_xy_mm"], dtype=float)
        old_local = [tuple(np.asarray(point, dtype=float) - centre) for point in tile["m6_hole_centres_board_xy_mm"]]
        if not all(np.isclose(abs(value), 78.0, atol=0.15) for point in old_local for value in point):
            raise ValueError("Unexpected v4 source hole locations for {}: {}".format(tile["tile_id"], old_local))
        converted = fill_and_redrill(source_mesh, old_local, TARGET_LOCAL_HOLES_MM)
        output_path = args.output_dir / str(tile["stl"])
        converted.export(output_path)
        verify = trimesh.load_mesh(output_path, force="mesh", process=True)
        if not isinstance(verify, trimesh.Trimesh) or not verify.is_watertight:
            raise RuntimeError("Output tile did not remain watertight: {}".format(output_path))
        tile["m6_hole_centres_board_xy_mm"] = tile_holes_in_board_frame(tile)
        tile["m6_hole_centres_tile_local_xy_mm"] = [list(point) for point in TARGET_LOCAL_HOLES_MM]
        tile["m6_hole_pitch_mm"] = [150.0, 150.0]
        tile["face_count"] = int(len(verify.faces))
        tile["watertight"] = True
        converted_tiles.append(tile)
        print("Converted {} -> {} faces".format(tile["tile_id"], len(verify.faces)), flush=True)
    manifest["tiles"] = converted_tiles
    manifest["urdf"] = str(source_manifest["urdf"])
    manifest["sampling_sites"] = str(source_manifest["sampling_sites"])
    manifest["preview"] = "tactile_gan_coverage_board_340mm_mountpitch150_preview.png"
    (args.output_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.source_dir / str(source_manifest["urdf"]), args.output_dir / str(source_manifest["urdf"]))
    shutil.copy2(args.source_dir / str(source_manifest["sampling_sites"]), args.output_dir / str(source_manifest["sampling_sites"]))
    render_layout(args.output_dir / str(manifest["preview"]), manifest)
    write_readme(args.output_dir / "README.md")
    print("Output: {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
