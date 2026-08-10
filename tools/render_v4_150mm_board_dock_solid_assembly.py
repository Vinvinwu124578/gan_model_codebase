#!/usr/bin/env python3
"""Render the original v4 150 mm tile and lightweight v1 TacTip dock as solids.

This is a visualization-only tool.  It reads the existing STL files without
modifying their geometry or mounting coordinates.  The dense height-field tile
is reduced to a coherent top envelope solely for rendering, while the board
base and the original dock remain opaque solid meshes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD_DIR = ROOT / "outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DEFAULT_DOCK_DIR = DEFAULT_BOARD_DIR / "tactip_calibration_dock_lightweight_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile-stl",
        type=Path,
        default=DEFAULT_BOARD_DIR / "tactile_gan_coverage_board_340mm_tile_nw.stl",
    )
    parser.add_argument(
        "--dock-stl",
        type=Path,
        default=DEFAULT_DOCK_DIR / "v4_150mm_tactip_calibration_dock_lightweight.stl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DOCK_DIR / "solid_assembly_preview",
    )
    parser.add_argument(
        "--surface-pitch-mm",
        type=float,
        default=1.25,
        help="Rendering-grid pitch; it does not alter either STL.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.tile_stl = args.tile_stl.expanduser().resolve()
    args.dock_stl = args.dock_stl.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.tile_stl.is_file():
        parser.error("--tile-stl does not exist: {}".format(args.tile_stl))
    if not args.dock_stl.is_file():
        parser.error("--dock-stl does not exist: {}".format(args.dock_stl))
    if not 0.5 <= float(args.surface_pitch_mm) <= 3.0:
        parser.error("--surface-pitch-mm must be in [0.5, 3.0]")
    return args


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Could not load {} as a mesh".format(path))
    return mesh


def board_top_envelope(tile: trimesh.Trimesh, pitch_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the dense STL top envelope to a continuous render grid."""
    lower, upper = tile.bounds
    x_values = np.arange(lower[0], upper[0] + pitch_mm * 0.5, pitch_mm)
    y_values = np.arange(lower[1], upper[1] + pitch_mm * 0.5, pitch_mm)
    y_values[-1] = upper[1]
    x_values[-1] = upper[0]
    z_grid = np.full((len(y_values), len(x_values)), float(lower[2]), dtype=float)

    vertices = np.asarray(tile.vertices, dtype=float)
    x_index = np.clip(np.rint((vertices[:, 0] - lower[0]) / pitch_mm).astype(int), 0, len(x_values) - 1)
    y_index = np.clip(np.rint((vertices[:, 1] - lower[1]) / pitch_mm).astype(int), 0, len(y_values) - 1)
    np.maximum.at(z_grid, (y_index, x_index), vertices[:, 2])

    # The inside of a through-hole has no top vertex.  Preserve the solid
    # surrounding surface for a readable opaque rendering; the known M6 holes
    # are drawn separately as dark disks in the top view.
    board_nominal_top = 6.0
    z_grid[z_grid < board_nominal_top] = board_nominal_top
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, z_grid


def mesh_triangles(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.triangles, dtype=float)


def board_base_triangles(bounds: np.ndarray, top_z_mm: float = 6.0) -> np.ndarray:
    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    base = trimesh.creation.box(
        extents=(upper[0] - lower[0], upper[1] - lower[1], top_z_mm - lower[2]),
        transform=trimesh.transformations.translation_matrix(
            ((lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, (lower[2] + top_z_mm) / 2.0)
        ),
    )
    return mesh_triangles(base)


def top_surface_triangles(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    rows, columns = z_grid.shape
    triangles: list[np.ndarray] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            p00 = np.asarray((x_grid[row, column], y_grid[row, column], z_grid[row, column]))
            p10 = np.asarray((x_grid[row, column + 1], y_grid[row, column + 1], z_grid[row, column + 1]))
            p01 = np.asarray((x_grid[row + 1, column], y_grid[row + 1, column], z_grid[row + 1, column]))
            p11 = np.asarray((x_grid[row + 1, column + 1], y_grid[row + 1, column + 1], z_grid[row + 1, column + 1]))
            triangles.extend((np.asarray((p00, p10, p11)), np.asarray((p00, p11, p01))))
    return np.asarray(triangles, dtype=float)


def add_poly(axis: object, triangles: np.ndarray, colour: tuple[float, float, float, float]) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    collection = Poly3DCollection(triangles, facecolor=colour, edgecolor="none", linewidth=0.0)
    axis.add_collection3d(collection)


def draw_holes(axis: object, z_mm: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    for x_mm in (-75.0, 75.0):
        for y_mm in (-75.0, 75.0):
            axis.plot(
                x_mm + 3.0 * np.cos(theta),
                y_mm + 3.0 * np.sin(theta),
                np.full_like(theta, z_mm + 0.05),
                color="#0d2435",
                linewidth=1.4,
            )


def render_png(path: Path, tile: trimesh.Trimesh, dock: trimesh.Trimesh, pitch_mm: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_grid, y_grid, z_grid = board_top_envelope(tile, pitch_mm)
    top = top_surface_triangles(x_grid, y_grid, z_grid)
    base = board_base_triangles(tile.bounds)
    dock_triangles = mesh_triangles(dock)
    all_vertices = np.vstack((tile.vertices, dock.vertices))
    lower, upper = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
    midpoint = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) * 0.52, 1.0)

    figure = plt.figure(figsize=(15.5, 5.4), dpi=190)
    views = (
        (31.0, -56.0, "Assembled physical view"),
        (78.0, -90.0, "Top view: shared 150 mm M6 pair"),
        (17.0, 17.0, "Dock load path and solid board"),
    )
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        add_poly(axis, base, (0.06, 0.27, 0.47, 1.0))
        add_poly(axis, top, (0.12, 0.43, 0.67, 1.0))
        add_poly(axis, dock_triangles, (0.90, 0.30, 0.04, 1.0))
        draw_holes(axis, 6.05)
        axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
        axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
        axis.set_zlim(-2.0, max(upper[2] + 12.0, 30.0))
        axis.set_box_aspect((1.15, 1.15, 0.42))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(title, fontsize=11, weight="bold", pad=8)
    figure.suptitle(
        "Original v4 150 mm tile + lightweight v1 TacTip dock (opaque solid rendering)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_html(path: Path, tile: trimesh.Trimesh, dock: trimesh.Trimesh, pitch_mm: float) -> None:
    import plotly.graph_objects as go

    x_grid, y_grid, z_grid = board_top_envelope(tile, pitch_mm)
    lower, upper = tile.bounds
    board_base = trimesh.creation.box(
        extents=(upper[0] - lower[0], upper[1] - lower[1], 6.0 - lower[2]),
        transform=trimesh.transformations.translation_matrix(
            ((lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, (lower[2] + 6.0) / 2.0)
        ),
    )

    def mesh_trace(mesh: trimesh.Trimesh, name: str, colour: str) -> go.Mesh3d:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=colour,
            flatshading=True,
            opacity=1.0,
            name=name,
            hoverinfo="skip",
        )

    figure = go.Figure(
        data=[
            mesh_trace(board_base, "v4 tile base", "#164a70"),
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0.0, "#2c77a6"], [1.0, "#2c77a6"]],
                showscale=False,
                opacity=1.0,
                name="v4 tactile top surface",
                hovertemplate="tile X %{x:.1f} mm<br>tile Y %{y:.1f} mm<br>Z %{z:.1f} mm<extra>v4 board</extra>",
            ),
            mesh_trace(dock, "original lightweight v1 dock", "#e85d04"),
        ]
    )
    figure.update_layout(
        title="Original v4 150 mm tile + lightweight v1 TacTip dock",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        scene={
            "xaxis": {"title": "tile local X (mm)", "backgroundcolor": "#f7fafc"},
            "yaxis": {"title": "tile local Y (mm)", "backgroundcolor": "#f7fafc"},
            "zaxis": {"title": "Z (mm)", "backgroundcolor": "#f7fafc"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.35, "y": -1.6, "z": 0.85}},
        },
        legend={"orientation": "h", "y": 1.05},
        margin={"l": 0, "r": 0, "t": 62, "b": 0},
    )
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def main() -> int:
    args = parse_args()
    png_path = args.output_dir / "v4_150mm_tile_lightweight_v1_dock_solid_assembly.png"
    html_path = args.output_dir / "v4_150mm_tile_lightweight_v1_dock_solid_assembly.html"
    if (png_path.exists() or html_path.exists()) and not args.overwrite:
        raise FileExistsError("Output already exists. Use --overwrite to replace the preview.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile = load_mesh(args.tile_stl)
    dock = load_mesh(args.dock_stl)
    render_png(png_path, tile, dock, float(args.surface_pitch_mm))
    render_html(html_path, tile, dock, float(args.surface_pitch_mm))
    print("Wrote {}".format(png_path))
    print("Wrote {}".format(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
