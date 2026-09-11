"""Read-only CAD inspection. Outputs do not authorize or define robot motion."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "v4_150mm_tactip_calibration_dock_camera_style_rest_stop.stl"
mesh = trimesh.load_mesh(SOURCE)
triangles = mesh.triangles
upward = np.flatnonzero(mesh.face_normals[:, 2] > .99999)
surfaces = []
for elevation in sorted(set(np.round(triangles[upward, 0, 2], 5))):
    faces = upward[np.abs(triangles[upward, 0, 2] - elevation) < 1e-4]
    adjacency = mesh.face_adjacency[np.isin(mesh.face_adjacency, faces).all(axis=1)]
    for component in trimesh.graph.connected_components(adjacency, nodes=faces):
        vertices = triangles[component].reshape(-1, 3)
        surfaces.append({
            "z_numeric": float(elevation), "face_count": len(component),
            "area_numeric_squared": float(mesh.area_faces[component].sum()),
            "bounds_numeric": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        })


def vertical_intersections(x, y):
    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    v0 = triangles[:, 1, :2] - triangles[:, 0, :2]
    v1 = triangles[:, 2, :2] - triangles[:, 0, :2]
    v2 = np.array([x, y]) - triangles[:, 0, :2]
    determinant = cross(v0, v1)
    good = np.abs(determinant) > 1e-12
    a, b = np.zeros(len(triangles)), np.zeros(len(triangles))
    a[good] = cross(v2[good], v1[good]) / determinant[good]
    b[good] = cross(v0[good], v2[good]) / determinant[good]
    inside = good & (a >= -1e-8) & (b >= -1e-8) & ((a + b) <= 1 + 1e-8)
    z = triangles[:, 0, 2] + a * (triangles[:, 1, 2] - triangles[:, 0, 2]) + b * (triangles[:, 2, 2] - triangles[:, 0, 2])
    values = sorted(set((round(float(z[i]), 6), round(float(mesh.face_normals[i, 2]), 6)) for i in np.flatnonzero(inside)))
    return [{"z_numeric": elevation, "normal_z": normal} for elevation, normal in values]


# Fit the many outer-rim vertices independently of the designated probe point.
# The rim bounding extrema provide an initial circle; polygon vertices near its
# 30-unit outer radius are then fit by least squares.
rim = mesh.vertices[np.abs(mesh.vertices[:, 2] - 62) < 1e-5, :2]
outer = rim[np.abs(np.linalg.norm(rim - [0, -139], axis=1) - 30) < .001]
circle_coefficients = np.linalg.lstsq(np.column_stack([2 * outer[:, 0], 2 * outer[:, 1], np.ones(len(outer))]),
                                      np.sum(outer ** 2, axis=1), rcond=None)[0]
circle_centre = circle_coefficients[:2]
circle_radius = float(np.sqrt(circle_coefficients[2] + np.dot(circle_centre, circle_centre)))
circle_residual = float(np.max(np.abs(np.linalg.norm(outer - circle_centre, axis=1) - circle_radius)))
probes = [{"xy_numeric": list(xy), "intersections": vertical_intersections(*xy)} for xy in
          [(0, -139), (0, -140), (0, -137), (0, -142), (10, -139), (20, -139), (14, -131)]]
report = {
    "schema": "dock_stl_geometry_inspection.v1", "purpose": "geometry inspection only; not a fixture or motion configuration",
    "source_file": SOURCE.relative_to(ROOT.parent).as_posix(), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "units": "STL is unitless; millimetres are assumed from the repository design convention and dimensions",
    "face_count": len(mesh.faces), "vertex_count": len(mesh.vertices), "watertight": bool(mesh.is_watertight),
    "connected_solid_count": len(mesh.split(only_watertight=False)), "bounds_numeric": mesh.bounds.tolist(),
    "upward_horizontal_connected_surfaces": surfaces,
    "outer_rim_circle_fit": {"z_numeric": 62.0, "vertex_count": len(outer), "centre_xy_numeric": circle_centre.tolist(),
                             "radius_numeric": circle_radius, "max_radial_residual_numeric": circle_residual},
    "vertical_probes": probes,
    "conclusions": {
        "at_assumed_ring_centre_xy": [0.0, -139.0],
        "top_support_z_numeric": 27.0,
        "broad_underlying_platform_z_numeric": 25.5,
        "relief_above_platform_numeric": 1.5,
        "difference_from_legacy_nominal_tip_z_12": 15.0,
        "support_interpretation": "The highest material intersection at X=0,Y=-139 is upward-facing Z=27, so 27 is the CAD support peak there. Z=25.5 is the surrounding platform, not that peak.",
        "legacy_comparison": "12 is the old design's nominal seated tip coordinate, not a measured physical tip location. 27-12=15 is a CAD-coordinate difference.",
        "ring_faces": "Horizontal annular surfaces remain at Z=52 and Z=62; these surfaces alone do not identify the actual sensor seating or preload state.",
    },
    "not_determined_by_stl": [
        "Actual robot User/Tool frame, flange-to-TCP transform, taught dock TCP or board mounting pose.",
        "Printed-part dimensions, deformation, shrinkage, mounting tilt or measurement uncertainty.",
        "True TacTip apex-to-flange distance, skin preload, skin indentation or which rigid component seats against the ring.",
        "Actual contact centre under tilt, visual detection bias, camera thresholds or tactile image reference.",
        "Whether every board point is reachable, collision-free or accurate to 0.1 mm.",
        "Encoded units; STL itself does not define millimetres.",
    ],
}
(ROOT / "geometry_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(14, 10), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
ax_top = fig.add_subplot(grid[0, 0])
for elevation, colour, label in [(25.5, "#88b7cc", "Platform Z=25.5"), (27, "#e87f32", "Relief / support peak Z=27")]:
    selected = upward[np.abs(triangles[upward, 0, 2] - elevation) < 1e-5]
    ax_top.add_collection(PolyCollection(triangles[selected, :, :2], facecolors=colour, edgecolors="none", label=label))
ax_top.scatter([0], [-139], s=70, marker="x", linewidths=2, color="black", label="Ring centre (0, -139)")
ax_top.set(xlim=(-32, 32), ylim=(-151, -84), xlabel="X (assumed mm)", ylabel="Y (assumed mm)", title="Top view of raised support and contact relief")
ax_top.set_aspect("equal")
ax_top.legend(loc="upper right", fontsize=8)
ax_top.grid(alpha=.2)


def section(axis, normal, origin, indices, horizontal_label, title, xlim):
    lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    axis.add_collection(LineCollection(lines[:, :, indices], colors="#203c56", linewidths=1.4))
    axis.set(xlim=xlim, ylim=(-2, 69), xlabel=horizontal_label + " (assumed mm)", ylabel="Z (assumed mm)", title=title)
    axis.axhline(12, color="#969696", linestyle="--", linewidth=1, label="Legacy nominal tip Z=12")
    axis.axhline(25.5, color="#31809c", linestyle=":", linewidth=1, label="Platform Z=25.5")
    axis.axhline(27, color="#d66c28", linestyle="--", linewidth=1, label="Support peak Z=27")
    axis.grid(alpha=.2)
    return lines


ax_cross = fig.add_subplot(grid[0, 1])
section(ax_cross, [0, 1, 0], [0, -139, 0], [0, 2], "X", "Cross-section at Y=-139", (-39, 39))
ax_cross.scatter([0], [27], c="#d66c28", s=60)
ax_cross.annotate("CAD support: (0, -139, 27)", xy=(0, 27), xytext=(-35, 36), arrowprops={"arrowstyle": "->"}, fontsize=9)
ax_cross.legend(loc="upper left", fontsize=8)
ax_side = fig.add_subplot(grid[1, :])
section(ax_side, [1, 0, 0], [0, 0, 0], [1, 2], "Y", "Longitudinal section at X=0", (-190, -70))
ax_side.scatter([-139], [27], c="#d66c28", s=55)
ax_side.annotate("27 - 12 = 15 mm CAD difference", xy=(-139, 27), xytext=(-180, 39), arrowprops={"arrowstyle": "->"})
fig.suptitle("Actual supplied dock STL: geometric support inspection\nCAD geometry only; physical TCP, preload and print error remain unmeasured", fontsize=14)
fig.savefig(ROOT / "geometry_sections.png", dpi=180)
plt.close(fig)
print(json.dumps({"report": str(ROOT / "geometry_report.json"), "plot": str(ROOT / "geometry_sections.png"),
                  "centre_intersections": vertical_intersections(0, -139), "circle": report["outer_rim_circle_fit"]}, indent=2))
