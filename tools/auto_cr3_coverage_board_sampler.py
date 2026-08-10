#!/usr/bin/env python3
"""Plan and safely collect varied TacTip data from one coverage-board tile.

This script is the real-robot counterpart of the four-tile GAN coverage
board.  It uses the printed calibration dock made by
``design_coverage_board_tactip_calibration_dock.py`` instead of fitting
arbitrary mesh points for every board placement:

1. Seat Tool(2) TacTip in the dock and store one TCP with
   ``--teach-dock-from-current``.
2. On collection, verify that the robot is back at that seated TCP.
3. Leave the dock vertically, perform an automatic visual-contact check on
   the dock's flat reference land, and apply its small rigid translation
   correction to the tile plan.
4. Collect varying seed locations, post-contact depths, and tilt angles with
   fresh tactile baselines and two-frame visual-contact confirmation.

No CR3 movement is sent unless both ``--execute`` and
``--yes-i-confirm-cr3-is-safe`` are supplied.  A no-contact event returns by
the verified high route and stops the batch by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from auto_cr3_gelsight_pair_sampler import GelSightCapture
from auto_cr3_visual_contact_search import (
    ImageFeature,
    capture_record,
    finite_pose,
    list_pose,
    move_and_verify,
    position_error_mm,
    rotation_error_deg,
    robust_threshold,
)
from design_tactile_gan_coverage_board_modular import surface_height_for_profile
from live_cr3_gelsight_sampler import DobotCR3LiveClient, parse_camera_source
from tactip_runtime_preprocess import add_tactip_preprocess_args, create_tactip_preprocessor


DEFAULT_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v5_highprotrusion_deepcontact_70mm_mountgrid50")
DEFAULT_DOCK_DIR = DEFAULT_BOARD_DIR / "coverage_board_tactip_calibration_dock_v1"
DEFAULT_DOCK_DESIGN = DEFAULT_DOCK_DIR / "coverage_board_tactip_calibration_dock_design.json"
DEFAULT_RUN_ROOT = DEFAULT_BOARD_DIR / "cr3_coverage_board_runs"
VALID_TILES = ("tile_nw", "tile_ne", "tile_sw", "tile_se")
SUPPORTED_DOCK_SCHEMAS = {
    "coverage_board_tactip_calibration_dock.v1",
    "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v1",
    "tactile_gan_coverage_board.v4.verified_pitch_dock.v2",
}
DEFAULT_POST_CONTACT_DEPTH_MIN_MM = 1.0
DEFAULT_POST_CONTACT_DEPTH_MAX_MM = 10.0
# The visual contact detector can identify first contact up to one 0.5 mm
# search step after the nominal surface.  Leave 1 mm of headroom above the
# requested 10 mm capture range for that detection uncertainty.
DEFAULT_MAX_EXTRA_BELOW_PLANNED_CONTACT_MM = 11.0


@dataclass(frozen=True)
class FixtureTransform:
    """Rigid mapping from tile-local coordinates to CR3 base coordinates."""

    dock_tcp: tuple[float, float, float, float, float, float]
    dock_tcp_local_mm: tuple[float, float, float]
    tile_to_base: np.ndarray
    dock_rotation: np.ndarray
    board_yaw_deg: float

    def position(self, local_xyz_mm: Sequence[float]) -> np.ndarray:
        delta = np.asarray(local_xyz_mm, dtype=float) - np.asarray(self.dock_tcp_local_mm, dtype=float)
        return np.asarray(self.dock_tcp[:3], dtype=float) + self.tile_to_base @ delta

    def orientation(self, tilt_x_deg: float, tilt_y_deg: float) -> np.ndarray:
        axis_x = self.tile_to_base[:, 0]
        axis_y = self.tile_to_base[:, 1]
        rotate_x = Rotation.from_rotvec(axis_x * math.radians(float(tilt_x_deg))).as_matrix()
        rotate_y = Rotation.from_rotvec(axis_y * math.radians(float(tilt_y_deg))).as_matrix()
        return rotate_y @ rotate_x @ self.dock_rotation

    def pose(self, local_xyz_mm: Sequence[float], tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0) -> tuple[float, float, float, float, float, float]:
        position = self.position(local_xyz_mm)
        raw_angles = Rotation.from_matrix(self.orientation(tilt_x_deg, tilt_y_deg)).as_euler("XYZ", degrees=True)
        # Keep the numerical Euler representation near the seated datum.  For
        # example, +178 degrees and -182 degrees encode the same orientation,
        # but the latter is the small physical change from a -179 degree
        # resting roll and is less ambiguous to inspect in a CR3 plan.
        reference = np.asarray(self.dock_tcp[3:], dtype=float)
        angles = np.asarray(
            [angle + 360.0 * round((ref - angle) / 360.0) for angle, ref in zip(raw_angles, reference)],
            dtype=float,
        )
        return finite_pose(tuple(position.tolist()) + tuple(angles.tolist()), "generated fixture pose")

    def press_axis(self, tilt_x_deg: float, tilt_y_deg: float) -> np.ndarray:
        axis = self.orientation(tilt_x_deg, tilt_y_deg) @ np.asarray((0.0, 0.0, 1.0), dtype=float)
        return axis / np.linalg.norm(axis)

    def local_vector(self, base_vector_mm: Sequence[float]) -> np.ndarray:
        return self.tile_to_base.T @ np.asarray(base_vector_mm, dtype=float)


@dataclass(frozen=True)
class BoardSample:
    index: int
    sample_id: str
    tile_id: str
    source_seed_site_id: str
    region_id: str
    category: str
    stimulus: str
    replicate: int
    local_contact_mm: tuple[float, float, float]
    expected_surface_z_mm: float
    post_contact_depth_mm: float
    tilt_x_deg: float
    tilt_y_deg: float
    jitter_x_mm: float
    jitter_y_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument("--tile", choices=VALID_TILES, required=True, help="One physical board tile currently mounted with the dock.")
    parser.add_argument("--dock-design", type=Path, default=DEFAULT_DOCK_DESIGN)
    parser.add_argument("--fixture-profile", type=Path, help="Saved seated Tool(2) TCP. Defaults to the per-tile profile in the dock directory.")
    parser.add_argument("--teach-dock-from-current", action="store_true", help="Read the currently seated Tool(2) TCP and write a fixture profile. This mode does not move CR3.")
    parser.add_argument("--output-dir", type=Path, help="Collection/planning directory. A timestamped tile directory is used by default.")
    parser.add_argument("--profile", choices=("quick", "standard", "dense"), default="standard")
    parser.add_argument(
        "--samples-per-tile",
        type=int,
        help=(
            "Exact dense dataset size after optional --region/--site filters and "
            "controller IK filtering. For the four-region tile, "
            "--samples-per-tile 2000 retains 500 safe, jittered, IK-reachable "
            "contacts per region."
        ),
    )
    parser.add_argument("--region", action="append", dest="regions", help="Optional region filter, e.g. --region R03. Repeatable.")
    parser.add_argument("--site", action="append", dest="sites", help="Optional CSV seed-site filter, e.g. --site R03_S05. Repeatable.")
    parser.add_argument(
        "--dense-spatial-layout",
        choices=("region_grid", "seed_jitter"),
        default="region_grid",
        help=(
            "Dense XY layout. region_grid (default) spreads contacts across each "
            "feature's usable safe window; seed_jitter preserves the former tight "
            "jitter around CSV seed points."
        ),
    )
    parser.add_argument(
        "--dense-region-anchor-count",
        type=int,
        default=25,
        help="Number of distinct XY anchors per region for region_grid; use a square number. Default: 25 (5x5).",
    )
    parser.add_argument(
        "--dense-region-half-span-mm",
        type=float,
        help=(
            "Optional half-span of the region-grid centre positions. By default the "
            "sampler derives the largest safe value from the 60 mm feature zone and "
            "40 mm TacTip diameter (10 mm)."
        ),
    )
    parser.add_argument("--max-samples", type=int, help="Optional cap after filtering and profile expansion.")
    parser.add_argument(
        "--first-contact-test",
        action="store_true",
        help=(
            "Require one selected site and replace its normal recipe with the "
            "configured minimum post-contact depth (1.0 mm by default) at zero tilt. Intended for a first "
            "real contact check after a new fixture setup."
        ),
    )
    parser.add_argument(
        "--report-max-samples",
        type=int,
        default=120,
        help="Maximum image cards in collection_report.html; the CSV/JSON always contain every sample.",
    )
    parser.add_argument("--seed", type=int, default=1729, help="Recorded plan seed; the base plan itself is deterministic.")
    parser.add_argument("--board-yaw-deg", type=float, default=0.0, help="One-time fine yaw correction from the keyed dock frame. Keep 0 for the printed fixture orientation.")
    parser.add_argument("--robot-ip", default="192.168.31.88")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--move-port", type=int, default=30003)
    parser.add_argument("--robot-timeout-sec", type=float, default=6.0)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--tool", type=int, default=2, help="CR3 TacTip Tool frame; the fixture was designed for Tool 2.")
    parser.add_argument("--speed", type=float, default=3.0, help="CR3 speed percent; limited to 5 for visual-contact collection.")
    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-read-timeout-sec", type=float, default=10.0)
    parser.add_argument("--capture-min-gray-p95", type=float, default=20.0)
    parser.add_argument("--capture-retry-count", type=int, default=8)
    parser.add_argument("--capture-retry-sec", type=float, default=0.05)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--step-mm", type=float, default=0.5, help="Visual-contact downward increment. Maximum 0.5 mm.")
    parser.add_argument(
        "--min-post-contact-depth-mm",
        type=float,
        default=DEFAULT_POST_CONTACT_DEPTH_MIN_MM,
        help="Minimum final indentation after visual first contact. Default: 1.0 mm.",
    )
    parser.add_argument(
        "--max-post-contact-depth-mm",
        type=float,
        default=DEFAULT_POST_CONTACT_DEPTH_MAX_MM,
        help="Maximum final indentation after visual first contact. Default: 10.0 mm.",
    )
    parser.add_argument(
        "--allow-depth-above-csv-limit",
        action="store_true",
        help=(
            "Permit --execute when the requested 1-10 mm distribution exceeds a "
            "site's CSV recommended maximum. Use only after physically validating "
            "the mounted board and TacTip at those depths."
        ),
    )
    parser.add_argument("--approach-clearance-mm", type=float, default=25.0, help="No-contact clearance above each nominal local surface.")
    parser.add_argument("--safe-height-mm", type=float, default=100.0, help="Fixture-local safe height over the tile bottom plane.")
    parser.add_argument("--dock-exit-lift-mm", type=float, default=65.0, help="Fixture-local lift from seated apex before any lateral travel.")
    parser.add_argument(
        "--max-extra-below-planned-contact-mm",
        type=float,
        default=DEFAULT_MAX_EXTRA_BELOW_PLANNED_CONTACT_MM,
        help="Hard visual-search allowance past a nominal board surface. Default: 11 mm for 10 mm final indentation.",
    )
    parser.add_argument("--baseline-frames", type=int, default=9)
    parser.add_argument("--noise-probes", type=int, default=3)
    parser.add_argument("--probe-frames", type=int, default=3)
    parser.add_argument("--capture-frames", type=int, default=9)
    parser.add_argument(
        "--save-search-frames",
        action="store_true",
        help=(
            "Keep baseline/noise/search diagnostic images. By default they are "
            "deleted after contact analysis so a 2000-sample run retains only "
            "final tactile captures."
        ),
    )
    parser.add_argument("--consecutive-hits", type=int, default=2)
    parser.add_argument("--noise-multiplier", type=float, default=3.0)
    # ``search_visual_contact`` now uses local LK marker displacement in
    # pixels after whole-image motion removal, rather than intensity change.
    parser.add_argument("--min-contact-mean", type=float, default=0.10)
    parser.add_argument("--min-contact-p95", type=float, default=0.45)
    parser.add_argument("--dock-position-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--dock-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--motion-position-tolerance-mm", type=float, default=0.75)
    parser.add_argument("--motion-rotation-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--reference-max-lateral-correction-mm", type=float, default=3.0)
    parser.add_argument("--reference-max-vertical-correction-mm", type=float, default=4.0)
    parser.add_argument(
        "--disable-ik-preflight",
        action="store_true",
        help="Disable the controller-side no-motion inverse-kinematics check before every MovL.",
    )
    parser.add_argument(
        "--ik-candidate-multiplier",
        type=float,
        default=2.0,
        help=(
            "For an exact dense run, prepare this many deterministic candidate "
            "sites before controller IK filtering. The sampler retains the requested "
            "count of reachable routes and uses later candidates as replacements."
        ),
    )
    parser.add_argument("--skip-reference-pad-check", action="store_true", help="Skip the automatic visual-contact check on the dock's flat reference land.")
    parser.add_argument("--continue-on-no-contact", action="store_true", help="Return high and continue after a site with no stable visual contact. Default stops safely.")
    parser.add_argument("--return-to-dock", action="store_true", help="After a completed batch, re-seat TacTip in the dock at the verified starting TCP.")
    parser.add_argument(
        "--ik-preflight-only",
        action="store_true",
        help=(
            "Ask the CR3 controller to solve the complete route without sending any "
            "motion command. Dense runs produce an IK-filtered replacement plan; use "
            "this after seating TacTip in the dock and before a first real collection."
        ),
    )
    parser.add_argument(
        "--reseat-dock-only",
        action="store_true",
        help=(
            "Return only from the recorded dock-exit pose to the saved seated "
            "TacTip dock TCP. The current TCP must already match either the dock "
            "or its vertically lifted exit pose."
        ),
    )
    parser.add_argument(
        "--recover-reference-to-dock",
        action="store_true",
        help=(
            "Recover from an interrupted dock-reference visual search. The current "
            "TCP must be on the known reference approach line; CR3 then retracts via "
            "reference-high, dock-high, dock-exit, and the seated dock TCP."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Send CR3 motions after all preflight checks pass.")
    parser.add_argument("--yes-i-confirm-cr3-is-safe", action="store_true")
    add_tactip_preprocess_args(parser)
    args = parser.parse_args()
    args.board_dir = args.board_dir.expanduser().resolve()
    args.dock_design = args.dock_design.expanduser().resolve()
    if args.fixture_profile is None:
        args.fixture_profile = args.dock_design.parent / "profiles" / "{}_fixture_profile.json".format(args.tile)
    args.fixture_profile = args.fixture_profile.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if not args.board_dir.is_dir():
        parser.error("--board-dir does not exist: {}".format(args.board_dir))
    if not args.dock_design.is_file():
        parser.error("--dock-design does not exist: {}".format(args.dock_design))
    if args.teach_dock_from_current and (args.execute or args.ik_preflight_only or args.reseat_dock_only or args.recover_reference_to_dock):
        parser.error("--teach-dock-from-current only records a datum; run preflight or collection separately")
    if args.ik_preflight_only and (args.execute or args.reseat_dock_only or args.recover_reference_to_dock):
        parser.error("--ik-preflight-only cannot be combined with a motion mode")
    if args.reseat_dock_only and not args.execute:
        parser.error("--reseat-dock-only requires --execute and --yes-i-confirm-cr3-is-safe")
    if args.recover_reference_to_dock and not args.execute:
        parser.error("--recover-reference-to-dock requires --execute and --yes-i-confirm-cr3-is-safe")
    if args.reseat_dock_only and args.recover_reference_to_dock:
        parser.error("Use only one recovery mode at a time")
    if args.execute and not args.yes_i_confirm_cr3_is_safe:
        parser.error("--execute requires --yes-i-confirm-cr3-is-safe")
    if args.execute and bool(args.no_tactip_preprocess):
        parser.error("Formal visual-contact collection requires shared TacTip preprocessing; omit --no-tactip-preprocess")
    if args.max_samples is not None and int(args.max_samples) < 1:
        parser.error("--max-samples must be at least 1")
    if args.samples_per_tile is not None and int(args.samples_per_tile) < 1:
        parser.error("--samples-per-tile must be at least 1")
    if args.samples_per_tile is not None and args.max_samples is not None:
        parser.error("Use either --samples-per-tile or --max-samples, not both")
    grid_side = int(round(math.sqrt(int(args.dense_region_anchor_count))))
    if int(args.dense_region_anchor_count) < 4 or grid_side * grid_side != int(args.dense_region_anchor_count):
        parser.error("--dense-region-anchor-count must be a square integer of at least 4, such as 16, 25, 36, 49, or 64")
    if args.first_contact_test and (args.max_samples != 1 or args.samples_per_tile is not None):
        parser.error("--first-contact-test requires --max-samples 1 and no --samples-per-tile")
    if int(args.report_max_samples) < 0:
        parser.error("--report-max-samples must be non-negative")
    for name in (
        "robot_timeout_sec",
        "speed",
        "camera_read_timeout_sec",
        "settle_sec",
        "step_mm",
        "min_post_contact_depth_mm",
        "max_post_contact_depth_mm",
        "approach_clearance_mm",
        "safe_height_mm",
        "dock_exit_lift_mm",
        "max_extra_below_planned_contact_mm",
        "noise_multiplier",
        "min_contact_mean",
        "min_contact_p95",
        "dock_position_tolerance_mm",
        "dock_rotation_tolerance_deg",
        "motion_position_tolerance_mm",
        "motion_rotation_tolerance_deg",
        "reference_max_lateral_correction_mm",
        "reference_max_vertical_correction_mm",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if not 1.0 <= float(args.speed) <= 5.0:
        parser.error("--speed must be in [1, 5] for visual-contact collection")
    if not 0.0 < float(args.step_mm) <= 0.5:
        parser.error("--step-mm must be in (0, 0.5]")
    if float(args.min_post_contact_depth_mm) < DEFAULT_POST_CONTACT_DEPTH_MIN_MM:
        parser.error("--min-post-contact-depth-mm must be at least 1.0 mm")
    if float(args.max_post_contact_depth_mm) > DEFAULT_POST_CONTACT_DEPTH_MAX_MM:
        parser.error("--max-post-contact-depth-mm must be at most 10.0 mm")
    if float(args.max_post_contact_depth_mm) < float(args.min_post_contact_depth_mm):
        parser.error("--max-post-contact-depth-mm must be greater than or equal to --min-post-contact-depth-mm")
    if not 0.0 <= float(args.max_extra_below_planned_contact_mm) <= 12.0:
        parser.error("--max-extra-below-planned-contact-mm must be in [0, 12]")
    if float(args.max_extra_below_planned_contact_mm) + 1e-9 < float(args.max_post_contact_depth_mm):
        parser.error("--max-extra-below-planned-contact-mm must be at least --max-post-contact-depth-mm")
    if not 1.0 <= float(args.ik_candidate_multiplier) <= 5.0:
        parser.error("--ik-candidate-multiplier must be in [1, 5]")
    if args.dense_region_half_span_mm is not None and float(args.dense_region_half_span_mm) <= 0.0:
        parser.error("--dense-region-half-span-mm must be positive")
    if min(int(args.baseline_frames), int(args.probe_frames), int(args.capture_frames)) < 2 or int(args.noise_probes) < 2:
        parser.error("--baseline-frames, --probe-frames, --capture-frames >= 2 and --noise-probes >= 2 are required")
    if int(args.consecutive_hits) < 2:
        parser.error("--consecutive-hits must be at least 2")
    if not 0 <= int(args.user) <= 9 or not 0 <= int(args.tool) <= 9:
        parser.error("--user and --tool must be in [0, 9]")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("{} is not a JSON object".format(path))
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_board_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    manifest_paths = sorted(args.board_dir.glob("*_manifest.json"))
    if len(manifest_paths) != 1:
        raise FileNotFoundError("Expected exactly one board manifest under {}, found {}".format(args.board_dir, len(manifest_paths)))
    manifest = read_json(manifest_paths[0])
    tile = next((dict(item) for item in manifest.get("tiles", []) if str(item.get("tile_id")) == args.tile), None)
    if tile is None:
        raise ValueError("Board manifest has no {} definition".format(args.tile))
    csv_path = args.board_dir / str(manifest.get("sampling_sites", ""))
    if not csv_path.is_file():
        raise FileNotFoundError("Sampling-site CSV is missing: {}".format(csv_path))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Sampling-site CSV has no rows")
    return manifest, tile, rows


def load_dock_design(args: argparse.Namespace) -> dict[str, Any]:
    design = read_json(args.dock_design)
    schema = str(design.get("schema"))
    if schema not in SUPPORTED_DOCK_SCHEMAS:
        raise ValueError(
            "{} has unsupported calibration-dock schema {!r}; expected one of {}".format(
                args.dock_design,
                schema,
                ", ".join(sorted(SUPPORTED_DOCK_SCHEMAS)),
            )
        )
    reference = dict(design.get("tactip_reference", {}))
    local = reference.get("nominal_seated_tool_tcp_local_mm")
    if not isinstance(local, list) or len(local) != 3 or not np.isfinite(np.asarray(local, dtype=float)).all():
        raise ValueError("Dock design lacks a finite nominal seated Tool(2) TCP")
    if int(reference.get("tool", -1)) != int(args.tool):
        raise ValueError("Dock design is for Tool({}), but --tool is {}".format(reference.get("tool"), args.tool))
    return design


def fixture_from_profile(profile: dict[str, Any], dock_design: dict[str, Any], args: argparse.Namespace) -> FixtureTransform:
    if str(profile.get("schema")) != "coverage_board_tactip_fixture_profile.v1":
        raise ValueError("Fixture profile has an unsupported schema")
    if str(profile.get("tile_id")) != args.tile:
        raise ValueError("Fixture profile belongs to {}, not {}".format(profile.get("tile_id"), args.tile))
    if int(profile.get("tool", -1)) != int(args.tool):
        raise ValueError("Fixture profile was taught with Tool({}), not Tool({})".format(profile.get("tool"), args.tool))
    expected_design_hash = str(profile.get("dock_design_sha256", ""))
    actual_design_hash = sha256_file(args.dock_design)
    if expected_design_hash != actual_design_hash:
        raise ValueError("Fixture profile belongs to a different dock design; teach a fresh seated datum")
    dock_tcp = finite_pose(profile.get("dock_tcp", ()), "fixture dock TCP")
    local = tuple(float(value) for value in dict(dock_design["tactip_reference"])["nominal_seated_tool_tcp_local_mm"])
    dock_rotation = Rotation.from_euler("XYZ", dock_tcp[3:], degrees=True).as_matrix()
    # Tool +Z is the physical press-down axis.  The keyed fixture convention
    # makes tile +X = Tool +X, tile +Y = Tool -Y, and tile +Z = Tool -Z.
    adaptor = np.diag((1.0, -1.0, -1.0))
    yaw = Rotation.from_euler("Z", float(profile.get("board_yaw_deg", 0.0)), degrees=True).as_matrix()
    tile_to_base = dock_rotation @ adaptor @ yaw
    if np.linalg.det(tile_to_base) < 0.99:
        raise RuntimeError("Fixture axis construction is not a proper rotation")
    return FixtureTransform(
        dock_tcp=dock_tcp,
        dock_tcp_local_mm=local,
        tile_to_base=tile_to_base,
        dock_rotation=dock_rotation,
        board_yaw_deg=float(profile.get("board_yaw_deg", 0.0)),
    )


def teach_dock_profile(args: argparse.Namespace, dock_design: dict[str, Any]) -> int:
    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    try:
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        joints, pose, raw_joints, raw_pose = robot.read_state()
        dock_tcp = finite_pose(pose, "currently seated Tool(2) TCP")
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tile_id": args.tile,
            "tool": int(args.tool),
            "user": int(args.user),
            "board_dir": str(args.board_dir),
            "dock_design": str(args.dock_design),
            "dock_design_sha256": sha256_file(args.dock_design),
            "dock_tcp": list_pose(dock_tcp),
            "board_yaw_deg": float(args.board_yaw_deg),
            "coordinate_convention": {
                "tile_plus_x": "Tool local +X at the seated datum",
                "tile_plus_y": "Tool local -Y at the seated datum",
                "tile_plus_z": "Tool local -Z at the seated datum (physical up)",
            },
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "joint_count": len(joints or ()),
            "set_user_tool_replies": replies,
            "dock_local_tool_tcp_mm": dict(dock_design["tactip_reference"])["nominal_seated_tool_tcp_local_mm"],
            "note": "This command only reads the currently seated Tool(2) TCP; it does not send a motion command.",
        }
        write_json(args.fixture_profile, profile)
        print("Saved seated fixture profile: {}".format(args.fixture_profile))
        print("Tool({}) TCP: {}".format(args.tool, " ".join("{:.4f}".format(value) for value in dock_tcp)))
        return 0
    finally:
        robot.close()


def filter_tile_rows(tile: dict[str, Any], rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    regions = {str(value) for value in tile.get("region_ids", [])}
    requested_regions = set(args.regions or [])
    requested_sites = set(args.sites or [])
    selected = [row for row in rows if row.get("region_id") in regions]
    if requested_regions:
        missing = requested_regions - regions
        if missing:
            raise ValueError("Requested region(s) are not on {}: {}".format(args.tile, ", ".join(sorted(missing))))
        selected = [row for row in selected if row.get("region_id") in requested_regions]
    if requested_sites:
        found = {str(row.get("site_id")) for row in selected}
        missing = requested_sites - found
        if missing:
            raise ValueError("Requested seed site(s) are not on {}: {}".format(args.tile, ", ".join(sorted(missing))))
        selected = [row for row in selected if row.get("site_id") in requested_sites]
    selected.sort(key=lambda row: (str(row["region_id"]), str(row["site_id"])))
    if not selected:
        raise ValueError("No sampling CSV rows remain for this tile/filter")
    return selected


TILTS_BY_CATEGORY = {
    "flat": ((0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)),
    "edge": ((0.0, 0.0), (4.0, 0.0), (-4.0, 0.0), (0.0, 3.0), (0.0, -3.0)),
    "curvature": ((0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)),
    "multi_touch": ((0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)),
    "small_feature": ((0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)),
}


def profile_replicates(name: str) -> tuple[set[int] | None, int]:
    if name == "quick":
        return {1, 5, 9}, 1
    if name == "standard":
        return None, 2
    if name == "dense":
        return None, 4
    raise ValueError("Unsupported profile {}".format(name))


def seed_number(site_id: str) -> int:
    try:
        return int(site_id.rsplit("S", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not parse seed number from {}".format(site_id)) from exc


def sampling_label(args: argparse.Namespace) -> str:
    if args.samples_per_tile is not None:
        return "{}-sample-dense".format(int(args.samples_per_tile))
    return str(args.profile)


def radical_inverse(index: int, base: int) -> float:
    """One deterministic low-discrepancy coordinate in [0, 1)."""

    result = 0.0
    factor = 1.0 / float(base)
    value = max(0, int(index))
    while value:
        result += factor * float(value % base)
        value //= base
        factor /= float(base)
    return result


def low_discrepancy_fraction(index: int, base: int, seed: int, stream: int) -> float:
    """Scramble a Halton coordinate without sacrificing reproducibility."""

    # A deterministic Cranley-Patterson rotation avoids every seed site using
    # an identical ring of offsets while preserving a reproducible plan.
    scramble = ((int(seed) * (int(stream) * 37 + 19) + int(stream) * 101) % 997) / 997.0
    return (radical_inverse(int(index) + 1, int(base)) + scramble) % 1.0


def planned_depth_bounds(args: argparse.Namespace) -> tuple[float, float]:
    """Return the globally requested final indentation interval in millimetres."""

    return float(args.min_post_contact_depth_mm), float(args.max_post_contact_depth_mm)


def uniformly_distributed_depth(index: int, seed: int, stream: int, args: argparse.Namespace) -> float:
    """Sample the requested 1--10 mm range with a deterministic low-discrepancy sequence.

    The sequence distributes successive contacts across the whole interval
    rather than repeating a small category-specific set of indentation levels.
    It is deterministic from ``--seed`` and therefore reproducible in the
    saved plan and after controller-IK candidate replacement.
    """

    lower, upper = planned_depth_bounds(args)
    if abs(upper - lower) <= 1e-9:
        return lower
    return lower + (upper - lower) * low_discrepancy_fraction(index, 5, seed, stream)


def evenly_allocate(total: int, labels: Sequence[str]) -> dict[str, int]:
    if not labels:
        raise ValueError("Cannot allocate samples across zero regions")
    quotient, remainder = divmod(int(total), len(labels))
    return {str(label): quotient + (1 if index < remainder else 0) for index, label in enumerate(labels)}


def dense_region_grid_half_span_mm(manifest: dict[str, Any], args: argparse.Namespace) -> float:
    """Largest safe centre-position span after accounting for TacTip's footprint."""

    safe_feature_half_span = float(manifest.get("safe_central_feature_zone_mm", 0.0)) / 2.0
    tactip_radius = float(manifest.get("nominal_tactip_compliant_diameter_mm", 0.0)) / 2.0
    automatic = safe_feature_half_span - tactip_radius
    if automatic <= 0.0:
        raise ValueError(
            "The board's safe feature zone ({:.1f} mm) is not wider than the TacTip diameter ({:.1f} mm)".format(
                2.0 * safe_feature_half_span,
                2.0 * tactip_radius,
            )
        )
    requested = automatic if args.dense_region_half_span_mm is None else float(args.dense_region_half_span_mm)
    if requested > automatic + 1e-6:
        raise ValueError(
            "--dense-region-half-span-mm {:.2f} exceeds the {:.2f} mm safe centre-position half-span "
            "derived from this board's feature zone and TacTip footprint".format(requested, automatic)
        )
    return float(requested)


def effective_dense_spatial_layout(args: argparse.Namespace) -> str:
    """Keep explicit single-site experiments local even when grid is the default."""

    if args.dense_spatial_layout == "region_grid" and bool(args.sites):
        return "seed_jitter"
    return str(args.dense_spatial_layout)


def dense_region_grid_offsets(
    anchor_count: int,
    half_span_mm: float,
    seed: int,
    region_index: int,
) -> list[tuple[float, float]]:
    """Build a reproducibly jittered square anchor lattice inside one safe region."""

    side = int(round(math.sqrt(int(anchor_count))))
    cell = 2.0 * float(half_span_mm) / float(side)
    max_jitter = 0.10 * cell
    offsets: list[tuple[float, float]] = []
    for row in range(side):
        for column in range(side):
            anchor_index = row * side + column
            centre_x = -float(half_span_mm) + (float(column) + 0.5) * cell
            centre_y = -float(half_span_mm) + (float(row) + 0.5) * cell
            jitter_x = (2.0 * low_discrepancy_fraction(anchor_index, 2, seed, 50001 + region_index) - 1.0) * max_jitter
            jitter_y = (2.0 * low_discrepancy_fraction(anchor_index, 3, seed, 51001 + region_index) - 1.0) * max_jitter
            offsets.append((float(centre_x + jitter_x), float(centre_y + jitter_y)))
    return offsets


def coprime_stride(count: int) -> int:
    """Return a stable stride that visits every grid anchor before repetition."""

    candidate = max(1, int(math.sqrt(count)) * 2 + 1)
    while math.gcd(candidate, count) != 1:
        candidate += 1
    return candidate


def closest_seed_row(rows: Sequence[dict[str, str]], board_xy: np.ndarray) -> dict[str, str]:
    """Attach a broad-coverage contact to its nearest CSV metadata seed."""

    return min(
        rows,
        key=lambda row: float(
            np.linalg.norm(
                board_xy - np.asarray((float(row["board_x_mm"]), float(row["board_y_mm"])), dtype=float)
            )
        ),
    )


def dense_tilt(category: str, index: int, seed: int, stream: int) -> tuple[float, float]:
    """Return a balanced, bounded tilt inside the validated category limits."""

    presets = TILTS_BY_CATEGORY[category]
    nominal_x, nominal_y = presets[int(index) % len(presets)]
    if abs(nominal_x) < 1e-9 and abs(nominal_y) < 1e-9:
        return 0.0, 0.0
    # Keep the direction/maximum angle from the existing validated recipe but
    # vary its magnitude to increase coverage between the canonical presets.
    magnitude = 0.55 + 0.45 * low_discrepancy_fraction(index, 7, seed, stream)
    return float(nominal_x) * magnitude, float(nominal_y) * magnitude


def build_dense_samples(
    manifest: dict[str, Any],
    tile: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    requested_count: int | None = None,
) -> list[BoardSample]:
    """Build an exact, reproducible dense plan with broad spatial coverage.

    The default ``region_grid`` layout uses a deterministic set of widely
    separated anchors in the safe centre window of every feature region.  It
    accounts for the nominal TacTip footprint so the sensor does not cross a
    neighbouring board cell.  ``seed_jitter`` remains available for targeted
    local repeat experiments and is selected automatically for ``--site``.
    """

    if requested_count is None:
        requested_count = int(args.samples_per_tile)
    requested_count = int(requested_count)
    if requested_count < 1:
        raise ValueError("Dense sample count must be positive")
    local_centre = np.asarray(tile["center_board_xy_mm"], dtype=float)
    regions_by_id = {str(region["region_id"]): dict(region) for region in manifest.get("regions", [])}
    selected_regions = sorted({str(row["region_id"]) for row in rows})
    allocation = evenly_allocate(requested_count, selected_regions)
    height_profile = str(manifest.get("height_profile", ""))
    if not height_profile:
        raise ValueError("Board manifest lacks height_profile required for dense surface-height evaluation")
    safe_half_span = float(manifest.get("safe_central_feature_zone_mm", 60.0)) / 2.0
    use_region_grid = effective_dense_spatial_layout(args) == "region_grid"
    region_grid_half_span = dense_region_grid_half_span_mm(manifest, args) if use_region_grid else None
    samples: list[BoardSample] = []

    for region_index, region_id in enumerate(selected_regions):
        region_rows = sorted((row for row in rows if str(row["region_id"]) == region_id), key=lambda row: str(row["site_id"]))
        if region_id not in regions_by_id:
            raise ValueError("Board manifest lacks geometry metadata for {}".format(region_id))
        if not region_rows:
            raise ValueError("No CSV seed sites are available for {}".format(region_id))
        region = regions_by_id[region_id]
        centre_board = np.asarray(region["center_board_xy_mm"], dtype=float)
        count = allocation[region_id]
        anchors = (
            dense_region_grid_offsets(
                int(args.dense_region_anchor_count),
                float(region_grid_half_span),
                args.seed,
                region_index,
            )
            if use_region_grid
            else ()
        )
        anchor_stride = coprime_stride(len(anchors)) if anchors else 1
        for ordinal in range(count):
            if anchors:
                anchor_index = (ordinal * anchor_stride + (args.seed + 13 * region_index) % len(anchors)) % len(anchors)
                local_u, local_v = anchors[anchor_index]
                board_xy = centre_board + np.asarray((local_u, local_v), dtype=float)
                row = closest_seed_row(region_rows, board_xy)
                repeat = ordinal // len(anchors) + 1
                sample_id = "{}_{}_g{:02d}_d{:04d}".format(args.tile, region_id, anchor_index + 1, repeat)
                jitter_x = float(board_xy[0]) - float(row["board_x_mm"])
                jitter_y = float(board_xy[1]) - float(row["board_y_mm"])
            else:
                row_index = ordinal % len(region_rows)
                row = region_rows[row_index]
                repeat = ordinal // len(region_rows) + 1
                seed_id_for_stream = str(row["site_id"])
                stream = 1009 * (region_index + 1) + 97 * seed_number(seed_id_for_stream)
                jitter_radius = float(row["sampling_jitter_radius_mm"])
                radial = jitter_radius * math.sqrt(low_discrepancy_fraction(repeat, 2, args.seed, stream))
                angle = 2.0 * math.pi * low_discrepancy_fraction(repeat, 3, args.seed, stream + 1)
                jitter_x = radial * math.cos(angle)
                jitter_y = radial * math.sin(angle)
                board_xy = np.asarray((float(row["board_x_mm"]) + jitter_x, float(row["board_y_mm"]) + jitter_y), dtype=float)
                local_u, local_v = board_xy - centre_board
                sample_id = "{}_{}_d{:04d}".format(args.tile, seed_id_for_stream, repeat)
            seed_id = str(row["site_id"])
            seed_id_number = seed_number(seed_id)
            stream = 1009 * (region_index + 1) + 97 * seed_id_number
            if abs(float(local_u)) > safe_half_span + 1e-6 or abs(float(local_v)) > safe_half_span + 1e-6:
                raise RuntimeError(
                    "Dense layout left the {} mm safe central zone for {}: ({:.3f}, {:.3f})".format(
                        2.0 * safe_half_span, seed_id, float(local_u), float(local_v)
                    )
                )
            surface_z = surface_height_for_profile(height_profile, str(row["stimulus"]), float(local_u), float(local_v))
            # Depth is a global, balanced 1--10 mm protocol.  Per-site CSV
            # limits are still preserved in the plan metadata and gate real
            # execution unless the operator explicitly authorises an override.
            depth = uniformly_distributed_depth(ordinal, args.seed, 7001 + region_index, args)
            tilt_x, tilt_y = dense_tilt(str(row["category"]), repeat + seed_id_number, args.seed, stream + 3)
            local_xy = board_xy - local_centre
            samples.append(
                BoardSample(
                    index=len(samples) + 1,
                    sample_id=sample_id,
                    tile_id=args.tile,
                    source_seed_site_id=seed_id,
                    region_id=region_id,
                    category=str(row["category"]),
                    stimulus=str(row["stimulus"]),
                    replicate=repeat,
                    local_contact_mm=(float(local_xy[0]), float(local_xy[1]), float(surface_z)),
                    expected_surface_z_mm=float(surface_z),
                    post_contact_depth_mm=float(depth),
                    tilt_x_deg=float(tilt_x),
                    tilt_y_deg=float(tilt_y),
                    jitter_x_mm=float(jitter_x),
                    jitter_y_mm=float(jitter_y),
                )
            )
    if len(samples) != requested_count:
        raise RuntimeError("Dense plan generated {} samples, expected {}".format(len(samples), requested_count))
    return samples


def build_samples(
    manifest: dict[str, Any],
    tile: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[BoardSample]:
    if args.samples_per_tile is not None:
        return build_dense_samples(manifest, tile, rows, args)
    local_centre = np.asarray(tile["center_board_xy_mm"], dtype=float)
    selected_seed_numbers, replicate_count = profile_replicates(args.profile)
    expanded: list[BoardSample] = []
    for row_index, row in enumerate(rows):
        number = seed_number(str(row["site_id"]))
        if selected_seed_numbers is not None and number not in selected_seed_numbers:
            continue
        category = str(row["category"])
        tilts = TILTS_BY_CATEGORY[category]
        for replicate in range(replicate_count):
            config_index = row_index + replicate * 3 + number
            depth = uniformly_distributed_depth(len(expanded), args.seed, 9001 + row_index, args)
            tilt_x, tilt_y = tilts[config_index % len(tilts)]
            local_xy = np.asarray((float(row["board_x_mm"]), float(row["board_y_mm"])), dtype=float) - local_centre
            expanded.append(
                BoardSample(
                    index=len(expanded) + 1,
                    sample_id="{}_{}_r{:02d}".format(args.tile, row["site_id"], replicate + 1),
                    tile_id=args.tile,
                    source_seed_site_id=str(row["site_id"]),
                    region_id=str(row["region_id"]),
                    category=category,
                    stimulus=str(row["stimulus"]),
                    replicate=replicate + 1,
                    local_contact_mm=(float(local_xy[0]), float(local_xy[1]), float(row["expected_surface_z_mm"])),
                    expected_surface_z_mm=float(row["expected_surface_z_mm"]),
                    post_contact_depth_mm=depth,
                    tilt_x_deg=float(tilt_x),
                    tilt_y_deg=float(tilt_y),
                    jitter_x_mm=0.0,
                    jitter_y_mm=0.0,
                )
            )
    if args.max_samples is not None:
        expanded = expanded[: int(args.max_samples)]
    if not expanded:
        raise ValueError("No samples remain after applying --profile/filter options")
    return [BoardSample(index=index, **{key: value for key, value in asdict(sample).items() if key != "index"}) for index, sample in enumerate(expanded, start=1)]


def build_ik_candidate_pool(
    manifest: dict[str, Any],
    tile: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    requested_samples: list[BoardSample],
) -> list[BoardSample]:
    """Return deterministic dense replacements for controller IK filtering.

    Each region starts with the ordinary dense plan's low-discrepancy sequence.
    Later candidates continue that same sequence, so a rejected
    workspace-boundary point can be replaced without moving a contact into an
    unknown area of the printed tile.
    """

    if args.samples_per_tile is None:
        return list(requested_samples)
    pool_count = max(
        len(requested_samples),
        int(math.ceil(float(len(requested_samples)) * float(args.ik_candidate_multiplier))),
    )
    return build_dense_samples(manifest, tile, rows, args, requested_count=pool_count)


def reindex_samples(samples: Sequence[BoardSample]) -> list[BoardSample]:
    """Keep output sample indexes consecutive after candidate replacement."""

    return [replace(sample, index=index) for index, sample in enumerate(samples, start=1)]


def depth_distribution_summary(samples: Sequence[BoardSample], args: argparse.Namespace) -> dict[str, Any]:
    """Describe the requested depth coverage, including 1 mm-wide histogram bins."""

    lower, upper = planned_depth_bounds(args)
    values = np.asarray([float(sample.post_contact_depth_mm) for sample in samples], dtype=float)
    edges = np.linspace(lower, upper, 10, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    histogram = {
        "{:.0f}-{:.0f}mm".format(float(edges[index]), float(edges[index + 1])): int(count)
        for index, count in enumerate(counts)
    }
    return {
        "requested_range_mm": [lower, upper],
        "sample_count": int(len(values)),
        "actual_min_mm": None if len(values) == 0 else float(values.min()),
        "actual_max_mm": None if len(values) == 0 else float(values.max()),
        "actual_mean_mm": None if len(values) == 0 else float(values.mean()),
        "histogram_1mm_bins": histogram,
    }


def spatial_coverage_summary(samples: Sequence[BoardSample]) -> dict[str, Any]:
    """Report distinct XY anchors and their spacing for each tactile region."""

    summary: dict[str, Any] = {}
    for region_id in sorted({sample.region_id for sample in samples}):
        points = np.asarray(
            [sample.local_contact_mm[:2] for sample in samples if sample.region_id == region_id],
            dtype=float,
        )
        unique = np.unique(np.round(points, decimals=6), axis=0)
        minimum_spacing = None
        if len(unique) > 1:
            deltas = unique[:, None, :] - unique[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            distances[np.diag_indices_from(distances)] = np.inf
            minimum_spacing = float(distances.min())
        summary[region_id] = {
            "sample_count": int(len(points)),
            "distinct_xy_anchor_count": int(len(unique)),
            "minimum_distinct_xy_spacing_mm": minimum_spacing,
            "x_span_mm": float(unique[:, 0].max() - unique[:, 0].min()) if len(unique) else 0.0,
            "y_span_mm": float(unique[:, 1].max() - unique[:, 1].min()) if len(unique) else 0.0,
        }
    return summary


def csv_depth_limit_summary(samples: Sequence[BoardSample], rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Summarise planned depths that exceed a site's declared CSV limit."""

    limits = {
        str(row["site_id"]): (
            float(row["recommended_depth_min_mm"]),
            float(row["recommended_depth_max_mm"]),
        )
        for row in rows
    }
    above: list[BoardSample] = []
    below: list[BoardSample] = []
    for sample in samples:
        try:
            lower, upper = limits[sample.source_seed_site_id]
        except KeyError as exc:
            raise ValueError("No CSV depth range for {}".format(sample.source_seed_site_id)) from exc
        if float(sample.post_contact_depth_mm) > upper + 1e-6:
            above.append(sample)
        if float(sample.post_contact_depth_mm) < lower - 1e-6:
            below.append(sample)

    def group_count(items: Sequence[BoardSample], attribute: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(item, attribute)) for item in items).items()))

    max_excess = max(
        (
            float(sample.post_contact_depth_mm) - limits[sample.source_seed_site_id][1]
            for sample in above
        ),
        default=0.0,
    )
    return {
        "above_csv_limit_count": len(above),
        "below_csv_limit_count": len(below),
        "above_csv_limit_by_category": group_count(above, "category"),
        "above_csv_limit_by_region": group_count(above, "region_id"),
        "max_excess_mm": float(max_excess),
    }


def apply_first_contact_test(samples: list[BoardSample], args: argparse.Namespace) -> list[BoardSample]:
    """Make a single low-risk, zero-tilt check from an approved seed site."""

    if not args.first_contact_test:
        return samples
    if len(samples) != 1:
        raise ValueError("--first-contact-test requires exactly one planned sample")
    sample = samples[0]
    minimum_depth, _ = planned_depth_bounds(args)
    return [
        replace(
            sample,
            sample_id="{}_contact_check".format(sample.sample_id),
            post_contact_depth_mm=minimum_depth,
            tilt_x_deg=0.0,
            tilt_y_deg=0.0,
        )
    ]


def make_route(
    sample: BoardSample,
    fixture: FixtureTransform,
    args: argparse.Namespace,
    correction_base_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    local_contact = np.asarray(sample.local_contact_mm, dtype=float)
    contact_pose = fixture.pose(local_contact, sample.tilt_x_deg, sample.tilt_y_deg)
    correction = np.asarray(correction_base_mm, dtype=float)
    contact_position = np.asarray(contact_pose[:3], dtype=float) + correction
    orientation = tuple(contact_pose[3:])
    press_axis = fixture.press_axis(sample.tilt_x_deg, sample.tilt_y_deg)
    approach_position = contact_position - press_axis * float(args.approach_clearance_mm)
    approach_pose = finite_pose(tuple(approach_position) + orientation, "site approach")
    dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
    dock_exit = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
    # Keep the first high waypoint directly above the keyed dock.  The former
    # board-centre waypoint can be outside the CR3 workspace even though the
    # seated datum is reachable.
    dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
    high_pose = fixture.pose((local_contact[0], local_contact[1], float(args.safe_height_mm)), sample.tilt_x_deg, sample.tilt_y_deg)
    # The reference correction concerns the board plane, not the dock. It is
    # applied after the globally clear route is reached.
    high_pose = finite_pose(tuple(np.asarray(high_pose[:3]) + correction) + tuple(high_pose[3:]), "site high")
    return {
        "contact_tcp": finite_pose(tuple(contact_position) + orientation, "site contact"),
        "approach_tcp": approach_pose,
        "site_high_tcp": high_pose,
        "press_axis_base": press_axis,
        "outbound": (("dock_exit", dock_exit), ("dock_high", dock_high), ("site_high", high_pose), ("approach", approach_pose)),
        "return": (("site_high", high_pose), ("dock_high", dock_high), ("dock_exit", dock_exit)),
    }


def maximum_capture_tcp(route: dict[str, Any], args: argparse.Namespace) -> tuple[float, float, float, float, float, float]:
    """Return the deepest TCP the visual-contact search could ever command.

    The camera-based search may detect zero contact anywhere between the
    approach pose and the fixed allowance below the nominal surface.  Its
    subsequent requested indentation can never exceed this endpoint, so it is
    the conservative pose to pass to controller IK during planning.
    """

    approach = np.asarray(route["approach_tcp"][:3], dtype=float)
    contact = np.asarray(route["contact_tcp"][:3], dtype=float)
    clearance = float(np.linalg.norm(contact - approach))
    if clearance <= 0.0:
        raise ValueError("Site approach and contact TCPs cannot be identical")
    axis = np.asarray(route["press_axis_base"], dtype=float)
    deepest = approach + axis * (clearance + float(args.max_extra_below_planned_contact_mm))
    return finite_pose(
        tuple(deepest.tolist()) + tuple(route["approach_tcp"][3:]),
        "maximum visual-contact capture",
    )


def route_ik_targets(route: dict[str, Any], args: argparse.Namespace) -> tuple[tuple[str, tuple[float, float, float, float, float, float]], ...]:
    """All non-common route poses that must be IK-reachable for one sample."""

    return (
        ("site_high", route["site_high_tcp"]),
        ("approach", route["approach_tcp"]),
        ("planned_contact", route["contact_tcp"]),
        ("deepest_capture_limit", maximum_capture_tcp(route, args)),
        # This is deliberately checked again using the IK solution of the
        # deepest endpoint. It validates the branch needed for the upward
        # retreat, not merely the geometrically identical outbound high pose.
        ("site_retract", route["site_high_tcp"]),
    )


def reference_route(fixture: FixtureTransform, dock_design: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reference = dict(dock_design["reference_pad"])
    local_contact = np.asarray(reference["centre_local_mm"], dtype=float)
    # ``centre_local_mm`` describes the centre of the printed reference-pad
    # solid.  The TCP must target its exposed top face, whose absolute local
    # height is recorded separately in every dock design.
    local_contact[2] = float(reference.get("top_z_mm", local_contact[2]))
    base_contact = fixture.pose(local_contact)
    press_axis = fixture.press_axis(0.0, 0.0)
    approach_position = np.asarray(base_contact[:3]) - press_axis * float(args.approach_clearance_mm)
    approach = finite_pose(tuple(approach_position) + tuple(base_contact[3:]), "reference-pad approach")
    dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
    exit_pose = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
    dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
    high = fixture.pose((local_contact[0], local_contact[1], float(args.safe_height_mm)))
    return {
        "local_contact_mm": local_contact,
        "contact_tcp": base_contact,
        "approach_tcp": approach,
        "press_axis_base": press_axis,
        "outbound": (("dock_exit", exit_pose), ("dock_high", dock_high), ("reference_high", high), ("reference_approach", approach)),
        "return": (("reference_high", high), ("dock_high", dock_high), ("dock_exit", exit_pose)),
    }


def route_records(route: Iterable[tuple[str, tuple[float, float, float, float, float, float]]]) -> list[dict[str, Any]]:
    return [{"label": label, "tcp": list_pose(pose)} for label, pose in route]


def write_plan_csv(
    path: Path,
    samples: list[BoardSample],
    fixture: FixtureTransform | None,
    args: argparse.Namespace,
    correction_base_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    fields = [
        "sample_index", "sample_id", "tile_id", "source_seed_site_id", "region_id", "category", "stimulus", "replicate",
        "local_x_mm", "local_y_mm", "expected_surface_z_mm", "post_contact_depth_mm", "tilt_x_deg", "tilt_y_deg", "jitter_x_mm", "jitter_y_mm",
        "planned_contact_tcp_x", "planned_contact_tcp_y", "planned_contact_tcp_z", "planned_contact_tcp_Rx", "planned_contact_tcp_Ry", "planned_contact_tcp_Rz",
        "planned_approach_tcp_x", "planned_approach_tcp_y", "planned_approach_tcp_z", "planned_approach_tcp_Rx", "planned_approach_tcp_Ry", "planned_approach_tcp_Rz",
        "planned_site_high_tcp_x", "planned_site_high_tcp_y", "planned_site_high_tcp_z", "planned_site_high_tcp_Rx", "planned_site_high_tcp_Ry", "planned_site_high_tcp_Rz",
        "planned_deepest_capture_limit_tcp_x", "planned_deepest_capture_limit_tcp_y", "planned_deepest_capture_limit_tcp_z", "planned_deepest_capture_limit_tcp_Rx", "planned_deepest_capture_limit_tcp_Ry", "planned_deepest_capture_limit_tcp_Rz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row: dict[str, Any] = {
                "sample_index": sample.index,
                "sample_id": sample.sample_id,
                "tile_id": sample.tile_id,
                "source_seed_site_id": sample.source_seed_site_id,
                "region_id": sample.region_id,
                "category": sample.category,
                "stimulus": sample.stimulus,
                "replicate": sample.replicate,
                "local_x_mm": "{:.4f}".format(sample.local_contact_mm[0]),
                "local_y_mm": "{:.4f}".format(sample.local_contact_mm[1]),
                "expected_surface_z_mm": "{:.4f}".format(sample.expected_surface_z_mm),
                "post_contact_depth_mm": "{:.4f}".format(sample.post_contact_depth_mm),
                "tilt_x_deg": "{:.3f}".format(sample.tilt_x_deg),
                "tilt_y_deg": "{:.3f}".format(sample.tilt_y_deg),
                "jitter_x_mm": "{:.4f}".format(sample.jitter_x_mm),
                "jitter_y_mm": "{:.4f}".format(sample.jitter_y_mm),
            }
            if fixture is not None:
                route = make_route(sample, fixture, args, correction_base_mm)
                for prefix, pose in (
                    ("planned_contact_tcp", route["contact_tcp"]),
                    ("planned_approach_tcp", route["approach_tcp"]),
                    ("planned_site_high_tcp", route["site_high_tcp"]),
                    ("planned_deepest_capture_limit_tcp", maximum_capture_tcp(route, args)),
                ):
                    for axis, value in zip(("x", "y", "z", "Rx", "Ry", "Rz"), pose):
                        row["{}_{}".format(prefix, axis)] = "{:.8f}".format(value)
            writer.writerow(row)


def write_plan_preview(path: Path, tile: dict[str, Any], board_dir: Path, dock_design: dict[str, Any], samples: list[BoardSample]) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for the interactive plan preview") from exc
    tile_path = board_dir / str(tile["stl"])
    mesh = trimesh.load_mesh(tile_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Could not load tile mesh for preview: {}".format(tile_path))
    face_limit = 28000
    if len(mesh.faces) > face_limit:
        face_indices = np.linspace(0, len(mesh.faces) - 1, face_limit, dtype=int)
        faces = mesh.faces[face_indices]
        vertices_used, remap = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = mesh.vertices[vertices_used]
        faces = remap.reshape((-1, 3))
    else:
        vertices, faces = mesh.vertices, mesh.faces
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#2a6fbb", opacity=0.43, name="printed tile",
            hoverinfo="skip",
        )
    )
    dock_path = Path(str(dock_design.get("stl", "")))
    if dock_path.is_file():
        dock = trimesh.load_mesh(dock_path, force="mesh", process=False)
        if isinstance(dock, trimesh.Trimesh):
            dock_faces = dock.faces
            if len(dock_faces) > 12000:
                indices = np.linspace(0, len(dock_faces) - 1, 12000, dtype=int)
                dock_faces = dock_faces[indices]
                dock_vertices_used, dock_remap = np.unique(dock_faces.reshape(-1), return_inverse=True)
                dock_vertices = dock.vertices[dock_vertices_used]
                dock_faces = dock_remap.reshape((-1, 3))
            else:
                dock_vertices = dock.vertices
            figure.add_trace(
                go.Mesh3d(
                    x=dock_vertices[:, 0], y=dock_vertices[:, 1], z=dock_vertices[:, 2],
                    i=dock_faces[:, 0], j=dock_faces[:, 1], k=dock_faces[:, 2],
                    color="#e27024", opacity=0.87, name="calibration dock", hoverinfo="skip",
                )
            )
    categories = sorted({sample.category for sample in samples})
    palette = {"flat": "#1ca6a6", "edge": "#e65640", "curvature": "#7856e8", "multi_touch": "#f0a529", "small_feature": "#dd5fa3"}
    for category in categories:
        group = [sample for sample in samples if sample.category == category]
        figure.add_trace(
            go.Scatter3d(
                x=[sample.local_contact_mm[0] for sample in group],
                y=[sample.local_contact_mm[1] for sample in group],
                z=[sample.local_contact_mm[2] for sample in group],
                mode="markers",
                name="{} planned contacts ({})".format(category, len(group)),
                marker={"size": 4.8, "color": palette.get(category, "#ffffff"), "line": {"color": "#ffffff", "width": 0.6}},
                customdata=[[sample.sample_id, sample.source_seed_site_id, sample.post_contact_depth_mm, sample.tilt_x_deg, sample.tilt_y_deg] for sample in group],
                hovertemplate="%{customdata[0]}<br>seed=%{customdata[1]}<br>post-contact depth=%{customdata[2]:.2f} mm<br>tilt X/Y=%{customdata[3]:+.1f}/%{customdata[4]:+.1f} deg<br>tile local X/Y/Z=%{x:.2f}/%{y:.2f}/%{z:.2f} mm<extra></extra>",
            )
        )
    for region_id in sorted({sample.region_id for sample in samples}):
        unique_by_xy: dict[tuple[float, float], BoardSample] = {}
        for sample in samples:
            if sample.region_id != region_id:
                continue
            key = (round(float(sample.local_contact_mm[0]), 6), round(float(sample.local_contact_mm[1]), 6))
            unique_by_xy.setdefault(key, sample)
        anchors = list(unique_by_xy.values())
        if len(anchors) == len([sample for sample in samples if sample.region_id == region_id]):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=[sample.local_contact_mm[0] for sample in anchors],
                y=[sample.local_contact_mm[1] for sample in anchors],
                z=[sample.local_contact_mm[2] for sample in anchors],
                mode="markers",
                name="{} distinct XY anchors ({})".format(region_id, len(anchors)),
                marker={"size": 7.5, "color": "#172033", "symbol": "square-open", "line": {"color": "#ffffff", "width": 1.1}},
                customdata=[[sample.source_seed_site_id, sample.replicate] for sample in anchors],
                hovertemplate="{} spatial anchor<br>nearest CSV seed=%{{customdata[0]}}<br>tile local X/Y/Z=%{{x:.2f}}/%{{y:.2f}}/%{{z:.2f}} mm<extra></extra>".format(region_id),
            )
        )
    reference = dict(dock_design["reference_pad"])["centre_local_mm"]
    figure.add_trace(go.Scatter3d(x=[reference[0]], y=[reference[1]], z=[reference[2]], mode="markers", name="automatic reference pad", marker={"size": 7, "color": "#f6d743", "symbol": "diamond"}, hovertemplate="Automatic flat reference pad<extra></extra>"))
    figure.update_layout(
        title="{}: contact plan in tile-local coordinates".format(str(tile["tile_id"])),
        paper_bgcolor="#f7f9fc",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        scene={
            "xaxis": {"title": "tile local X (mm)"},
            "yaxis": {"title": "tile local Y (mm)"},
            "zaxis": {"title": "tile local Z (mm)"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.62, "z": 1.15}},
        },
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def base_to_tile_local(fixture: FixtureTransform, base_xyz_mm: Sequence[float]) -> np.ndarray:
    """Invert the dock-defined rigid transform for a local preview."""

    return np.asarray(fixture.dock_tcp_local_mm, dtype=float) + fixture.tile_to_base.T @ (
        np.asarray(base_xyz_mm, dtype=float) - np.asarray(fixture.dock_tcp[:3], dtype=float)
    )


def write_ik_candidate_preview(
    path: Path,
    tile: dict[str, Any],
    board_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
    args: argparse.Namespace,
    requested_samples: Sequence[BoardSample],
    candidate_samples: Sequence[BoardSample],
) -> None:
    """Visualize the dense replacement pool and representative safe routes."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for the interactive IK candidate preview") from exc

    tile_path = board_dir / str(tile["stl"])
    mesh = trimesh.load_mesh(tile_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Could not load tile mesh for IK preview: {}".format(tile_path))
    faces = mesh.faces
    vertices = mesh.vertices
    if len(faces) > 28000:
        face_indices = np.linspace(0, len(faces) - 1, 28000, dtype=int)
        faces = faces[face_indices]
        vertices_used, remap = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[vertices_used]
        faces = remap.reshape((-1, 3))
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#2a6fbb", opacity=0.38, name="printed tile", hoverinfo="skip",
        )
    )
    dock_path = Path(str(dock_design.get("stl", "")))
    if dock_path.is_file():
        dock = trimesh.load_mesh(dock_path, force="mesh", process=False)
        if isinstance(dock, trimesh.Trimesh):
            dock_faces = dock.faces
            dock_vertices = dock.vertices
            if len(dock_faces) > 12000:
                dock_indices = np.linspace(0, len(dock_faces) - 1, 12000, dtype=int)
                dock_faces = dock_faces[dock_indices]
                dock_used, dock_remap = np.unique(dock_faces.reshape(-1), return_inverse=True)
                dock_vertices = dock_vertices[dock_used]
                dock_faces = dock_remap.reshape((-1, 3))
            figure.add_trace(
                go.Mesh3d(
                    x=dock_vertices[:, 0], y=dock_vertices[:, 1], z=dock_vertices[:, 2],
                    i=dock_faces[:, 0], j=dock_faces[:, 1], k=dock_faces[:, 2],
                    color="#e27024", opacity=0.82, name="calibration dock", hoverinfo="skip",
                )
            )

    requested_ids = {sample.sample_id for sample in requested_samples}
    reserve = [sample for sample in candidate_samples if sample.sample_id not in requested_ids]
    palette = {"flat": "#27a7a7", "edge": "#ea704b", "curvature": "#8062e5", "multi_touch": "#eead32", "small_feature": "#db6097"}
    for category in sorted({sample.category for sample in requested_samples}):
        group = [sample for sample in requested_samples if sample.category == category]
        figure.add_trace(
            go.Scatter3d(
                x=[sample.local_contact_mm[0] for sample in group],
                y=[sample.local_contact_mm[1] for sample in group],
                z=[sample.local_contact_mm[2] for sample in group],
                mode="markers", name="planned {} ({})".format(category, len(group)),
                marker={"size": 3.4, "color": palette.get(category, "#ffffff"), "opacity": 0.88},
                customdata=[[sample.sample_id, sample.source_seed_site_id, sample.post_contact_depth_mm] for sample in group],
                hovertemplate="%{customdata[0]}<br>seed=%{customdata[1]}<br>post-depth=%{customdata[2]:.2f} mm<extra></extra>",
            )
        )
    if reserve:
        figure.add_trace(
            go.Scatter3d(
                x=[sample.local_contact_mm[0] for sample in reserve],
                y=[sample.local_contact_mm[1] for sample in reserve],
                z=[sample.local_contact_mm[2] for sample in reserve],
                mode="markers", name="IK replacement reserve ({})".format(len(reserve)),
                marker={"size": 2.6, "color": "#91d36d", "opacity": 0.43},
                customdata=[[sample.sample_id, sample.region_id] for sample in reserve],
                hovertemplate="reserve %{customdata[0]}<br>region=%{customdata[1]}<extra></extra>",
            )
        )

    seen_regions: set[str] = set()
    representatives: list[BoardSample] = []
    for sample in requested_samples:
        if sample.region_id not in seen_regions:
            representatives.append(sample)
            seen_regions.add(sample.region_id)
        if len(representatives) >= 4:
            break
    for sample in representatives:
        route = make_route(sample, fixture, args)
        dock_high = dict(route["outbound"])["dock_high"]
        route_poses = (
            dock_high,
            route["site_high_tcp"],
            route["approach_tcp"],
            route["contact_tcp"],
            maximum_capture_tcp(route, args),
            route["site_high_tcp"],
            dock_high,
        )
        local_route = np.asarray([base_to_tile_local(fixture, pose[:3]) for pose in route_poses], dtype=float)
        figure.add_trace(
            go.Scatter3d(
                x=local_route[:, 0], y=local_route[:, 1], z=local_route[:, 2],
                mode="lines+markers", name="route example {}".format(sample.region_id),
                line={"color": "#f3e057", "width": 4},
                marker={"size": 3.8, "color": "#f3e057"},
                hovertemplate="{} route endpoint<extra></extra>".format(sample.sample_id),
            )
        )
    figure.update_layout(
        title=(
            "{}: {} planned contacts + {} deterministic IK replacements; "
            "yellow paths are representative route envelopes"
        ).format(tile["tile_id"], len(requested_samples), len(reserve)),
        paper_bgcolor="#f7f9fc",
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        scene={
            "xaxis": {"title": "tile local X (mm)"},
            "yaxis": {"title": "tile local Y (mm)"},
            "zaxis": {"title": "tile local Z (mm)"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.62, "z": 1.15}},
        },
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def prepare_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        path = args.output_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = DEFAULT_RUN_ROOT / "{}_{}_{}".format(args.tile, sampling_label(args), stamp)
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def discard_intermediate_capture(output_dir: Path, record: dict[str, Any]) -> None:
    """Remove diagnostic images once their marker-motion statistics are used.

    A dense run can take many visual-search frames per final capture.  Keeping
    every raw/preprocessed diagnostic image would dominate the final dataset
    size, so only the actual post-contact capture is retained by default.
    """

    root = output_dir.resolve()
    raw_relative = str(record.get("raw_image", ""))
    raw_path = (root / raw_relative).resolve() if raw_relative else None
    candidates: list[Path] = []
    if raw_path is not None:
        try:
            raw_path.relative_to(root)
        except ValueError:
            raw_path = None
        if raw_path is not None:
            name = raw_path.name
            candidates.append(raw_path)
            candidates.extend(
                root / "tactip_preprocessed" / directory / name
                for directory in ("gray", "ring_suppressed", "model_input", "model_input_256", "model_roi", "overlay")
            )
    motion_relative = str(record.get("motion_visualization", ""))
    if motion_relative:
        motion_path = (root / motion_relative).resolve()
        try:
            motion_path.relative_to(root)
        except ValueError:
            motion_path = None
        if motion_path is not None:
            candidates.append(motion_path)
    for path in candidates:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A failed cleanup must never affect the physical contact safety
            # path; it only leaves an extra diagnostic image on disk.
            pass
    for key in ("raw_image", "preprocessed_gray", "model_input", "motion_visualization"):
        record.pop(key, None)
    record["intermediate_frame_saved"] = False


def prepare_site_baseline(
    camera: GelSightCapture,
    args: argparse.Namespace,
    output_dir: Path,
    preprocessor: Any,
    label: str,
) -> tuple[ImageFeature, float, dict[str, float], list[dict[str, Any]]]:
    baseline_record, baseline, camera_time = capture_record(camera, args, output_dir, preprocessor, "{}_baseline".format(label), int(args.baseline_frames), 0.0)
    frames = [baseline_record]
    if not args.save_search_frames:
        discard_intermediate_capture(output_dir, baseline_record)
    noise_records: list[dict[str, Any]] = []
    for index in range(int(args.noise_probes)):
        record, _feature, camera_time = capture_record(
            camera,
            args,
            output_dir,
            preprocessor,
            "{}_noise_{:02d}".format(label, index + 1),
            int(args.probe_frames),
            camera_time,
            baseline,
        )
        frames.append(record)
        noise_records.append(record)
        if not args.save_search_frames:
            discard_intermediate_capture(output_dir, record)
    threshold = {
        "mean": robust_threshold([float(item["marker_motion"]["mean"]) for item in noise_records], float(args.min_contact_mean), float(args.noise_multiplier)),
        "p95": robust_threshold([float(item["marker_motion"]["p95"]) for item in noise_records], float(args.min_contact_p95), float(args.noise_multiplier)),
    }
    return baseline, camera_time, threshold, frames


def search_visual_contact(
    robot: DobotCR3LiveClient,
    camera: GelSightCapture,
    preprocessor: Any,
    args: argparse.Namespace,
    output_dir: Path,
    label: str,
    approach_tcp: tuple[float, float, float, float, float, float],
    planned_contact_tcp: tuple[float, float, float, float, float, float],
    press_axis: np.ndarray,
    post_contact_depth_mm: float,
) -> dict[str, Any]:
    planned_clearance = float(np.linalg.norm(np.asarray(planned_contact_tcp[:3]) - np.asarray(approach_tcp[:3])))
    max_contact_depth = planned_clearance + float(args.max_extra_below_planned_contact_mm) - float(post_contact_depth_mm)
    if max_contact_depth <= 0.0:
        raise RuntimeError("No contact-search room remains after requested post-contact depth")
    record: dict[str, Any] = {
        "label": label,
        "approach_tcp": list_pose(approach_tcp),
        "planned_contact_tcp": list_pose(planned_contact_tcp),
        "press_axis_base": [float(value) for value in press_axis],
        "planned_approach_to_contact_mm": planned_clearance,
        "post_contact_depth_mm": float(post_contact_depth_mm),
        "max_contact_depth_from_approach_mm": max_contact_depth,
        "frames": [],
        "motion": [],
        "status": "started",
    }
    time.sleep(float(args.settle_sec))
    baseline, camera_time, threshold, frames = prepare_site_baseline(camera, args, output_dir, preprocessor, label)
    record["frames"].extend(frames)
    record["threshold"] = threshold
    print("{} thresholds: mean={:.3f}, p95={:.3f}".format(label, threshold["mean"], threshold["p95"]), flush=True)
    first_hit: dict[str, Any] | None = None
    consecutive_hits = 0
    last_motion: dict[str, Any] | None = None
    last_depth: float | None = None
    depths = list(np.arange(float(args.step_mm), max_contact_depth + float(args.step_mm) * 0.5, float(args.step_mm)))
    for index, depth in enumerate(depths, start=1):
        target_position = np.asarray(approach_tcp[:3], dtype=float) + press_axis * float(depth)
        target = finite_pose(tuple(target_position) + tuple(approach_tcp[3:]), "visual search target")
        motion = move_and_verify(robot, "{} press {:03d}".format(label, index), target, args)
        record["motion"].append(motion)
        last_motion, last_depth = motion, float(depth)
        time.sleep(float(args.settle_sec))
        frame, _feature, camera_time = capture_record(
            camera,
            args,
            output_dir,
            preprocessor,
            "{}_search_{:03d}_{:05.2f}mm".format(label, index, float(depth)),
            int(args.probe_frames),
            camera_time,
            baseline,
        )
        frame["search_depth_mm"] = float(depth)
        frame["actual_tcp"] = motion["actual_tcp"]
        stats = frame["marker_motion"]
        hit = float(stats["mean"]) >= float(threshold["mean"]) and float(stats["p95"]) >= float(threshold["p95"])
        frame["hit"] = bool(hit)
        record["frames"].append(frame)
        if not args.save_search_frames:
            discard_intermediate_capture(output_dir, frame)
        print("{} depth={:.2f} mean={:.3f} p95={:.3f} hit={}".format(label, float(depth), float(stats["mean"]), float(stats["p95"]), bool(hit)), flush=True)
        if hit:
            consecutive_hits += 1
            if first_hit is None:
                first_hit = frame
            if consecutive_hits >= int(args.consecutive_hits):
                break
        else:
            first_hit = None
            consecutive_hits = 0
    if first_hit is not None and consecutive_hits == 1 and last_motion is not None and last_depth is not None:
        time.sleep(float(args.settle_sec))
        confirm, _feature, camera_time = capture_record(
            camera,
            args,
            output_dir,
            preprocessor,
            "{}_terminal_confirm_{:05.2f}mm".format(label, last_depth),
            int(args.probe_frames),
            camera_time,
            baseline,
        )
        confirm["search_depth_mm"] = last_depth
        confirm["actual_tcp"] = last_motion["actual_tcp"]
        confirm["stationary_confirmation"] = True
        stats = confirm["marker_motion"]
        hit = float(stats["mean"]) >= float(threshold["mean"]) and float(stats["p95"]) >= float(threshold["p95"])
        confirm["hit"] = bool(hit)
        record["frames"].append(confirm)
        if not args.save_search_frames:
            discard_intermediate_capture(output_dir, confirm)
        if hit:
            consecutive_hits += 1
    if first_hit is None or consecutive_hits < int(args.consecutive_hits):
        record.update({"status": "no_contact", "reason": "No stable marker-motion contact inside the fixed search allowance."})
        return record
    visual_depth = float(first_hit["search_depth_mm"])
    visual_tcp = finite_pose(first_hit["actual_tcp"], "visual-contact TCP")
    capture_depth = visual_depth + float(post_contact_depth_mm)
    if capture_depth > planned_clearance + float(args.max_extra_below_planned_contact_mm) + 1e-6:
        record.update({"status": "capture_limit_reached", "reason": "Contact was detected too deep for the requested post-contact indentation."})
        return record
    if float(post_contact_depth_mm) <= 1e-9:
        record.update({"status": "contact_found", "visual_contact_tcp": list_pose(visual_tcp), "visual_contact_depth_from_approach_mm": visual_depth, "actual_capture_tcp": list_pose(visual_tcp), "capture_depth_from_approach_mm": visual_depth})
        return record
    capture_position = np.asarray(approach_tcp[:3]) + press_axis * capture_depth
    capture_target = finite_pose(tuple(capture_position) + tuple(approach_tcp[3:]), "final capture TCP")
    capture_motion = move_and_verify(robot, "{} final capture".format(label), capture_target, args)
    record["motion"].append(capture_motion)
    time.sleep(float(args.settle_sec))
    capture, _feature, _camera_time = capture_record(
        camera,
        args,
        output_dir,
        preprocessor,
        "{}_capture_{:.2f}mm".format(label, float(post_contact_depth_mm)),
        int(args.capture_frames),
        camera_time,
        baseline,
    )
    capture["actual_tcp"] = capture_motion["actual_tcp"]
    capture["capture_depth_from_approach_mm"] = capture_depth
    record.update(
        {
            "status": "captured",
            "visual_contact_tcp": list_pose(visual_tcp),
            "visual_contact_depth_from_approach_mm": visual_depth,
            "delta_from_planned_zero_along_press_mm": visual_depth - planned_clearance,
            "actual_capture_tcp": capture_motion["actual_tcp"],
            "capture_depth_from_approach_mm": capture_depth,
            "capture": capture,
        }
    )
    return record


def execute_route(robot: DobotCR3LiveClient, route: Iterable[tuple[str, tuple[float, float, float, float, float, float]]], label_prefix: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, pose in route:
        records.append(move_and_verify(robot, "{} {}".format(label_prefix, label), pose, args))
    return records


def write_collection_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "sample_index", "sample_id", "seed_site", "region_id", "category", "stimulus", "status", "post_contact_depth_mm", "tilt_x_deg", "tilt_y_deg",
        "local_x_mm", "local_y_mm", "expected_surface_z_mm", "jitter_x_mm", "jitter_y_mm",
        "visual_contact_depth_from_approach_mm", "delta_from_planned_zero_along_press_mm", "capture_mean", "capture_p95", "raw_image", "model_input",
        "visual_contact_tcp_x", "visual_contact_tcp_y", "visual_contact_tcp_z", "visual_contact_tcp_Rx", "visual_contact_tcp_Ry", "visual_contact_tcp_Rz",
        "actual_capture_tcp_x", "actual_capture_tcp_y", "actual_capture_tcp_z", "actual_capture_tcp_Rx", "actual_capture_tcp_Ry", "actual_capture_tcp_Rz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            sample = dict(item["sample"])
            run = dict(item.get("result", {}))
            capture = dict(run.get("capture", {}))
            row: dict[str, Any] = {
                "sample_index": sample["index"],
                "sample_id": sample["sample_id"],
                "seed_site": sample["source_seed_site_id"],
                "region_id": sample["region_id"],
                "category": sample["category"],
                "stimulus": sample["stimulus"],
                "status": run.get("status", ""),
                "post_contact_depth_mm": sample["post_contact_depth_mm"],
                "tilt_x_deg": sample["tilt_x_deg"],
                "tilt_y_deg": sample["tilt_y_deg"],
                "local_x_mm": sample["local_contact_mm"][0],
                "local_y_mm": sample["local_contact_mm"][1],
                "expected_surface_z_mm": sample["expected_surface_z_mm"],
                "jitter_x_mm": sample.get("jitter_x_mm", 0.0),
                "jitter_y_mm": sample.get("jitter_y_mm", 0.0),
                "visual_contact_depth_from_approach_mm": run.get("visual_contact_depth_from_approach_mm", ""),
                "delta_from_planned_zero_along_press_mm": run.get("delta_from_planned_zero_along_press_mm", ""),
                "capture_mean": dict(capture.get("marker_motion", {})).get("mean", ""),
                "capture_p95": dict(capture.get("marker_motion", {})).get("p95", ""),
                "raw_image": capture.get("raw_image", ""),
                "model_input": capture.get("model_input", ""),
            }
            for prefix, values in (("visual_contact_tcp", run.get("visual_contact_tcp", ())), ("actual_capture_tcp", run.get("actual_capture_tcp", ()))):
                if isinstance(values, (list, tuple)) and len(values) == 6:
                    for axis, value in zip(("x", "y", "z", "Rx", "Ry", "Rz"), values):
                        row["{}_{}".format(prefix, axis)] = "{:.8f}".format(float(value))
            writer.writerow(row)


def build_report(path: Path, payload: dict[str, Any], max_cards: int) -> None:
    cards: list[str] = []
    for item in list(payload.get("samples", []))[: max(0, int(max_cards))]:
        sample = dict(item["sample"])
        result = dict(item.get("result", {}))
        capture = dict(result.get("capture", {}))
        raw = html.escape(str(capture.get("raw_image", "")))
        model_input = html.escape(str(capture.get("model_input", "")))
        body = "{} | depth {:.2f} mm | tilt X/Y {:+.1f}/{:+.1f} deg".format(sample["source_seed_site_id"], float(sample["post_contact_depth_mm"]), float(sample["tilt_x_deg"]), float(sample["tilt_y_deg"]))
        images = ""
        if raw:
            images += '<figure><figcaption>Raw capture</figcaption><img src="{}"></figure>'.format(raw)
        if model_input:
            images += '<figure><figcaption>Shared GAN model input</figcaption><img src="{}"></figure>'.format(model_input)
        cards.append("<article><h2>{}</h2><p><b>{}</b><br>{}</p><div class=\"images\">{}</div></article>".format(html.escape(str(sample["sample_id"])), html.escape(str(result.get("status", ""))), html.escape(body), images))
    captured = sum(1 for item in payload.get("samples", []) if dict(item.get("result", {})).get("status") == "captured")
    page = """<!doctype html><html><head><meta charset=\"utf-8\"><title>Coverage-board CR3 collection</title><style>
body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232}h1{margin:0;font-size:23px}.summary{padding:14px 28px;color:#b7c5d6}.grid{padding:0 28px 28px;display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}article{background:#182232;border:1px solid #33475f;border-radius:7px;padding:13px}article h2{margin:0;font-size:16px}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.images figure{margin:0;background:#0d141e;padding:6px}.images figcaption{font-size:11px;color:#cbd7e5;margin-bottom:5px}.images img{width:100%;display:block;background:#000;image-rendering:pixelated}
</style></head><body><header><h1>Coverage-board CR3 tactile collection</h1></header><div class=\"summary\">Tile: <b>{}</b>. Status: <b>{}</b>. Captured: <b>{}</b> / {}. Every capture used a fresh visual-contact baseline.</div><main class=\"grid\">{}</main></body></html>""".format(
        html.escape(str(payload.get("tile_id", ""))),
        html.escape(str(payload.get("status", ""))),
        captured,
        len(payload.get("samples", [])),
        "\n".join(cards),
    )
    path.write_text(page, encoding="utf-8")


def write_run_readme(path: Path, args: argparse.Namespace, fixture_profile: Path | None, tile_size_mm: Sequence[float]) -> None:
    text = """# CR3 coverage-board sampling run

This directory contains a deterministic plan for one {} x {} mm board tile. It
varies the tactile XY location in each of the tile's four regions, the
post-contact indentation depth, and small tilt angles. Exact sample-count
runs use the selected dense spatial layout. The default `region_grid` spreads
contacts over footprint-safe region anchors; `seed_jitter` is retained for
targeted local repeats.

## Safety contract

- Tool(2) must be seated in the printed calibration dock before an execute run.
- The script checks that its current TCP matches the saved seated datum before
  it makes any motion.
- It exits upward, moves only at the configured safe height, approaches each
  site from above, and detects actual contact from marker motion.
- Before the first board-site motion, an execute run asks the controller to
  solve site-high, approach, nominal contact, maximum possible capture, and
  retreat poses. Dense runs replace rejected candidates from a bounded,
  deterministic safe-site pool and save `ik_route_filter.json`.
- On a missing contact it returns by the high route and stops by default.

## Current run

- Tile: `{}`
- Profile: `{}`
- Requested exact sample count: `{}`
- Dense XY layout: `{}`
- Distinct XY anchors per region: `{}`
- Final indentation after visual contact: uniformly distributed across `{:.1f}-{:.1f} mm`
- Fixture profile: `{}`
- `planned_sampling_points_3d.html` is in tile-local coordinates; it is the
  mechanically fixed frame defined by the printed dock.
    """.format(
        "{:.0f}".format(float(tile_size_mm[0])),
        "{:.0f}".format(float(tile_size_mm[1])),
        args.tile,
        sampling_label(args),
        "not requested" if args.samples_per_tile is None else int(args.samples_per_tile),
        effective_dense_spatial_layout(args),
        int(args.dense_region_anchor_count),
        float(args.min_post_contact_depth_mm),
        float(args.max_post_contact_depth_mm),
        fixture_profile or "not supplied (offline plan only)",
    )
    path.write_text(text, encoding="utf-8")


def check_ik_target(
    robot: DobotCR3LiveClient,
    label: str,
    target: Sequence[float],
    args: argparse.Namespace,
    joint_near: Sequence[float] | None,
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    """Check one controller IK target and preserve branch diagnostics.

    ``InverseSolution(..., 1, {near joints})`` is intentionally the acceptance
    criterion because ``move_and_verify`` will use that same branch hint before
    it sends a real MovL.  A second no-hint query is diagnostic only: it tells
    us whether a target is outside the workspace or merely incompatible with
    the current joint branch, but it never authorizes a route by itself.
    """

    pose = finite_pose(target, label + " IK target")
    near = tuple(float(value) for value in joint_near) if joint_near is not None else None
    record: dict[str, Any] = {
        "label": label,
        "target_tcp": list_pose(pose),
        "joint_near_deg": list(near) if near is not None else None,
    }
    try:
        solution, raw = robot.inverse_solution(pose, args.user, args.tool, joint_near=near)
    except Exception as near_exc:
        record["near_hint_error"] = "{}: {}".format(type(near_exc).__name__, near_exc)
        if near is None:
            record["status"] = "unreachable"
            record["failure_kind"] = "no_ik_solution"
            return None, record
        try:
            fallback_solution, fallback_raw = robot.inverse_solution(pose, args.user, args.tool, joint_near=None)
        except Exception as fallback_exc:
            record["status"] = "unreachable"
            record["failure_kind"] = "no_ik_solution"
            record["no_hint_error"] = "{}: {}".format(type(fallback_exc).__name__, fallback_exc)
            return None, record
        record.update(
            {
                "status": "branch_hint_failed",
                "failure_kind": "near_joint_branch",
                "no_hint_solution_joints_deg": list(fallback_solution),
                "no_hint_raw_reply": fallback_raw,
            }
        )
        return None, record
    record.update(
        {
            "status": "reachable",
            "inverse_joint_solution_deg": list(solution),
            "raw_reply": raw,
        }
    )
    return tuple(float(value) for value in solution), record


def check_ik_sequence(
    robot: DobotCR3LiveClient,
    targets: Sequence[tuple[str, tuple[float, float, float, float, float, float]]],
    args: argparse.Namespace,
    joint_near: Sequence[float] | None,
) -> tuple[tuple[float, ...] | None, list[dict[str, Any]]]:
    """Solve a route in order, carrying each solution to the next target."""

    current_near = tuple(float(value) for value in joint_near) if joint_near is not None else None
    checks: list[dict[str, Any]] = []
    for index, (label, target) in enumerate(targets):
        solution, record = check_ik_target(robot, label, target, args, current_near)
        checks.append(record)
        if solution is None:
            for skipped_label, skipped_target in targets[index + 1:]:
                checks.append(
                    {
                        "label": skipped_label,
                        "target_tcp": list_pose(skipped_target),
                        "status": "not_checked_after_upstream_failure",
                        "failure_kind": record.get("failure_kind", "upstream_failure"),
                    }
                )
            return None, checks
        current_near = solution
    return current_near, checks


def preflight_sample_route(
    robot: DobotCR3LiveClient,
    sample: BoardSample,
    fixture: FixtureTransform,
    args: argparse.Namespace,
    correction_base_mm: Sequence[float],
    dock_high_joints: Sequence[float],
) -> dict[str, Any]:
    """Validate the complete contact-and-retract portion of one candidate."""

    route = make_route(sample, fixture, args, correction_base_mm)
    targets = route_ik_targets(route, args)
    final_joints, checks = check_ik_sequence(robot, targets, args, dock_high_joints)
    failure = next((item for item in checks if item.get("status") not in {"reachable"}), None)
    return {
        "candidate": asdict(sample),
        "status": "accepted" if final_joints is not None else "rejected",
        "failure_label": None if failure is None else failure.get("label"),
        "failure_kind": None if failure is None else failure.get("failure_kind"),
        "checks": checks,
        "route": {
            "site_high_tcp": list_pose(route["site_high_tcp"]),
            "approach_tcp": list_pose(route["approach_tcp"]),
            "contact_tcp": list_pose(route["contact_tcp"]),
            "deepest_capture_limit_tcp": list_pose(maximum_capture_tcp(route, args)),
        },
    }


def select_ik_reachable_samples(
    robot: DobotCR3LiveClient,
    requested_samples: Sequence[BoardSample],
    candidate_samples: Sequence[BoardSample],
    fixture: FixtureTransform,
    args: argparse.Namespace,
    correction_base_mm: Sequence[float],
    current_joints: Sequence[float],
) -> tuple[list[BoardSample], dict[str, Any]]:
    """Select an exact per-region set of controller-IK-reachable candidates.

    This runs after the real reference-pad correction and before the first
    board site motion.  It has no motion side effects.  A dense plan can use
    later safe-layout candidates to replace rejected workspace-boundary poses
    while preserving its requested count and regional balance.
    """

    requested_by_region = Counter(sample.region_id for sample in requested_samples)
    report: dict[str, Any] = {
        "schema": "cr3_coverage_board_route_ik_filter.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "tool": int(args.tool),
        "user": int(args.user),
        "requested_count": len(requested_samples),
        "requested_by_region": dict(sorted(requested_by_region.items())),
        "candidate_pool_count": len(candidate_samples),
        "correction_base_mm": [float(value) for value in correction_base_mm],
        "status": "started",
        "shared_transit_checks": [],
        "candidates": [],
    }
    if args.disable_ik_preflight:
        selected = reindex_samples(requested_samples)
        report.update(
            {
                "status": "disabled_by_flag",
                "selected_count": len(selected),
                "selected_sample_ids": [sample.sample_id for sample in selected],
                "note": "--disable-ik-preflight disabled both route filtering and per-MovL controller IK checks.",
            }
        )
        return selected, report

    dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
    dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
    dock_high_joints, shared_checks = check_ik_sequence(
        robot,
        (("dock_high", dock_high),),
        args,
        current_joints,
    )
    report["shared_transit_checks"] = shared_checks
    if dock_high_joints is None:
        report.update(
            {
                "status": "shared_transit_unreachable",
                "selected_count": 0,
                "selected_sample_ids": [],
            }
        )
        return [], report

    accepted: list[BoardSample] = []
    accepted_by_region: Counter[str] = Counter()
    for candidate in candidate_samples:
        region_id = candidate.region_id
        if accepted_by_region[region_id] >= requested_by_region.get(region_id, 0):
            continue
        candidate_record = preflight_sample_route(
            robot,
            candidate,
            fixture,
            args,
            correction_base_mm,
            dock_high_joints,
        )
        report["candidates"].append(candidate_record)
        if candidate_record["status"] != "accepted":
            continue
        accepted.append(candidate)
        accepted_by_region[region_id] += 1
        if all(accepted_by_region[region] >= count for region, count in requested_by_region.items()):
            break

    selected = reindex_samples(accepted)
    rejected = [item for item in report["candidates"] if item["status"] == "rejected"]
    branch_failures = sum(1 for item in rejected if item.get("failure_kind") == "near_joint_branch")
    rejected_by_region = Counter(str(dict(item["candidate"])["region_id"]) for item in rejected)
    rejected_by_label = Counter(str(item.get("failure_label", "unknown")) for item in rejected)
    report.update(
        {
            "status": "complete" if len(selected) == len(requested_samples) else "insufficient_reachable_candidates",
            "checked_candidate_count": len(report["candidates"]),
            "selected_count": len(selected),
            "selected_by_region": dict(sorted(Counter(sample.region_id for sample in selected).items())),
            "selected_sample_ids": [sample.sample_id for sample in selected],
            "rejected_count": len(rejected),
            "rejected_by_region": dict(sorted(rejected_by_region.items())),
            "rejected_by_failure_label": dict(sorted(rejected_by_label.items())),
            "near_joint_branch_failure_count": branch_failures,
            "no_ik_solution_count": sum(1 for item in rejected if item.get("failure_kind") == "no_ik_solution"),
        }
    )
    return selected, report


def run_ik_preflight(
    args: argparse.Namespace,
    output_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
    samples: list[BoardSample],
    candidate_samples: Sequence[BoardSample] | None = None,
) -> int:
    """Check full route IK and generate a reachable dense plan without motion."""

    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    report_path = output_dir / "ik_preflight.json"
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_ik_preflight.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "fixture_profile": str(args.fixture_profile),
        "tool": int(args.tool),
        "user": int(args.user),
        "motion_commands_sent": 0,
        "checks": [],
        "reference_route_checks": [],
        "route_filter": None,
        "status": "started",
    }
    try:
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        joints, current_pose, raw_joints, raw_pose = robot.read_state()
        dock_position_error = position_error_mm(fixture.dock_tcp, current_pose)
        dock_rotation_error = rotation_error_deg(fixture.dock_tcp, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "expected_dock_tcp": list_pose(fixture.dock_tcp),
            "position_error_mm": dock_position_error,
            "rotation_error_deg": dock_rotation_error,
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "set_user_tool_replies": replies,
        }
        if dock_position_error > float(args.dock_position_tolerance_mm) or dock_rotation_error > float(args.dock_rotation_tolerance_deg):
            payload["status"] = "not_seated_at_dock"
            payload["error"] = (
                "Current Tool({}) TCP is {:.3f} mm / {:.3f} deg from the saved dock datum "
                "(limits {:.3f} mm / {:.3f} deg)."
            ).format(
                int(args.tool),
                dock_position_error,
                dock_rotation_error,
                float(args.dock_position_tolerance_mm),
                float(args.dock_rotation_tolerance_deg),
            )
            write_json(report_path, payload)
            print("IK preflight stopped: {}".format(payload["error"]))
            print("IK preflight report: {}".format(report_path))
            return 2

        dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
        dock_exit = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
        dock_exit_joints, dock_exit_check = check_ik_target(robot, "dock_exit", dock_exit, args, joints)
        payload["checks"].append(dock_exit_check)
        if dock_exit_joints is None:
            payload["status"] = "dock_exit_unreachable"
            write_json(report_path, payload)
            print("IK preflight report: {}".format(report_path))
            return 2

        reference_ok = True
        if not args.skip_reference_pad_check:
            reference = reference_route(fixture, dock_design, args)
            reference_targets = tuple(("reference_" + label, pose) for label, pose in reference["outbound"]) + (
                ("reference_contact", reference["contact_tcp"]),
                ("reference_deepest_capture_limit", maximum_capture_tcp(reference, args)),
            ) + tuple(("reference_return_" + label, pose) for label, pose in reference["return"])
            _reference_final, reference_checks = check_ik_sequence(robot, reference_targets, args, joints)
            payload["reference_route_checks"] = reference_checks
            reference_ok = all(check.get("status") == "reachable" for check in reference_checks)

        selected_samples, route_filter = select_ik_reachable_samples(
            robot,
            samples,
            list(candidate_samples) if candidate_samples is not None else samples,
            fixture,
            args,
            (0.0, 0.0, 0.0),
            dock_exit_joints,
        )
        route_filter_path = output_dir / "ik_route_filter.json"
        write_json(route_filter_path, route_filter)
        payload["route_filter"] = {
            key: value for key, value in route_filter.items() if key not in {"candidates", "shared_transit_checks"}
        }
        payload["route_filter"]["report_path"] = str(route_filter_path.name)
        filtered_plan_csv = output_dir / "ik_filtered_sampling_plan.csv"
        write_plan_csv(filtered_plan_csv, selected_samples, fixture, args)
        write_json(
            output_dir / "ik_filtered_sampling_plan.json",
            {
                "schema": "cr3_coverage_board_ik_filtered_plan.v1",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requested_sample_count": len(samples),
                "selected_sample_count": len(selected_samples),
                "ik_filter_report": str(route_filter_path.name),
                "samples": [asdict(sample) for sample in selected_samples],
            },
        )
        payload["status"] = "passed" if reference_ok and route_filter.get("status") in {"complete", "disabled_by_flag"} else "failed"
        write_json(report_path, payload)
        print("IK preflight report: {}".format(report_path))
        print("IK-filtered plan: {}".format(filtered_plan_csv))
        return 0 if payload["status"] == "passed" else 2
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        write_json(report_path, payload)
        print("IK preflight report: {}".format(report_path))
        raise
    finally:
        robot.close()


def run_reseat_dock(args: argparse.Namespace, output_dir: Path, fixture: FixtureTransform) -> int:
    """Return only from the known lifted dock-exit pose to the seated dock."""

    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    report_path = output_dir / "reseat_dock.json"
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_reseat_dock.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "fixture_profile": str(args.fixture_profile),
        "tool": int(args.tool),
        "user": int(args.user),
        "status": "started",
    }
    try:
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        _joints, current_pose, raw_joints, raw_pose = robot.read_state()
        dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
        dock_exit = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
        dock_error = position_error_mm(fixture.dock_tcp, current_pose)
        exit_error = position_error_mm(dock_exit, current_pose)
        dock_rotation_error = rotation_error_deg(fixture.dock_tcp, current_pose)
        exit_rotation_error = rotation_error_deg(dock_exit, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "expected_dock_tcp": list_pose(fixture.dock_tcp),
            "expected_dock_exit_tcp": list_pose(dock_exit),
            "dock_position_error_mm": dock_error,
            "dock_rotation_error_deg": dock_rotation_error,
            "dock_exit_position_error_mm": exit_error,
            "dock_exit_rotation_error_deg": exit_rotation_error,
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "set_user_tool_replies": replies,
        }
        position_limit = float(args.dock_position_tolerance_mm)
        rotation_limit = float(args.dock_rotation_tolerance_deg)
        if dock_error <= position_limit and dock_rotation_error <= rotation_limit:
            payload["status"] = "already_seated"
            write_json(report_path, payload)
            print("TacTip is already seated at the saved dock TCP.")
            print("Reseat report: {}".format(report_path))
            return 0
        if exit_error > position_limit or exit_rotation_error > rotation_limit:
            payload["status"] = "unsafe_start_pose"
            payload["error"] = (
                "Current TCP is neither the saved dock nor the known dock-exit pose; "
                "it is {:.3f} mm / {:.3f} deg from dock-exit (limits {:.3f} / {:.3f})."
            ).format(exit_error, exit_rotation_error, position_limit, rotation_limit)
            write_json(report_path, payload)
            print("Reseat refused: {}".format(payload["error"]))
            print("Reseat report: {}".format(report_path))
            return 2
        payload["motion"] = move_and_verify(robot, "reseat_dock", fixture.dock_tcp, args)
        payload["status"] = "completed"
        write_json(report_path, payload)
        print("TacTip returned to the saved dock TCP.")
        print("Reseat report: {}".format(report_path))
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        write_json(report_path, payload)
        print("Reseat report: {}".format(report_path))
        raise
    finally:
        robot.close()


def run_recover_reference_to_dock(
    args: argparse.Namespace,
    output_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
) -> int:
    """Recover a stopped reference-pad search through its known clear route."""

    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    report_path = output_dir / "reference_recovery.json"
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_reference_recovery.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "fixture_profile": str(args.fixture_profile),
        "tool": int(args.tool),
        "user": int(args.user),
        "status": "started",
    }
    try:
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        _joints, current_pose, raw_joints, raw_pose = robot.read_state()
        reference = reference_route(fixture, dock_design, args)
        reference_contact = np.asarray(reference["contact_tcp"][:3], dtype=float)
        press_axis = np.asarray(reference["press_axis_base"], dtype=float)
        offset = np.asarray(current_pose[:3], dtype=float) - reference_contact
        axial_mm = float(np.dot(offset, press_axis))
        lateral_mm = float(np.linalg.norm(offset - axial_mm * press_axis))
        orientation_error = rotation_error_deg(reference["contact_tcp"], current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "reference_contact_tcp": list_pose(reference["contact_tcp"]),
            "axial_offset_mm": axial_mm,
            "lateral_offset_mm": lateral_mm,
            "rotation_error_deg": orientation_error,
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "set_user_tool_replies": replies,
        }
        # The reference search begins 25 mm above contact and is allowed to go
        # at most 5 mm through it.  A small lateral allowance covers controller
        # tracking error without accepting an arbitrary unknown start pose.
        if not (-float(args.approach_clearance_mm) - 1.0 <= axial_mm <= float(args.max_extra_below_planned_contact_mm) + 1.0) or lateral_mm > 3.0 or orientation_error > float(args.dock_rotation_tolerance_deg):
            payload["status"] = "unsafe_start_pose"
            payload["error"] = (
                "Current TCP is not on the allowed reference-search line: axial {:.3f} mm, "
                "lateral {:.3f} mm, rotation {:.3f} deg."
            ).format(axial_mm, lateral_mm, orientation_error)
            write_json(report_path, payload)
            print("Reference recovery refused: {}".format(payload["error"]))
            print("Reference recovery report: {}".format(report_path))
            return 2

        route_by_label = dict(reference["outbound"])
        recovery_route = (
            ("recover_reference_high", route_by_label["reference_high"]),
            ("recover_dock_high", route_by_label["dock_high"]),
            ("recover_dock_exit", route_by_label["dock_exit"]),
            ("recover_dock_seated", fixture.dock_tcp),
        )
        payload["motions"] = execute_route(robot, recovery_route, "reference recovery", args)
        payload["status"] = "completed"
        write_json(report_path, payload)
        print("Recovered through the reference high route and re-seated TacTip.")
        print("Reference recovery report: {}".format(report_path))
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        write_json(report_path, payload)
        print("Reference recovery report: {}".format(report_path))
        raise
    finally:
        robot.close()


def run_collection(
    args: argparse.Namespace,
    output_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
    samples: list[BoardSample],
    candidate_samples: Sequence[BoardSample] | None = None,
) -> int:
    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    camera = GelSightCapture(parse_camera_source(args.camera_source), args.width, args.height, args.fps, args.camera_read_timeout_sec)
    preprocessor = create_tactip_preprocessor(args, output_dir)
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_collection.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "fixture_profile": str(args.fixture_profile),
        "dock_design": str(args.dock_design),
        "settings": {
            "profile": sampling_label(args),
            "samples_per_tile": int(args.samples_per_tile) if args.samples_per_tile is not None else None,
            "speed_percent": float(args.speed),
            "tool": int(args.tool),
            "user": int(args.user),
            "step_mm": float(args.step_mm),
            "ik_candidate_multiplier": float(args.ik_candidate_multiplier),
        },
        "reference_check": None,
        "ik_route_filter": None,
        "samples": [],
        "status": "started",
    }
    metadata_path = output_dir / "collection.json"
    csv_path = output_dir / "samples.csv"
    report_path = output_dir / "collection_report.html"
    write_json(metadata_path, payload)
    failed = False
    correction = np.zeros(3, dtype=float)
    try:
        if preprocessor is None:
            raise RuntimeError("TacTip preprocessing is required for collection")
        camera.open()
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        joints, current_pose, raw_joints, raw_pose = robot.read_state()
        start_position_error = position_error_mm(fixture.dock_tcp, current_pose)
        start_rotation_error = rotation_error_deg(fixture.dock_tcp, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "expected_dock_tcp": list_pose(fixture.dock_tcp),
            "position_error_mm": start_position_error,
            "rotation_error_deg": start_rotation_error,
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "joint_count": len(joints or ()),
            "set_user_tool_replies": replies,
        }
        if start_position_error > float(args.dock_position_tolerance_mm) or start_rotation_error > float(args.dock_rotation_tolerance_deg):
            raise RuntimeError("CR3 is not seated at the fixture datum: {:.3f} mm / {:.3f} deg away (limits {:.3f} mm / {:.3f} deg). It will not move from an unknown state.".format(start_position_error, start_rotation_error, float(args.dock_position_tolerance_mm), float(args.dock_rotation_tolerance_deg)))
        if not args.skip_reference_pad_check:
            reference = reference_route(fixture, dock_design, args)
            ref_motion = execute_route(robot, reference["outbound"], "reference", args)
            ref_result = search_visual_contact(
                robot,
                camera,
                preprocessor,
                args,
                output_dir,
                "reference_pad",
                reference["approach_tcp"],
                reference["contact_tcp"],
                reference["press_axis_base"],
                0.0,
            )
            ref_result["outbound_motion"] = ref_motion
            ref_result["return_motion"] = execute_route(robot, reference["return"], "reference return", args)
            payload["reference_check"] = ref_result
            write_json(metadata_path, payload)
            if ref_result.get("status") != "contact_found":
                if args.return_to_dock:
                    # A no-contact result has already completed the known high
                    # return route to dock-exit.  Re-seat only in this narrow,
                    # verified case so an unsuccessful first check does not
                    # leave TacTip suspended over the fixture.
                    ref_result["reseat_motion"] = execute_route(
                        robot,
                        (("reseat_dock", fixture.dock_tcp),),
                        "reference no-contact",
                        args,
                    )
                    payload["finish_pose"] = "seated_dock_tcp_after_reference_no_contact"
                    write_json(metadata_path, payload)
                raise RuntimeError("The dock reference pad did not produce stable visual contact; inspect camera/preprocessing and fixture before sampling.")
            correction = np.asarray(ref_result["visual_contact_tcp"][:3], dtype=float) - np.asarray(reference["contact_tcp"][:3], dtype=float)
            correction_local = fixture.local_vector(correction)
            lateral = float(np.linalg.norm(correction_local[:2]))
            vertical = abs(float(correction_local[2]))
            ref_result["correction_base_mm"] = correction.tolist()
            ref_result["correction_tile_mm"] = correction_local.tolist()
            if lateral > float(args.reference_max_lateral_correction_mm) or vertical > float(args.reference_max_vertical_correction_mm):
                raise RuntimeError("Reference-pad correction is unexpectedly large: lateral {:.3f} mm / vertical {:.3f} mm (limits {:.3f} / {:.3f}). Do not sample until dock mounting and Tool(2) are checked.".format(lateral, vertical, float(args.reference_max_lateral_correction_mm), float(args.reference_max_vertical_correction_mm)))
            print("Reference correction tile X/Y/Z = {:+.3f} / {:+.3f} / {:+.3f} mm".format(*correction_local), flush=True)
        else:
            dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
            exit_pose = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
            execute_route(robot, (("dock_exit", exit_pose),), "start", args)
            payload["reference_check"] = {"status": "skipped", "correction_base_mm": correction.tolist()}
        dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
        dock_exit = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
        filter_joints, filter_pose, filter_raw_joints, filter_raw_pose = robot.read_state()
        filter_position_error = position_error_mm(dock_exit, filter_pose)
        filter_rotation_error = rotation_error_deg(dock_exit, filter_pose)
        if (
            filter_position_error > float(args.motion_position_tolerance_mm)
            or filter_rotation_error > float(args.motion_rotation_tolerance_deg)
        ):
            raise RuntimeError(
                "CR3 is not at the verified dock-exit pose before route IK filtering: "
                "{:.3f} mm / {:.3f} deg (limits {:.3f} / {:.3f})."
                .format(
                    filter_position_error,
                    filter_rotation_error,
                    float(args.motion_position_tolerance_mm),
                    float(args.motion_rotation_tolerance_deg),
                )
            )
        route_filter_path = output_dir / "ik_route_filter.json"
        selected_samples, ik_filter = select_ik_reachable_samples(
            robot,
            samples,
            list(candidate_samples) if candidate_samples is not None else samples,
            fixture,
            args,
            correction,
            filter_joints,
        )
        ik_filter["preflight_start_state"] = {
            "actual_tcp": list_pose(filter_pose),
            "expected_dock_exit_tcp": list_pose(dock_exit),
            "position_error_mm": filter_position_error,
            "rotation_error_deg": filter_rotation_error,
            "raw_get_angle": filter_raw_joints,
            "raw_get_pose": filter_raw_pose,
        }
        write_json(route_filter_path, ik_filter)
        payload["ik_route_filter"] = {
            key: value for key, value in ik_filter.items() if key not in {"candidates", "shared_transit_checks"}
        }
        payload["ik_route_filter"]["report_path"] = str(route_filter_path.name)
        filtered_plan_json = output_dir / "ik_filtered_sampling_plan.json"
        filtered_plan_csv = output_dir / "ik_filtered_sampling_plan.csv"
        write_json(
            filtered_plan_json,
            {
                "schema": "cr3_coverage_board_ik_filtered_plan.v1",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requested_sample_count": len(samples),
                "selected_sample_count": len(selected_samples),
                "ik_filter_report": str(route_filter_path.name),
                "samples": [asdict(sample) for sample in selected_samples],
            },
        )
        write_plan_csv(filtered_plan_csv, selected_samples, fixture, args, correction)
        write_json(metadata_path, payload)
        if ik_filter.get("status") not in {"complete", "disabled_by_flag"}:
            raise RuntimeError(
                "Route IK filtering found only {} reachable candidates for {} requested samples. "
                "No board-site MovL was sent. Inspect {} and increase "
                "--ik-candidate-multiplier only after checking the fixture/workspace."
                .format(len(selected_samples), len(samples), route_filter_path)
            )
        samples = selected_samples
        adjusted_csv = output_dir / "runtime_adjusted_sampling_plan.csv"
        write_plan_csv(adjusted_csv, samples, fixture, args, correction)
        print(
            "IK route filter accepted {} / {} requested samples after checking {} candidates. Report: {}"
            .format(len(samples), int(ik_filter["requested_count"]), int(ik_filter.get("checked_candidate_count", 0)), route_filter_path),
            flush=True,
        )
        stopped_after_no_contact = False
        for sample in samples:
            route = make_route(sample, fixture, args, correction)
            print("Begin {} ({}/{})".format(sample.sample_id, sample.index, len(samples)), flush=True)
            outbound = execute_route(robot, route["outbound"][1:] if payload["samples"] or not args.skip_reference_pad_check else route["outbound"][1:], sample.sample_id, args)
            # First regular site begins at dock-exit after the reference check;
            # later sites also end at dock-exit. The safe-transit waypoint is
            # deliberately retained for every site.
            result = search_visual_contact(
                robot,
                camera,
                preprocessor,
                args,
                output_dir,
                sample.sample_id,
                route["approach_tcp"],
                route["contact_tcp"],
                route["press_axis_base"],
                sample.post_contact_depth_mm,
            )
            result["outbound_motion"] = outbound
            result["return_motion"] = execute_route(robot, route["return"], "{} return".format(sample.sample_id), args)
            item = {"sample": asdict(sample), "route": {"contact_tcp": list_pose(route["contact_tcp"]), "approach_tcp": list_pose(route["approach_tcp"]), "press_axis_base": [float(value) for value in route["press_axis_base"]]}, "result": result}
            payload["samples"].append(item)
            write_json(metadata_path, payload)
            if result.get("status") != "captured":
                print("{} ended as {} and returned to dock-exit route.".format(sample.sample_id, result.get("status")), flush=True)
                if not args.continue_on_no_contact:
                    stopped_after_no_contact = True
                    break
            else:
                print("Captured {} and returned high.".format(sample.sample_id), flush=True)
        if args.return_to_dock and not stopped_after_no_contact:
            execute_route(robot, (("reseat_dock", fixture.dock_tcp),), "finish", args)
            payload["finish_pose"] = "seated_dock_tcp"
        else:
            payload["finish_pose"] = "dock_exit_tcp"
        payload["status"] = "stopped_after_no_contact" if stopped_after_no_contact else "completed"
        write_json(metadata_path, payload)
        write_collection_csv(csv_path, payload["samples"])
        build_report(report_path, payload, int(args.report_max_samples))
        print("Collection metadata: {}".format(metadata_path))
        print("Sample table: {}".format(csv_path))
        print("Collection report: {}".format(report_path))
        return 0 if payload["status"] == "completed" else 2
    except Exception as exc:
        failed = True
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        try:
            write_json(metadata_path, payload)
            write_collection_csv(csv_path, payload["samples"])
            build_report(report_path, payload, int(args.report_max_samples))
        except Exception:
            pass
        raise
    finally:
        if preprocessor is not None:
            preprocessor.close()
        camera.close()
        robot.close()
        if failed:
            print("Collection stopped after an error. No automatic recovery move was issued; inspect CR3 state before continuing.", file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    dock_design = load_dock_design(args)
    if args.teach_dock_from_current:
        return teach_dock_profile(args, dock_design)
    manifest, tile, csv_rows = load_board_data(args)
    filtered_rows = filter_tile_rows(tile, csv_rows, args)
    samples = apply_first_contact_test(build_samples(manifest, tile, filtered_rows, args), args)
    ik_candidate_samples = build_ik_candidate_pool(manifest, tile, filtered_rows, args, samples)
    sample_depth_summary = depth_distribution_summary(samples, args)
    candidate_depth_summary = depth_distribution_summary(ik_candidate_samples, args)
    sample_spatial_summary = spatial_coverage_summary(samples)
    candidate_spatial_summary = spatial_coverage_summary(ik_candidate_samples)
    csv_depth_summary = csv_depth_limit_summary(ik_candidate_samples, filtered_rows)
    fixture: FixtureTransform | None = None
    if args.fixture_profile.is_file():
        fixture = fixture_from_profile(read_json(args.fixture_profile), dock_design, args)
    elif args.execute:
        raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
    output_dir = prepare_run_dir(args)
    plan_path = output_dir / "sampling_plan.csv"
    preview_path = output_dir / "planned_sampling_points_3d.html"
    plan_payload = {
        "schema": "cr3_coverage_board_sampling_plan.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "board_manifest": str(next(args.board_dir.glob("*_manifest.json"))),
        "board_manifest_sha256": sha256_file(next(args.board_dir.glob("*_manifest.json"))),
        "dock_design": str(args.dock_design),
        "dock_design_sha256": sha256_file(args.dock_design),
        "fixture_profile": str(args.fixture_profile) if fixture is not None else None,
        "tile": tile,
        "profile": sampling_label(args),
        "samples_per_tile": int(args.samples_per_tile) if args.samples_per_tile is not None else None,
        "seed": int(args.seed),
        "sample_count": len(samples),
        "post_contact_depth_distribution": sample_depth_summary,
        "ik_candidate_depth_distribution": candidate_depth_summary,
        "dense_spatial_layout": effective_dense_spatial_layout(args),
        "dense_region_anchor_count": int(args.dense_region_anchor_count),
        "planned_spatial_coverage": sample_spatial_summary,
        "ik_candidate_spatial_coverage": candidate_spatial_summary,
        "csv_depth_limit_summary": csv_depth_summary,
        "ik_candidate_pool_count": len(ik_candidate_samples),
        "ik_candidate_multiplier": float(args.ik_candidate_multiplier),
        "samples": [asdict(sample) for sample in samples],
        "notes": [
            "Tile-local X/Y/Z are exact coordinates from the generated board manifest and dock geometry.",
            "Exact-count plans use the selected spatial layout and recompute analytical surface height for every planned XY point. The default region_grid uses the footprint-safe central region, not repeated micro-jitter around nine seeds.",
            "A base-frame CR3 pose is emitted only after a valid seated fixture profile is supplied.",
            "The real run performs its own visual reference-pad check and records any accepted translation correction.",
            "An execute run uses controller IK to filter site-high, approach, contact, maximum-capture, and retreat poses after the reference correction. Dense runs select replacements from the deterministic candidate pool.",
            "The post-contact depth protocol is globally distributed across the requested range after visual first contact. CSV depth-limit excess is recorded and requires an explicit execution override.",
        ],
    }
    write_json(output_dir / "sampling_plan.json", plan_payload)
    write_plan_csv(plan_path, samples, fixture, args)
    write_plan_preview(preview_path, tile, args.board_dir, dock_design, samples)
    ik_preview_path: Path | None = None
    if fixture is not None:
        ik_preview_path = output_dir / "ik_candidate_pool_preview.html"
        write_ik_candidate_preview(
            ik_preview_path,
            tile,
            args.board_dir,
            dock_design,
            fixture,
            args,
            samples,
            ik_candidate_samples,
        )
    write_run_readme(
        output_dir / "README.md",
        args,
        args.fixture_profile if fixture is not None else None,
        tuple(float(value) for value in manifest.get("tile_size_mm", (0.0, 0.0))),
    )
    print("Plan directory: {}".format(output_dir))
    print("Tile {}: {} planned tactile samples ({})".format(args.tile, len(samples), sampling_label(args)))
    print("Interactive local plan: {}".format(preview_path))
    if ik_preview_path is not None:
        print("Interactive IK candidate preview: {}".format(ik_preview_path))
    if args.reseat_dock_only:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
        return run_reseat_dock(args, output_dir, fixture)
    if args.recover_reference_to_dock:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
        return run_recover_reference_to_dock(args, output_dir, dock_design, fixture)
    if args.ik_preflight_only:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
        return run_ik_preflight(args, output_dir, dock_design, fixture, samples, ik_candidate_samples)
    if not args.execute:
        if fixture is None:
            print("DRY RUN: no CR3/camera action. No seated profile was supplied, so only tile-local plan coordinates were emitted.")
        else:
            print("DRY RUN: no CR3/camera action. The plan includes CR3 base-frame poses derived from the seated fixture profile.")
        return 0
    if csv_depth_summary["above_csv_limit_count"] and not args.allow_depth_above_csv_limit:
        lower, upper = planned_depth_bounds(args)
        raise ValueError(
            "The requested {:.1f}-{:.1f} mm distribution has {} candidate(s) above the board CSV's "
            "recommended depth limits (maximum excess {:.2f} mm). No CR3 motion was sent. "
            "After physically validating the mounted board at 10 mm, rerun with "
            "--allow-depth-above-csv-limit.".format(
                lower,
                upper,
                int(csv_depth_summary["above_csv_limit_count"]),
                float(csv_depth_summary["max_excess_mm"]),
            )
        )
    assert fixture is not None
    return run_collection(args, output_dir, dock_design, fixture, samples, ik_candidate_samples)


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print("COVERAGE_BOARD_SAMPLER_FAILED: {}: {}".format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
