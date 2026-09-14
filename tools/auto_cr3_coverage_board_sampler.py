#!/usr/bin/env python3
"""Plan and safely collect varied TacTip data from one coverage-board tile.

This script is the real-robot counterpart of the four-tile GAN coverage
board.  It uses the printed calibration dock made by
``design_v4_150mm_tactip_calibration_dock.py`` instead of fitting arbitrary
mesh points for every board placement:

1. Seat Tool(2) TacTip in the dock and store one TCP with
   ``--teach-dock-from-current``.  Docks containing the camera-style physical
   rest crossbar use that contact as an explicit height datum.
2. For a rest-crossbar dock, ``--calibrate-height-from-rest-stop`` records a
   tactile-image reference while the TacTip apex is physically supported by
   the crossbar. Before an execute run, the current dock image must match
   that reference as well as the saved TCP.
3. Leave the dock vertically, travel at the configured safe height directly
   above the target site, and use visual contact only at that target.  The
   former dock reference-pad touch is opt-in via ``--use-reference-pad-check``.
4. Collect varying seed locations, post-contact depths, and tilt angles with
   fresh tactile baselines and two-frame visual-contact confirmation.  With
   ``--continuous-board-transit``, the dock is checked once and consecutive
   sites are joined at the common collision-clear height instead of returning
   to the dock after every sample.

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
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from auto_cr3_gelsight_pair_sampler import GelSightCapture
from auto_cr3_visual_contact_search import (
    ImageFeature,
    capture_record,
    finite_pose,
    list_pose,
    load_feature,
    move_and_verify,
    position_error_mm,
    rotation_error_deg,
    robust_threshold,
)
from design_tactile_gan_coverage_board_modular import surface_height_for_profile
from live_cr3_gelsight_sampler import DobotCR3LiveClient, parse_camera_source
from tactip_runtime_preprocess import add_tactip_preprocess_args, create_tactip_preprocessor


DEFAULT_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
DEFAULT_DOCK_DIR = DEFAULT_BOARD_DIR / "tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm"
DEFAULT_DOCK_DESIGN = DEFAULT_DOCK_DIR / "v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
DEFAULT_RUN_ROOT = DEFAULT_BOARD_DIR / "cr3_coverage_board_runs"
VALID_TILES = ("tile_nw", "tile_ne", "tile_sw", "tile_se")
SUPPORTED_DOCK_SCHEMAS = {
    "coverage_board_dock_geometry_preview.v1",
    "coverage_board_tactip_calibration_dock.v1",
    "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v1",
    "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2",
}
SUPPORTED_FIXTURE_PROFILE_SCHEMAS = {
    "coverage_board_tactip_fixture_profile.v1",
    "coverage_board_tactip_fixture_profile.v2",
    "coverage_board_tactip_fixture_profile.v3",
}
DEFAULT_POST_CONTACT_DEPTH_MIN_MM = 1.0
DEFAULT_POST_CONTACT_DEPTH_MAX_MM = 10.0
DEFAULT_FIRST_CONTACT_MAX_EXTRA_BELOW_NOMINAL_MM = 2.0
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
    calibration_mode = parser.add_mutually_exclusive_group()
    calibration_mode.add_argument(
        "--teach-dock-from-current",
        action="store_true",
        help=(
            "Read the currently seated Tool(2) TCP and write a fixture profile without moving CR3. "
            "For a dock with a rest crossbar this automatically records a rest-stop height datum."
        ),
    )
    calibration_mode.add_argument(
        "--calibrate-height-from-rest-stop",
        action="store_true",
        help=(
            "No-motion height calibration for the raised camera-style rest crossbar. "
            "Seat TacTip until its apex rests on the raised marker, then record a stable Tool(2) TCP "
            "and a repeated tactile-image reference of that crossbar contact."
        ),
    )
    parser.add_argument(
        "--rest-stop-stability-tolerance-mm",
        type=float,
        default=0.30,
        help="Maximum position change between two no-motion rest-stop readings. Default: 0.30 mm.",
    )
    parser.add_argument(
        "--rest-stop-stability-tolerance-deg",
        type=float,
        default=0.30,
        help="Maximum rotation change between two no-motion rest-stop readings. Default: 0.30 deg.",
    )
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
    parser.add_argument(
        "--skip-previews", action="store_true",
        help="Skip STL loading and the three HTML previews. CSV/JSON plans and all robot checks remain enabled.",
    )
    parser.add_argument(
        "--fixture-local-preview", action="store_true",
        help="Draw a geometry-only route in tile coordinates without a robot fixture profile. Offline only; not a CR3 base-frame or IK validation.",
    )
    parser.add_argument("--zero-tilt", action="store_true", help="Use zero tilt for every sample; required for a zero-tilt height calibration.")
    parser.add_argument(
        "--height-measurement", action="store_true",
        help="Single-site first-contact measurement with no extra indentation; requires --first-contact-test. Coarse search then at most 0.02 mm refinement inside the measured bracket. Keeps images.",
    )
    parser.add_argument(
        "--height-calibration", type=Path,
        help="Validated multi-point height calibration JSON bound to this fixture, dock and board. No manual offset or reference-pad correction may be combined with it.",
    )
    parser.add_argument("--board-yaw-deg", type=float, default=0.0, help="One-time fine yaw correction from the keyed dock frame. Keep 0 for the printed fixture orientation.")
    parser.add_argument("--robot-ip", default="192.168.31.88")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--move-port", type=int, default=30003)
    parser.add_argument("--robot-timeout-sec", type=float, default=6.0)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--tool", type=int, default=2, help="CR3 TacTip Tool frame; the fixture was designed for Tool 2.")
    parser.add_argument("--speed", type=float, default=3.0, help="CR3 speed percent; limited to 5 for visual-contact collection.")
    parser.add_argument("--camera-source", default="0")
    # Preserve the 4:3 TacTip optical frame while retaining more marker detail
    # than the camera's legacy 640x480 mode. The runtime preprocessor scales
    # its validated 640x480 geometry to this capture size before making the
    # unchanged 256x256 model input.
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
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
    depth_policy = parser.add_mutually_exclusive_group()
    depth_policy.add_argument(
        "--respect-csv-depth-limits",
        dest="respect_csv_depth_limits",
        action="store_true",
        default=True,
        help=(
            "Constrain every planned indentation to the intersection of the requested range and "
            "that CSV seed site's recommended depth range. This is the default."
        ),
    )
    depth_policy.add_argument(
        "--ignore-csv-depth-limits",
        dest="respect_csv_depth_limits",
        action="store_false",
        help=(
            "Use the global requested depth range for every site. Real execution still requires "
            "--allow-depth-above-csv-limit when a point exceeds its CSV recommendation."
        ),
    )
    parser.add_argument(
        "--allow-depth-above-csv-limit",
        action="store_true",
        help=(
            "Permit --execute when --ignore-csv-depth-limits produces a point above "
            "its CSV recommended maximum. Use only after physically validating the "
            "mounted board and TacTip at those depths."
        ),
    )
    parser.add_argument("--approach-clearance-mm", type=float, default=25.0, help="No-contact clearance above each nominal local surface.")
    parser.add_argument(
        "--board-height-offset-mm", type=float, default=0.0,
        help=(
            "Measured board-surface correction along tile +Z: positive raises the surface, negative lowers it. "
            "Applies only to board contact/approach, not dock, reference pad or safe transit height. "
            "Default 0; validate any correction with a zero-tilt --first-contact-test."
        ),
    )
    parser.add_argument("--safe-height-mm", type=float, default=100.0, help="Fixture-local safe height over the tile bottom plane.")
    parser.add_argument("--dock-exit-lift-mm", type=float, default=65.0, help="Fixture-local lift from seated apex before any lateral travel.")
    parser.add_argument(
        "--max-extra-below-planned-contact-mm",
        type=float,
        default=None,
        help=(
            "Hard visual-search allowance past a nominal board surface. Defaults to 2 mm for "
            "--first-contact-test and 11 mm for normal 1-10 mm batch collection; the automatic default "
            "also covers the explicitly requested indentation plus contact-search margin."
        ),
    )
    parser.add_argument(
        "--contact-search-margin-mm",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed model-surface error below the nominal contact before visual contact is "
            "declared missing. The total deepest capture limit is each sample's requested indentation "
            "plus this margin, capped by --max-extra-below-planned-contact-mm. Default: 1.0 mm; "
            "values above 4 mm require --allow-deep-contact-localization."
        ),
    )
    parser.add_argument(
        "--allow-deep-contact-localization",
        action="store_true",
        help=(
            "Explicitly allow a one-site visual height-localization search with up to 10 mm "
            "surface-error margin. Only valid with --first-contact-test; normal sampling remains capped at 4 mm."
        ),
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
        "--disable-ik-cache",
        action="store_true",
        help=(
            "Disable exact successful IK-query reuse inside one route-filter run. "
            "The default cache includes the full target pose, near-joint branch, User and Tool; "
            "it never replaces the fresh checks before real MovL commands."
        ),
    )
    parser.add_argument(
        "--ik-diagnose-failures",
        action="store_true",
        help=(
            "After a near-joint IK rejection, also ask for an unhinted diagnostic solution. "
            "Off by default to avoid extra controller round trips; a failed near-joint check "
            "always rejects the candidate regardless of this option."
        ),
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
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--skip-reference-pad-check",
        action="store_true",
        default=True,
        help="Skip the dock reference-pad touch. This is the default direct-to-site route.",
    )
    reference_group.add_argument(
        "--use-reference-pad-check",
        dest="skip_reference_pad_check",
        action="store_false",
        help="Opt in to the extra dock reference-pad visual-contact check before sampling.",
    )
    parser.add_argument(
        "--skip-dock-tactile-reference-check",
        action="store_true",
        help=(
            "Bypass the required rest-crossbar tactile-image match before an execute run. "
            "Diagnostic use only: normal collection must start from a TCP and tactile image "
            "that both match the saved dock reference."
        ),
    )
    parser.add_argument("--continue-on-no-contact", action="store_true", help="Return high and continue after a site with no stable visual contact. Default stops safely.")
    parser.add_argument(
        "--continuous-board-transit",
        action="store_true",
        help=(
            "Verify the seated dock once, then remain over the board between samples: retract to "
            "site-high, move directly to the next site-high at --safe-height-mm, and approach the "
            "next contact. Without this flag, every sample returns through dock-high to dock-exit."
        ),
    )
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
    parser.add_argument(
        "--verify-dock-tactile-reference-only",
        action="store_true",
        help=(
            "Read the seated Tool(2) TCP and compare one live TacTip frame against the saved "
            "raised-crossbar reference, without sending any robot movement command."
        ),
    )
    parser.add_argument(
        "--capture-camera-only",
        action="store_true",
        help=(
            "Save one formal TacTip capture using the same median-frame and preprocessing path as "
            "collection. Does not connect to or move the CR3."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Send CR3 motions after all preflight checks pass.")
    parser.add_argument("--yes-i-confirm-cr3-is-safe", action="store_true")
    add_tactip_preprocess_args(parser)
    args = parser.parse_args()
    args._height_calibration = None
    if args.height_measurement:
        args.motion_position_tolerance_mm = min(float(args.motion_position_tolerance_mm), 0.1)
        args.motion_rotation_tolerance_deg = min(float(args.motion_rotation_tolerance_deg), 0.2)
        args.consecutive_hits = max(int(args.consecutive_hits), 3)
        args.save_search_frames = True
    if args.max_extra_below_planned_contact_mm is None:
        requested_depth = float(args.min_post_contact_depth_mm) if args.first_contact_test else float(args.max_post_contact_depth_mm)
        args.max_extra_below_planned_contact_mm = (
            max(
                DEFAULT_FIRST_CONTACT_MAX_EXTRA_BELOW_NOMINAL_MM if args.first_contact_test else DEFAULT_MAX_EXTRA_BELOW_PLANNED_CONTACT_MM,
                requested_depth + float(args.contact_search_margin_mm),
            )
        )
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
    calibration_only = bool(args.teach_dock_from_current or args.calibrate_height_from_rest_stop)
    if args.fixture_local_preview and (
        args.execute or args.ik_preflight_only or calibration_only or args.reseat_dock_only
        or args.recover_reference_to_dock or args.verify_dock_tactile_reference_only or args.capture_camera_only
        or args.height_calibration or args.skip_previews
    ):
        parser.error("--fixture-local-preview is offline geometry only and cannot be combined with hardware modes, calibrated height, or --skip-previews")
    if args.height_calibration and (float(args.board_height_offset_mm) != 0.0 or not args.skip_reference_pad_check or calibration_only):
        parser.error("--height-calibration cannot be combined with manual height offset, reference-pad correction or dock teaching")
    if args.height_measurement and (
        not args.first_contact_test or args.height_calibration or float(args.board_height_offset_mm) != 0.0
        or not args.skip_reference_pad_check
    ):
        parser.error("--height-measurement requires --first-contact-test and an uncorrected board (no height calibration/offset or reference-pad correction)")
    if args.height_measurement and (args.profile == "dense" or len(args.sites or []) != 1):
        parser.error("--height-measurement requires exactly one --site and a non-dense profile for repeatable reference coordinates")
    if args.height_measurement and (calibration_only or args.capture_camera_only or args.fixture_local_preview
        or args.reseat_dock_only or args.recover_reference_to_dock or args.verify_dock_tactile_reference_only):
        parser.error("--height-measurement cannot be combined with teaching, camera-only, geometry-preview or recovery modes")
    if calibration_only and (
        args.execute
        or args.ik_preflight_only
        or args.reseat_dock_only
        or args.recover_reference_to_dock
        or args.verify_dock_tactile_reference_only
        or args.capture_camera_only
    ):
        parser.error("Dock / rest-stop calibration only records a datum; run preflight or collection separately")
    if args.ik_preflight_only and (
        args.execute or args.reseat_dock_only or args.recover_reference_to_dock or args.verify_dock_tactile_reference_only
    ):
        parser.error("--ik-preflight-only cannot be combined with a motion mode")
    if args.reseat_dock_only and not args.execute:
        parser.error("--reseat-dock-only requires --execute and --yes-i-confirm-cr3-is-safe")
    if args.recover_reference_to_dock and not args.execute:
        parser.error("--recover-reference-to-dock requires --execute and --yes-i-confirm-cr3-is-safe")
    if args.reseat_dock_only and args.recover_reference_to_dock:
        parser.error("Use only one recovery mode at a time")
    if args.verify_dock_tactile_reference_only and (
        args.execute or args.reseat_dock_only or args.recover_reference_to_dock
    ):
        parser.error("--verify-dock-tactile-reference-only cannot be combined with a motion mode")
    if args.capture_camera_only and (
        args.execute
        or args.ik_preflight_only
        or args.reseat_dock_only
        or args.recover_reference_to_dock
        or args.verify_dock_tactile_reference_only
    ):
        parser.error("--capture-camera-only cannot be combined with a robot or dock verification mode")
    if args.execute and not args.yes_i_confirm_cr3_is_safe:
        parser.error("--execute requires --yes-i-confirm-cr3-is-safe")
    if args.execute and bool(args.no_tactip_preprocess):
        parser.error("Formal visual-contact collection requires shared TacTip preprocessing; omit --no-tactip-preprocess")
    if args.calibrate_height_from_rest_stop and bool(args.no_tactip_preprocess):
        parser.error("Rest-stop height calibration records a tactile reference image; omit --no-tactip-preprocess")
    if args.capture_camera_only and bool(args.no_tactip_preprocess):
        parser.error("--capture-camera-only writes the formal preprocessing outputs; omit --no-tactip-preprocess")
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
        "rest_stop_stability_tolerance_mm",
        "rest_stop_stability_tolerance_deg",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0:
            parser.error("--{} must be finite and positive".format(name.replace("_", "-")))
    if not math.isfinite(float(args.board_height_offset_mm)):
        parser.error("--board-height-offset-mm must be finite")
    if not math.isfinite(float(args.board_yaw_deg)):
        parser.error("--board-yaw-deg must be finite")
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
    max_contact_search_margin = 10.0 if args.allow_deep_contact_localization else 4.0
    if not 0.0 <= float(args.contact_search_margin_mm) <= max_contact_search_margin:
        parser.error("--contact-search-margin-mm must be in [0, {:.0f}]{}".format(
            max_contact_search_margin,
            " when --allow-deep-contact-localization is set" if args.allow_deep_contact_localization else "",
        ))
    if args.allow_deep_contact_localization and not args.first_contact_test:
        parser.error("--allow-deep-contact-localization is only valid with --first-contact-test")
    if args.allow_deep_contact_localization and not args.skip_reference_pad_check:
        parser.error("Deep contact localization is for one board site only; omit --use-reference-pad-check so its enlarged search budget cannot reach the dock reference pad")
    required_capture_depth = float(args.min_post_contact_depth_mm) if args.first_contact_test else float(args.max_post_contact_depth_mm)
    if float(args.max_extra_below_planned_contact_mm) + 1e-9 < required_capture_depth + float(args.contact_search_margin_mm):
        parser.error(
            "--max-extra-below-planned-contact-mm must cover the deepest requested indentation PLUS "
            "--contact-search-margin-mm; reduce one of those values or set a sufficient hard cap (at most 12 mm)"
        )
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
    if (schema == "coverage_board_dock_geometry_preview.v1" or design.get("geometry_preview_only")) and not args.fixture_local_preview:
        raise ValueError("This STL-derived dock description is geometry-preview only; it cannot teach a TCP, calibrate height, or connect to hardware. Supply the verified real dock design.")
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


def rest_stop_contract(dock_design: dict[str, Any]) -> dict[str, Any]:
    """Validate the physical height datum supplied by a camera-style v2 dock."""
    if dock_design.get("geometry_preview_only") or dock_design.get("schema") == "coverage_board_dock_geometry_preview.v1":
        raise ValueError("A geometry-preview dock cannot serve as a physical height datum")
    rest_stop = dock_design.get("rest_pose_stop")
    if not isinstance(rest_stop, dict):
        raise ValueError(
            "This dock design has no physical rest crossbar. Use --teach-dock-from-current "
            "or select the v2 camera-style-rest-stop dock design."
        )
    local = rest_stop.get("contact_centre_tile_local_mm")
    if not isinstance(local, list) or len(local) != 3 or not np.isfinite(np.asarray(local, dtype=float)).all():
        raise ValueError("Dock rest_pose_stop lacks a finite contact_centre_tile_local_mm")
    top_z = float(rest_stop.get("top_surface_z_mm", float("nan")))
    if not math.isfinite(top_z) or not np.isclose(float(local[2]), top_z, atol=0.01):
        raise ValueError("Dock rest-stop contact local Z does not match its recorded top surface")
    nominal = np.asarray(dict(dock_design["tactip_reference"])["nominal_seated_tool_tcp_local_mm"], dtype=float)
    if not np.allclose(np.asarray(local, dtype=float), nominal, atol=0.01):
        raise ValueError(
            "The rest-stop contact point must equal the dock's nominal seated Tool(2) TCP; "
            "the generated v2 dock should satisfy this exactly."
        )
    return dict(rest_stop)


def fixture_from_profile(profile: dict[str, Any], dock_design: dict[str, Any], args: argparse.Namespace) -> FixtureTransform:
    profile_schema = str(profile.get("schema"))
    if profile_schema not in SUPPORTED_FIXTURE_PROFILE_SCHEMAS:
        raise ValueError("Fixture profile has an unsupported schema")
    if str(profile.get("tile_id")) != args.tile:
        raise ValueError("Fixture profile belongs to {}, not {}".format(profile.get("tile_id"), args.tile))
    if int(profile.get("tool", -1)) != int(args.tool):
        raise ValueError("Fixture profile was taught with Tool({}), not Tool({})".format(profile.get("tool"), args.tool))
    if int(profile.get("user", -1)) != int(args.user):
        raise ValueError("Fixture profile User({}) differs from requested User({}); teach a fresh datum in the selected frame".format(profile.get("user"), args.user))
    expected_design_hash = str(profile.get("dock_design_sha256", ""))
    actual_design_hash = sha256_file(args.dock_design)
    if expected_design_hash != actual_design_hash:
        raise ValueError("Fixture profile belongs to a different dock design; teach a fresh seated datum")
    if "rest_pose_stop" in dock_design:
        rest_stop = rest_stop_contract(dock_design)
        height_calibration = profile.get("height_calibration")
        if not isinstance(height_calibration, dict):
            raise ValueError(
                "This camera-style rest-stop dock requires a height-calibrated profile. Seat TacTip on the crossbar and run "
                "--calibrate-height-from-rest-stop."
            )
        recorded_local = height_calibration.get("rest_stop_contact_tile_local_mm")
        if not isinstance(recorded_local, list) or len(recorded_local) != 3 or not np.allclose(
            np.asarray(recorded_local, dtype=float),
            np.asarray(rest_stop["contact_centre_tile_local_mm"], dtype=float),
            atol=0.01,
        ):
            raise ValueError("Fixture profile rest-stop coordinate does not match this dock design")
    dock_tcp = finite_pose(profile.get("dock_tcp", ()), "fixture dock TCP")
    local = tuple(float(value) for value in dict(dock_design["tactip_reference"])["nominal_seated_tool_tcp_local_mm"])
    dock_rotation = Rotation.from_euler("XYZ", dock_tcp[3:], degrees=True).as_matrix()
    # Tool +Z is the physical press-down axis.  The keyed fixture convention
    # makes tile +X = Tool +X, tile +Y = Tool -Y, and tile +Z = Tool -Z.
    adaptor = np.diag((1.0, -1.0, -1.0))
    board_yaw = float(profile.get("board_yaw_deg", 0.0))
    if not math.isfinite(board_yaw):
        raise ValueError("Fixture profile board_yaw_deg must be finite; teach a fresh datum")
    yaw = Rotation.from_euler("Z", board_yaw, degrees=True).as_matrix()
    tile_to_base = dock_rotation @ adaptor @ yaw
    if not np.isfinite(tile_to_base).all() or np.linalg.det(tile_to_base) < 0.99:
        raise RuntimeError("Fixture axis construction is not a proper rotation")
    return FixtureTransform(
        dock_tcp=dock_tcp,
        dock_tcp_local_mm=local,
        tile_to_base=tile_to_base,
        dock_rotation=dock_rotation,
        board_yaw_deg=float(profile.get("board_yaw_deg", 0.0)),
    )


def read_stable_rest_stop_pose(
    robot: DobotCR3LiveClient,
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float, float, float, float], dict[str, Any]]:
    """Read two stationary poses so height zero is not saved while the arm settles."""
    joints_a, pose_a, raw_joints_a, raw_pose_a = robot.read_state()
    time.sleep(0.25)
    joints_b, pose_b, raw_joints_b, raw_pose_b = robot.read_state()
    first = finite_pose(pose_a, "first rest-stop Tool(2) TCP")
    second = finite_pose(pose_b, "second rest-stop Tool(2) TCP")
    position_delta = position_error_mm(first, second)
    rotation_delta = rotation_error_deg(first, second)
    if position_delta > float(args.rest_stop_stability_tolerance_mm) or rotation_delta > float(args.rest_stop_stability_tolerance_deg):
        raise RuntimeError(
            "TacTip was not stationary on the rest crossbar: {:.3f} mm / {:.3f} deg change between readings "
            "(limits {:.3f} mm / {:.3f} deg). Re-seat it fully, wait for it to settle, and retry."
            .format(
                position_delta,
                rotation_delta,
                float(args.rest_stop_stability_tolerance_mm),
                float(args.rest_stop_stability_tolerance_deg),
            )
        )
    return second, {
        "sample_interval_sec": 0.25,
        "position_delta_mm": position_delta,
        "rotation_delta_deg": rotation_delta,
        "position_tolerance_mm": float(args.rest_stop_stability_tolerance_mm),
        "rotation_tolerance_deg": float(args.rest_stop_stability_tolerance_deg),
        "first_tcp": list_pose(first),
        "second_tcp": list_pose(second),
        "first_raw_get_angle": raw_joints_a,
        "first_raw_get_pose": raw_pose_a,
        "second_raw_get_angle": raw_joints_b,
        "second_raw_get_pose": raw_pose_b,
        "second_joint_count": len(joints_b or ()),
        "first_joint_count": len(joints_a or ()),
    }


def add_model_roi_path(record: dict[str, Any]) -> dict[str, Any]:
    """Persist the model ROI path alongside the shared preprocessing outputs."""
    raw_name = Path(str(record["raw_image"])).name
    record["model_roi"] = "tactip_preprocessed/model_roi/{}".format(raw_name)
    return record


def tactile_texture_similarity(reference: ImageFeature, candidate: ImageFeature) -> dict[str, float]:
    """Compare two rest-stop images after robust per-frame brightness normalization."""
    allowed = reference.allowed_mask & candidate.allowed_mask
    if int(np.count_nonzero(allowed)) < 2000:
        raise RuntimeError("Rest-stop reference and current tactile ROI do not overlap sufficiently")
    reference_values = reference.gray[allowed].astype(np.float32)
    candidate_values = candidate.gray[allowed].astype(np.float32)

    def normalize(values: np.ndarray) -> np.ndarray:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return np.clip((values - median) / max(1.4826 * mad, 1.0), -5.0, 5.0)

    a = normalize(reference_values)
    b = normalize(candidate_values)
    correlation = float(np.corrcoef(a, b)[0, 1]) if float(np.std(a)) > 1.0e-6 and float(np.std(b)) > 1.0e-6 else 0.0
    return {
        "correlation": correlation,
        "normalized_mae": float(np.mean(np.abs(a - b)) / 10.0),
        "shared_roi_pixels": int(np.count_nonzero(allowed)),
    }


def rest_stop_acceptance_from_repeat(repeat_record: dict[str, Any]) -> dict[str, float]:
    """Set a robust image-match acceptance band from stationary crossbar captures."""
    marker_motion = dict(repeat_record.get("marker_motion", {}))
    texture = dict(repeat_record.get("texture_similarity", {}))
    if not marker_motion or not texture:
        raise RuntimeError("Rest-stop tactile repeat capture did not produce comparison statistics")
    return {
        "max_marker_motion_mean_px": max(0.25, 3.0 * float(marker_motion["mean"]) + 0.10),
        "max_marker_motion_p95_px": max(0.75, 3.0 * float(marker_motion["p95"]) + 0.25),
        # The primary seating checks are the physical Tool(2) TCP and local
        # marker motion.  The raw texture includes LED/glass reflections that
        # can drift over minutes even while the seated TacTip is unchanged, so
        # keep correlation as a secondary gross-mismatch guard rather than
        # calibrating an unrealistically narrow sub-minute band.
        "min_texture_correlation": 0.75,
        "max_texture_normalized_mae": max(0.08, 3.0 * float(texture["normalized_mae"]) + 0.04),
    }


def load_rest_stop_reference_feature(profile: dict[str, Any]) -> tuple[dict[str, Any], ImageFeature]:
    reference = profile.get("tactile_rest_stop_reference")
    if not isinstance(reference, dict):
        raise RuntimeError(
            "This rest-stop fixture profile has no tactile crossbar reference. "
            "With TacTip seated on the crossbar, rerun --calibrate-height-from-rest-stop before executing collection."
        )
    root_text = reference.get("reference_root")
    primary = reference.get("primary_capture")
    if not isinstance(root_text, str) or not isinstance(primary, dict):
        raise RuntimeError("Rest-stop tactile reference is incomplete; recalibrate while TacTip is seated on the crossbar")
    root = Path(root_text).expanduser().resolve()
    gray_relative = primary.get("preprocessed_gray")
    roi_relative = primary.get("model_roi")
    if not isinstance(gray_relative, str) or not isinstance(roi_relative, str):
        raise RuntimeError("Rest-stop tactile reference has no processed gray/ROI paths; recalibrate the dock")
    gray_path = root / gray_relative
    roi_path = root / roi_relative
    if not gray_path.is_file() or not roi_path.is_file():
        raise FileNotFoundError("Rest-stop tactile reference files are missing under {}; recalibrate the dock".format(root))
    return reference, load_feature(gray_path, roi_path)


def verify_rest_stop_tactile_reference(
    camera: GelSightCapture,
    args: argparse.Namespace,
    output_dir: Path,
    preprocessor: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Require the currently seated TacTip image to match the saved crossbar contact."""
    if args.skip_dock_tactile_reference_check:
        return {
            "status": "skipped_by_flag",
            "reason": "--skip-dock-tactile-reference-check was supplied",
        }
    reference, baseline = load_rest_stop_reference_feature(profile)
    record, feature, _camera_time = capture_record(
        camera,
        args,
        output_dir,
        preprocessor,
        "dock_crossbar_tactile_verification",
        max(3, int(args.probe_frames)),
        0.0,
        baseline,
    )
    add_model_roi_path(record)
    record["texture_similarity"] = tactile_texture_similarity(baseline, feature)
    acceptance = dict(reference.get("acceptance", {}))
    required = (
        "max_marker_motion_mean_px",
        "max_marker_motion_p95_px",
        "min_texture_correlation",
        "max_texture_normalized_mae",
    )
    if not all(name in acceptance for name in required):
        raise RuntimeError("Rest-stop tactile reference has no acceptance band; recalibrate the dock")
    motion = dict(record["marker_motion"])
    texture = dict(record["texture_similarity"])
    failures: list[str] = []
    if float(motion["mean"]) > float(acceptance["max_marker_motion_mean_px"]):
        failures.append("local marker motion mean {:.3f}px > {:.3f}px".format(float(motion["mean"]), float(acceptance["max_marker_motion_mean_px"])))
    if float(motion["p95"]) > float(acceptance["max_marker_motion_p95_px"]):
        failures.append("local marker motion p95 {:.3f}px > {:.3f}px".format(float(motion["p95"]), float(acceptance["max_marker_motion_p95_px"])))
    if float(texture["correlation"]) < float(acceptance["min_texture_correlation"]):
        failures.append("texture correlation {:.3f} < {:.3f}".format(float(texture["correlation"]), float(acceptance["min_texture_correlation"])))
    if float(texture["normalized_mae"]) > float(acceptance["max_texture_normalized_mae"]):
        failures.append("normalized texture MAE {:.3f} > {:.3f}".format(float(texture["normalized_mae"]), float(acceptance["max_texture_normalized_mae"])))
    record["reference_root"] = str(reference["reference_root"])
    record["acceptance"] = acceptance
    record["status"] = "passed" if not failures else "failed"
    record["failures"] = failures
    return record


def teach_rest_stop_height_profile(args: argparse.Namespace, dock_design: dict[str, Any]) -> int:
    rest_stop = rest_stop_contract(dock_design)
    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    reference_root = (
        args.fixture_profile.parent
        / "tactile_rest_stop_references"
        / "{}_{}".format(args.fixture_profile.stem, time.strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    camera = GelSightCapture(parse_camera_source(args.camera_source), args.width, args.height, args.fps, args.camera_read_timeout_sec)
    preprocessor: Any | None = None
    try:
        reference_root.mkdir(parents=True, exist_ok=False)
        preprocessor = create_tactip_preprocessor(args, reference_root)
        if preprocessor is None:
            raise RuntimeError("Rest-stop height calibration requires TacTip preprocessing")
        camera.open()
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        dock_tcp, stability = read_stable_rest_stop_pose(robot, args)
        primary_capture, primary_feature, camera_time = capture_record(
            camera,
            args,
            reference_root,
            preprocessor,
            "rest_crossbar_reference",
            max(3, int(args.baseline_frames)),
            0.0,
        )
        add_model_roi_path(primary_capture)
        time.sleep(float(args.settle_sec))
        repeat_capture, repeat_feature, _camera_time = capture_record(
            camera,
            args,
            reference_root,
            preprocessor,
            "rest_crossbar_repeat",
            max(3, int(args.probe_frames)),
            camera_time,
            primary_feature,
        )
        add_model_roi_path(repeat_capture)
        repeat_capture["texture_similarity"] = tactile_texture_similarity(primary_feature, repeat_feature)
        acceptance = rest_stop_acceptance_from_repeat(repeat_capture)
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v3",
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
            "height_calibration": {
                "method": "physical_rest_crossbar_two_read_no_motion",
                "rest_stop_contact_tile_local_mm": list(rest_stop["contact_centre_tile_local_mm"]),
                "rest_stop_marker_peak_z_mm": float(rest_stop["top_surface_z_mm"]),
                "distance_below_ring_top_mm": float(rest_stop["distance_below_ring_top_mm"]),
                "measured_tool_tcp_at_crossbar": list_pose(dock_tcp),
                "stability": stability,
                "meaning": (
                    "The TacTip contact apex was physically supported by the printed crossbar. "
                    "This pose is the tile-local Z height datum used for every generated board contact."
                ),
            },
            "tactile_rest_stop_reference": {
                "schema": "coverage_board_tactip_rest_stop_tactile_reference.v1",
                "reference_root": str(reference_root),
                "primary_capture": primary_capture,
                "repeat_capture": repeat_capture,
                "acceptance": acceptance,
                "meaning": (
                    "This is the camera image while the TacTip apex physically contacts the raised dock crossbar. "
                    "A real execute run verifies this image before it leaves the dock, preventing motion if the "
                    "TacTip is merely near the rest pose instead of seated against the crossbar."
                ),
            },
            "set_user_tool_replies": replies,
            "dock_local_tool_tcp_mm": dict(dock_design["tactip_reference"])["nominal_seated_tool_tcp_local_mm"],
            "note": "This command reads stationary Tool(2) poses and captures two tactile images while TacTip rests on the physical crossbar; it does not send a motion command.",
        }
        write_json(args.fixture_profile, profile)
        print("Saved rest-stop height-calibrated fixture profile: {}".format(args.fixture_profile))
        print("Rest-stop tactile reference: {}".format(reference_root))
        print("Tool({}) crossbar TCP: {}".format(args.tool, " ".join("{:.4f}".format(value) for value in dock_tcp)))
        print(
            "Rest-stop tile local XYZ: {} | stable change: {:.3f} mm / {:.3f} deg".format(
                " ".join("{:.3f}".format(float(value)) for value in rest_stop["contact_centre_tile_local_mm"]),
                float(stability["position_delta_mm"]),
                float(stability["rotation_delta_deg"]),
            )
        )
        return 0
    finally:
        if preprocessor is not None:
            preprocessor.close()
        camera.close()
        robot.close()


def teach_dock_profile(args: argparse.Namespace, dock_design: dict[str, Any]) -> int:
    """Teach legacy docks, or automatically use the physical rest stop on v2."""
    if "rest_pose_stop" in dock_design:
        return teach_rest_stop_height_profile(args, dock_design)
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


def depth_bounds_for_seed(row: dict[str, str], args: argparse.Namespace) -> tuple[float, float]:
    """Return the permitted depth interval for one CSV seed site.

    Physical boards have category-specific indentation limits.  The safe
    protocol takes the intersection of those limits and the operator's global
    requested interval, rather than planning a deep sample and refusing it
    only after the robot has been connected.
    """

    requested_lower, requested_upper = planned_depth_bounds(args)
    if not bool(args.respect_csv_depth_limits):
        return requested_lower, requested_upper
    csv_lower = float(row["recommended_depth_min_mm"])
    csv_upper = float(row["recommended_depth_max_mm"])
    lower = max(requested_lower, csv_lower)
    upper = min(requested_upper, csv_upper)
    if lower > upper + 1e-9:
        raise ValueError(
            "Requested {:.1f}-{:.1f} mm has no overlap with {} CSV limit {:.1f}-{:.1f} mm".format(
                requested_lower,
                requested_upper,
                row["site_id"],
                csv_lower,
                csv_upper,
            )
        )
    return float(lower), float(upper)


def uniformly_distributed_depth(
    index: int,
    seed: int,
    stream: int,
    args: argparse.Namespace,
    row: dict[str, str] | None = None,
) -> float:
    """Sample one permitted depth interval with a deterministic low-discrepancy sequence.

    When CSV depth protection is enabled, the interval is adapted to the
    nearest source seed's printed-board limit.  It remains deterministic from
    ``--seed`` and therefore reproducible in the saved plan and after
    controller-IK candidate replacement.
    """

    lower, upper = planned_depth_bounds(args) if row is None else depth_bounds_for_seed(row, args)
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
            # Keep the broad 1--10 mm protocol where the printed geometry
            # permits it, while never exceeding the nearest seed's validated
            # CSV depth limit.
            depth = uniformly_distributed_depth(ordinal, args.seed, 7001 + region_index, args, row)
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
            depth = uniformly_distributed_depth(len(expanded), args.seed, 9001 + row_index, args, row)
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


def board_surface_correction_mm(sample: BoardSample, args: argparse.Namespace) -> float:
    calibration = getattr(args, "_height_calibration", None)
    if calibration is None:
        return float(args.board_height_offset_mm)
    if abs(sample.tilt_x_deg) > 1e-6 or abs(sample.tilt_y_deg) > 1e-6:
        raise ValueError("This height calibration is valid only at zero tilt; use --zero-tilt. Tilted sampling needs independent TCP/apex and orientation validation.")
    from board_height_calibration import height_correction_at
    return height_correction_at(calibration, sample.local_contact_mm[0], sample.local_contact_mm[1])


def make_route(
    sample: BoardSample,
    fixture: FixtureTransform,
    args: argparse.Namespace,
    correction_base_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    local_contact = np.asarray(sample.local_contact_mm, dtype=float).copy()
    height_correction = board_surface_correction_mm(sample, args)
    local_contact[2] += height_correction
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
    # A surface correction must never lower the globally clear transit plane.
    high_correction = correction.copy()
    normal = fixture.tile_to_base[:, 2]
    high_correction -= normal * min(0.0, float(np.dot(high_correction, normal)))
    high_pose = finite_pose(tuple(np.asarray(high_pose[:3]) + high_correction) + tuple(high_pose[3:]), "site high")
    if float(np.dot(np.asarray(high_pose[:3]) - approach_position, normal)) <= 0.0:
        raise ValueError("Safe height must be above the corrected approach; inspect board height, clearance and fixture calibration")
    return {
        "contact_tcp": finite_pose(tuple(contact_position) + orientation, "site contact"),
        "approach_tcp": approach_pose,
        "site_high_tcp": high_pose,
        "press_axis_base": press_axis,
        "board_height_offset_mm": float(args.board_height_offset_mm),
        "calibrated_height_correction_mm": height_correction if getattr(args, "_height_calibration", None) is not None else None,
        "effective_surface_z_mm": float(local_contact[2] + np.dot(correction, normal)),
        "outbound": (("dock_exit", dock_exit), ("dock_high", dock_high), ("site_high", high_pose), ("approach", approach_pose)),
        "return": (("site_high", high_pose), ("dock_high", dock_high), ("dock_exit", dock_exit)),
    }


def maximum_below_nominal_contact_mm(args: argparse.Namespace, post_contact_depth_mm: float) -> float:
    """Return the hard deepest allowance for one tactile sample.

    A planned 6 mm capture may finish at most 7 mm below the nominal printed
    surface with the default 1 mm model/contact margin.  The older global
    allowance remains a second hard ceiling for an operator who deliberately
    configures a smaller value.
    """

    return min(
        float(args.max_extra_below_planned_contact_mm),
        max(0.0, float(post_contact_depth_mm)) + float(args.contact_search_margin_mm),
    )


def maximum_capture_tcp(
    route: dict[str, Any],
    args: argparse.Namespace,
    post_contact_depth_mm: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Return the deepest TCP the visual-contact search could ever command.

    The camera-based search may detect zero contact anywhere between the
    approach pose and the per-sample surface-error margin below the nominal
    surface. Its subsequent requested indentation can never exceed this
    endpoint, so it is the conservative pose to pass to controller IK during
    planning.
    """

    approach = np.asarray(route["approach_tcp"][:3], dtype=float)
    contact = np.asarray(route["contact_tcp"][:3], dtype=float)
    clearance = float(np.linalg.norm(contact - approach))
    if clearance <= 0.0:
        raise ValueError("Site approach and contact TCPs cannot be identical")
    axis = np.asarray(route["press_axis_base"], dtype=float)
    deepest = approach + axis * (clearance + maximum_below_nominal_contact_mm(args, post_contact_depth_mm))
    return finite_pose(
        tuple(deepest.tolist()) + tuple(route["approach_tcp"][3:]),
        "maximum visual-contact capture",
    )


def route_ik_targets(
    route: dict[str, Any],
    args: argparse.Namespace,
    post_contact_depth_mm: float,
) -> tuple[tuple[str, tuple[float, float, float, float, float, float]], ...]:
    """All non-common route poses that must be IK-reachable for one sample."""

    return (
        ("site_high", route["site_high_tcp"]),
        ("approach", route["approach_tcp"]),
        ("planned_contact", route["contact_tcp"]),
        ("deepest_capture_limit", maximum_capture_tcp(route, args, post_contact_depth_mm)),
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
        "local_x_mm", "local_y_mm", "expected_surface_z_mm", "board_height_offset_mm", "calibrated_height_correction_mm", "effective_surface_z_mm", "post_contact_depth_mm", "tilt_x_deg", "tilt_y_deg", "jitter_x_mm", "jitter_y_mm",
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
                "board_height_offset_mm": "{:.4f}".format(args.board_height_offset_mm),
                "calibrated_height_correction_mm": board_surface_correction_mm(sample, args) if getattr(args, "_height_calibration", None) is not None else "",
                "effective_surface_z_mm": "{:.4f}".format(sample.local_contact_mm[2] + board_surface_correction_mm(sample, args)),
                "post_contact_depth_mm": "{:.4f}".format(sample.post_contact_depth_mm),
                "tilt_x_deg": "{:.3f}".format(sample.tilt_x_deg),
                "tilt_y_deg": "{:.3f}".format(sample.tilt_y_deg),
                "jitter_x_mm": "{:.4f}".format(sample.jitter_x_mm),
                "jitter_y_mm": "{:.4f}".format(sample.jitter_y_mm),
            }
            if fixture is not None:
                route = make_route(sample, fixture, args, correction_base_mm)
                row["effective_surface_z_mm"] = "{:.4f}".format(route["effective_surface_z_mm"])
                for prefix, pose in (
                    ("planned_contact_tcp", route["contact_tcp"]),
                    ("planned_approach_tcp", route["approach_tcp"]),
                    ("planned_site_high_tcp", route["site_high_tcp"]),
                    ("planned_deepest_capture_limit_tcp", maximum_capture_tcp(route, args, sample.post_contact_depth_mm)),
                ):
                    for axis, value in zip(("x", "y", "z", "Rx", "Ry", "Rz"), pose):
                        row["{}_{}".format(prefix, axis)] = "{:.8f}".format(value)
            writer.writerow(row)


@lru_cache(maxsize=4)
def _cached_preview_mesh_arrays(
    resolved_path: str,
    modification_time_ns: int,
    file_size: int,
    face_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and thin display geometry once; metadata arguments invalidate changes."""
    import trimesh

    mesh = trimesh.load_mesh(Path(resolved_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Could not load mesh for preview: {}".format(resolved_path))
    vertices, faces = mesh.vertices, mesh.faces
    if len(faces) > face_limit:
        indices = np.linspace(0, len(faces) - 1, face_limit, dtype=int)
        faces = faces[indices]
        used, remap = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used]
        faces = remap.reshape((-1, 3))
    # Keep only the displayed arrays, not an entire large STL's backing data.
    # All previews share these arrays, so accidental in-place edits must fail.
    vertices = np.array(vertices, copy=True)
    faces = np.array(faces, copy=True)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


def load_preview_mesh_arrays(path: Path, face_limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache only visual geometry; never used for route or contact decisions."""
    if int(face_limit) < 1:
        raise ValueError("Preview face limit must be positive")
    resolved = path.resolve()
    metadata = resolved.stat()
    return _cached_preview_mesh_arrays(str(resolved), metadata.st_mtime_ns, metadata.st_size, int(face_limit))


def write_plan_preview(path: Path, tile: dict[str, Any], board_dir: Path, dock_design: dict[str, Any], samples: list[BoardSample]) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for the interactive plan preview") from exc
    tile_path = board_dir / str(tile["stl"])
    vertices, faces = load_preview_mesh_arrays(tile_path, 28000)
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
        dock_vertices, dock_faces = load_preview_mesh_arrays(dock_path, 12000)
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
    if "reference_pad" in dock_design:
        reference = dict(dock_design["reference_pad"])["centre_local_mm"]
        figure.add_trace(go.Scatter3d(x=[reference[0]], y=[reference[1]], z=[reference[2]], mode="markers", name="optional reference pad", marker={"size": 7, "color": "#f6d743", "symbol": "diamond"}, hovertemplate="Optional reference pad<extra></extra>"))
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
    vertices, faces = load_preview_mesh_arrays(tile_path, 28000)
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
        dock_vertices, dock_faces = load_preview_mesh_arrays(dock_path, 12000)
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
                z=[sample.local_contact_mm[2] + board_surface_correction_mm(sample, args) for sample in group],
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
                z=[sample.local_contact_mm[2] + board_surface_correction_mm(sample, args) for sample in reserve],
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
            maximum_capture_tcp(route, args, sample.post_contact_depth_mm),
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


def write_motion_route_preview(
    path: Path,
    tile: dict[str, Any],
    board_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
    args: argparse.Namespace,
    samples: Sequence[BoardSample],
) -> None:
    """Write a full dock-to-contact TCP route preview without contacting CR3.

    The planned visual-contact limit is drawn as a distinct red segment.  It
    is a conservative motion bound, not a claim that the robot should always
    travel that far below the nominal surface.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for the full motion preview") from exc

    tile_path = board_dir / str(tile["stl"])
    vertices, faces = load_preview_mesh_arrays(tile_path, 28000)

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#2a6fbb", opacity=0.34, name="printed tile", hoverinfo="skip", showlegend=False,
        )
    )
    dock_path = Path(str(dock_design.get("stl", "")))
    if dock_path.is_file():
        dock_vertices, dock_faces = load_preview_mesh_arrays(dock_path, 12000)
        figure.add_trace(
            go.Mesh3d(
                x=dock_vertices[:, 0], y=dock_vertices[:, 1], z=dock_vertices[:, 2],
                i=dock_faces[:, 0], j=dock_faces[:, 1], k=dock_faces[:, 2],
                color="#e27024", opacity=0.82, name="calibration dock", hoverinfo="skip", showlegend=False,
            )
        )

    shown_legend_categories: set[str] = set()
    tcp_frame_label = "design-frame TCP" if args.fixture_local_preview else "base TCP"

    def add_route(
        name: str,
        poses: Sequence[tuple[float, float, float, float, float, float]],
        labels: Sequence[str],
        color: str,
        width: float = 5.0,
        category: str = "Transit",
        dash: str = "solid",
    ) -> None:
        local = np.asarray([base_to_tile_local(fixture, pose[:3]) for pose in poses], dtype=float)
        customdata = [
            "{}<br>{}<br>{}: ({:.1f}, {:.1f}, {:.1f}) mm".format(name, label, tcp_frame_label, pose[0], pose[1], pose[2])
            for label, pose in zip(labels, poses)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=local[:, 0], y=local[:, 1], z=local[:, 2], mode="lines+markers", name=category,
                legendgroup=category, showlegend=category not in shown_legend_categories,
                line={"color": color, "width": width, "dash": dash}, marker={"color": color, "size": 4.8},
                customdata=customdata,
                hovertemplate="%{customdata}<br>tile local: (%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>",
            )
        )
        shown_legend_categories.add(category)

    seated = fixture.dock_tcp
    dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
    dock_exit = fixture.pose((dock_local[0], dock_local[1], dock_local[2] + float(args.dock_exit_lift_mm)))
    dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
    if not args.skip_reference_pad_check:
        reference = reference_route(fixture, dock_design, args)
        reference_outbound = dict(reference["outbound"])
        reference_limit = maximum_capture_tcp(reference, args, 0.0)
        add_route(
            "reference-pad clear route",
            (seated, dock_exit, dock_high, reference_outbound["reference_high"], reference["approach_tcp"], reference["contact_tcp"]),
            ("seated TacTip", "vertical dock exit", "dock high", "reference high", "reference approach", "reference nominal contact"),
            "#2563eb",
        )
        add_route(
            "reference visual-search cap (only if contact is not detected earlier)",
            (reference["contact_tcp"], reference_limit),
            ("reference nominal contact", "reference hard search limit"),
            "#e53935",
            4.0,
            category="Press/search bound",
        )

    tactip_radius = float(dock_design.get("tactip_reference", {}).get("soft_tip_through_diameter_mm", 44.0)) / 2.0
    shown_samples = list(samples[:6])
    previous_site_high: tuple[float, float, float, float, float, float] | None = None
    for index, sample in enumerate(shown_samples):
        route = make_route(sample, fixture, args)
        deepest = maximum_capture_tcp(route, args, sample.post_contact_depth_mm)
        color = "#2563eb"
        if args.continuous_board_transit and previous_site_high is not None:
            outbound_poses = (previous_site_high, route["site_high_tcp"], route["approach_tcp"], route["contact_tcp"])
            outbound_labels = ("previous site high", "next site high", "site approach", "planned contact")
        else:
            outbound_poses = (seated, dock_exit, dock_high, route["site_high_tcp"], route["approach_tcp"], route["contact_tcp"])
            outbound_labels = ("seated TacTip", "vertical dock exit", "dock high", "site high", "site approach", "planned contact")
        add_route(
            "{} TCP route".format(sample.sample_id),
            outbound_poses,
            outbound_labels,
            color,
        )
        add_route(
            "{} visual-search cap".format(sample.sample_id),
            (route["contact_tcp"], deepest),
            ("planned contact", "hard visual-search limit"),
            "#e53935",
            3.0,
            category="Press/search bound",
        )
        if args.continuous_board_transit:
            return_poses = (deepest, route["site_high_tcp"])
            return_labels = ("hard visual-search limit", "site retract")
        else:
            return_poses = (deepest, route["site_high_tcp"], dock_high, dock_exit, seated)
            return_labels = ("hard visual-search limit", "site retract", "dock high", "dock exit", "re-seated TacTip")
        add_route(
            "{} conservative return envelope".format(sample.sample_id),
            return_poses,
            return_labels,
            "#0f766e",
            4.0,
            category="Return",
            dash="dash",
        )
        previous_site_high = route["site_high_tcp"]
        contact_local = base_to_tile_local(fixture, route["contact_tcp"][:3])
        angles = np.linspace(0.0, 2.0 * math.pi, 49)
        figure.add_trace(
            go.Scatter3d(
                x=contact_local[0] + tactip_radius * np.cos(angles),
                y=contact_local[1] + tactip_radius * np.sin(angles),
                z=np.full_like(angles, contact_local[2]),
                mode="lines", name="Tip footprint",
                legendgroup="Tip footprint", showlegend=index == 0,
                line={"color": "#7c3aed", "width": 3},
                customdata=[sample.sample_id] * len(angles),
                hovertemplate="%{customdata}<br>Tip footprint<br>tile local: (%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>",
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=[dock_local[0]], y=[dock_local[1]], z=[dock_local[2]], mode="markers", name="design dock datum (not measured)" if args.fixture_local_preview else "saved seated Tool(2) TCP",
            marker={"color": "#111827", "size": 8, "symbol": "diamond"},
            showlegend=False,
            hovertemplate=("Design dock datum (not measured)" if args.fixture_local_preview else "Saved seated Tool(2) TCP") + "<extra></extra>",
        )
    )
    title_suffix = "first {} / {} samples".format(len(shown_samples), len(samples)) if len(samples) > len(shown_samples) else "{} samples".format(len(samples))
    route_mode = "direct-to-site" if args.skip_reference_pad_check else "reference-pad check enabled"
    if args.continuous_board_transit:
        route_mode += " · continuous transit"
    preview_status = "GEOMETRY ONLY · no robot calibration / IK" if args.fixture_local_preview else "Saved fixture · preview only; IK not verified here"
    figure.update_layout(
        title={
            "text": "{} · TCP route preview<br><sup>{} · {}</sup><br><sup>{} · no CR3 commands sent</sup>".format(
                tile["tile_id"], preview_status, title_suffix, route_mode,
            ),
            "x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top", "font": {"size": 20},
        },
        paper_bgcolor="#f7f9fc",
        margin={"l": 0, "r": 0, "t": 126, "b": 60},
        legend={"orientation": "h", "y": -0.06, "x": 0.5, "xanchor": "center", "groupclick": "togglegroup", "font": {"size": 12}},
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


def bounded_search_depths(max_depth_mm: float, step_mm: float) -> list[float]:
    """Visit the exact endpoint with steps no larger than step_mm, never overshoot."""
    if not math.isfinite(max_depth_mm) or not math.isfinite(step_mm) or max_depth_mm <= 0.0 or step_mm <= 0.0:
        raise ValueError("Search depth and step must be finite and positive")
    count = int(math.floor(max_depth_mm / step_mm))
    depths = [index * step_mm for index in range(1, count + 1) if index * step_mm < max_depth_mm]
    depths.append(float(max_depth_mm))
    return depths


def refine_visual_contact_bracket(
    robot: DobotCR3LiveClient,
    camera: GelSightCapture,
    preprocessor: Any,
    args: argparse.Namespace,
    output_dir: Path,
    label: str,
    approach_tcp: Sequence[float],
    press_axis: np.ndarray,
    baseline: ImageFeature,
    threshold: dict[str, float],
    no_contact_tcp: Sequence[float],
    contact_tcp: Sequence[float],
    camera_time: float,
) -> dict[str, Any]:
    """Refine a previously explored contact interval without extending its bound.

    This brackets an image-defined contact event, not a zero-force physical
    surface. Release must be observed against the original unloaded baseline.
    Invalid observations stop this attempt; motion/camera errors propagate
    with ``refinement_record`` attached and never trigger recovery motion.
    """
    result: dict[str, Any] = {
        "status": "unusable", "no_contact_tcp": None, "contact_tcp": None,
        "camera_time": float(camera_time), "frames": [], "motion": [],
    }

    def unusable(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    try:
        approach = finite_pose(approach_tcp, "refinement approach")
        lower_pose = finite_pose(no_contact_tcp, "coarse no-contact TCP")
        upper_pose = finite_pose(contact_tcp, "coarse contact TCP")
        axis = np.asarray(press_axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-8, rtol=0.0):
            return unusable("Refinement press axis must be a finite unit three-vector")
        fine_step = min(0.02, float(getattr(args, "contact_refine_step_mm", 0.02)))
        bracket_limit = min(0.04, float(getattr(args, "contact_refine_max_bracket_mm", 0.04)))
        position_tolerance = min(0.01, float(args.motion_position_tolerance_mm))
        rotation_tolerance = min(0.1, float(args.motion_rotation_tolerance_deg))
        if not all(math.isfinite(value) and value > 0.0 for value in (fine_step, bracket_limit, position_tolerance, rotation_tolerance)):
            return unusable("Refinement step, bracket and motion tolerances must be finite and positive")
        if not math.isfinite(float(camera_time)) or not all(math.isfinite(float(threshold[key])) and float(threshold[key]) > 0.0 for key in ("mean", "p95")):
            return unusable("Refinement needs finite contact thresholds and camera time")
        strict_args = argparse.Namespace(**vars(args))
        strict_args.motion_position_tolerance_mm = position_tolerance
        strict_args.motion_rotation_tolerance_deg = rotation_tolerance
        origin = np.asarray(approach[:3], dtype=float)
        epsilon = 1e-8

        def depth_and_lateral(pose: Sequence[float]) -> tuple[float, float]:
            delta = np.asarray(pose[:3], dtype=float) - origin
            depth = float(np.dot(delta, axis))
            return depth, float(np.linalg.norm(delta - axis * depth))

        lower_depth, lower_lateral = depth_and_lateral(lower_pose)
        upper_depth, upper_lateral = depth_and_lateral(upper_pose)
        if lower_depth < -epsilon or upper_depth <= lower_depth + epsilon:
            return unusable("Coarse feedback does not define a positive interval ahead of the approach")
        if upper_depth - lower_depth > 1.0 + epsilon:
            return unusable("Coarse contact interval exceeds 1 mm; supply consecutive coarse observations")
        if max(lower_lateral, upper_lateral) > position_tolerance + epsilon:
            return unusable("Coarse feedback is not aligned with the refinement press axis")
        if max(rotation_error_deg(approach, lower_pose), rotation_error_deg(approach, upper_pose)) > rotation_tolerance + epsilon:
            return unusable("Coarse feedback orientation differs from the refinement approach")
        result.update({
            "coarse_no_contact_tcp": list_pose(lower_pose), "coarse_contact_tcp": list_pose(upper_pose),
            "coarse_depth_interval_mm": [lower_depth, upper_depth], "fine_step_mm": fine_step,
            "max_bracket_mm": bracket_limit, "motion_position_tolerance_mm": position_tolerance,
        })

        def checked_move(target: Sequence[float], stage: str) -> tuple[tuple[float, ...], float] | None:
            target_pose = finite_pose(target, stage + " target")
            target_depth, _ = depth_and_lateral(target_pose)
            if target_depth < lower_depth - epsilon or target_depth > upper_depth + epsilon:
                result["reason"] = "Refinement target would leave the previously explored interval"
                return None
            try:
                motion = move_and_verify(robot, "{} {}".format(label, stage), target_pose, strict_args)
            except Exception as exc:
                result["motion"].append({"label": stage, "target_tcp": list_pose(target_pose), "status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)})
                raise
            result["motion"].append(motion)
            actual = finite_pose(motion["actual_tcp"], stage + " actual TCP")
            actual_depth, lateral = depth_and_lateral(actual)
            if position_error_mm(target_pose, actual) > position_tolerance + epsilon or rotation_error_deg(target_pose, actual) > rotation_tolerance + epsilon:
                result["reason"] = "Refinement feedback exceeds the tightened motion tolerance"
                return None
            if actual_depth < lower_depth - epsilon or actual_depth > upper_depth + epsilon or lateral > position_tolerance + epsilon:
                result["reason"] = "Refinement feedback left the previously explored axial interval"
                return None
            return actual, actual_depth

        def observe(actual: Sequence[float], depth: float, stage: str) -> bool:
            time.sleep(float(args.settle_sec))
            previous_time = float(result["camera_time"])
            try:
                frame, _feature, next_time = capture_record(
                    camera, args, output_dir, preprocessor, "{}_{}".format(label, stage),
                    max(2, int(args.probe_frames)), previous_time, baseline,
                )
            except Exception as exc:
                result["frames"].append({"refinement_stage": stage, "actual_tcp": list_pose(actual), "depth_from_approach_mm": depth, "status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)})
                raise
            result["frames"].append(frame)
            frame.update({"refinement_stage": stage, "actual_tcp": list_pose(actual), "depth_from_approach_mm": depth})
            if not math.isfinite(float(next_time)) or float(next_time) <= previous_time:
                raise RuntimeError("Contact refinement did not receive a fresh camera observation")
            result["camera_time"] = float(next_time)
            stats = frame["marker_motion"]
            values = [float(stats[key]) for key in ("mean", "p95")]
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise RuntimeError("Contact refinement received invalid marker-motion statistics")
            hit = values[0] >= float(threshold["mean"]) and values[1] >= float(threshold["p95"])
            frame["hit"] = hit
            return hit

        released = checked_move(lower_pose, "refine_release")
        if released is None:
            return result
        released_pose, released_depth = released
        required_hits = max(2, int(args.consecutive_hits))
        for repeat in range(required_hits):
            if observe(released_pose, released_depth, "refine_release_{:02d}".format(repeat + 1)):
                return unusable("TacTip did not release at the coarse no-contact endpoint; refinement stopped")
        result["no_contact_tcp"] = list_pose(released_pose)
        previous_actual_depth = released_depth
        previous_no_contact_depth = released_depth
        remaining = upper_depth - released_depth
        if remaining <= epsilon:
            return unusable("No positive explored interval remains after release")
        for index, relative_depth in enumerate(bounded_search_depths(remaining, fine_step), start=1):
            target_depth = min(upper_depth, released_depth + relative_depth)
            target = tuple(origin + axis * target_depth) + tuple(approach[3:])
            moved = checked_move(target, "refine_step_{:03d}".format(index))
            if moved is None:
                return result
            actual, actual_depth = moved
            if actual_depth <= previous_actual_depth + epsilon:
                return unusable("Refinement feedback did not advance along the press axis")
            previous_actual_depth = actual_depth
            if not observe(actual, actual_depth, "refine_probe_{:03d}".format(index)):
                result["no_contact_tcp"] = list_pose(actual)
                previous_no_contact_depth = actual_depth
                continue
            for repeat in range(1, required_hits):
                if not observe(actual, actual_depth, "refine_confirm_{:03d}_{:02d}".format(index, repeat)):
                    return unusable("Fine contact was not confirmed at the same pose; refinement stopped")
            width = actual_depth - previous_no_contact_depth
            result["observed_contact_tcp"] = list_pose(actual)
            result["actual_bracket_width_mm"] = width
            if width <= epsilon or width > bracket_limit + epsilon:
                return unusable("Confirmed contact does not form a positive bracket within the required width")
            result.update({"status": "refined", "contact_tcp": list_pose(actual)})
            return result
        return unusable("No confirmed fine contact inside the previously explored interval")
    except Exception as exc:
        result["reason"] = "{}: {}".format(type(exc).__name__, exc)
        setattr(exc, "refinement_record", result)
        raise


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
    press_axis = np.asarray(press_axis, dtype=float)
    if press_axis.shape != (3,) or not np.isfinite(press_axis).all() or not np.isclose(np.linalg.norm(press_axis), 1.0):
        raise ValueError("Press axis must be a finite unit vector")
    planned_clearance = float(np.linalg.norm(np.asarray(planned_contact_tcp[:3]) - np.asarray(approach_tcp[:3])))
    if not np.allclose(
        np.asarray(planned_contact_tcp[:3]) - np.asarray(approach_tcp[:3]),
        press_axis * planned_clearance, atol=1e-6,
    ):
        raise ValueError("Planned contact must lie ahead of approach on the press axis")
    maximum_below_nominal = maximum_below_nominal_contact_mm(args, post_contact_depth_mm)
    max_contact_depth = planned_clearance + maximum_below_nominal - float(post_contact_depth_mm)
    if max_contact_depth <= 0.0:
        raise RuntimeError("No contact-search room remains after requested post-contact depth")
    record: dict[str, Any] = {
        "label": label,
        "approach_tcp": list_pose(approach_tcp),
        "planned_contact_tcp": list_pose(planned_contact_tcp),
        "press_axis_base": [float(value) for value in press_axis],
        "planned_approach_to_contact_mm": planned_clearance,
        "post_contact_depth_mm": float(post_contact_depth_mm),
        "maximum_below_nominal_contact_mm": maximum_below_nominal,
        "max_contact_depth_from_approach_mm": max_contact_depth,
        "effective_contact_search_margin_mm": max_contact_depth - planned_clearance,
        "board_height_offset_mm": float(args.board_height_offset_mm) if label != "reference_pad" else 0.0,
        "frames": [],
        "motion": [],
        "status": "started",
    }
    time.sleep(float(args.settle_sec))
    baseline, camera_time, threshold, frames = prepare_site_baseline(camera, args, output_dir, preprocessor, label)
    record["frames"].extend(frames)
    record["threshold"] = threshold
    print("{} thresholds: mean={:.3f}, p95={:.3f}".format(label, threshold["mean"], threshold["p95"]), flush=True)
    print(
        "{} contact budget: approach={:.2f} mm, first-contact margin={:.2f} mm, indentation={:.2f} mm, total below-surface cap={:.2f} mm"
        .format(label, planned_clearance, max_contact_depth - planned_clearance, post_contact_depth_mm, maximum_below_nominal),
        flush=True,
    )
    first_hit: dict[str, Any] | None = None
    consecutive_hits = 0
    last_motion: dict[str, Any] | None = None
    last_depth: float | None = None
    last_no_contact_tcp: list[float] | None = None
    depths = bounded_search_depths(max_contact_depth, float(args.step_mm))
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
        frame["relative_to_planned_nominal_mm"] = float(depth - planned_clearance)
        frame["actual_tcp"] = motion["actual_tcp"]
        stats = frame["marker_motion"]
        hit = float(stats["mean"]) >= float(threshold["mean"]) and float(stats["p95"]) >= float(threshold["p95"])
        frame["hit"] = bool(hit)
        record["frames"].append(frame)
        if not args.save_search_frames:
            discard_intermediate_capture(output_dir, frame)
        print(
            "{} travel_from_approach={:.2f} mm relative_to_nominal={:+.2f} mm mean={:.3f} p95={:.3f} hit={}".format(
                label,
                float(depth),
                float(depth - planned_clearance),
                float(stats["mean"]),
                float(stats["p95"]),
                bool(hit),
            ),
            flush=True,
        )
        if hit:
            first_hit = frame
            consecutive_hits = 1
            # Confirm at this same pose; descending to confirm both shifts the
            # first-contact datum and can exhaust the search budget at its end.
            for confirm_index in range(1, int(args.consecutive_hits)):
                time.sleep(float(args.settle_sec))
                confirm, _feature, camera_time = capture_record(
                    camera, args, output_dir, preprocessor,
                    "{}_confirm_{:03d}_{:02d}".format(label, index, confirm_index),
                    int(args.probe_frames), camera_time, baseline,
                )
                confirm["search_depth_mm"] = float(depth)
                confirm["relative_to_planned_nominal_mm"] = float(depth - planned_clearance)
                confirm["actual_tcp"] = motion["actual_tcp"]
                confirm["stationary_confirmation"] = True
                stats = confirm["marker_motion"]
                confirm_hit = float(stats["mean"]) >= float(threshold["mean"]) and float(stats["p95"]) >= float(threshold["p95"])
                confirm["hit"] = bool(confirm_hit)
                record["frames"].append(confirm)
                if not args.save_search_frames:
                    discard_intermediate_capture(output_dir, confirm)
                if not confirm_hit:
                    first_hit = None
                    consecutive_hits = 0
                    break
                consecutive_hits += 1
            if consecutive_hits >= int(args.consecutive_hits):
                break
        else:
            first_hit = None
            consecutive_hits = 0
            last_no_contact_tcp = list_pose(motion["actual_tcp"])
    if last_motion is not None and last_depth is not None:
        actual_depth = float(np.dot(np.asarray(last_motion["actual_tcp"][:3]) - np.asarray(approach_tcp[:3]), press_axis))
        record["last_search_tcp"] = last_motion["actual_tcp"]
        record["last_commanded_search_depth_mm"] = last_depth
        record["last_actual_depth_from_approach_mm"] = actual_depth
        record["last_actual_relative_to_nominal_mm"] = actual_depth - planned_clearance
    if first_hit is None or consecutive_hits < int(args.consecutive_hits):
        reason = (
            "No stable marker-motion contact. Searched {:.2f} mm past the corrected model surface; "
            "last actual offset was {:+.2f} mm. Check the measured board height, seated datum, User/Tool and tactile frames. "
            "The post-contact indentation is not extra first-contact search travel."
        ).format(max_contact_depth - planned_clearance, float(record.get("last_actual_relative_to_nominal_mm", float("nan"))))
        record.update({"status": "no_contact", "reason": reason})
        print("{}: {}".format(label, reason), flush=True)
        return record
    visual_tcp = finite_pose(first_hit["actual_tcp"], "visual-contact TCP")
    record["coarse_contact_depth_from_approach_mm"] = float(first_hit["search_depth_mm"])
    if getattr(args, "height_measurement", False):
        if last_no_contact_tcp is None:
            record.update({"status": "measurement_unusable", "reason": "No observed unloaded endpoint before contact; no height bracket can be inferred."})
            return record
        try:
            refinement = refine_visual_contact_bracket(
                robot, camera, preprocessor, args, output_dir, label, approach_tcp,
                press_axis, baseline, threshold, last_no_contact_tcp, visual_tcp, camera_time,
            )
        except Exception as exc:
            record["contact_refinement"] = getattr(exc, "refinement_record", {"status": "failed"})
            record["status"] = "failed"
            setattr(exc, "contact_search_record", record)
            raise
        record["contact_refinement"] = refinement
        record["frames"].extend(refinement["frames"])
        record["motion"].extend(refinement["motion"])
        if refinement["status"] != "refined":
            record.update({"status": "measurement_unusable", "reason": refinement.get("reason", "Fine contact bracket could not be verified")})
            return record
        visual_tcp = finite_pose(refinement["contact_tcp"], "refined visual-contact TCP")
        last_no_contact_tcp = refinement["no_contact_tcp"]
        camera_time = float(refinement["camera_time"])
    record["first_contact_bracket"] = (
        {
            "no_contact_tcp": last_no_contact_tcp,
            "contact_tcp": list_pose(visual_tcp),
            "detection": "visual_marker_threshold",
            "note": "Brackets the visual detection threshold, not guaranteed physical zero contact.",
        }
        if last_no_contact_tcp is not None else None
    )
    visual_depth = float(np.dot(np.asarray(visual_tcp[:3]) - np.asarray(approach_tcp[:3]), press_axis))
    record["commanded_contact_depth_from_approach_mm"] = float(first_hit["search_depth_mm"])
    if getattr(args, "height_measurement", False):
        record["commanded_contact_depth_from_approach_mm"] = float(np.dot(np.asarray(refinement["motion"][-1]["target_tcp"][:3]) - np.asarray(approach_tcp[:3]), press_axis))
    capture_depth = visual_depth + float(post_contact_depth_mm)
    if capture_depth > planned_clearance + maximum_below_nominal + 1e-6:
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
                "post_contact_depth_mm": run.get("post_contact_depth_mm", sample["post_contact_depth_mm"]),
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
body{{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}}header{{padding:20px 28px;background:#182232}}h1{{margin:0;font-size:23px}}.summary{{padding:14px 28px;color:#b7c5d6}}.grid{{padding:0 28px 28px;display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}}article{{background:#182232;border:1px solid #33475f;border-radius:7px;padding:13px}}article h2{{margin:0;font-size:16px}}.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.images figure{{margin:0;background:#0d141e;padding:6px}}.images figcaption{{font-size:11px;color:#cbd7e5;margin-bottom:5px}}.images img{{width:100%;display:block;background:#000;image-rendering:pixelated}}
</style></head><body><header><h1>Coverage-board CR3 tactile collection</h1></header><div class=\"summary\">Tile: <b>{}</b>. Status: <b>{}</b>. Captured: <b>{}</b> / {}. Every capture used a fresh visual-contact baseline.</div><main class=\"grid\">{}</main></body></html>""".format(
        html.escape(str(payload.get("tile_id", ""))),
        html.escape(str(payload.get("status", ""))),
        captured,
        len(payload.get("samples", [])),
        "\n".join(cards),
    )
    path.write_text(page, encoding="utf-8")


def write_run_readme(path: Path, args: argparse.Namespace, fixture_profile: Path | None, tile_size_mm: Sequence[float]) -> None:
    depth_policy = (
        "uniformly distributed within each seed's CSV-safe intersection of the requested range"
        if args.respect_csv_depth_limits
        else "uniformly distributed across the global requested range (CSV limits ignored)"
    )
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
- In continuous-board mode, the seated TCP and tactile rest reference are
  checked once. Each sample then retracts to site-high and travels directly to
  the next site-high; it does not revisit the dock between samples.
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
- Final indentation after visual contact: {} (`{:.1f}-{:.1f} mm` requested)
- Visual-contact search margin below nominal surface: `{:.1f} mm` (also capped by the hard depth limit)
- Board surface height correction (tile +Z): `{:+.3f} mm`; dock and safe transit height are unchanged.
- Inter-sample route: `{}`
- Missing-contact policy: `{}`
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
        depth_policy,
        float(args.min_post_contact_depth_mm),
        float(args.max_post_contact_depth_mm),
        float(args.contact_search_margin_mm),
        float(args.board_height_offset_mm),
        "continuous site-high transit" if args.continuous_board_transit else "dock-high / dock-exit after every sample",
        "record, retract high, and continue" if args.continue_on_no_contact else "record, retract high, and stop",
        fixture_profile or "not supplied (offline plan only)",
    )
    if args.skip_previews:
        text += "\nHTML previews were skipped with --skip-previews. CSV/JSON plans and controller checks remain enabled.\n"
    path.write_text(text, encoding="utf-8")


def check_ik_target(
    robot: DobotCR3LiveClient,
    label: str,
    target: Sequence[float],
    args: argparse.Namespace,
    joint_near: Sequence[float] | None,
    context: dict[str, Any] | None = None,
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    """Check one controller IK target and preserve branch diagnostics.

    ``InverseSolution(..., 1, {near joints})`` is intentionally the acceptance
    criterion because ``move_and_verify`` will use that same branch hint before
    it sends a real MovL.  An optional no-hint query is diagnostic only: it tells
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
    started = time.perf_counter()
    # A cache lives only for this no-motion filtering call. Do not round poses
    # or joints: even the same TCP requires a new query with a different branch
    # hint, including the return from the deepest capture to site-high.
    cache_key = (tuple(pose), near, int(args.user), int(args.tool))
    cache = context.get("cache") if context is not None else None
    if context is not None:
        context["target_checks"] += 1
    if cache is not None and cache_key in cache:
        solution, raw = cache[cache_key]
        context["cache_hits"] += 1
        record.update(
            {
                "status": "reachable",
                "inverse_joint_solution_deg": list(solution),
                "raw_reply": raw,
                "ik_source": "exact_query_cache",
                "ik_elapsed_sec": time.perf_counter() - started,
            }
        )
        return solution, record
    try:
        if context is not None:
            context["controller_calls"] += 1
        solution, raw = robot.inverse_solution(pose, args.user, args.tool, joint_near=near)
    except (TimeoutError, ConnectionError, OSError):
        # A transport failure is not evidence that one candidate is unreachable.
        # Stop instead of multiplying the timeout across the remaining pool.
        raise
    except Exception as near_exc:
        if isinstance(near_exc, RuntimeError) and str(near_exc).startswith("No reply for command:"):
            # The bundled CR3 client reports a closed/no-response dashboard
            # connection using this exact RuntimeError prefix.
            raise
        record["near_hint_error"] = "{}: {}".format(type(near_exc).__name__, near_exc)
        if near is None or not bool(getattr(args, "ik_diagnose_failures", False)):
            record["status"] = "unreachable"
            record["failure_kind"] = "no_ik_solution" if near is None else "near_hint_failed_unclassified"
            record["no_hint_diagnostic"] = "not_applicable" if near is None else "disabled"
            record["ik_elapsed_sec"] = time.perf_counter() - started
            return None, record
        try:
            if context is not None:
                context["controller_calls"] += 1
                context["diagnostic_calls"] += 1
            fallback_solution, fallback_raw = robot.inverse_solution(pose, args.user, args.tool, joint_near=None)
        except (TimeoutError, ConnectionError, OSError):
            raise
        except Exception as fallback_exc:
            if isinstance(fallback_exc, RuntimeError) and str(fallback_exc).startswith("No reply for command:"):
                raise
            record["status"] = "unreachable"
            record["failure_kind"] = "no_ik_solution"
            record["no_hint_error"] = "{}: {}".format(type(fallback_exc).__name__, fallback_exc)
            record["ik_elapsed_sec"] = time.perf_counter() - started
            return None, record
        record.update(
            {
                "status": "branch_hint_failed",
                "failure_kind": "near_joint_branch",
                "no_hint_solution_joints_deg": list(fallback_solution),
                "no_hint_raw_reply": fallback_raw,
                "ik_elapsed_sec": time.perf_counter() - started,
            }
        )
        return None, record
    record.update(
        {
            "status": "reachable",
            "inverse_joint_solution_deg": list(solution),
            "raw_reply": raw,
            "ik_source": "controller",
            "ik_elapsed_sec": time.perf_counter() - started,
        }
    )
    solution = tuple(float(value) for value in solution)
    if cache is not None:
        cache[cache_key] = (solution, raw)
    return solution, record


def check_ik_sequence(
    robot: DobotCR3LiveClient,
    targets: Sequence[tuple[str, tuple[float, float, float, float, float, float]]],
    args: argparse.Namespace,
    joint_near: Sequence[float] | None,
    context: dict[str, Any] | None = None,
) -> tuple[tuple[float, ...] | None, list[dict[str, Any]]]:
    """Solve a route in order, carrying each solution to the next target."""

    current_near = tuple(float(value) for value in joint_near) if joint_near is not None else None
    checks: list[dict[str, Any]] = []
    for index, (label, target) in enumerate(targets):
        solution, record = check_ik_target(robot, label, target, args, current_near, context)
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
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete contact-and-retract portion of one candidate."""

    route = make_route(sample, fixture, args, correction_base_mm)
    targets = route_ik_targets(route, args, sample.post_contact_depth_mm)
    final_joints, checks = check_ik_sequence(robot, targets, args, dock_high_joints, context)
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
            "deepest_capture_limit_tcp": list_pose(maximum_capture_tcp(route, args, sample.post_contact_depth_mm)),
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

    started = time.perf_counter()
    context: dict[str, Any] = {
        "cache": None if bool(getattr(args, "disable_ik_cache", False)) else {},
        "target_checks": 0,
        "controller_calls": 0,
        "diagnostic_calls": 0,
        "cache_hits": 0,
    }
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
        "exact_query_cache_enabled": context["cache"] is not None,
        "failure_diagnostics_enabled": bool(getattr(args, "ik_diagnose_failures", False)),
    }
    last_progress = started

    def update_progress(selected_count: int, *, final: bool = False) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        elapsed = now - started
        report["performance"] = {
            "elapsed_sec": elapsed,
            "target_check_count": context["target_checks"],
            "controller_ik_call_count": context["controller_calls"],
            "diagnostic_ik_call_count": context["diagnostic_calls"],
            "cache_hit_count": context["cache_hits"],
            "cache_entry_count": len(context["cache"]) if context["cache"] is not None else 0,
        }
        if not final and now - last_progress < 2.0:
            return
        last_progress = now
        eta = elapsed * max(0, len(requested_samples) - selected_count) / selected_count if selected_count else None
        print(
            "IK preflight: accepted {}/{}; checked {}; rejected {}; controller calls {}; "
            "cache hits {}; elapsed {:.1f}s; estimated remaining {}".format(
                selected_count,
                len(requested_samples),
                len(report["candidates"]),
                len(report["candidates"]) - selected_count,
                context["controller_calls"],
                context["cache_hits"],
                elapsed,
                "{:.1f}s".format(eta) if eta is not None else "unknown",
            ),
            flush=True,
        )

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

    print(
        "IK preflight started: {} requested / {} candidates; five branch-aware targets per "
        "candidate; exact-query cache {}; extra failure diagnostics {}.".format(
            len(requested_samples),
            len(candidate_samples),
            "on" if context["cache"] is not None else "off",
            "on" if report["failure_diagnostics_enabled"] else "off",
        ),
        flush=True,
    )
    dock_local = np.asarray(fixture.dock_tcp_local_mm, dtype=float)
    dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
    dock_high_joints, shared_checks = check_ik_sequence(
        robot,
        (("dock_high", dock_high),),
        args,
        current_joints,
        context,
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
        update_progress(0, final=True)
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
            context,
        )
        report["candidates"].append(candidate_record)
        if candidate_record["status"] != "accepted":
            update_progress(len(accepted))
            continue
        accepted.append(candidate)
        accepted_by_region[region_id] += 1
        update_progress(len(accepted))
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
            "unclassified_near_hint_failure_count": sum(
                1 for item in rejected if item.get("failure_kind") == "near_hint_failed_unclassified"
            ),
        }
    )
    update_progress(len(selected), final=True)
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
                ("reference_deepest_capture_limit", maximum_capture_tcp(reference, args, 0.0)),
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
    """Return only from the known dock-exit or dock-high pose to the seated dock."""

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
        dock_high = fixture.pose((dock_local[0], dock_local[1], float(args.safe_height_mm)))
        dock_error = position_error_mm(fixture.dock_tcp, current_pose)
        exit_error = position_error_mm(dock_exit, current_pose)
        high_error = position_error_mm(dock_high, current_pose)
        dock_rotation_error = rotation_error_deg(fixture.dock_tcp, current_pose)
        exit_rotation_error = rotation_error_deg(dock_exit, current_pose)
        high_rotation_error = rotation_error_deg(dock_high, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "expected_dock_tcp": list_pose(fixture.dock_tcp),
            "expected_dock_exit_tcp": list_pose(dock_exit),
            "expected_dock_high_tcp": list_pose(dock_high),
            "dock_position_error_mm": dock_error,
            "dock_rotation_error_deg": dock_rotation_error,
            "dock_exit_position_error_mm": exit_error,
            "dock_exit_rotation_error_deg": exit_rotation_error,
            "dock_high_position_error_mm": high_error,
            "dock_high_rotation_error_deg": high_rotation_error,
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
        at_dock_exit = exit_error <= position_limit and exit_rotation_error <= rotation_limit
        at_dock_high = high_error <= position_limit and high_rotation_error <= rotation_limit
        if not at_dock_exit and not at_dock_high:
            payload["status"] = "unsafe_start_pose"
            payload["error"] = (
                "Current TCP is neither the saved dock, dock-exit, nor dock-high pose; "
                "it is {:.3f} mm / {:.3f} deg from dock-exit and {:.3f} mm / {:.3f} deg from dock-high "
                "(limits {:.3f} / {:.3f})."
            ).format(
                exit_error,
                exit_rotation_error,
                high_error,
                high_rotation_error,
                position_limit,
                rotation_limit,
            )
            write_json(report_path, payload)
            print("Reseat refused: {}".format(payload["error"]))
            print("Reseat report: {}".format(report_path))
            return 2
        if at_dock_high:
            payload["motion"] = execute_route(
                robot,
                (("recover_dock_exit", dock_exit), ("reseat_dock", fixture.dock_tcp)),
                "dock-high recovery",
                args,
            )
        else:
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
        "fixture_profile_sha256": sha256_file(args.fixture_profile),
        "dock_design_sha256": sha256_file(args.dock_design),
        "board_manifest_sha256": sha256_file(next(args.board_dir.glob("*_manifest.json"))),
        "settings": {
            "profile": sampling_label(args),
            "samples_per_tile": int(args.samples_per_tile) if args.samples_per_tile is not None else None,
            "speed_percent": float(args.speed),
            "tool": int(args.tool),
            "user": int(args.user),
            "step_mm": float(args.step_mm),
            "board_height_offset_mm": float(args.board_height_offset_mm),
            "height_calibration": str(args.height_calibration) if args.height_calibration else None,
            "height_calibration_sha256": sha256_file(args.height_calibration) if args.height_calibration else None,
            "height_measurement": bool(args.height_measurement),
            "contact_search_margin_mm": float(args.contact_search_margin_mm),
            "max_extra_below_planned_contact_mm": float(args.max_extra_below_planned_contact_mm),
            "ik_candidate_multiplier": float(args.ik_candidate_multiplier),
            "reference_pad_check_enabled": not bool(args.skip_reference_pad_check),
            "dock_tactile_reference_check_enabled": not bool(args.skip_dock_tactile_reference_check),
            "continuous_board_transit": bool(args.continuous_board_transit),
        },
        "reference_check": None,
        "dock_tactile_reference_check": None,
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
        if "rest_pose_stop" in dock_design:
            fixture_profile = read_json(args.fixture_profile)
            dock_tactile_check = verify_rest_stop_tactile_reference(
                camera,
                args,
                output_dir,
                preprocessor,
                fixture_profile,
            )
            payload["dock_tactile_reference_check"] = dock_tactile_check
            write_json(metadata_path, payload)
            if dock_tactile_check.get("status") not in {"passed", "skipped_by_flag"}:
                details = "; ".join(str(value) for value in dock_tactile_check.get("failures", []))
                raise RuntimeError(
                    "TacTip is at the saved TCP but its live image does not match the saved raised-crossbar "
                    "contact. It will not leave the dock. {}".format(details or "Re-seat TacTip on the crossbar and recalibrate if needed.")
                )
            print(
                "Dock tactile reference {}: marker mean={:.3f}px p95={:.3f}px texture corr={:.3f}".format(
                    dock_tactile_check.get("status"),
                    float(dict(dock_tactile_check.get("marker_motion", {})).get("mean", 0.0)),
                    float(dict(dock_tactile_check.get("marker_motion", {})).get("p95", 0.0)),
                    float(dict(dock_tactile_check.get("texture_similarity", {})).get("correlation", 0.0)),
                ),
                flush=True,
            )
        else:
            payload["dock_tactile_reference_check"] = {
                "status": "not_required",
                "reason": "Dock design has no raised rest crossbar.",
            }
            write_json(metadata_path, payload)
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
        last_route: dict[str, Any] | None = None
        for sample_number, sample in enumerate(samples):
            route = make_route(sample, fixture, args, correction)
            print("Begin {} ({}/{})".format(sample.sample_id, sample.index, len(samples)), flush=True)
            if args.continuous_board_transit and sample_number > 0:
                # The previous sample has already retracted to its site-high
                # pose.  Move directly between site-high poses, which all use
                # the shared fixture-local safe height, before descending at
                # the new XY location.  ``move_and_verify`` still performs a
                # fresh controller-side IK check for both waypoints.
                outbound_route = (
                    ("next_site_high", route["site_high_tcp"]),
                    ("approach", route["approach_tcp"]),
                )
            else:
                # Collection starts at dock-exit after the one-time seated
                # dock gate.  Legacy mode repeats this dock-high leg for every
                # sample because each sample also returns to dock-exit.
                outbound_route = route["outbound"][1:]
            outbound = execute_route(robot, outbound_route, sample.sample_id, args)
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
                0.0 if args.height_measurement else sample.post_contact_depth_mm,
            )
            result["outbound_motion"] = outbound
            if args.continuous_board_transit:
                return_route = (("site_high", route["site_high_tcp"]),)
            else:
                return_route = route["return"]
            result["return_motion"] = execute_route(robot, return_route, "{} return".format(sample.sample_id), args)
            item = {"sample": asdict(sample), "route": {"contact_tcp": list_pose(route["contact_tcp"]), "approach_tcp": list_pose(route["approach_tcp"]), "press_axis_base": [float(value) for value in route["press_axis_base"]], "board_height_offset_mm": route["board_height_offset_mm"], "calibrated_height_correction_mm": route["calibrated_height_correction_mm"], "effective_surface_z_mm": route["effective_surface_z_mm"]}, "result": result}
            payload["samples"].append(item)
            last_route = route
            write_json(metadata_path, payload)
            expected_status = "contact_found" if args.height_measurement else "captured"
            if result.get("status") != expected_status:
                finish_name = "site-high" if args.continuous_board_transit else "dock-exit route"
                print("{} ended as {} and returned to {}.".format(sample.sample_id, result.get("status"), finish_name), flush=True)
                if not args.continue_on_no_contact:
                    stopped_after_no_contact = True
                    break
            else:
                print("{}: {} and returned high.".format(sample.sample_id, expected_status), flush=True)
        if args.return_to_dock and not stopped_after_no_contact:
            if args.continuous_board_transit and last_route is not None:
                return_by_label = dict(last_route["return"])
                finish_route = (
                    ("dock_high", return_by_label["dock_high"]),
                    ("dock_exit", return_by_label["dock_exit"]),
                    ("reseat_dock", fixture.dock_tcp),
                )
            else:
                finish_route = (("reseat_dock", fixture.dock_tcp),)
            execute_route(robot, finish_route, "finish", args)
            payload["finish_pose"] = "seated_dock_tcp"
        else:
            payload["finish_pose"] = "last_site_high_tcp" if args.continuous_board_transit else "dock_exit_tcp"
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
        if hasattr(exc, "contact_search_record"):
            payload["failed_contact_search"] = exc.contact_search_record
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


def run_verify_dock_tactile_reference(
    args: argparse.Namespace,
    output_dir: Path,
    dock_design: dict[str, Any],
    fixture: FixtureTransform,
) -> int:
    """Run the TCP-plus-image dock gate without authorizing any CR3 motion."""
    profile = read_json(args.fixture_profile)
    if "rest_pose_stop" not in dock_design:
        raise ValueError("--verify-dock-tactile-reference-only requires a dock with rest_pose_stop")
    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    camera = GelSightCapture(parse_camera_source(args.camera_source), args.width, args.height, args.fps, args.camera_read_timeout_sec)
    preprocessor = create_tactip_preprocessor(args, output_dir)
    report_path = output_dir / "dock_tactile_reference_check.json"
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_dock_tactile_reference_check.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "fixture_profile": str(args.fixture_profile),
        "status": "started",
    }
    try:
        if preprocessor is None:
            raise RuntimeError("Dock tactile-reference verification requires TacTip preprocessing")
        camera.open()
        robot.connect()
        replies = robot.set_user_tool(args.user, args.tool)
        robot.require_motion_ready()
        joints, current_pose, raw_joints, raw_pose = robot.read_state()
        position_error = position_error_mm(fixture.dock_tcp, current_pose)
        rotation_error = rotation_error_deg(fixture.dock_tcp, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "expected_dock_tcp": list_pose(fixture.dock_tcp),
            "position_error_mm": position_error,
            "rotation_error_deg": rotation_error,
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "joint_count": len(joints or ()),
            "set_user_tool_replies": replies,
        }
        if position_error > float(args.dock_position_tolerance_mm) or rotation_error > float(args.dock_rotation_tolerance_deg):
            raise RuntimeError(
                "CR3 is not seated at the fixture datum: {:.3f} mm / {:.3f} deg away "
                "(limits {:.3f} mm / {:.3f} deg). No motion was sent."
                .format(
                    position_error,
                    rotation_error,
                    float(args.dock_position_tolerance_mm),
                    float(args.dock_rotation_tolerance_deg),
                )
            )
        payload["tactile_check"] = verify_rest_stop_tactile_reference(camera, args, output_dir, preprocessor, profile)
        payload["status"] = str(payload["tactile_check"].get("status", "failed"))
        write_json(report_path, payload)
        print("Dock tactile-reference report: {}".format(report_path))
        return 0 if payload["status"] == "passed" else 2
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        write_json(report_path, payload)
        print("Dock tactile-reference report: {}".format(report_path))
        raise
    finally:
        if preprocessor is not None:
            preprocessor.close()
        camera.close()
        robot.close()


def run_capture_camera_only(args: argparse.Namespace, output_dir: Path) -> int:
    """Save one collection-format TacTip image without opening a CR3 socket."""
    camera = GelSightCapture(parse_camera_source(args.camera_source), args.width, args.height, args.fps, args.camera_read_timeout_sec)
    preprocessor = create_tactip_preprocessor(args, output_dir)
    report_path = output_dir / "formal_camera_capture.json"
    payload: dict[str, Any] = {
        "schema": "cr3_coverage_board_formal_camera_capture.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tile_id": args.tile,
        "camera_source": str(args.camera_source),
        "width": int(args.width),
        "height": int(args.height),
        "fps": float(args.fps),
        "capture_frames": int(args.capture_frames),
        "cr3_connected": False,
        "cr3_motion_sent": False,
        "status": "started",
    }
    try:
        if preprocessor is None:
            raise RuntimeError("Formal camera capture requires TacTip preprocessing")
        camera.open()
        record, feature, _camera_time = capture_record(
            camera,
            args,
            output_dir,
            preprocessor,
            "formal_camera_capture",
            int(args.capture_frames),
            0.0,
        )
        add_model_roi_path(record)
        payload["capture"] = record
        payload["marker_safe_pixels"] = int(np.count_nonzero(feature.marker_support))
        payload["status"] = "captured"
        write_json(report_path, payload)
        print("Formal camera capture: {}".format(output_dir / str(record["raw_image"])))
        print("Formal camera capture report: {}".format(report_path))
        return 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        write_json(report_path, payload)
        print("Formal camera capture report: {}".format(report_path))
        raise
    finally:
        if preprocessor is not None:
            preprocessor.close()
        camera.close()


def run(args: argparse.Namespace) -> int:
    planning_started = time.monotonic()
    print("Loading fixture and building the sampling plan...", flush=True)
    dock_design = load_dock_design(args)
    if args.capture_camera_only:
        return run_capture_camera_only(args, prepare_run_dir(args))
    if args.calibrate_height_from_rest_stop:
        return teach_rest_stop_height_profile(args, dock_design)
    if args.teach_dock_from_current:
        return teach_dock_profile(args, dock_design)
    manifest, tile, csv_rows = load_board_data(args)
    filtered_rows = filter_tile_rows(tile, csv_rows, args)
    samples = apply_first_contact_test(build_samples(manifest, tile, filtered_rows, args), args)
    if args.height_measurement:
        if any(sample.stimulus != "flat_reference" for sample in samples):
            raise ValueError("Height measurement requires a known flat_reference site; curved/edge features do not identify board height")
        samples = [replace(sample, post_contact_depth_mm=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0) for sample in samples]
    ik_candidate_samples = build_ik_candidate_pool(manifest, tile, filtered_rows, args, samples)
    if args.zero_tilt:
        samples = [replace(sample, tilt_x_deg=0.0, tilt_y_deg=0.0) for sample in samples]
        ik_candidate_samples = [replace(sample, tilt_x_deg=0.0, tilt_y_deg=0.0) for sample in ik_candidate_samples]
    sample_depth_summary = depth_distribution_summary(samples, args)
    candidate_depth_summary = depth_distribution_summary(ik_candidate_samples, args)
    sample_spatial_summary = spatial_coverage_summary(samples)
    candidate_spatial_summary = spatial_coverage_summary(ik_candidate_samples)
    csv_depth_summary = csv_depth_limit_summary(ik_candidate_samples, filtered_rows)
    fixture: FixtureTransform | None = None
    fixture_profile_payload: dict[str, Any] | None = None
    if args.fixture_profile.is_file():
        fixture_profile_payload = read_json(args.fixture_profile)
        fixture = fixture_from_profile(fixture_profile_payload, dock_design, args)
    elif args.execute:
        raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
    if args.fixture_local_preview and fixture is not None:
        raise ValueError("--fixture-local-preview requires no real fixture profile; omit this flag to preview your measured fixture")
    if args.height_calibration:
        if fixture is None:
            raise ValueError("A real fixture profile is required to validate the height calibration binding")
        from board_height_calibration import load_height_calibration
        args._height_calibration = load_height_calibration(
            args.height_calibration, args.fixture_profile, args.dock_design,
            next(args.board_dir.glob("*_manifest.json")), args.tile, args.user, args.tool,
        )
        if float(args._height_calibration["quality"]["additive_error_budget_mm"]) > 0.1 + 1e-9:
            raise ValueError("Height calibration exceeds the required 0.1 mm observation error budget; a relaxed calibration file cannot authorize this run")
        if not math.isfinite(float(args._height_calibration["quality"]["additive_error_budget_mm"])):
            raise ValueError("Height calibration error budget must be finite")
        # Validate every requested and reserve XY before any camera/robot action.
        for sample in (*samples, *ik_candidate_samples):
            board_surface_correction_mm(sample, args)
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
        "height_calibration": str(args.height_calibration) if args.height_calibration else None,
        "height_calibration_sha256": sha256_file(args.height_calibration) if args.height_calibration else None,
        "route_preview_frame": "tile_local_geometry_only" if args.fixture_local_preview else "measured_fixture" if fixture is not None else "no_route_fixture",
        "board_height_offset_mm": float(args.board_height_offset_mm),
        "contact_search_margin_mm": float(args.contact_search_margin_mm),
        "max_extra_below_planned_contact_mm": float(args.max_extra_below_planned_contact_mm),
        "previews_enabled": not bool(args.skip_previews),
        "post_contact_depth_distribution": sample_depth_summary,
        "ik_candidate_depth_distribution": candidate_depth_summary,
        "depth_policy": "per_seed_csv_constrained" if args.respect_csv_depth_limits else "global_requested_range",
        "dense_spatial_layout": effective_dense_spatial_layout(args),
        "dense_region_anchor_count": int(args.dense_region_anchor_count),
        "planned_spatial_coverage": sample_spatial_summary,
        "ik_candidate_spatial_coverage": candidate_spatial_summary,
        "csv_depth_limit_summary": csv_depth_summary,
        "ik_candidate_pool_count": len(ik_candidate_samples),
        "ik_candidate_multiplier": float(args.ik_candidate_multiplier),
        "reference_pad_check_enabled": not bool(args.skip_reference_pad_check),
        "dock_tactile_reference_check_enabled": not bool(args.skip_dock_tactile_reference_check),
        "continuous_board_transit": bool(args.continuous_board_transit),
        "samples": [asdict(sample) for sample in samples],
        "notes": [
            "Tile-local X/Y/Z are exact coordinates from the generated board manifest and dock geometry.",
            "Exact-count plans use the selected spatial layout and recompute analytical surface height for every planned XY point. The default region_grid uses the footprint-safe central region, not repeated micro-jitter around nine seeds.",
            "A base-frame CR3 pose is emitted only after a valid seated fixture profile is supplied.",
            (
                "For a raised-crossbar dock, an execute run first verifies both the saved Tool TCP and a "
                "current tactile image against the saved crossbar-contact reference; no movement is sent on a mismatch."
            ),
            (
                "The real run skips the dock reference-pad touch and uses the saved dock frame directly."
                if args.skip_reference_pad_check
                else "The real run performs its visual reference-pad check and records any accepted translation correction."
            ),
            "An execute run uses controller IK to filter site-high, approach, contact, maximum-capture, and retreat poses after the reference correction. Dense runs select replacements from the deterministic candidate pool.",
            (
                "The post-contact depth protocol is distributed inside each source seed's CSV-safe intersection "
                "with the requested range after visual first contact."
                if args.respect_csv_depth_limits
                else "The post-contact depth protocol uses the global requested range; CSV depth-limit excess is recorded and requires an explicit execution override."
            ),
            "Visual contact may search only one configured surface-error margin past nominal before it stops; the final capture is also hard-capped at requested indentation plus that margin.",
            (
                "The dock gate runs once, then samples are connected at fixture-local safe height without per-sample dock returns."
                if args.continuous_board_transit
                else "Every sample returns through dock-high to dock-exit before the next site."
            ),
        ],
    }
    write_json(output_dir / "sampling_plan.json", plan_payload)
    write_plan_csv(plan_path, samples, fixture, args)
    if not args.skip_previews:
        print("Building HTML previews (use --skip-previews to omit mesh processing)...", flush=True)
        preview_samples = [replace(sample, local_contact_mm=(sample.local_contact_mm[0], sample.local_contact_mm[1], sample.local_contact_mm[2] + board_surface_correction_mm(sample, args))) for sample in samples]
        write_plan_preview(preview_path, tile, args.board_dir, dock_design, preview_samples)
    ik_preview_path: Path | None = None
    motion_preview_path: Path | None = None
    if fixture is not None and not args.skip_previews:
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
        motion_preview_path = output_dir / "full_tcp_motion_safety_preview.html"
        write_motion_route_preview(
            motion_preview_path,
            tile,
            args.board_dir,
            dock_design,
            fixture,
            args,
            samples,
        )
    elif args.fixture_local_preview:
        local = tuple(float(value) for value in dock_design["tactip_reference"]["nominal_seated_tool_tcp_local_mm"])
        preview_fixture = FixtureTransform(
            dock_tcp=local + (180.0, 0.0, 0.0), dock_tcp_local_mm=local,
            tile_to_base=np.eye(3), dock_rotation=np.diag((1.0, -1.0, -1.0)), board_yaw_deg=0.0,
        )
        motion_preview_path = output_dir / "fixture_local_motion_preview.html"
        write_motion_route_preview(motion_preview_path, tile, args.board_dir, dock_design, preview_fixture, args, samples)
        print("Geometry-only route: no robot base transform, IK check, or real calibration was generated.", flush=True)
    write_run_readme(
        output_dir / "README.md",
        args,
        args.fixture_profile if fixture is not None else None,
        tuple(float(value) for value in manifest.get("tile_size_mm", (0.0, 0.0))),
    )
    print("Plan directory: {}".format(output_dir))
    print("Tile {}: {} planned tactile samples ({})".format(args.tile, len(samples), sampling_label(args)))
    if not args.skip_previews:
        print("Interactive local plan: {}".format(preview_path))
    print("Planning completed in {:.1f}s; first-contact search margin {:.2f} mm; board-height correction {:+.2f} mm."
          .format(time.monotonic() - planning_started, args.contact_search_margin_mm, args.board_height_offset_mm), flush=True)
    if ik_preview_path is not None:
        print("Interactive IK candidate preview: {}".format(ik_preview_path))
    if motion_preview_path is not None:
        print("Full TCP safety route preview: {}".format(motion_preview_path))
    if args.reseat_dock_only:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
        return run_reseat_dock(args, output_dir, fixture)
    if args.recover_reference_to_dock:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --teach-dock-from-current first.".format(args.fixture_profile))
        return run_recover_reference_to_dock(args, output_dir, dock_design, fixture)
    if args.verify_dock_tactile_reference_only:
        if fixture is None:
            raise FileNotFoundError("No fixture profile at {}. Seat TacTip in the dock and run --calibrate-height-from-rest-stop first.".format(args.fixture_profile))
        return run_verify_dock_tactile_reference(args, output_dir, dock_design, fixture)
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
