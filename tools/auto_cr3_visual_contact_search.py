#!/usr/bin/env python3
"""Find a TacTip contact automatically from marker-pixel motion.

This tool is deliberately a *single-site* safety check for an existing,
collision-checked OBJ/CR3 plan.  It starts only at the plan's declared safe
transit TCP, follows that plan's route to its no-contact approach TCP, and
then descends along the exact planned press axis in small MovL increments.

No manual contact teaching is required.  The first stable marker-image change
relative to a no-contact baseline is treated as the visual contact point.  The
robot then returns to the approach pose and safe transit; this script does not
silently collect the rest of a plan or alter the saved calibration.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from auto_cr3_gelsight_pair_sampler import GelSightCapture, read_quality_checked_frame
from live_cr3_gelsight_sampler import DobotCR3LiveClient, parse_camera_source
from tactip_runtime_preprocess import add_tactip_preprocess_args, create_tactip_preprocessor


@dataclass(frozen=True)
class PlanSite:
    site_id: str
    pair_id: str
    approach_tcp: tuple[float, float, float, float, float, float]
    contact_tcp: tuple[float, float, float, float, float, float]
    capture_tcp: tuple[float, float, float, float, float, float]
    route: tuple[tuple[str, tuple[float, float, float, float, float, float]], ...]
    safe_transit_tcp: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class DiffStats:
    mean: float
    p95: float
    p99: float
    active_pixels: int


@dataclass(frozen=True)
class ImageFeature:
    gray: np.ndarray
    allowed_mask: np.ndarray
    marker_support: np.ndarray
    normalized_texture: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="Directory containing obj_contact_plan.csv and object_collision_routes.json.",
    )
    parser.add_argument("--site", default="site_001", help="Single planned site to search, for example site_001.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to a timestamped directory inside --plan-dir.")
    parser.add_argument("--robot-ip", default="192.168.31.88")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--move-port", type=int, default=30003)
    parser.add_argument("--robot-timeout-sec", type=float, default=6.0)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument(
        "--tool",
        type=int,
        default=2,
        help="CR3 Tool frame. The current camera setup uses Tool 2.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="CR3 speed percent. Visual contact search intentionally permits at most 5 percent.",
    )
    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-read-timeout-sec", type=float, default=10.0)
    parser.add_argument("--capture-min-gray-p95", type=float, default=20.0)
    parser.add_argument("--capture-retry-count", type=int, default=8)
    parser.add_argument("--capture-retry-sec", type=float, default=0.05)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument(
        "--step-mm",
        type=float,
        default=0.5,
        help="Downward increment along the plan's press axis. Maximum is 0.5 mm.",
    )
    parser.add_argument(
        "--max-extra-below-planned-contact-mm",
        type=float,
        default=4.0,
        help=(
            "Hard safety limit beyond the OBJ plan's zero-contact TCP. The default searches 4 mm farther only; "
            "the maximum is 8 mm for a user-requested follow-up search."
        ),
    )
    parser.add_argument("--baseline-frames", type=int, default=9)
    parser.add_argument("--noise-probes", type=int, default=3)
    parser.add_argument("--probe-frames", type=int, default=3)
    parser.add_argument("--consecutive-hits", type=int, default=2)
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=3.0,
        help="Robust baseline-noise multiplier used to form marker-motion thresholds.",
    )
    # These are marker *displacements in pixels*, after cancelling a global
    # camera shift.  They are deliberately much lower than the old
    # illumination-difference floors: a real TacTip contact begins as a small
    # movement of the dot lattice, not necessarily a large brightness change.
    parser.add_argument("--min-contact-mean", type=float, default=0.10)
    parser.add_argument("--min-contact-p95", type=float, default=0.45)
    parser.add_argument("--start-position-tolerance-mm", type=float, default=0.75)
    parser.add_argument("--start-rotation-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--motion-position-tolerance-mm", type=float, default=0.75)
    parser.add_argument("--motion-rotation-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--execute", action="store_true", help="Send the guarded CR3 movements after validation.")
    parser.add_argument("--yes-i-confirm-cr3-is-safe", action="store_true")
    add_tactip_preprocess_args(parser)
    args = parser.parse_args()

    if not args.plan_dir.is_dir():
        parser.error("--plan-dir does not exist: {}".format(args.plan_dir))
    for name in (
        "robot_timeout_sec",
        "camera_read_timeout_sec",
        "speed",
        "settle_sec",
        "step_mm",
        "noise_multiplier",
        "min_contact_mean",
        "min_contact_p95",
        "start_position_tolerance_mm",
        "start_rotation_tolerance_deg",
        "motion_position_tolerance_mm",
        "motion_rotation_tolerance_deg",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if not 1.0 <= float(args.speed) <= 5.0:
        parser.error("--speed must be in [1, 5] for vision-only contact search")
    if not 0.0 < float(args.step_mm) <= 0.5:
        parser.error("--step-mm must be in (0, 0.5]")
    if not 0.0 <= float(args.max_extra_below_planned_contact_mm) <= 8.0:
        parser.error("--max-extra-below-planned-contact-mm must be in [0, 8]")
    if int(args.baseline_frames) < 3 or int(args.probe_frames) < 2 or int(args.noise_probes) < 2:
        parser.error("--baseline-frames >= 3, --probe-frames >= 2, and --noise-probes >= 2 are required")
    if int(args.consecutive_hits) < 2:
        parser.error("--consecutive-hits must be at least 2 to reject one-frame camera flicker")
    if int(args.capture_retry_count) < 0 or float(args.capture_retry_sec) < 0.0:
        parser.error("Capture retry settings must be non-negative")
    if not 0 <= int(args.user) <= 9 or not 0 <= int(args.tool) <= 9:
        parser.error("--user and --tool must be in [0, 9]")
    if bool(args.no_tactip_preprocess):
        parser.error("This contact detector requires the default TacTip marker preprocessing; omit --no-tactip-preprocess")
    if args.execute and not args.yes_i_confirm_cr3_is_safe:
        parser.error("--execute requires --yes-i-confirm-cr3-is-safe")
    return args


def finite_pose(values: Sequence[float], label: str) -> tuple[float, float, float, float, float, float]:
    pose = tuple(float(value) for value in values)
    if len(pose) != 6 or not np.isfinite(pose).all():
        raise ValueError("{} must contain six finite TCP values".format(label))
    return pose  # type: ignore[return-value]


def pose_from_row(row: dict[str, str], prefix: str) -> tuple[float, float, float, float, float, float]:
    columns = ["{}_{}".format(prefix, axis) for axis in ("x", "y", "z", "Rx", "Ry", "Rz")]
    try:
        return finite_pose([float(row[column]) for column in columns], prefix)
    except KeyError as exc:
        raise ValueError("Contact plan is missing column {}".format(exc)) from exc


def pose_rotation(pose: Sequence[float]) -> Rotation:
    return Rotation.from_euler("XYZ", tuple(float(value) for value in pose[3:]), degrees=True)


def position_error_mm(expected: Sequence[float], actual: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(expected[:3], dtype=float) - np.asarray(actual[:3], dtype=float)))


def rotation_error_deg(expected: Sequence[float], actual: Sequence[float]) -> float:
    delta = pose_rotation(actual) * pose_rotation(expected).inv()
    return float(math.degrees(np.linalg.norm(delta.as_rotvec())))


def values_text(values: Sequence[float]) -> str:
    return " ".join("{:.4f}".format(float(value)) for value in values)


def list_pose(pose: Sequence[float]) -> list[float]:
    return [float(value) for value in pose]


def load_plan_site(plan_dir: Path, site_id: str) -> PlanSite:
    plan_path = plan_dir / "obj_contact_plan.csv"
    routes_path = plan_dir / "object_collision_routes.json"
    if not plan_path.is_file():
        raise FileNotFoundError("Expected {}".format(plan_path))
    if not routes_path.is_file():
        raise FileNotFoundError("Expected {}".format(routes_path))

    with plan_path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("site_id") == site_id]
    if not rows:
        raise ValueError("{} has no row for {}".format(plan_path, site_id))
    rows.sort(key=lambda row: (float(row.get("indentation_mm", "inf")), int(row.get("plan_index", "999999"))))
    row = rows[0]

    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    if routes.get("status") != "passed":
        raise ValueError("{} is not a passed collision-route report".format(routes_path))
    route_data = routes.get("routes", {}).get(site_id)
    if not isinstance(route_data, dict):
        raise ValueError("{} has no route for {}".format(routes_path, site_id))
    route_values = route_data.get("waypoints")
    if not isinstance(route_values, list) or not route_values:
        raise ValueError("{} has an empty route for {}".format(routes_path, site_id))

    route: list[tuple[str, tuple[float, float, float, float, float, float]]] = []
    for waypoint in route_values:
        if not isinstance(waypoint, dict):
            raise ValueError("Route waypoint is malformed")
        route.append((str(waypoint.get("label", "waypoint")), finite_pose(waypoint.get("tcp", ()), "route TCP")))
    approach = pose_from_row(row, "approach_tcp")
    contact = pose_from_row(row, "contact_tcp")
    capture = pose_from_row(row, "capture_tcp")
    route_end_error = position_error_mm(route[-1][1], approach)
    route_end_rotation_error = rotation_error_deg(route[-1][1], approach)
    if route_end_error > 0.02 or route_end_rotation_error > 0.02:
        raise ValueError(
            "Route for {} does not end at its plan approach ({:.3f} mm / {:.3f} deg)".format(
                site_id, route_end_error, route_end_rotation_error
            )
        )
    return PlanSite(
        site_id=site_id,
        pair_id=str(row.get("pair_id", "")),
        approach_tcp=approach,
        contact_tcp=contact,
        capture_tcp=capture,
        route=tuple(route),
        safe_transit_tcp=finite_pose(routes.get("safe_transit_tcp", ()), "safe transit TCP"),
    )


def press_axis_and_distances(site: PlanSite) -> tuple[np.ndarray, float, float]:
    approach = np.asarray(site.approach_tcp[:3], dtype=float)
    contact = np.asarray(site.contact_tcp[:3], dtype=float)
    capture = np.asarray(site.capture_tcp[:3], dtype=float)
    vector = contact - approach
    clearance = float(np.linalg.norm(vector))
    if clearance < 1.0:
        raise ValueError("Plan approach and contact TCPs are too close to define a press axis")
    axis = vector / clearance
    capture_depth = float(np.dot(capture - approach, axis))
    if capture_depth + 1.0e-3 < clearance:
        raise ValueError("Plan capture TCP is not beyond its zero-contact TCP along the press axis")
    return axis, clearance, capture_depth


def move_and_verify(
    client: DobotCR3LiveClient,
    label: str,
    target: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = finite_pose(target, label + " target")
    mode_before, raw_mode_before = client.read_robot_mode()
    if mode_before != 5:
        raise RuntimeError("{} will not start while RobotMode is {} ({})".format(label, mode_before, raw_mode_before))
    if not getattr(args, "disable_ik_preflight", False):
        joint_near, _raw_joint_near = client.read_joints()
        try:
            inverse_joints, raw_inverse = client.inverse_solution(
                expected,
                int(args.user),
                int(args.tool),
                joint_near=joint_near,
            )
        except Exception as exc:
            raise RuntimeError(
                "{} has no CR3 inverse-kinematics solution; MovL was not sent. {}"
                .format(label, exc)
            ) from exc
    else:
        inverse_joints, raw_inverse = (), "IK preflight disabled"
    joints, actual, raw_joints, raw_pose = client.move_pose(expected, args.speed, args.user, args.tool)
    if actual is None:
        raise RuntimeError("{} returned no GetPose after MovL".format(label))
    actual_pose = finite_pose(actual, label + " actual TCP")
    position_error = position_error_mm(expected, actual_pose)
    rotation_error = rotation_error_deg(expected, actual_pose)
    mode_after, raw_mode_after = client.read_robot_mode()
    record = {
        "label": label,
        "target_tcp": list_pose(expected),
        "actual_tcp": list_pose(actual_pose),
        "position_error_mm": position_error,
        "rotation_error_deg": rotation_error,
        "raw_get_angle": raw_joints,
        "raw_get_pose": raw_pose,
        "robot_mode_before": {"value": int(mode_before), "raw": raw_mode_before},
        "robot_mode_after": {"value": int(mode_after), "raw": raw_mode_after},
        "ik_preflight": {
            "enabled": not getattr(args, "disable_ik_preflight", False),
            "solution_joints": [float(value) for value in inverse_joints],
            "raw": raw_inverse,
        },
        "joint_count": len(joints or ()),
    }
    if mode_after != 5:
        raise RuntimeError("{} ended with RobotMode {} ({})".format(label, mode_after, raw_mode_after))
    if (
        position_error > float(args.motion_position_tolerance_mm)
        or rotation_error > float(args.motion_rotation_tolerance_deg)
    ):
        raise RuntimeError(
            "{} reached an unexpected TCP: position {:.3f} mm (limit {:.3f}), rotation {:.3f} deg (limit {:.3f})".format(
                label,
                position_error,
                float(args.motion_position_tolerance_mm),
                rotation_error,
                float(args.motion_rotation_tolerance_deg),
            )
        )
    return record


def median_capture(
    camera: GelSightCapture,
    args: argparse.Namespace,
    count: int,
    newer_than: float,
) -> tuple[np.ndarray, float, float]:
    frames: list[np.ndarray] = []
    p95_values: list[float] = []
    timestamp = newer_than
    for _index in range(count):
        frame, timestamp, p95 = read_quality_checked_frame(
            camera,
            float(args.capture_min_gray_p95),
            int(args.capture_retry_count),
            float(args.capture_retry_sec),
            timestamp,
        )
        frames.append(frame)
        p95_values.append(float(p95))
        time.sleep(0.035)
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8), timestamp, float(np.median(p95_values))


def write_frame(
    output_dir: Path,
    label: str,
    frame: np.ndarray,
    preprocessor: Any,
) -> tuple[Path, Path, Path]:
    raw_path = output_dir / "frames" / "{}.png".format(label)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(raw_path), frame):
        raise RuntimeError("Could not write {}".format(raw_path))
    processed = preprocessor.process_and_save(frame, raw_path)
    if processed is None:
        raise RuntimeError("TacTip preprocessing failed for {}".format(raw_path))
    preprocess_dir = output_dir / "tactip_preprocessed"
    gray_path = preprocess_dir / "gray" / raw_path.name
    roi_path = preprocess_dir / "model_roi" / raw_path.name
    if not gray_path.is_file() or not roi_path.is_file():
        raise RuntimeError("TacTip preprocessing outputs are incomplete for {}".format(raw_path.name))
    return raw_path, gray_path, roi_path


def load_feature(gray_path: Path, roi_path: Path) -> ImageFeature:
    gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or roi is None or gray.shape != roi.shape:
        raise RuntimeError("Could not load matching preprocessing outputs for {}".format(gray_path.name))
    allowed = roi > 0
    if int(np.count_nonzero(allowed)) < 2000:
        raise RuntimeError("Marker-safe preprocessing ROI is unexpectedly small for {}".format(gray_path.name))
    value = gray.astype(np.float32)
    response = value - cv2.GaussianBlur(value, (0, 0), sigmaX=6.0, sigmaY=6.0)
    values = response[allowed]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    normalized = (response - median) / max(1.4826 * mad, 1.0)
    marker_cutoff = float(np.percentile(values, 93.0))
    marker_seed = ((response >= marker_cutoff) & allowed).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    marker_support = (cv2.dilate(marker_seed, kernel) > 0) & allowed
    if int(np.count_nonzero(marker_support)) < 1200:
        marker_support = allowed.copy()
    return ImageFeature(gray=gray, allowed_mask=allowed, marker_support=marker_support, normalized_texture=normalized)


def diff_stats(baseline: ImageFeature, candidate: ImageFeature) -> tuple[DiffStats, np.ndarray]:
    """Measure local TacTip marker displacement, independent of illumination.

    The older implementation compared normalized image brightness.  That made
    the white outer lighting ring and camera exposure changes look like a
    contact even while the TacTip was in free space.  Here, the same marker
    corners are tracked from the no-contact frame with pyramidal LK optical
    flow.  The median whole-image motion is removed before reporting the
    local displacement distribution, so a small camera translation cannot
    trigger a contact by itself.
    """
    if baseline.gray.shape != candidate.gray.shape:
        raise RuntimeError("TacTip preprocessing frame geometry changed during the search")
    active = (baseline.marker_support | candidate.marker_support) & baseline.allowed_mask & candidate.allowed_mask
    if int(np.count_nonzero(active)) < 1200:
        active = baseline.allowed_mask & candidate.allowed_mask
    mask = active.astype(np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(
        baseline.gray,
        maxCorners=420,
        qualityLevel=0.006,
        minDistance=5.0,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) < 40:
        raise RuntimeError("Could not find enough TacTip marker corners for optical-flow contact detection")
    next_points, forward_ok, forward_error = cv2.calcOpticalFlowPyrLK(
        baseline.gray,
        candidate.gray,
        corners,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1.0e-4,
    )
    if next_points is None or forward_ok is None or forward_error is None:
        raise RuntimeError("TacTip optical flow did not return marker tracks")
    back_points, backward_ok, _backward_error = cv2.calcOpticalFlowPyrLK(
        candidate.gray,
        baseline.gray,
        next_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1.0e-4,
    )
    if back_points is None or backward_ok is None:
        raise RuntimeError("TacTip backward optical-flow validation failed")
    origins = corners.reshape(-1, 2)
    tracked = next_points.reshape(-1, 2)
    returned = back_points.reshape(-1, 2)
    forward_status = forward_ok.reshape(-1).astype(bool)
    backward_status = backward_ok.reshape(-1).astype(bool)
    lk_error = forward_error.reshape(-1)
    forward_backward_error = np.linalg.norm(returned - origins, axis=1)
    valid = (
        forward_status
        & backward_status
        & np.isfinite(tracked).all(axis=1)
        & np.isfinite(lk_error)
        & (lk_error <= 25.0)
        & np.isfinite(forward_backward_error)
        & (forward_backward_error <= 0.75)
    )
    if int(np.count_nonzero(valid)) < 30:
        raise RuntimeError("Too few stable TacTip marker tracks for optical-flow contact detection")
    origins = origins[valid]
    tracked = tracked[valid]
    raw_displacement = tracked - origins
    global_shift = np.median(raw_displacement, axis=0)
    local_displacement = raw_displacement - global_shift
    magnitudes = np.linalg.norm(local_displacement, axis=1)
    stats = DiffStats(
        mean=float(np.mean(magnitudes)),
        p95=float(np.percentile(magnitudes, 95)),
        p99=float(np.percentile(magnitudes, 99)),
        active_pixels=int(len(magnitudes)),
    )
    visualization = cv2.cvtColor(candidate.gray, cv2.COLOR_GRAY2BGR)
    visualization[~active] = 0
    arrow_scale = 5.0
    for origin, vector, magnitude in zip(origins, local_displacement, magnitudes):
        start = tuple(np.rint(origin).astype(int))
        end = tuple(np.rint(origin + vector * arrow_scale).astype(int))
        colour = (0, int(np.clip(120.0 + magnitude * 70.0, 120.0, 255.0)), 255)
        cv2.arrowedLine(visualization, start, end, colour, 1, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(
        visualization,
        "flow: {} tracks, global {:+.2f},{:+.2f}px".format(len(magnitudes), float(global_shift[0]), float(global_shift[1])),
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return stats, visualization


def robust_threshold(values: Sequence[float], floor: float, multiplier: float) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(float(floor), median + float(multiplier) * max(1.4826 * mad, 0.05))


def relative_path(path: Path, output_dir: Path) -> str:
    return str(path.resolve().relative_to(output_dir.resolve()))


def save_diff_image(output_dir: Path, label: str, image: np.ndarray) -> Path:
    path = output_dir / "analysis" / "{}_marker_motion.png".format(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write {}".format(path))
    return path


def capture_record(
    camera: GelSightCapture,
    args: argparse.Namespace,
    output_dir: Path,
    preprocessor: Any,
    label: str,
    frames: int,
    newer_than: float,
    baseline: ImageFeature | None = None,
) -> tuple[dict[str, Any], ImageFeature, float]:
    frame, timestamp, gray_p95 = median_capture(camera, args, frames, newer_than)
    raw_path, gray_path, roi_path = write_frame(output_dir, label, frame, preprocessor)
    feature = load_feature(gray_path, roi_path)
    record: dict[str, Any] = {
        "label": label,
        "raw_image": relative_path(raw_path, output_dir),
        "preprocessed_gray": relative_path(gray_path, output_dir),
        "model_input": "tactip_preprocessed/model_input/{}".format(raw_path.name),
        "frame_count": int(frames),
        "camera_time": float(timestamp),
        "capture_gray_p95": float(gray_p95),
        "marker_safe_pixels": int(np.count_nonzero(feature.marker_support)),
    }
    if baseline is not None:
        stats, diff_image = diff_stats(baseline, feature)
        diff_path = save_diff_image(output_dir, label, diff_image)
        record.update(
            {
                "marker_motion": {
                    "mean": stats.mean,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "active_pixels": stats.active_pixels,
                },
                "motion_visualization": relative_path(diff_path, output_dir),
            }
        )
    return record, feature, timestamp


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_html_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    frames = list(payload.get("frames", []))
    threshold = payload.get("threshold", {})
    rows: list[str] = []
    for frame in frames:
        motion = frame.get("marker_motion", {})
        hit = frame.get("hit")
        state = "-" if hit is None else ("contact evidence" if hit else "below threshold")
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(frame.get("label", ""))),
                "" if frame.get("search_depth_mm") is None else "{:.2f}".format(float(frame["search_depth_mm"])),
                "" if not motion else "{:.3f}".format(float(motion.get("mean", 0.0))),
                "" if not motion else "{:.3f}".format(float(motion.get("p95", 0.0))),
                "" if not motion else "{:.3f}".format(float(motion.get("p99", 0.0))),
                html.escape(state),
                int(frame.get("marker_safe_pixels", 0)),
            )
        )
    cards: list[str] = []
    for frame in frames:
        raw = html.escape(str(frame.get("raw_image", "")))
        model = html.escape(str(frame.get("model_input", "")))
        diff = frame.get("motion_visualization")
        diff_html = "<figure><figcaption>Marker-motion difference</figcaption><img src=\"{}\"></figure>".format(
            html.escape(str(diff))
        ) if diff else ""
        cards.append(
            "<article class=\"card\"><h2>{}</h2><div class=\"images\">"
            "<figure><figcaption>Raw capture</figcaption><img src=\"{}\"></figure>"
            "<figure><figcaption>Ring-free marker input</figcaption><img src=\"{}\"></figure>{}</div></article>".format(
                html.escape(str(frame.get("label", ""))), raw, model, diff_html
            )
        )
    detection = payload.get("detection") or {}
    title = "contact detected" if detection.get("detected") else "no contact detected"
    style = """body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232}h1{margin:0;font-size:23px}p{color:#b7c5d6;line-height:1.45}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;padding:16px 28px;background:#141f2d}.metric{border:1px solid #33475f;border-radius:6px;padding:11px;background:#182232}.metric b{display:block;font-size:18px;color:#fff}main{padding:18px 28px}.table-wrap{overflow:auto}.table{border-collapse:collapse;width:100%;font-size:13px}.table th,.table td{border-bottom:1px solid #304257;padding:8px;text-align:left;white-space:nowrap}.table th{color:#a7c8f0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:14px;margin-top:18px}.card{background:#182232;border:1px solid #33475f;border-radius:7px;padding:12px}.card h2{font-size:15px;margin:0 0 10px}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}.images figure{margin:0;background:#0d141e;padding:6px}.images figcaption{font-size:11px;color:#cbd7e5;min-height:26px}.images img{width:100%;display:block;background:#000;image-rendering:pixelated}"""
    page = """<!doctype html><html><head><meta charset=\"utf-8\"><title>Visual TacTip contact search</title>
<style>{}</style></head>
<body><header><h1>Visual TacTip contact search: {}</h1><p>Only the marker-safe, illumination-ring-suppressed image area is used for the decision. A contact needs {} consecutive steps above both thresholds; the first qualifying step is reported as the visual contact TCP.</p></header>
<section class=\"summary\"><div class=\"metric\">Status<b>{}</b></div><div class=\"metric\">Mean threshold<b>{:.3f}</b></div><div class=\"metric\">P95 threshold<b>{:.3f}</b></div><div class=\"metric\">Plan site<b>{}</b></div></section>
<main><div class=\"table-wrap\"><table class=\"table\"><thead><tr><th>Frame</th><th>Depth from approach (mm)</th><th>Mean</th><th>P95</th><th>P99</th><th>Decision</th><th>Marker-safe px</th></tr></thead><tbody>{}</tbody></table></div><div class=\"cards\">{}</div></main></body></html>""".format(
        style,
        html.escape(str(payload.get("site", ""))),
        int(payload.get("settings", {}).get("consecutive_hits", 2)),
        html.escape(title),
        float(threshold.get("mean", 0.0)),
        float(threshold.get("p95", 0.0)),
        html.escape(str(payload.get("site", ""))),
        "\n".join(rows),
        "\n".join(cards),
    )
    path = output_dir / "contact_search_report.html"
    path.write_text(page, encoding="utf-8")
    return path


def return_to_safe(
    client: DobotCR3LiveClient,
    site: PlanSite,
    args: argparse.Namespace,
    motion_records: list[dict[str, Any]],
) -> None:
    motion_records.append(move_and_verify(client, "retreat to approach", site.approach_tcp, args))
    for label, target in reversed(site.route[:-1]):
        motion_records.append(move_and_verify(client, "return waypoint {}".format(label), target, args))
    motion_records.append(move_and_verify(client, "return safe transit", site.safe_transit_tcp, args))


def dry_run(site: PlanSite, args: argparse.Namespace) -> int:
    axis, clearance, capture_depth = press_axis_and_distances(site)
    max_depth = clearance + float(args.max_extra_below_planned_contact_mm)
    print("DRY RUN: no CR3 or camera actions will occur.", flush=True)
    print("Plan site: {} ({})".format(site.site_id, site.pair_id), flush=True)
    print("Safe transit Tool({}): {}".format(args.tool, values_text(site.safe_transit_tcp)), flush=True)
    print("Route: {}".format(" -> ".join(label for label, _pose in site.route)), flush=True)
    print("Approach TCP: {}".format(values_text(site.approach_tcp)), flush=True)
    print("Planned zero-contact TCP: {}".format(values_text(site.contact_tcp)), flush=True)
    print("Planned 1 mm capture TCP: {}".format(values_text(site.capture_tcp)), flush=True)
    print("Press axis in CR3 base: {:.7f} {:.7f} {:.7f}".format(*axis), flush=True)
    print("Approach-to-zero distance: {:.3f} mm; planned capture depth: {:.3f} mm".format(clearance, capture_depth), flush=True)
    print(
        "Search: {:.3f} mm steps from 0 to {:.3f} mm from approach (at most {:.3f} mm beyond planned zero contact).".format(
            float(args.step_mm), max_depth, float(args.max_extra_below_planned_contact_mm)
        ),
        flush=True,
    )
    print("Add --execute --yes-i-confirm-cr3-is-safe to run the guarded single-site test.", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    plan_dir = args.plan_dir.expanduser().resolve()
    site = load_plan_site(plan_dir, str(args.site))
    if not args.execute:
        return dry_run(site, args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (plan_dir / "visual_contact_{}_{}".format(site.site_id, timestamp))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    axis, planned_clearance, planned_capture_depth = press_axis_and_distances(site)
    max_depth = planned_clearance + float(args.max_extra_below_planned_contact_mm)
    metadata_path = output_dir / "contact_search.json"
    payload: dict[str, Any] = {
        "schema": "cr3_tactip_visual_contact_search.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan_dir": str(plan_dir),
        "site": site.site_id,
        "pair_id": site.pair_id,
        "tool": int(args.tool),
        "user": int(args.user),
        "safe_transit_tcp": list_pose(site.safe_transit_tcp),
        "approach_tcp": list_pose(site.approach_tcp),
        "planned_contact_tcp": list_pose(site.contact_tcp),
        "planned_capture_tcp": list_pose(site.capture_tcp),
        "press_axis_base": [float(value) for value in axis],
        "planned_approach_to_contact_mm": planned_clearance,
        "planned_approach_to_capture_mm": planned_capture_depth,
        "settings": {
            "speed_percent": float(args.speed),
            "step_mm": float(args.step_mm),
            "max_extra_below_planned_contact_mm": float(args.max_extra_below_planned_contact_mm),
            "max_depth_from_approach_mm": max_depth,
            "baseline_frames": int(args.baseline_frames),
            "noise_probes": int(args.noise_probes),
            "probe_frames": int(args.probe_frames),
            "consecutive_hits": int(args.consecutive_hits),
            "noise_multiplier": float(args.noise_multiplier),
            "min_contact_mean": float(args.min_contact_mean),
            "min_contact_p95": float(args.min_contact_p95),
        },
        "frames": [],
        "motion": [],
        "threshold": {},
        "detection": {"detected": False},
        "status": "started",
    }
    write_json(metadata_path, payload)

    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout_sec)
    camera = GelSightCapture(
        parse_camera_source(args.camera_source), args.width, args.height, args.fps, args.camera_read_timeout_sec
    )
    preprocessor = create_tactip_preprocessor(args, output_dir)
    failed = False
    normal_return_done = False
    try:
        if preprocessor is None:
            raise RuntimeError("TacTip preprocessing is required for visual contact search")
        camera.open()
        robot.connect()
        select_replies = robot.set_user_tool(args.user, args.tool)
        mode, raw_mode = robot.read_robot_mode()
        if mode != 5:
            raise RuntimeError("RobotMode must be enabled and idle (5), got {} ({})".format(mode, raw_mode))
        joints, current_pose, raw_joints, raw_pose = robot.read_state()
        start_position_error = position_error_mm(site.safe_transit_tcp, current_pose)
        start_rotation_error = rotation_error_deg(site.safe_transit_tcp, current_pose)
        payload["start_state"] = {
            "actual_tcp": list_pose(current_pose),
            "raw_get_angle": raw_joints,
            "raw_get_pose": raw_pose,
            "joint_count": len(joints or ()),
            "user_tool_replies": select_replies,
            "position_error_to_safe_mm": start_position_error,
            "rotation_error_to_safe_deg": start_rotation_error,
        }
        if (
            start_position_error > float(args.start_position_tolerance_mm)
            or start_rotation_error > float(args.start_rotation_tolerance_deg)
        ):
            raise RuntimeError(
                "CR3 is not at the plan safe transit: {:.3f} mm / {:.3f} deg away. The script will not move from an unknown state.".format(
                    start_position_error, start_rotation_error
                )
            )

        for index, (label, target) in enumerate(site.route, start=1):
            payload["motion"].append(move_and_verify(robot, "route {} {}".format(index, label), target, args))
        time.sleep(float(args.settle_sec))

        baseline_record, baseline_feature, camera_time = capture_record(
            camera,
            args,
            output_dir,
            preprocessor,
            "baseline_approach",
            int(args.baseline_frames),
            0.0,
        )
        payload["frames"].append(baseline_record)
        noise_records: list[dict[str, Any]] = []
        for index in range(int(args.noise_probes)):
            record, _feature, camera_time = capture_record(
                camera,
                args,
                output_dir,
                preprocessor,
                "noise_{:02d}".format(index + 1),
                int(args.probe_frames),
                camera_time,
                baseline_feature,
            )
            payload["frames"].append(record)
            noise_records.append(record)
        means = [float(record["marker_motion"]["mean"]) for record in noise_records]
        p95s = [float(record["marker_motion"]["p95"]) for record in noise_records]
        threshold = {
            "mean": robust_threshold(means, float(args.min_contact_mean), float(args.noise_multiplier)),
            "p95": robust_threshold(p95s, float(args.min_contact_p95), float(args.noise_multiplier)),
            "noise_mean_values": means,
            "noise_p95_values": p95s,
        }
        payload["threshold"] = threshold
        print(
            "Visual baseline ready: marker-motion thresholds mean={:.3f}, p95={:.3f}".format(
                float(threshold["mean"]), float(threshold["p95"])
            ),
            flush=True,
        )

        first_hit: dict[str, Any] | None = None
        consecutive_hits = 0
        depths = np.arange(float(args.step_mm), max_depth + float(args.step_mm) * 0.5, float(args.step_mm))
        for index, depth in enumerate(depths, start=1):
            target_position = np.asarray(site.approach_tcp[:3], dtype=float) + axis * float(depth)
            target = finite_pose(tuple(target_position) + tuple(site.approach_tcp[3:]), "search target")
            motion = move_and_verify(robot, "press step {:03d}".format(index), target, args)
            payload["motion"].append(motion)
            time.sleep(float(args.settle_sec))
            record, _feature, camera_time = capture_record(
                camera,
                args,
                output_dir,
                preprocessor,
                "step_{:03d}_{:05.2f}mm".format(index, float(depth)),
                int(args.probe_frames),
                camera_time,
                baseline_feature,
            )
            record["search_depth_mm"] = float(depth)
            record["actual_tcp"] = motion["actual_tcp"]
            stats = record["marker_motion"]
            hit = float(stats["mean"]) >= float(threshold["mean"]) and float(stats["p95"]) >= float(threshold["p95"])
            record["hit"] = bool(hit)
            payload["frames"].append(record)
            print(
                "depth={:.2f} mm marker-change mean={:.3f} p95={:.3f} hit={}".format(
                    float(depth), float(stats["mean"]), float(stats["p95"]), bool(hit)
                ),
                flush=True,
            )
            if hit:
                consecutive_hits += 1
                if first_hit is None:
                    first_hit = record
                if consecutive_hits >= int(args.consecutive_hits):
                    contact_tcp = finite_pose(first_hit["actual_tcp"], "first visual contact TCP")
                    contact_depth = float(first_hit["search_depth_mm"])
                    payload["detection"] = {
                        "detected": True,
                        "first_visible_contact_depth_from_approach_mm": contact_depth,
                        "first_visible_contact_tcp": list_pose(contact_tcp),
                        "confirmation_depth_from_approach_mm": float(depth),
                        "confirmation_tcp": list_pose(motion["actual_tcp"]),
                        "delta_from_planned_zero_along_press_mm": contact_depth - planned_clearance,
                        "note": (
                            "This is the first stable marker-motion threshold crossing, not a force-sensor zero. "
                            "It is intentionally saved as a local site result and does not overwrite the global OBJ calibration."
                        ),
                    }
                    print(
                        "VISUAL CONTACT DETECTED at {:.2f} mm from approach; returning to safe transit.".format(contact_depth),
                        flush=True,
                    )
                    break
            else:
                consecutive_hits = 0
                first_hit = None

        return_to_safe(robot, site, args, payload["motion"])
        normal_return_done = True
        payload["status"] = "contact_detected" if payload["detection"].get("detected") else "no_contact_within_limit"
        if not payload["detection"].get("detected"):
            payload["detection"]["note"] = (
                "No stable marker motion was detected within the hard search limit. The CR3 returned to safe transit; "
                "the script did not descend farther."
            )
        write_json(metadata_path, payload)
        report = write_html_report(output_dir, payload)
        print("Contact-search record: {}".format(metadata_path), flush=True)
        print("Visual report: {}".format(report), flush=True)
        return 0 if payload["detection"].get("detected") else 2
    except Exception as exc:
        failed = True
        payload["status"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        try:
            write_json(metadata_path, payload)
            write_html_report(output_dir, payload)
        except Exception:
            pass
        raise
    finally:
        if preprocessor is not None:
            preprocessor.close()
        camera.close()
        robot.close()
        if failed and not normal_return_done:
            print(
                "Visual contact search stopped after an error. It did not issue an automatic recovery move; inspect the CR3 state before resuming.",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
