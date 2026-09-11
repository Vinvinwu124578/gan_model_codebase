#!/usr/bin/env python3
"""Execute a calibrated Tactile Gym pair plan on a real Dobot CR3.

The pair plan is the single source of truth for simulation and physical data.
Each row is a TCP offset in the calibrated zero-contact task frame. This tool
assumes a horizontal surface with positive CR3 base Z pointing away from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# Direct `python tools/...py` puts only tools/ on sys.path. Include this
# checkout's sibling tactile_sim2real package before importing live helpers.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dobot_cr3_client import DASHBOARD_PORT, DEFAULT_IP, MOVE_PORT
from live_cr3_gelsight_sampler import (
    JOINT_COLUMNS,
    PAIR_PLAN_COMPLETED_STATUSES,
    SIM2REAL_MANIFEST_COLUMNS,
    SIM2REAL_OBJECT_COLUMNS,
    SIM2REAL_POSE_COLUMNS,
    SIM2REAL_SHEAR_COLUMNS,
    SIM2REAL_TARGET_COLUMNS,
    SIM2REAL_TCP_COLUMNS,
    CameraWorker,
    DobotCR3LiveClient,
    format_values,
    parse_camera_source,
    request_macos_camera_access,
)
from tactile_sim2real.pairing import (
    ACTUAL_TCP_COLUMNS,
    PAIR_RECORD_COLUMNS,
    REAL_TARGET_TCP_COLUMNS,
    PairPlan,
    PairPlanRow,
    pose_error,
    task_offset_to_real_tcp,
    translation_basis_from_values,
    verify_real_pose,
    write_header_if_missing,
)
from tactip_runtime_preprocess import (
    TacTipRuntimePreprocessor,
    add_tactip_preprocess_args,
    create_tactip_preprocessor,
)


RAW_POSE_COLUMNS = ["pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6"]
RAW_SAMPLE_COLUMNS = [
    "sample_id",
    "timestamp",
    "elapsed_sec",
    "image_file",
    "camera_time",
    "capture_gray_p95",
    *JOINT_COLUMNS,
    *RAW_POSE_COLUMNS,
    "raw_get_angle",
    "raw_get_pose",
]


class GelSightCapture:
    def __init__(self, source, width: int, height: int, fps: float, timeout_sec: float) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_sec = timeout_sec
        self.worker = CameraWorker(
            source=source,
            width=width,
            height=height,
            fps=fps,
            backend="any",
            mirror=False,
            rotate=0,
            synthetic=False,
        )

    def open(self) -> None:
        granted, message = request_macos_camera_access()
        if not granted:
            raise RuntimeError(message)
        self.worker.start()
        self.read_frame()

    def read_frame(self, newer_than: float = 0.0):
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            frame, frame_time, error = self.worker.get_latest()
            if frame is not None and frame_time > newer_than:
                return frame, frame_time
            time.sleep(0.02)
        _frame, _frame_time, error = self.worker.get_latest()
        detail = ": {}".format(error) if error else ""
        raise RuntimeError("Timed out waiting for a GelSight frame{}".format(detail))

    def close(self) -> None:
        self.worker.stop()


def read_quality_checked_frame(
    camera: GelSightCapture,
    min_gray_p95: float,
    retry_count: int,
    retry_sec: float,
    newer_than: float,
):
    """Return a fresh, sufficiently exposed frame or fail while the caller can retract."""
    requested_after = newer_than
    last_p95 = float("nan")
    for attempt in range(retry_count + 1):
        image, camera_time = camera.read_frame(newer_than=requested_after)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        p95 = float(np.percentile(gray, 95))
        if p95 >= min_gray_p95:
            return image, camera_time, p95
        last_p95 = p95
        requested_after = camera_time
        if attempt < retry_count:
            time.sleep(retry_sec)
    raise RuntimeError(
        "GelSight exposure check failed after {} frame(s): gray p95 {:.1f} is below {:.1f}".format(
            retry_count + 1, last_p95, min_gray_p95
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-real-tcp",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "Rx", "Ry", "Rz"),
        help="Calibrated zero-contact CR3 TCP that defines the real task frame.",
    )
    parser.add_argument("--robot-ip", default=DEFAULT_IP)
    parser.add_argument("--dashboard-port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--move-port", type=int, default=MOVE_PORT)
    parser.add_argument("--robot-timeout", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--tool", type=int, default=0)
    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-read-timeout-sec", type=float, default=10.0)
    add_tactip_preprocess_args(parser)
    parser.add_argument(
        "--capture-min-gray-p95",
        type=float,
        default=20.0,
        help="Reject frames whose grayscale 95th percentile is below this exposure threshold.",
    )
    parser.add_argument("--capture-retry-count", type=int, default=8)
    parser.add_argument("--capture-retry-sec", type=float, default=0.05)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument(
        "--clearance-base-z-mm",
        type=float,
        default=1.0,
        help="Safe clearance above the reference plane, along positive CR3 base Z.",
    )
    parser.add_argument(
        "--contact-radius-mm",
        type=float,
        default=20.0,
        help="Conservative radius of the tactile contact face, used for tilted-contact safety checks.",
    )
    parser.add_argument(
        "--rotation-clearance-margin-mm",
        type=float,
        default=1.0,
        help="Additional normal clearance used while approaching a tilted contact.",
    )
    parser.add_argument(
        "--max-edge-indentation-mm",
        type=float,
        default=None,
        help="Maximum allowed deepest edge indentation. Defaults to --max-plan-z-mm.",
    )
    parser.add_argument("--start-position-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--start-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--pair-position-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--pair-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--max-plan-xy-mm", type=float, default=2.0)
    parser.add_argument("--max-plan-z-mm", type=float, default=1.0)
    parser.add_argument("--max-plan-rotation-deg", type=float, default=2.0)
    parser.add_argument(
        "--translation-basis-base",
        type=float,
        nargs=9,
        metavar=("B00", "B01", "B02", "B10", "B11", "B12", "B20", "B21", "B22"),
        help=(
            "Optional row-major, right-handed 3x3 translation basis in CR3 base axes. "
            "Its columns map task X/Y/Z translations onto the calibrated real setup."
        ),
    )
    parser.add_argument(
        "--real-rz-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Multiplier used only while mapping planned sim Rz onto the physical CR3 TCP.",
    )
    parser.add_argument(
        "--real-rx-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Multiplier used only while mapping planned sim Rx onto the physical CR3 TCP.",
    )
    parser.add_argument(
        "--real-ry-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Multiplier used only while mapping planned sim Ry onto the physical CR3 TCP.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Send MovL commands after all preflight checks pass.")
    parser.add_argument("--yes-i-confirm-cr3-is-safe", action="store_true")
    args = parser.parse_args()
    for name in (
        "robot_timeout",
        "speed",
        "camera_read_timeout_sec",
        "settle_sec",
        "clearance_base_z_mm",
        "rotation_clearance_margin_mm",
        "start_position_tolerance_mm",
        "start_rotation_tolerance_deg",
        "pair_position_tolerance_mm",
        "pair_rotation_tolerance_deg",
        "max_plan_xy_mm",
        "max_plan_z_mm",
        "max_plan_rotation_deg",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.contact_radius_mm < 0:
        parser.error("--contact-radius-mm must be non-negative")
    if args.capture_min_gray_p95 < 0:
        parser.error("--capture-min-gray-p95 must be non-negative")
    if args.capture_retry_count < 0:
        parser.error("--capture-retry-count must be non-negative")
    if args.capture_retry_sec < 0:
        parser.error("--capture-retry-sec must be non-negative")
    if args.max_edge_indentation_mm is not None and args.max_edge_indentation_mm <= 0:
        parser.error("--max-edge-indentation-mm must be positive")
    if not 1 <= args.speed <= 100:
        parser.error("--speed must be in [1, 100]")
    if not 0 <= args.user <= 9 or not 0 <= args.tool <= 9:
        parser.error("--user and --tool must be in [0, 9]")
    if args.execute and not args.yes_i_confirm_cr3_is_safe:
        parser.error("--execute requires --yes-i-confirm-cr3-is-safe")
    return args


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _row in csv.DictReader(file))


def completed_pair_ids(path: Path, plan: PairPlan) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("plan_path") == str(plan.path) and row.get("verification_status") in PAIR_PLAN_COMPLETED_STATUSES:
                completed.add(row.get("pair_id", ""))
    return completed


def ensure_raw_output(path: Path, args: argparse.Namespace, plan: PairPlan) -> tuple[Path, Path, int]:
    path.mkdir(parents=True, exist_ok=True)
    image_dir = path / "images"
    image_dir.mkdir(exist_ok=True)
    samples_path = path / "samples.csv"
    if not samples_path.exists():
        with samples_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=RAW_SAMPLE_COLUMNS).writeheader()
    else:
        with samples_path.open(newline="", encoding="utf-8") as file:
            if csv.DictReader(file).fieldnames != RAW_SAMPLE_COLUMNS:
                raise RuntimeError("{} has an incompatible samples.csv header".format(path))
    metadata_path = path / "meta.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "adapter": Path(__file__).name,
                    "pair_plan": str(plan.path),
                    "reference_real_tcp": list(args.reference_real_tcp),
                    "clearance_base_z_mm": args.clearance_base_z_mm,
                    "contact_radius_mm": args.contact_radius_mm,
                    "rotation_clearance_margin_mm": args.rotation_clearance_margin_mm,
                    "max_edge_indentation_mm": args.max_edge_indentation_mm,
                    "capture_min_gray_p95": args.capture_min_gray_p95,
                    "capture_retry_count": args.capture_retry_count,
                    "capture_retry_sec": args.capture_retry_sec,
                    "speed_percent": args.speed,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return image_dir, samples_path, csv_row_count(samples_path)


def ensure_dataset_output(path: Path, args: argparse.Namespace, plan: PairPlan) -> tuple[Path, Path, Path, int, set[str]]:
    path.mkdir(parents=True, exist_ok=True)
    image_dir = path / "sensor_images"
    image_dir.mkdir(exist_ok=True)
    targets_path = path / "targets.csv"
    manifest_path = path / "manifest.csv"
    pair_records_path = path / "pair_records.csv"
    write_header_if_missing(targets_path, SIM2REAL_TARGET_COLUMNS)
    write_header_if_missing(manifest_path, SIM2REAL_MANIFEST_COLUMNS)
    write_header_if_missing(pair_records_path, PAIR_RECORD_COLUMNS)
    metadata_path = path / "auto_pair_sampler_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "adapter": Path(__file__).name,
                "pair_plan": str(plan.path),
                "reference_real_tcp": list(args.reference_real_tcp),
                "clearance_base_z_mm": args.clearance_base_z_mm,
                "contact_radius_mm": args.contact_radius_mm,
                "rotation_clearance_margin_mm": args.rotation_clearance_margin_mm,
                "max_edge_indentation_mm": args.max_edge_indentation_mm,
                "capture_min_gray_p95": args.capture_min_gray_p95,
                "capture_retry_count": args.capture_retry_count,
                "capture_retry_sec": args.capture_retry_sec,
                "speed_percent": args.speed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return image_dir, targets_path, manifest_path, csv_row_count(targets_path), completed_pair_ids(pair_records_path, plan)


def tilted_edge_offset_mm(task_pose: tuple[float, ...], contact_radius_mm: float) -> float:
    """Return the lowest-edge offset caused by the requested pitch/roll."""
    if contact_radius_mm <= 0:
        return 0.0
    normal = Rotation.from_euler("XYZ", task_pose[3:], degrees=True).apply((0.0, 0.0, 1.0))
    return float(contact_radius_mm * math.hypot(float(normal[0]), float(normal[1])))


def task_to_real_tcp(
    reference_tcp: tuple[float, ...], task_offset: tuple[float, ...], args: argparse.Namespace
) -> tuple[float, ...]:
    return task_offset_to_real_tcp(
        reference_tcp,
        task_offset,
        translation_basis_base=args.translation_basis_base,
        real_rx_sign=args.real_rx_sign,
        real_ry_sign=args.real_ry_sign,
        real_rz_sign=args.real_rz_sign,
    )


def motion_poses(reference_tcp: tuple[float, ...], row: PairPlanRow, args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    task_pose = row.required_pose()
    target = task_to_real_tcp(reference_tcp, task_pose, args)
    tilted_edge_offset = tilted_edge_offset_mm(task_pose, args.contact_radius_mm)
    approach_clearance = args.clearance_base_z_mm
    if tilted_edge_offset > 1e-6:
        approach_clearance += tilted_edge_offset + args.rotation_clearance_margin_mm
    clearance_offset = list(task_pose)
    clearance_offset[2] = -approach_clearance
    clearance = task_to_real_tcp(reference_tcp, tuple(clearance_offset), args)
    return target, clearance


def validate_row(row: PairPlanRow, reference_tcp: tuple[float, ...], args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    task_pose = row.required_pose()
    if abs(task_pose[0]) > args.max_plan_xy_mm or abs(task_pose[1]) > args.max_plan_xy_mm:
        raise ValueError("{} exceeds the configured task-plane XY bound".format(row.pair_id))
    if task_pose[2] < 0 or task_pose[2] > args.max_plan_z_mm:
        raise ValueError("{} has an unsafe task-frame Z offset {:.3f} mm".format(row.pair_id, task_pose[2]))
    if any(abs(value) > args.max_plan_rotation_deg for value in task_pose[3:]):
        raise ValueError("{} exceeds the configured rotation bound".format(row.pair_id))
    deepest_edge_indentation = task_pose[2] + tilted_edge_offset_mm(task_pose, args.contact_radius_mm)
    edge_limit = args.max_edge_indentation_mm if args.max_edge_indentation_mm is not None else args.max_plan_z_mm
    if deepest_edge_indentation > edge_limit + 1e-6:
        raise ValueError(
            "{} would indent the tactile-face edge {:.3f} mm, above the {:.3f} mm limit".format(
                row.pair_id, deepest_edge_indentation, edge_limit
            )
        )
    target, clearance = motion_poses(reference_tcp, row, args)
    expected = row.expected_real_tcp()
    if expected is None:
        raise ValueError("{} has no calibrated real_target_tcp values".format(row.pair_id))
    position_error, rotation_error = pose_error(target, expected)
    if position_error > 0.001 or rotation_error > 0.001:
        raise ValueError(
            "{} real target does not match the calibrated task-frame transform ({:.4f} mm, {:.4f} deg)".format(
                row.pair_id, position_error, rotation_error
            )
        )
    return target, clearance


def write_capture(
    raw_image_dir: Path,
    raw_samples_path: Path,
    raw_index: int,
    dataset_image_dir: Path,
    targets_path: Path,
    manifest_path: Path,
    pair_records_path: Path,
    dataset_index: int,
    row: PairPlanRow,
    plan: PairPlan,
    image,
    camera_time: float,
    capture_gray_p95: float,
    started_at: float,
    joints: tuple[float, ...],
    actual_pose: tuple[float, ...],
    raw_joints: str,
    raw_pose: str,
    verification_status: str,
    position_error_mm: float,
    rotation_error_deg: float,
    preprocessor: TacTipRuntimePreprocessor | None = None,
) -> tuple[str, str]:
    sample_id = "{:06d}".format(raw_index)
    raw_image_name = "gelsight_{}.png".format(sample_id)
    raw_image_path = raw_image_dir / raw_image_name
    if not cv2.imwrite(str(raw_image_path), image):
        raise RuntimeError("Could not write {}".format(raw_image_path))
    if preprocessor is not None:
        preprocessor.process_and_save(image, raw_image_path)

    sensor_image = "image_{}.png".format(dataset_index)
    shutil.copy2(raw_image_path, dataset_image_dir / sensor_image)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with raw_samples_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=RAW_SAMPLE_COLUMNS).writerow(
            {
                "sample_id": sample_id,
                "timestamp": timestamp,
                "elapsed_sec": "{:.6f}".format(time.time() - started_at),
                "image_file": "images/{}".format(raw_image_name),
                "camera_time": "{:.6f}".format(camera_time),
                "capture_gray_p95": "{:.3f}".format(capture_gray_p95),
                **dict(zip(JOINT_COLUMNS, format_values(joints))),
                **dict(zip(RAW_POSE_COLUMNS, format_values(actual_pose))),
                "raw_get_angle": raw_joints,
                "raw_get_pose": raw_pose,
            }
        )

    target_row = {column: "" for column in SIM2REAL_TARGET_COLUMNS}
    target_row.update(
        {
            "sensor_image": sensor_image,
            "object_label": row.object_label,
            "source_sample_id": sample_id,
            "timestamp": timestamp,
            "camera_time": "{:.6f}".format(camera_time),
        }
    )
    for column in SIM2REAL_POSE_COLUMNS + SIM2REAL_SHEAR_COLUMNS + SIM2REAL_OBJECT_COLUMNS:
        target_row[column] = row.value(column)
    for column, value in zip(JOINT_COLUMNS, format_values(joints)):
        target_row[column] = value
    for column, value in zip(SIM2REAL_TCP_COLUMNS, format_values(actual_pose)):
        target_row[column] = value
    with targets_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=SIM2REAL_TARGET_COLUMNS).writerow(target_row)

    with manifest_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=SIM2REAL_MANIFEST_COLUMNS).writerow(
            {
                "dataset_index": str(dataset_index),
                "sensor_image": sensor_image,
                "source_sample_id": sample_id,
                "source_image": "images/{}".format(raw_image_name),
                "sim_target_image": "",
                "pairing_status": "plan_recorded",
                "timestamp": timestamp,
            }
        )

    pair_record = {column: "" for column in PAIR_RECORD_COLUMNS}
    pair_record.update(
        {
            "pair_id": row.pair_id,
            "plan_index": str(row.index),
            "collector": "cr3_gelsight",
            "sensor_image": sensor_image,
            "source_sample_id": sample_id,
            "timestamp": timestamp,
            "object_label": row.object_label,
            "position_error_mm": "{:.8f}".format(position_error_mm),
            "rotation_error_deg": "{:.8f}".format(rotation_error_deg),
            "verification_status": verification_status,
            "plan_path": str(plan.path),
        }
    )
    for column in SIM2REAL_POSE_COLUMNS + SIM2REAL_SHEAR_COLUMNS + SIM2REAL_OBJECT_COLUMNS:
        pair_record[column] = row.value(column)
    for column, value in zip(ACTUAL_TCP_COLUMNS, format_values(actual_pose)):
        pair_record[column] = value
    for column, value in zip(REAL_TARGET_TCP_COLUMNS, row.values_for(REAL_TARGET_TCP_COLUMNS)):
        pair_record[column] = value
    with pair_records_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=PAIR_RECORD_COLUMNS).writerow(pair_record)
    return sample_id, sensor_image


def run(args: argparse.Namespace) -> int:
    if args.translation_basis_base is not None:
        args.translation_basis_base = translation_basis_from_values(args.translation_basis_base)
    plan = PairPlan.load(args.pair_plan)
    reference_tcp = tuple(float(value) for value in args.reference_real_tcp)
    selected_rows = list(plan.rows)
    for row in selected_rows:
        validate_row(row, reference_tcp, args)

    if not args.execute:
        print("Dry run only. Add --execute --yes-i-confirm-cr3-is-safe to move the CR3.")
        for row in selected_rows[: args.max_samples]:
            target, clearance = motion_poses(reference_tcp, row, args)
            print("{} target={} clearance={}".format(row.pair_id, format_values(target), format_values(clearance)))
        return 0

    raw_image_dir, raw_samples_path, raw_count = ensure_raw_output(args.raw_output_dir.resolve(), args, plan)
    dataset_image_dir, targets_path, manifest_path, dataset_count, completed = ensure_dataset_output(
        args.dataset_dir.resolve(), args, plan
    )
    pair_records_path = args.dataset_dir.resolve() / "pair_records.csv"
    rows = [row for row in selected_rows if row.pair_id not in completed]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        print("No unfinished pair-plan rows remain.")
        return 0

    preprocessor = create_tactip_preprocessor(args, args.raw_output_dir.resolve())
    camera = GelSightCapture(
        source=parse_camera_source(args.camera_source),
        width=args.width,
        height=args.height,
        fps=args.fps,
        timeout_sec=args.camera_read_timeout_sec,
    )
    robot = DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout)
    started_at = time.time()
    recovery_clearance = None
    needs_retract = False
    try:
        camera.open()
        robot.connect()
        mode, _raw_mode = robot.read_robot_mode()
        if mode != 5:
            raise RuntimeError("RobotMode must be 5 (enabled and idle), got {}".format(mode))
        _joints, current_pose, _raw_joints, _raw_pose = robot.read_state()
        start_position_error, start_rotation_error = pose_error(current_pose, reference_tcp)
        center_clearance = task_to_real_tcp(
            reference_tcp,
            (0.0, 0.0, -args.clearance_base_z_mm, 0.0, 0.0, 0.0),
            args,
        )
        clearance_position_error, clearance_rotation_error = pose_error(current_pose, center_clearance)
        if (
            start_position_error > args.start_position_tolerance_mm
            or start_rotation_error > args.start_rotation_tolerance_deg
        ) and (
            clearance_position_error > args.start_position_tolerance_mm
            or clearance_rotation_error > args.start_rotation_tolerance_deg
        ):
            raise RuntimeError(
                "Current TCP is neither the calibrated contact pose ({:.3f} mm, {:.3f} deg) nor the center clearance pose ({:.3f} mm, {:.3f} deg)".format(
                    start_position_error,
                    start_rotation_error,
                    clearance_position_error,
                    clearance_rotation_error,
                )
            )
        print("Start pose verified.", flush=True)

        for row in rows:
            target, clearance = motion_poses(reference_tcp, row, args)
            recovery_clearance = clearance
            print("Approach {}".format(row.pair_id), flush=True)
            robot.move_pose(clearance, args.speed, args.user, args.tool)
            needs_retract = True
            time.sleep(args.settle_sec)
            print("Contact {}".format(row.pair_id), flush=True)
            robot.move_pose(target, args.speed, args.user, args.tool)
            time.sleep(args.settle_sec)
            joints, actual_pose, raw_joints, raw_pose = robot.read_state()
            print("Verify {}".format(row.pair_id), flush=True)
            verification_status, position_error_mm, rotation_error_deg = verify_real_pose(
                row, actual_pose, args.pair_position_tolerance_mm, args.pair_rotation_tolerance_deg
            )
            if verification_status != "pose_verified":
                raise RuntimeError(
                    "{} did not reach its planned pose ({:.3f} mm, {:.3f} deg)".format(
                        row.pair_id, position_error_mm or -1.0, rotation_error_deg or -1.0
                    )
                )
            print("Capture {}".format(row.pair_id), flush=True)
            image, camera_time, capture_gray_p95 = read_quality_checked_frame(
                camera,
                args.capture_min_gray_p95,
                args.capture_retry_count,
                args.capture_retry_sec,
                newer_than=time.time(),
            )
            raw_count += 1
            dataset_count += 1
            sample_id, sensor_image = write_capture(
                raw_image_dir,
                raw_samples_path,
                raw_count,
                dataset_image_dir,
                targets_path,
                manifest_path,
                pair_records_path,
                dataset_count,
                row,
                plan,
                image,
                camera_time,
                capture_gray_p95,
                started_at,
                joints,
                actual_pose,
                raw_joints,
                raw_pose,
                verification_status,
                position_error_mm or 0.0,
                rotation_error_deg or 0.0,
                preprocessor,
            )
            print(
                "Captured {} as {} (gray p95 {:.1f})".format(row.pair_id, sensor_image, capture_gray_p95),
                flush=True,
            )
            robot.move_pose(clearance, args.speed, args.user, args.tool)
            needs_retract = False

        final_clearance = task_to_real_tcp(
            reference_tcp,
            (0.0, 0.0, -args.clearance_base_z_mm, 0.0, 0.0, 0.0),
            args,
        )
        robot.move_pose(final_clearance, args.speed, args.user, args.tool)
        print("Completed {} physical pair samples; CR3 is at the center clearance pose.".format(len(rows)))
        return 0
    finally:
        if needs_retract and recovery_clearance is not None:
            try:
                robot.move_pose(recovery_clearance, args.speed, args.user, args.tool)
            except Exception as exc:
                print("WARNING: automatic retraction failed: {}".format(exc), file=sys.stderr, flush=True)
        camera.close()
        robot.close()
        if preprocessor is not None:
            preprocessor.close()


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print("AUTO_PAIR_SAMPLER_FAILED: {}: {}".format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
