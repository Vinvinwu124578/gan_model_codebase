"""Manual live GelSight sampler and jog panel for a Dobot CR3.

The script opens a small Tk window, streams the GelSight camera, provides
optional CR3 jog/move controls, and saves one image plus the current CR3 joint
state whenever the operator clicks Sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import numpy as np

try:
    from dobot_cr3_client import DASHBOARD_PORT, DEFAULT_IP, MOVE_PORT, parse_reply_values, reply_error_code
except ImportError as exc:  # pragma: no cover - clearer runtime error for wrong cwd.
    raise SystemExit(
        "Could not import dobot_cr3_client.py. Run this script from the Tactistruct tools folder "
        "or use tools/run_live_cr3_gelsight_sampler.ps1."
    ) from exc

PAIRING_IMPORT_ERROR = None
try:
    from tactile_sim2real.pairing import (
        ACTUAL_TCP_COLUMNS,
        PAIR_RECORD_COLUMNS,
        REAL_TARGET_TCP_COLUMNS,
        PairPlan,
        PairPlanRow,
        verify_real_pose,
        write_header_if_missing,
    )
except ImportError as exc:  # The normal live sampler remains usable without the optional adapter package.
    PAIRING_IMPORT_ERROR = exc
    ACTUAL_TCP_COLUMNS = []
    PAIR_RECORD_COLUMNS = []
    REAL_TARGET_TCP_COLUMNS = []
    PairPlan = None
    PairPlanRow = None
    verify_real_pose = None
    write_header_if_missing = None

from tactip_runtime_preprocess import add_tactip_preprocess_args, create_tactip_preprocessor


JOINT_COLUMNS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
POSE_COLUMNS = ["pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6"]
SIM2REAL_POSE_COLUMNS = ["pose_x", "pose_y", "pose_z", "pose_Rx", "pose_Ry", "pose_Rz"]
SIM2REAL_SHEAR_COLUMNS = ["shear_x", "shear_y", "shear_z", "shear_Rx", "shear_Ry", "shear_Rz"]
SIM2REAL_OBJECT_COLUMNS = ["object_x", "object_y", "object_z", "object_Rx", "object_Ry", "object_Rz"]
SIM2REAL_TCP_COLUMNS = ["tcp_x", "tcp_y", "tcp_z", "tcp_Rx", "tcp_Ry", "tcp_Rz"]
SIM2REAL_TARGET_COLUMNS = [
    "sensor_image",
    "object_label",
    *SIM2REAL_POSE_COLUMNS,
    *SIM2REAL_SHEAR_COLUMNS,
    *SIM2REAL_OBJECT_COLUMNS,
    "source_sample_id",
    "timestamp",
    "camera_time",
    *JOINT_COLUMNS,
    *SIM2REAL_TCP_COLUMNS,
]
SIM2REAL_MANIFEST_COLUMNS = [
    "dataset_index",
    "sensor_image",
    "source_sample_id",
    "source_image",
    "sim_target_image",
    "pairing_status",
    "timestamp",
]
PAIR_PLAN_COMPLETED_STATUSES = {"pose_verified", "pose_unverified"}
ROBOT_MODE_NAMES = {
    1: "initializing",
    2: "brake open",
    4: "disabled",
    5: "enabled and idle",
    6: "drag mode",
    7: "running",
    8: "drag recording",
    9: "alarm",
    10: "paused",
    11: "jogging",
}


def timestamp_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


DEFAULT_SHARED_OUTPUT_NAME = "manual_cr3_gelsight_shared"


def parse_camera_source(value: str):
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def request_macos_camera_access(timeout_sec: float = 60.0) -> tuple[bool, str]:
    """Request camera permission on the main thread before OpenCV opens AVFoundation."""
    if sys.platform != "darwin":
        return True, ""
    try:
        from AppKit import NSApplication
        from AVFoundation import (
            AVCaptureDevice,
            AVAuthorizationStatusAuthorized,
            AVAuthorizationStatusNotDetermined,
            AVMediaTypeVideo,
        )
        from Foundation import NSDate, NSRunLoop
    except ImportError:
        return False, "macOS camera support is missing. Run: pip install -r requirements.txt"

    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if status == AVAuthorizationStatusAuthorized:
        return True, ""
    if status != AVAuthorizationStatusNotDetermined:
        return False, "Camera access was denied in macOS Privacy & Security settings."

    result: list[bool] = []

    def completed(granted: bool) -> None:
        result.append(bool(granted))

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, completed)
    deadline = time.monotonic() + max(1.0, timeout_sec)
    run_loop = NSRunLoop.currentRunLoop()
    while not result and time.monotonic() < deadline:
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    if result and result[0]:
        return True, ""
    if result:
        return False, "Camera access was not granted."
    return False, "Camera permission request timed out."


def format_values(values: Optional[tuple[float, ...]]) -> list[str]:
    if values is None:
        return [""] * 6
    return ["{:.8f}".format(float(v)) for v in values[:6]]


def frame_to_photo_image(frame_bgr, preview_scale: float) -> tk.PhotoImage:
    import cv2

    if preview_scale > 0 and abs(preview_scale - 1.0) > 1e-6:
        height, width = frame_bgr.shape[:2]
        size = (max(1, int(width * preview_scale)), max(1, int(height * preview_scale)))
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    ppm = b"P6\n%d %d\n255\n" % (width, height) + frame_rgb.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")


class DobotCR3LiveClient:
    def __init__(self, ip: str, dashboard_port: int, move_port: int, timeout: float) -> None:
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.timeout = timeout
        self.dashboard_sock: Optional[socket.socket] = None
        self.move_sock: Optional[socket.socket] = None
        self.dashboard_lock = threading.Lock()
        self.move_lock = threading.Lock()
        self.warning_lock = threading.Lock()
        self.pending_warnings: list[str] = []
        self.motion_initialized = False
        self.jog_speed_percent: Optional[int] = None

    def connect(self) -> None:
        self.close()
        self.dashboard_sock = socket.create_connection((self.ip, self.dashboard_port), timeout=self.timeout)
        self.dashboard_sock.settimeout(self.timeout)
        self.move_sock = socket.create_connection((self.ip, self.move_port), timeout=self.timeout)
        self.move_sock.settimeout(self.timeout)
        self.motion_initialized = False
        self.jog_speed_percent = None

    def close(self) -> None:
        for attr in ("move_sock", "dashboard_sock"):
            sock = getattr(self, attr)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                setattr(self, attr, None)
        self.motion_initialized = False
        self.jog_speed_percent = None

    def _ensure_move_sock(self) -> socket.socket:
        if self.move_sock is None:
            self.move_sock = socket.create_connection((self.ip, self.move_port), timeout=self.timeout)
            self.move_sock.settimeout(self.timeout)
        return self.move_sock

    def _ensure_dashboard_sock(self) -> socket.socket:
        if self.dashboard_sock is None:
            self.dashboard_sock = socket.create_connection((self.ip, self.dashboard_port), timeout=self.timeout)
            self.dashboard_sock.settimeout(self.timeout)
        return self.dashboard_sock

    def _close_dashboard_sock_unlocked(self) -> None:
        if self.dashboard_sock is not None:
            try:
                self.dashboard_sock.close()
            except OSError:
                pass
            self.dashboard_sock = None

    def reset_dashboard_connection(self) -> None:
        with self.dashboard_lock:
            self._close_dashboard_sock_unlocked()

    def _send_recv(self, sock: socket.socket, command: str, timeout: Optional[float] = None) -> str:
        old_timeout = sock.gettimeout()
        sock.settimeout(self.timeout if timeout is None else timeout)
        try:
            sock.sendall(command.encode("utf-8"))
            data = sock.recv(4096)
        finally:
            sock.settimeout(old_timeout)
        if not data:
            raise RuntimeError("No reply for command: {}".format(command))
        reply = data.decode("utf-8", errors="replace").strip()
        code = reply_error_code(reply)
        if code not in (None, 0):
            raise RuntimeError("Command failed: {}\nReply: {}".format(command, reply))
        return reply

    def dashboard_cmd(self, command: str, timeout: Optional[float] = None) -> str:
        with self.dashboard_lock:
            try:
                return self._send_recv(self._ensure_dashboard_sock(), command, timeout=timeout)
            except (OSError, RuntimeError):
                if self.dashboard_sock is not None:
                    try:
                        self.dashboard_sock.close()
                    except OSError:
                        pass
                    self.dashboard_sock = None
                return self._send_recv(self._ensure_dashboard_sock(), command, timeout=timeout)

    def move_cmd(self, command: str, timeout: Optional[float] = None) -> str:
        with self.move_lock:
            try:
                return self._send_recv(self._ensure_move_sock(), command, timeout=timeout)
            except (OSError, RuntimeError):
                if self.move_sock is not None:
                    try:
                        self.move_sock.close()
                    except OSError:
                        pass
                    self.move_sock = None
                return self._send_recv(self._ensure_move_sock(), command, timeout=timeout)

    def jog_cmd(self, command: str) -> str:
        with self.move_lock:
            try:
                sock = self._ensure_move_sock()
                sock.sendall(command.encode("utf-8"))
                old_timeout = sock.gettimeout()
                sock.settimeout(0.2)
                try:
                    data = sock.recv(4096)
                    if data:
                        return data.decode("utf-8", errors="replace").strip()
                except socket.timeout:
                    return command
                finally:
                    sock.settimeout(old_timeout)
            except OSError:
                if self.move_sock is not None:
                    try:
                        self.move_sock.close()
                    except OSError:
                        pass
                    self.move_sock = None
                sock = self._ensure_move_sock()
                sock.sendall(command.encode("utf-8"))
            return command

    def read_state(self) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        raw_joints = self.dashboard_cmd("GetAngle()")
        raw_pose = self.dashboard_cmd("GetPose()")
        joints = self._parse_values(raw_joints, 6, "GetAngle")
        pose = self._parse_values(raw_pose, 6, "GetPose")
        return joints, pose, raw_joints, raw_pose

    def inverse_solution(
        self,
        pose: tuple[float, ...],
        user: int,
        tool: int,
        joint_near: Optional[tuple[float, ...]] = None,
    ) -> tuple[tuple[float, ...], str]:
        """Ask the controller to validate a TCP pose without executing motion."""
        target = tuple(float(value) for value in pose)
        if len(target) != 6:
            raise ValueError("InverseSolution requires six TCP values")
        command = "InverseSolution({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:d},{:d}".format(
            *target,
            int(user),
            int(tool),
        )
        if joint_near is None:
            command += ",0)"
        else:
            near = tuple(float(value) for value in joint_near)
            if len(near) != 6:
                raise ValueError("InverseSolution joint-near hint requires six joint values")
            command += ",1,{{{}}})".format(",".join("{:.6f}".format(value) for value in near))
        raw = self.dashboard_cmd(command, timeout=3.0)
        return self._parse_values(raw, 6, "InverseSolution"), raw

    def read_joints(self) -> tuple[tuple[float, ...], str]:
        raw_joints = self.dashboard_cmd("GetAngle()")
        return self._parse_values(raw_joints, 6, "GetAngle"), raw_joints

    def read_joints_for_recovery(self, attempts: int = 6) -> tuple[tuple[float, ...], str]:
        errors = []
        for attempt in range(max(1, attempts)):
            try:
                if attempt > 0:
                    self.reset_dashboard_connection()
                    time.sleep(0.15)
                return self.read_joints()
            except Exception as exc:
                errors.append("{}: {}".format(type(exc).__name__, exc))
        try:
            raw_joints = self.move_cmd("GetAngle()", timeout=1.0)
            return self._parse_values(raw_joints, 6, "GetAngle"), raw_joints
        except Exception as exc:
            errors.append("move-port {}: {}".format(type(exc).__name__, exc))
        raise RuntimeError(
            "Could not read GetAngle() for Recovery after retries. "
            "Recovery will not move without a verified current joint state. "
            "Last errors: {}".format(" | ".join(errors[-4:]))
        )

    def read_pose(self) -> tuple[tuple[float, ...], str]:
        raw_pose = self.dashboard_cmd("GetPose()")
        return self._parse_values(raw_pose, 6, "GetPose"), raw_pose

    def read_robot_mode(self) -> tuple[int, str]:
        raw_mode = self.dashboard_cmd("RobotMode()", timeout=2.0)
        mode = int(round(self._parse_values(raw_mode, 1, "RobotMode")[0]))
        return mode, raw_mode

    def require_motion_ready(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + max(0.1, timeout)
        last_mode = None
        while time.monotonic() < deadline:
            mode, _raw_mode = self.read_robot_mode()
            last_mode = mode
            if mode == 5:
                return
            if mode == 9:
                raise RuntimeError("RobotMode is alarm (9). Clear the alarm on the controller, then try again.")
            time.sleep(0.15)
        mode_name = ROBOT_MODE_NAMES.get(last_mode, "unknown")
        raise RuntimeError(
            "RobotMode is {} ({}), not enabled and idle (5). "
            "Use Exit Drag (verify) when the robot is in drag mode, then try jogging again.".format(last_mode, mode_name)
        )

    def set_speed(self, speed: float) -> str:
        if not 1 <= speed <= 100:
            raise ValueError("SpeedFactor must be in [1, 100].")
        speed_i = round(speed)
        replies = [
            self._best_effort_dashboard_cmd("SpeedFactor({:d})".format(speed_i), timeout=1.0),
            self._best_effort_dashboard_cmd("SpeedJ({:d})".format(speed_i), timeout=1.0),
            self._best_effort_dashboard_cmd("SpeedL({:d})".format(speed_i), timeout=1.0),
        ]
        self.jog_speed_percent = speed_i
        return "\n".join(replies)

    def set_user_tool(self, user: int, tool: int) -> list[str]:
        return [self.dashboard_cmd("User({:d})".format(user)), self.dashboard_cmd("Tool({:d})".format(tool))]

    def _best_effort_dashboard_cmd(self, command: str, timeout: float = 1.0, record_warning: bool = True) -> str:
        with self.dashboard_lock:
            try:
                if self.dashboard_sock is None:
                    self.dashboard_sock = socket.create_connection((self.ip, self.dashboard_port), timeout=timeout)
                    self.dashboard_sock.settimeout(timeout)
                return self._send_recv(self.dashboard_sock, command, timeout=timeout)
            except Exception as exc:
                self._close_dashboard_sock_unlocked()
                warning = "WARN {} failed: {}: {}".format(command, type(exc).__name__, exc)
                if record_warning:
                    with self.warning_lock:
                        self.pending_warnings.append(warning)
                return warning

    def _record_warning(self, warning: str) -> str:
        with self.warning_lock:
            self.pending_warnings.append(warning)
        return warning

    def pop_warnings(self) -> list[str]:
        with self.warning_lock:
            warnings = list(self.pending_warnings)
            self.pending_warnings.clear()
        return warnings

    def initialize_motion(self, speed: float, user: int, tool: int) -> list[str]:
        speed_i = round(speed)
        replies = [
            self._best_effort_dashboard_cmd("EnableRobot()", timeout=3.0),
            self._best_effort_dashboard_cmd("User({:d})".format(user), timeout=1.0),
            self._best_effort_dashboard_cmd("Tool({:d})".format(tool), timeout=1.0),
            self._best_effort_dashboard_cmd("SpeedFactor({:d})".format(speed_i), timeout=1.0),
            self._best_effort_dashboard_cmd("SpeedJ({:d})".format(speed_i), timeout=1.0),
            self._best_effort_dashboard_cmd("SpeedL({:d})".format(speed_i), timeout=1.0),
        ]
        self.motion_initialized = True
        self.jog_speed_percent = speed_i
        return replies

    def prepare_motion(self, speed: float, user: int, tool: int) -> None:
        if not self.motion_initialized:
            self.initialize_motion(speed, user, tool)
            return
        speed_i = round(speed)
        self._best_effort_dashboard_cmd("SpeedFactor({:d})".format(speed_i), timeout=1.0)

    def prepare_jog_motion(self, speed: float, user: int, tool: int) -> None:
        speed_i = round(speed)
        if not self.motion_initialized:
            self.initialize_motion(speed, user, tool)
            return
        if self.jog_speed_percent == speed_i:
            return
        self._best_effort_dashboard_cmd("SpeedFactor({:d})".format(speed_i), timeout=0.2)
        self._best_effort_dashboard_cmd("SpeedJ({:d})".format(speed_i), timeout=0.2)
        self._best_effort_dashboard_cmd("SpeedL({:d})".format(speed_i), timeout=0.2)
        self.motion_initialized = True
        self.jog_speed_percent = speed_i

    def clear_error(self) -> str:
        return self.dashboard_cmd("ClearError()")

    def enable(self) -> str:
        return self.dashboard_cmd("EnableRobot()", timeout=20.0)

    def disable(self) -> str:
        return self.dashboard_cmd("DisableRobot()")

    def stop_drag(self) -> str:
        return self.dashboard_cmd("StopDrag()")

    def exit_drag_and_verify(self, timeout: float = 3.0) -> tuple[str, str]:
        raw_stop = self.stop_drag()
        deadline = time.monotonic() + max(0.1, timeout)
        last_mode = None
        raw_mode = ""
        while time.monotonic() < deadline:
            last_mode, raw_mode = self.read_robot_mode()
            if last_mode == 5:
                return raw_stop, raw_mode
            time.sleep(0.15)
        mode_name = ROBOT_MODE_NAMES.get(last_mode, "unknown")
        raise RuntimeError(
            "StopDrag() was sent, but RobotMode is still {} ({}). "
            "Release the end-effector hand-guiding button and disable any external DI configured for drag, "
            "then exit Drag in DobotStudio Pro and try again.".format(last_mode, mode_name)
        )

    def reset(self) -> str:
        return self.dashboard_cmd("ResetRobot()")

    def emergency_stop(self) -> str:
        return self.dashboard_cmd("EmergencyStop()")

    def recover_j5_from_singularity(
        self,
        step_deg: float = 5.0,
        speed: float = 5.0,
        user: int = 0,
        tool: int = 0,
    ) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        speed_i = max(1, min(20, round(speed)))
        prep_replies = [
            self._best_effort_dashboard_cmd("ClearError()", timeout=0.5, record_warning=False),
            self._best_effort_dashboard_cmd("EnableRobot()", timeout=3.0, record_warning=False),
            self._best_effort_dashboard_cmd("User({:d})".format(user), timeout=0.5, record_warning=False),
            self._best_effort_dashboard_cmd("Tool({:d})".format(tool), timeout=0.5, record_warning=False),
            self._best_effort_dashboard_cmd("SpeedFactor({:d})".format(speed_i), timeout=0.5, record_warning=False),
            self._best_effort_dashboard_cmd("SpeedJ({:d})".format(speed_i), timeout=0.5, record_warning=False),
            self._best_effort_dashboard_cmd("SpeedL({:d})".format(speed_i), timeout=0.5, record_warning=False),
        ]
        failed_prep = [line for line in prep_replies if isinstance(line, str) and line.startswith("WARN")]
        if failed_prep:
            self._record_warning(
                "WARN Recovery J5 dashboard prep had {} warning(s); using command-level SpeedJ={}/AccJ={} on JointMovJ.".format(
                    len(failed_prep), speed_i, speed_i
                )
            )

        self.reset_dashboard_connection()
        time.sleep(0.25)
        joints, raw_joints_before = self.read_joints_for_recovery()
        target = list(joints)
        direction = -1.0 if target[4] >= 0.0 else 1.0
        target[4] += direction * abs(float(step_deg))
        target[4] = max(-175.0, min(175.0, target[4]))

        raw_move = self.move_cmd(
            "JointMovJ({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},SpeedJ={:d},AccJ={:d})".format(
                *target, speed_i, speed_i
            ),
            timeout=10.0,
        )
        try:
            raw_sync = self.move_cmd("Sync()", timeout=60.0)
        except Exception as exc:
            raw_sync = self._record_warning(
                "WARN Recovery J5 Sync() failed after JointMovJ: {}: {}".format(type(exc).__name__, exc)
            )
            time.sleep(0.5)
        try:
            joints_now, pose_now, raw_joints, raw_pose = self.read_state()
            return joints_now, pose_now, raw_joints, raw_pose
        except Exception:
            return tuple(float(v) for v in target), None, raw_joints_before, "{}\n{}".format(raw_move, raw_sync)

    def move_joints(self, joints: tuple[float, ...], speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        self.prepare_motion(speed, user, tool)
        self.require_motion_ready()
        raw_move = self.move_cmd("JointMovJ({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f})".format(*joints), timeout=10.0)
        raw_sync = self.move_cmd("Sync()", timeout=180.0)
        try:
            joints_now, pose_now, raw_joints, raw_pose = self.read_state()
            return joints_now, pose_now, raw_joints, raw_pose
        except Exception:
            return tuple(float(v) for v in joints), None, raw_move, raw_sync

    def move_pose(self, pose: tuple[float, ...], speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        self.prepare_motion(speed, user, tool)
        self.require_motion_ready()
        raw_move = self.move_cmd("MovL({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f})".format(*pose), timeout=10.0)
        raw_sync = self.move_cmd("Sync()", timeout=180.0)
        try:
            joints_now, pose_now, raw_joints, raw_pose = self.read_state()
            return joints_now, pose_now, raw_joints, raw_pose
        except Exception:
            return None, tuple(float(v) for v in pose), raw_move, raw_sync

    def jog_joint(self, joint_index: int, direction: float, step: float, speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        joints, _raw_joints = self.read_joints()
        target = list(joints)
        target[joint_index] += float(direction) * float(step)
        return self.move_joints(tuple(target), speed, user, tool)

    def jog_pose_axis(self, axis_index: int, direction: float, step: float, speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        pose, _raw_pose = self.read_pose()
        target = list(pose)
        target[axis_index] += float(direction) * float(step)
        return self.move_pose(tuple(target), speed, user, tool)

    def start_move_jog(self, axis: str, speed: float, user: int, tool: int) -> str:
        self.prepare_jog_motion(speed, user, tool)
        self.require_motion_ready()
        return self.jog_cmd("MoveJog({})".format(axis))

    def stop_move_jog(self) -> str:
        return self.jog_cmd("MoveJog()")

    @staticmethod
    def _parse_values(reply: str, expected: int, label: str) -> tuple[float, ...]:
        try:
            return parse_reply_values(reply, expected, label)
        except RuntimeError as exc:
            if "Control mode is not TCP" in reply or "{}" in reply:
                raise RuntimeError(
                    "{} returned no robot state. The CR3 is not in TCP/IP secondary-development mode. "
                    "In DobotStudio Pro, connect to the robot and switch the robot mode to TCP/IP, "
                    "then click Sample again. Raw reply: {}".format(label, reply)
                ) from exc
            raise


class DryRunCR3Reader:
    def __init__(self) -> None:
        self.sample_index = 0
        self.joints = tuple(float(v) for v in (0.0, 30.0, 110.0, -50.0, -90.0, 0.0))
        self.pose = tuple(float(v) for v in (250.0, 0.0, 120.0, 180.0, 0.0, 0.0))
        self.speed = 50.0

    def read_state(self) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        self.sample_index += 1
        return self.joints, self.pose, "DRY_RUN_GetAngle", "DRY_RUN_GetPose"

    def read_joints(self) -> tuple[tuple[float, ...], str]:
        return self.joints, "DRY_RUN_GetAngle"

    def read_pose(self) -> tuple[tuple[float, ...], str]:
        return self.pose, "DRY_RUN_GetPose"

    def set_speed(self, speed: float) -> str:
        self.speed = speed
        return "DRY_RUN SpeedFactor/SpeedJ/SpeedL({:.1f})".format(speed)

    def set_user_tool(self, user: int, tool: int) -> list[str]:
        return ["DRY_RUN User({})".format(user), "DRY_RUN Tool({})".format(tool)]

    def initialize_motion(self, speed: float, user: int, tool: int) -> list[str]:
        self.speed = speed
        return [
            "DRY_RUN ClearError()",
            "DRY_RUN EnableRobot()",
            "DRY_RUN User({})".format(user),
            "DRY_RUN Tool({})".format(tool),
            "DRY_RUN SpeedFactor({:.1f})".format(speed),
        ]

    def clear_error(self) -> str:
        return "DRY_RUN ClearError()"

    def enable(self) -> str:
        return "DRY_RUN EnableRobot()"

    def disable(self) -> str:
        return "DRY_RUN DisableRobot()"

    def stop_drag(self) -> str:
        return "DRY_RUN StopDrag()"

    def exit_drag_and_verify(self, timeout: float = 3.0) -> str:
        return self.stop_drag()

    def reset(self) -> str:
        return "DRY_RUN ResetRobot()"

    def emergency_stop(self) -> str:
        return "DRY_RUN EmergencyStop()"

    def recover_j5_from_singularity(
        self,
        step_deg: float = 5.0,
        speed: float = 5.0,
        user: int = 0,
        tool: int = 0,
    ) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        joints = list(self.joints)
        direction = -1.0 if joints[4] >= 0.0 else 1.0
        joints[4] += direction * abs(float(step_deg))
        joints[4] = max(-175.0, min(175.0, joints[4]))
        self.joints = tuple(joints)
        self.speed = speed
        return self.read_state()

    def move_joints(self, joints: tuple[float, ...], speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        self.joints = tuple(float(v) for v in joints)
        self.speed = speed
        return self.read_state()

    def move_pose(self, pose: tuple[float, ...], speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        self.pose = tuple(float(v) for v in pose)
        self.speed = speed
        return self.read_state()

    def jog_joint(self, joint_index: int, direction: float, step: float, speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        joints = list(self.joints)
        joints[joint_index] += float(direction) * float(step)
        return self.move_joints(tuple(joints), speed, user, tool)

    def jog_pose_axis(self, axis_index: int, direction: float, step: float, speed: float, user: int, tool: int) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
        pose = list(self.pose)
        pose[axis_index] += float(direction) * float(step)
        return self.move_pose(tuple(pose), speed, user, tool)

    def start_move_jog(self, axis: str, speed: float, user: int, tool: int) -> str:
        self.speed = speed
        return "DRY_RUN MoveJog({})".format(axis)

    def stop_move_jog(self) -> str:
        return "DRY_RUN MoveJog()"

    def pop_warnings(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


class CameraWorker:
    def __init__(
        self,
        source,
        width: int,
        height: int,
        fps: float,
        backend: str,
        mirror: bool,
        rotate: int,
        synthetic: bool,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.mirror = mirror
        self.rotate = rotate
        self.synthetic = synthetic
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_time = 0.0
        self.error = ""
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.capture = None
        self.frame_index = 0

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._run, name="gelsight-camera", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._close_capture()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.latest_time, self.error
            return self.latest_frame.copy(), self.latest_time, self.error

    def _run(self) -> None:
        while self.running:
            try:
                if self.synthetic:
                    frame = self._synthetic_frame()
                    ok = True
                else:
                    if self.capture is None:
                        self._open_capture()
                    ok, frame = self.capture.read()
                if not ok or frame is None:
                    self._set_error("Could not read frame from camera source {}".format(self.source))
                    self._close_capture()
                    time.sleep(0.5)
                    continue
                frame = self._postprocess(frame)
                with self.lock:
                    self.latest_frame = frame
                    self.latest_time = time.time()
                    self.error = ""
                time.sleep(max(0.0, 1.0 / self.fps) if self.fps > 0 else 0.0)
            except Exception as exc:
                self._set_error("Waiting for camera source {}: {}".format(self.source, exc))
                self._close_capture()
                time.sleep(1.0)

    def _close_capture(self) -> None:
        capture = self.capture
        self.capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def _open_capture(self) -> None:
        import cv2

        if self.backend == "dshow" and isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif self.backend == "msmf" and isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
        else:
            self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera source {}".format(self.source))
        if self.width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def _postprocess(self, frame):
        import cv2

        if self.width > 0 and self.height > 0:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _synthetic_frame(self):
        import cv2

        self.frame_index += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        center = (self.width // 2, self.height // 2)
        radius = 30 + (self.frame_index % 80)
        cv2.circle(frame, center, radius, (50, 140, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "DRY RUN GelSight frame {}".format(self.frame_index),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _set_error(self, message: str) -> None:
        with self.lock:
            self.error = message


class LiveSamplerApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.image_dir = self.output_dir / "images"
        self.csv_path = self.output_dir / "samples.csv"
        self.meta_path = self.output_dir / "meta.json"
        self.sim2real_dataset_dir = (
            Path(args.sim2real_dataset_dir).expanduser().resolve() if args.sim2real_dataset_dir else None
        )
        self.sim2real_image_dir = self.sim2real_dataset_dir / "sensor_images" if self.sim2real_dataset_dir else None
        self.sim2real_targets_path = self.sim2real_dataset_dir / "targets.csv" if self.sim2real_dataset_dir else None
        self.sim2real_manifest_path = self.sim2real_dataset_dir / "manifest.csv" if self.sim2real_dataset_dir else None
        self.pair_plan = None
        self.pair_records_path = self.sim2real_dataset_dir / "pair_records.csv" if self.sim2real_dataset_dir else None
        self.pair_last_status = ""
        if args.pair_plan:
            self.pair_plan = PairPlan.load(Path(args.pair_plan))
            self.pair_plan.start_at(args.pair_plan_start)
        self.sim2real_index = 0
        self.sample_index = 0
        self.photo = None
        self.started_at = time.time()
        self.last_status = ""
        self._prepare_output()
        self.preprocessor = create_tactip_preprocessor(args, self.output_dir)

        source = parse_camera_source(args.camera_source)
        self.camera = CameraWorker(
            source=source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=args.camera_backend,
            mirror=args.mirror,
            rotate=args.rotate,
            synthetic=args.dry_run_camera,
        )
        self.robot = (
            DryRunCR3Reader()
            if args.dry_run_robot
            else DobotCR3LiveClient(args.robot_ip, args.dashboard_port, args.move_port, args.robot_timeout)
        )
        self.robot_busy = False
        self.active_jog_axis = None
        self.jog_busy = False
        self.closed = False
        self.state_poll_inflight = False
        self.state_poll_after_id = None
        self.last_state_status = "robot state: waiting"
        self.capture_once_done = False
        self.capture_once_deadline = (
            time.monotonic() + args.capture_once_timeout_sec if args.capture_once else None
        )

        self._build_ui()
        self._load_current_pair_target()
        self.root.after_idle(self._start_camera)
        self.root.after(30, self._update_frame)
        if args.capture_once:
            self.root.after(100, self._capture_once_when_ready)
        if int(self.args.robot_state_poll_ms) > 0:
            self.state_poll_after_id = self.root.after(250, self._poll_robot_state)

    def _start_camera(self) -> None:
        if self.closed:
            return
        if not self.args.dry_run_camera:
            granted, message = request_macos_camera_access()
            if not granted:
                self.status_var.set(message)
                return
        self.camera.start()

    def _capture_once_when_ready(self) -> None:
        if self.closed or self.capture_once_done:
            return
        frame, _frame_time, error = self.camera.get_latest()
        if frame is None:
            if self.capture_once_deadline is not None and time.monotonic() >= self.capture_once_deadline:
                message = "capture-once timed out waiting for a camera frame"
                if error:
                    message += ": {}".format(error)
                self.last_status = message
                print(message, file=sys.stderr, flush=True)
                self.root.after(0, self.close)
                return
            self.root.after(100, self._capture_once_when_ready)
            return
        self.capture_once_done = True
        self.sample()
        self.root.after(100, self.close)

    def _prepare_output(self) -> None:
        if self.output_dir.exists() and self.args.reset_output:
            shutil.rmtree(self.output_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists() and not self.args.reset_output:
            with self.csv_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.sample_index = len(rows)
        else:
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "sample_id",
                        "timestamp",
                        "elapsed_sec",
                        "image_file",
                        "camera_time",
                        "robot_user",
                        "robot_tool",
                    ]
                    + JOINT_COLUMNS
                    + POSE_COLUMNS
                    + ["raw_get_angle", "raw_get_pose"]
                )
        meta = {
            "script": Path(__file__).name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "robot": {
                "type": "Dobot CR3",
                "ip": self.args.robot_ip,
                "dashboard_port": self.args.dashboard_port,
                "move_port": self.args.move_port,
                "default_speed_percent": self.args.robot_speed,
                "default_user": self.args.robot_user,
                "default_tool": self.args.robot_tool,
                "default_joint_step_deg": self.args.joint_step_deg,
                "default_tcp_step_mm": self.args.tcp_step_mm,
                "default_rot_step_deg": self.args.rot_step_deg,
                "robot_state_poll_ms": self.args.robot_state_poll_ms,
                "set_speed_on_motion": bool(self.args.set_speed_on_motion),
                "confirm_motion": bool(self.args.confirm_motion),
                "joint_units": "degrees",
                "pose_units": "mm_degrees",
                "dry_run_robot": bool(self.args.dry_run_robot),
            },
            "camera": {
                "source": self.args.camera_source,
                "width": self.args.width,
                "height": self.args.height,
                "fps": self.args.fps,
                "backend": self.args.camera_backend,
                "mirror": bool(self.args.mirror),
                "rotate": self.args.rotate,
                "dry_run_camera": bool(self.args.dry_run_camera),
            },
            "output_dir": str(self.output_dir),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._prepare_sim2real_output(meta)

    def _prepare_sim2real_output(self, meta: dict) -> None:
        if self.sim2real_dataset_dir is None:
            return
        if self.sim2real_dataset_dir.exists() and self.args.reset_sim2real_dataset:
            shutil.rmtree(self.sim2real_dataset_dir)
        self.sim2real_image_dir.mkdir(parents=True, exist_ok=True)
        for path, columns in (
            (self.sim2real_targets_path, SIM2REAL_TARGET_COLUMNS),
            (self.sim2real_manifest_path, SIM2REAL_MANIFEST_COLUMNS),
        ):
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=columns).writeheader()
        if self.pair_plan is not None:
            write_header_if_missing(self.pair_records_path, PAIR_RECORD_COLUMNS)
            if not self.args.pair_plan_start:
                completed_pair_ids = set()
                with self.pair_records_path.open(newline="", encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        if (
                            row.get("plan_path") == str(self.pair_plan.path)
                            and row.get("verification_status") in PAIR_PLAN_COMPLETED_STATUSES
                        ):
                            completed_pair_ids.add(row.get("pair_id", ""))
                self.pair_plan.resume_after(completed_pair_ids)

        max_index = 0
        with self.sim2real_targets_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                image_stem = Path(row.get("sensor_image", "")).stem
                if image_stem.startswith("image_") and image_stem[6:].isdigit():
                    max_index = max(max_index, int(image_stem[6:]))
        self.sim2real_index = max_index

        sensor_params_path = self.sim2real_dataset_dir / "sensor_image_params.json"
        if not sensor_params_path.exists() or self.args.reset_sim2real_dataset:
            sensor_params = {
                "type": "gelsight",
                "source": self.args.camera_source,
                "width": self.args.width,
                "height": self.args.height,
                "fps": self.args.fps,
                "color_order": "BGR",
                "image_format": "png",
            }
            sensor_params_path.write_text(json.dumps(sensor_params, indent=2), encoding="utf-8")

        env_params_path = self.sim2real_dataset_dir / "env_params.json"
        if not env_params_path.exists() or self.args.reset_sim2real_dataset:
            env_params = {
                "robot": "cr3",
                "coordinate_frame": "robot_base",
                "tcp_units": "mm_degrees",
                "note": "pose_* and shear_* labels are intentionally blank until calibrated in a task work frame.",
            }
            env_params_path.write_text(json.dumps(env_params, indent=2), encoding="utf-8")

        collect_params_path = self.sim2real_dataset_dir / "collect_params.json"
        if not collect_params_path.exists() or self.args.reset_sim2real_dataset:
            collect_params = {
                "adapter": "live_cr3_gelsight_sampler",
                "object_label": self.args.sim2real_object_label,
                "paired_training_ready": False,
                "pairing_note": "Add a matching simulated image path to manifest.csv before pix2pix training.",
            }
            collect_params_path.write_text(json.dumps(collect_params, indent=2), encoding="utf-8")

        adapter_meta = {
            "source_sampler_meta": meta,
            "dataset_dir": str(self.sim2real_dataset_dir),
            "created_or_resumed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.pair_plan is not None:
            adapter_meta["pair_plan"] = {
                "path": str(self.pair_plan.path),
                "position_tolerance_mm": self.args.pair_position_tolerance_mm,
                "rotation_tolerance_deg": self.args.pair_rotation_tolerance_deg,
            }
        (self.sim2real_dataset_dir / "adapter_metadata.json").write_text(
            json.dumps(adapter_meta, indent=2), encoding="utf-8"
        )

    def _export_sample_to_sim2real(
        self,
        image_path: Path,
        sample_id: str,
        timestamp: str,
        frame_time: float,
        joints: Optional[tuple[float, ...]],
        pose: Optional[tuple[float, ...]],
        pair_row: Optional[PairPlanRow] = None,
        pair_status: str = "unpaired",
    ) -> str:
        if self.sim2real_dataset_dir is None:
            return ""
        self.sim2real_index += 1
        image_name = "image_{}.png".format(self.sim2real_index)
        target_image_path = self.sim2real_image_dir / image_name
        shutil.copy2(image_path, target_image_path)

        target_row = {column: "" for column in SIM2REAL_TARGET_COLUMNS}
        target_row.update(
            {
                "sensor_image": image_name,
                "object_label": pair_row.object_label if pair_row and pair_row.object_label else self.args.sim2real_object_label,
                "source_sample_id": sample_id,
                "timestamp": timestamp,
                "camera_time": "{:.6f}".format(frame_time),
            }
        )
        if pair_row is not None:
            for column in SIM2REAL_POSE_COLUMNS + SIM2REAL_SHEAR_COLUMNS + SIM2REAL_OBJECT_COLUMNS:
                target_row[column] = pair_row.value(column)
        for column, value in zip(JOINT_COLUMNS, format_values(joints)):
            target_row[column] = value
        for column, value in zip(SIM2REAL_TCP_COLUMNS, format_values(pose)):
            target_row[column] = value
        with self.sim2real_targets_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=SIM2REAL_TARGET_COLUMNS).writerow(target_row)

        manifest_status = "unpaired"
        if pair_row is not None:
            manifest_status = "plan_recorded" if pair_status in PAIR_PLAN_COMPLETED_STATUSES else pair_status
        manifest_row = {
            "dataset_index": str(self.sim2real_index),
            "sensor_image": image_name,
            "source_sample_id": sample_id,
            "source_image": str(image_path.relative_to(self.output_dir)),
            "sim_target_image": "",
            "pairing_status": manifest_status,
            "timestamp": timestamp,
        }
        with self.sim2real_manifest_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=SIM2REAL_MANIFEST_COLUMNS).writerow(manifest_row)
        return image_name

    def _write_pair_record(
        self,
        row: PairPlanRow,
        sensor_image: str,
        sample_id: str,
        timestamp: str,
        actual_pose: Optional[tuple[float, ...]],
        verification_status: str,
        position_error_mm: Optional[float],
        rotation_error_deg: Optional[float],
    ) -> None:
        if self.pair_records_path is None:
            raise RuntimeError("Pair records require --sim2real-dataset-dir.")
        record = {column: "" for column in PAIR_RECORD_COLUMNS}
        record.update(
            {
                "pair_id": row.pair_id,
                "plan_index": str(row.index),
                "collector": "cr3_gelsight",
                "sensor_image": sensor_image,
                "source_sample_id": sample_id,
                "timestamp": timestamp,
                "object_label": row.object_label or self.args.sim2real_object_label,
                "position_error_mm": "" if position_error_mm is None else "{:.8f}".format(position_error_mm),
                "rotation_error_deg": "" if rotation_error_deg is None else "{:.8f}".format(rotation_error_deg),
                "verification_status": verification_status,
                "plan_path": str(self.pair_plan.path),
            }
        )
        for column in SIM2REAL_POSE_COLUMNS + SIM2REAL_SHEAR_COLUMNS + SIM2REAL_OBJECT_COLUMNS:
            record[column] = row.value(column)
        for column, value in zip(ACTUAL_TCP_COLUMNS, format_values(actual_pose)):
            record[column] = value
        for column, value in zip(REAL_TARGET_TCP_COLUMNS, row.values_for(REAL_TARGET_TCP_COLUMNS)):
            record[column] = value
        with self.pair_records_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=PAIR_RECORD_COLUMNS).writerow(record)

    def _current_pair_row(self) -> Optional[PairPlanRow]:
        return self.pair_plan.current if self.pair_plan is not None else None

    def _pair_plan_status_line(self) -> str:
        if self.pair_plan is None:
            return ""
        row = self.pair_plan.current
        if row is None:
            return "pair plan: complete"
        target_state = "calibrated TCP target loaded" if row.expected_real_tcp() is not None else "no calibrated TCP target"
        return "pair plan: {} ({}/{}) | {}".format(
            row.pair_id,
            row.index,
            len(self.pair_plan.rows),
            target_state,
        )

    def _load_current_pair_target(self) -> None:
        if self.pair_plan is None or not hasattr(self, "pose_vars"):
            return
        row = self.pair_plan.current
        if row is None:
            self.pair_last_status = "pair plan complete; no additional planned samples will be accepted"
            return
        expected_pose = row.expected_real_tcp()
        if expected_pose is None:
            self.pair_last_status = "pair {} has no calibrated real TCP target; capture will be unverified".format(
                row.pair_id
            )
            return
        for variable, value in zip(self.pose_vars, expected_pose):
            variable.set("{:.3f}".format(value))
        self.pair_last_status = "pair {} TCP target loaded into MovL fields; review it before moving".format(row.pair_id)

    def _build_ui(self) -> None:
        self.root.title("CR3 GelSight live sampler")
        self.root.geometry("1280x980")
        self.root.minsize(980, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.image_label = tk.Label(
            self.root,
            bg="black",
            fg="white",
            text="Waiting for GelSight frame ...",
            anchor="center",
        )
        self.image_label.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=8, pady=8)

        self.status_var = tk.StringVar(value="Starting camera ...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        self.status_label.grid(row=1, column=0, columnspan=8, sticky="ew", padx=8)

        self._build_robot_control_ui(row=2)

        self.sample_button = tk.Button(self.root, text="Sample (Space)", command=self.sample)
        self.sample_button.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        self.open_button = tk.Button(self.root, text="Open Output", command=self.open_output)
        self.open_button.grid(row=4, column=2, columnspan=2, sticky="ew", padx=8, pady=8)
        self.quit_button = tk.Button(self.root, text="Quit (Esc)", command=self.close)
        self.quit_button.grid(row=4, column=4, columnspan=2, sticky="ew", padx=8, pady=8)

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(5, weight=1)
        self.root.grid_rowconfigure(0, weight=1, minsize=360)
        self.root.bind("<space>", lambda _event: self.sample())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("q", lambda _event: self.close())
        self.root.bind_all("<ButtonRelease-1>", self._stop_hold_jog, add="+")

    def _build_robot_control_ui(self, row: int) -> None:
        control = tk.LabelFrame(self.root, text="Dobot CR3 control")
        control.grid(row=row, column=0, columnspan=8, sticky="ew", padx=8, pady=4)
        for col in range(12):
            control.grid_columnconfigure(col, weight=1 if col in (11,) else 0)

        self.speed_var = tk.StringVar(value="{:.1f}".format(self.args.robot_speed))
        self.user_var = tk.StringVar(value=str(self.args.robot_user))
        self.tool_var = tk.StringVar(value=str(self.args.robot_tool))
        self.joint_step_var = tk.StringVar(value="{:.3f}".format(self.args.joint_step_deg))
        self.tcp_step_var = tk.StringVar(value="{:.3f}".format(self.args.tcp_step_mm))
        self.rot_step_var = tk.StringVar(value="{:.3f}".format(self.args.rot_step_deg))

        tk.Label(control, text="speed %").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.speed_var, width=7).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        tk.Label(control, text="user").grid(row=0, column=2, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.user_var, width=5).grid(row=0, column=3, sticky="w", padx=4, pady=3)
        tk.Label(control, text="tool").grid(row=0, column=4, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.tool_var, width=5).grid(row=0, column=5, sticky="w", padx=4, pady=3)
        tk.Label(control, text="joint step deg").grid(row=0, column=6, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.joint_step_var, width=7).grid(row=0, column=7, sticky="w", padx=4, pady=3)
        tk.Label(control, text="tcp step mm").grid(row=0, column=8, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.tcp_step_var, width=7).grid(row=0, column=9, sticky="w", padx=4, pady=3)
        tk.Label(control, text="rot step deg").grid(row=0, column=10, sticky="w", padx=4, pady=3)
        tk.Entry(control, textvariable=self.rot_step_var, width=7).grid(row=0, column=11, sticky="w", padx=4, pady=3)

        self.robot_buttons = []
        for col, (label, command) in enumerate(
            (
                ("Read Pose", self.read_robot_state),
                ("Speed -5%", lambda: self.adjust_speed(-5.0)),
                ("Speed +5%", lambda: self.adjust_speed(+5.0)),
                ("Apply User/Tool", self.apply_user_tool),
                ("Clear Error", lambda: self.dashboard_action("clear error", "clear_error")),
                ("Enable", lambda: self.dashboard_action("enable", "enable")),
                ("Exit Drag (verify)", self.exit_drag),
                ("Recovery J5", self.recover_j5),
                ("Disable", lambda: self.dashboard_action("disable", "disable")),
                ("Reset", lambda: self.dashboard_action("reset", "reset")),
                ("E-Stop", self.emergency_stop),
            )
        ):
            button = tk.Button(control, text=label, command=command)
            button.grid(row=1, column=col, sticky="ew", padx=4, pady=3)
            self.robot_buttons.append(button)

        target_frame = tk.Frame(control)
        target_frame.grid(row=2, column=0, columnspan=12, sticky="ew", pady=3)
        for col in range(12):
            target_frame.grid_columnconfigure(col, weight=1)

        self.joint_vars = [tk.StringVar(value=v) for v in ("0.000", "30.000", "110.000", "-50.000", "-90.000", "0.000")]
        self.pose_vars = [tk.StringVar(value="0.000") for _ in range(6)]

        joint_box = tk.LabelFrame(target_frame, text="Joint target J1-J6 (deg)")
        joint_box.grid(row=0, column=0, columnspan=6, sticky="ew", padx=(0, 4))
        self.jog_buttons = []
        self.joint_entries = []
        for idx, var in enumerate(self.joint_vars):
            tk.Label(joint_box, text="J{}".format(idx + 1)).grid(row=idx, column=0, sticky="w", padx=4, pady=2)
            entry = tk.Entry(joint_box, textvariable=var, width=11)
            entry.grid(row=idx, column=1, sticky="ew", padx=4, pady=2)
            self.joint_entries.append(entry)
            minus_button = self._make_hold_jog_button(joint_box, "-", "joint", idx, -1.0)
            plus_button = self._make_hold_jog_button(joint_box, "+", "joint", idx, +1.0)
            minus_button.grid(row=idx, column=2, sticky="ew", padx=2, pady=2)
            plus_button.grid(row=idx, column=3, sticky="ew", padx=2, pady=2)
            self.jog_buttons.extend([minus_button, plus_button])
        move_joints_button = tk.Button(joint_box, text="Move JointMovJ", command=self.move_joints)
        move_joints_button.grid(row=6, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        self.robot_buttons.append(move_joints_button)
        self.robot_buttons.extend(self.jog_buttons)

        pose_box = tk.LabelFrame(target_frame, text="TCP pose target (mm / deg)")
        pose_box.grid(row=0, column=6, columnspan=6, sticky="ew", padx=(4, 0))
        self.pose_jog_buttons = []
        self.pose_entries = []
        for idx, (name, var) in enumerate(zip(("X", "Y", "Z", "Rx", "Ry", "Rz"), self.pose_vars)):
            tk.Label(pose_box, text=name).grid(row=idx, column=0, sticky="w", padx=4, pady=2)
            entry = tk.Entry(pose_box, textvariable=var, width=11)
            entry.grid(row=idx, column=1, sticky="ew", padx=4, pady=2)
            self.pose_entries.append(entry)
            minus_button = self._make_hold_jog_button(pose_box, "-", "pose", idx, -1.0)
            plus_button = self._make_hold_jog_button(pose_box, "+", "pose", idx, +1.0)
            minus_button.grid(row=idx, column=2, sticky="ew", padx=2, pady=2)
            plus_button.grid(row=idx, column=3, sticky="ew", padx=2, pady=2)
            self.pose_jog_buttons.extend([minus_button, plus_button])
        move_pose_button = tk.Button(pose_box, text="Move MovL", command=self.move_pose)
        move_pose_button.grid(row=6, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        self.robot_buttons.append(move_pose_button)
        self.robot_buttons.extend(self.pose_jog_buttons)

    def _make_hold_jog_button(self, parent, text: str, kind: str, index: int, direction: float) -> tk.Button:
        button = tk.Button(parent, text=text, width=4)
        button.bind("<ButtonPress-1>", lambda _event, k=kind, i=index, d=direction: self._start_hold_jog(k, i, d))
        button.bind("<ButtonRelease-1>", self._stop_hold_jog)
        return button

    def _update_frame(self) -> None:
        frame, frame_time, error = self.camera.get_latest()
        if frame is not None:
            try:
                self.photo = frame_to_photo_image(frame, self.args.preview_scale)
                self.image_label.configure(image=self.photo, text="")
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
        elif error:
            self.image_label.configure(image="", text="GelSight camera error:\n{}".format(error))
        else:
            self.image_label.configure(image="", text="Waiting for GelSight frame ...")
        age = time.time() - frame_time if frame_time > 0 else 0.0
        status = [
            "samples: {}".format(self.sample_index),
            "output: {}".format(self.output_dir),
            "camera age: {:.2f}s".format(age) if frame is not None else "camera: no frame yet",
        ]
        pair_plan_status = self._pair_plan_status_line()
        if pair_plan_status:
            status.append(pair_plan_status)
        if self.pair_last_status:
            status.append(self.pair_last_status)
        if self.last_status:
            status.append(self.last_status)
        if self.last_state_status:
            status.append(self.last_state_status)
        if error:
            status.append("camera error: {}".format(error))
        self.status_var.set("\n".join(status))
        self.root.after(max(10, int(self.args.ui_interval_ms)), self._update_frame)

    def _schedule_robot_state_poll(self) -> None:
        if self.closed or int(self.args.robot_state_poll_ms) <= 0:
            return
        delay = max(100, int(self.args.robot_state_poll_ms))
        self.state_poll_after_id = self.root.after(delay, self._poll_robot_state)

    def _poll_robot_state(self) -> None:
        self.state_poll_after_id = None
        if self.closed:
            return
        if self.state_poll_inflight:
            self._schedule_robot_state_poll()
            return
        if self.robot_busy and self.active_jog_axis is None:
            self._schedule_robot_state_poll()
            return
        self.state_poll_inflight = True

        def worker() -> None:
            try:
                result = self.robot.read_state()
                try:
                    self.root.after(0, lambda result=result: self._finish_robot_state_poll(result, None))
                except tk.TclError:
                    pass
            except Exception as exc:
                try:
                    self.root.after(0, lambda exc=exc: self._finish_robot_state_poll(None, exc))
                except tk.TclError:
                    pass

        threading.Thread(target=worker, name="cr3-state-poll", daemon=True).start()

    def _finish_robot_state_poll(self, result, exc: Optional[Exception]) -> None:
        self.state_poll_inflight = False
        if self.closed:
            return
        if exc is None and result is not None:
            self._show_robot_state(result, preserve_focused=True)
            self.last_state_status = "robot state: live {:.1f}s".format(time.time() - self.started_at)
        else:
            self.last_state_status = "robot state: read failed"
        self._schedule_robot_state_poll()

    def _parse_six(self, variables: list[tk.StringVar], names: tuple[str, ...]) -> tuple[float, ...]:
        values = []
        for var, name in zip(variables, names):
            text = var.get().strip()
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError("{} must be a number, got {!r}".format(name, text)) from exc
        return tuple(values)

    def _read_speed_user_tool(self) -> tuple[float, int, int]:
        try:
            speed = float(self.speed_var.get().strip())
        except ValueError as exc:
            raise ValueError("speed must be a number") from exc
        if not 1 <= speed <= 100:
            raise ValueError("speed must be in [1, 100]")
        try:
            user = int(self.user_var.get().strip())
            tool = int(self.tool_var.get().strip())
        except ValueError as exc:
            raise ValueError("user/tool must be integers") from exc
        if not 0 <= user <= 9 or not 0 <= tool <= 9:
            raise ValueError("user/tool must be in [0, 9]")
        return speed, user, tool

    def _read_joint_step(self) -> float:
        try:
            step = abs(float(self.joint_step_var.get().strip()))
        except ValueError as exc:
            raise ValueError("joint step deg must be a number") from exc
        if step <= 0:
            raise ValueError("joint step deg must be > 0")
        return step

    def _read_tcp_step(self) -> float:
        try:
            step = abs(float(self.tcp_step_var.get().strip()))
        except ValueError as exc:
            raise ValueError("tcp step mm must be a number") from exc
        if step <= 0:
            raise ValueError("tcp step mm must be > 0")
        return step

    def _read_rot_step(self) -> float:
        try:
            step = abs(float(self.rot_step_var.get().strip()))
        except ValueError as exc:
            raise ValueError("rot step deg must be a number") from exc
        if step <= 0:
            raise ValueError("rot step deg must be > 0")
        return step

    def _set_robot_buttons_enabled(self, enabled: bool) -> None:
        for button in getattr(self, "robot_buttons", []):
            button.configure(state="normal" if enabled else "disabled")

    def _set_var_unless_focused(self, var: tk.StringVar, value: float, widgets: list[tk.Entry], index: int, preserve_focused: bool) -> None:
        if preserve_focused:
            focused = self.root.focus_get()
            if index < len(widgets) and focused is widgets[index]:
                return
        var.set("{:.3f}".format(float(value)))

    def _show_robot_state(self, result, preserve_focused: bool = False) -> None:
        joints, pose, _raw_joints, _raw_pose = result
        if joints is not None:
            for idx, (var, value) in enumerate(zip(self.joint_vars, joints)):
                self._set_var_unless_focused(var, value, self.joint_entries, idx, preserve_focused)
        if pose is not None:
            for idx, (var, value) in enumerate(zip(self.pose_vars, pose)):
                self._set_var_unless_focused(var, value, self.pose_entries, idx, preserve_focused)

    def _collect_warning_lines(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            lower = value.lower()
            if "warn" in lower or "warning" in lower:
                return [line.strip() for line in value.splitlines() if line.strip()]
            return []
        if isinstance(value, dict):
            lines = []
            for item in value.values():
                lines.extend(self._collect_warning_lines(item))
            return lines
        if isinstance(value, (list, tuple, set)):
            lines = []
            for item in value:
                lines.extend(self._collect_warning_lines(item))
            return lines
        return []

    def _show_robot_warnings(self, label: str, result=None) -> None:
        warning_lines = []
        pop_warnings = getattr(self.robot, "pop_warnings", None)
        if callable(pop_warnings):
            warning_lines.extend(pop_warnings())
        warning_lines.extend(self._collect_warning_lines(result))
        unique_lines = []
        seen = set()
        for line in warning_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        if not unique_lines:
            return
        self.last_status = "{} warning: {}".format(label, " | ".join(unique_lines[:2]))
        messagebox.showwarning("Robot warning", "{}:\n\n{}".format(label, "\n".join(unique_lines[:8])))

    def _run_robot_task(self, label: str, func, update_state: bool = False) -> None:
        if self.robot_busy:
            messagebox.showwarning("Robot busy", "The previous robot command is still running.")
            return
        if self.active_jog_axis is not None:
            messagebox.showwarning("Robot jogging", "Release the jog button before sending another robot command.")
            return
        self.robot_busy = True
        self._set_robot_buttons_enabled(False)
        self.last_status = "{} ...".format(label)

        def worker() -> None:
            try:
                result = func()
                self.root.after(0, lambda result=result: self._finish_robot_task(label, result, None, update_state))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._finish_robot_task(label, None, exc, update_state))

        threading.Thread(target=worker, name="cr3-command", daemon=True).start()

    def _finish_robot_task(self, label: str, result, exc: Optional[Exception], update_state: bool) -> None:
        self.robot_busy = False
        self._set_robot_buttons_enabled(True)
        if exc is not None:
            self._stop_hold_jog()
            self.last_status = "{} failed: {}: {}".format(label, type(exc).__name__, exc)
            messagebox.showerror("Robot command failed", self.last_status)
            return
        if update_state and result is not None:
            self._show_robot_state(result)
        self._show_robot_warnings(label, result)
        self.last_status = "{} done".format(label)
        if label == "exit drag":
            messagebox.showinfo("Drag mode exited", "RobotMode is now 5 (enabled and idle). You can jog the robot.")

    def _start_hold_jog(self, kind: str, index: int, direction: float):
        if self.robot_busy:
            messagebox.showwarning("Robot busy", "The previous robot command is still running.")
            return "break"
        if self.active_jog_axis is not None:
            return "break"
        try:
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return "break"
        axis = self._move_jog_axis(kind, index, direction)
        self.active_jog_axis = axis
        try:
            result = self.robot.start_move_jog(axis, speed, user, tool)
            self._show_robot_warnings("MoveJog({})".format(axis), result)
            self.last_status = "MoveJog({}) running; release button to stop".format(axis)
        except Exception as exc:
            self.active_jog_axis = None
            if "RobotMode is 6 (drag mode)" in str(exc):
                self.last_status = "Robot is in drag mode"
                if messagebox.askyesno(
                    "Drag mode active",
                    "The robot is in drag mode. Exit drag and verify idle now?\n\n"
                    "The robot will hold its current pose. After verification succeeds, press the jog button again.",
                ):
                    self.exit_drag(confirm=False)
                return "break"
            self.last_status = "MoveJog({}) failed: {}: {}".format(axis, type(exc).__name__, exc)
            messagebox.showerror("Robot command failed", self.last_status)
        return "break"

    def _stop_hold_jog(self, _event=None):
        if self.active_jog_axis is None:
            return
        axis = self.active_jog_axis
        self.active_jog_axis = None
        try:
            result = self.robot.stop_move_jog()
            self._show_robot_warnings("MoveJog stop", result)
            self.last_status = "MoveJog({}) stopped".format(axis)
        except Exception as exc:
            self.last_status = "MoveJog stop failed: {}: {}".format(type(exc).__name__, exc)
            messagebox.showerror("Robot command failed", self.last_status)

    def _finish_jog_start(self, axis: str, result, exc: Optional[Exception]) -> None:
        if exc is not None:
            self.active_jog_axis = None
            self.last_status = "MoveJog({}) failed: {}: {}".format(axis, type(exc).__name__, exc)
            messagebox.showerror("Robot command failed", self.last_status)
            return
        self.last_status = "MoveJog({}) running".format(axis)

    def _finish_jog_stop(self, axis: str, result, exc: Optional[Exception]) -> None:
        if exc is not None:
            self.last_status = "MoveJog stop failed: {}: {}".format(type(exc).__name__, exc)
            messagebox.showerror("Robot command failed", self.last_status)
            return
        self.last_status = "MoveJog({}) stopped".format(axis)

    @staticmethod
    def _move_jog_axis(kind: str, index: int, direction: float) -> str:
        sign = "+" if direction > 0 else "-"
        if kind == "joint":
            return "J{}{}".format(index + 1, sign)
        return "{}{}".format(("X", "Y", "Z", "Rx", "Ry", "Rz")[index], sign)

    def read_robot_state(self) -> None:
        self._run_robot_task("read robot state", self.robot.read_state, update_state=True)

    def adjust_speed(self, delta: float) -> None:
        try:
            speed = float(self.speed_var.get().strip())
        except ValueError:
            speed = float(self.args.robot_speed)
        speed = max(1.0, min(100.0, speed + float(delta)))
        self.speed_var.set("{:.1f}".format(speed))
        self._run_robot_task("set speed {:.1f}%".format(speed), lambda: self.robot.set_speed(speed))

    def apply_user_tool(self) -> None:
        try:
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        def work():
            return self.robot.initialize_motion(speed, user, tool)

        self._run_robot_task("initialize robot motion", work)

    def dashboard_action(self, label: str, method_name: str) -> None:
        self._run_robot_task(label, lambda: getattr(self.robot, method_name)())

    def emergency_stop(self) -> None:
        if self.args.confirm_motion and not messagebox.askyesno("Confirm emergency stop", "Send EmergencyStop() to the CR3 controller?"):
            return
        self.dashboard_action("emergency stop", "emergency_stop")

    def exit_drag(self, confirm: bool = True) -> None:
        if confirm and self.args.confirm_motion and not messagebox.askyesno(
            "Confirm Exit Drag",
            "Exit drag mode and hold the robot at its current pose?",
        ):
            return
        self._run_robot_task("exit drag", self.robot.exit_drag_and_verify)

    def recover_j5(self) -> None:
        if self.args.confirm_motion and not messagebox.askyesno(
            "Confirm Recovery J5",
            "This will ClearError, EnableRobot, set 5% speed, then move only J5 by 5 deg toward 0 deg.\n\n"
            "Make sure the robot workspace is clear. Continue?",
        ):
            return

        try:
            _speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        def work():
            return self.robot.recover_j5_from_singularity(step_deg=5.0, speed=5.0, user=user, tool=tool)

        self._run_robot_task("Recovery J5", work, update_state=True)

    def move_joints(self) -> None:
        try:
            joints = self._parse_six(self.joint_vars, ("J1", "J2", "J3", "J4", "J5", "J6"))
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        if self.args.confirm_motion and not messagebox.askyesno(
            "Confirm JointMovJ",
            "Move CR3 with JointMovJ to:\n{}\n\nContinue?".format(", ".join("{:.3f}".format(v) for v in joints)),
        ):
            return
        self._run_robot_task("JointMovJ", lambda: self.robot.move_joints(joints, speed, user, tool), update_state=True)

    def jog_joint(self, joint_index: int, direction: float) -> None:
        try:
            step = self._read_joint_step()
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        label = "Jog J{} {}{:.3f} deg".format(joint_index + 1, "+" if direction > 0 else "-", step)
        if self.args.confirm_motion and not messagebox.askyesno(
            "Confirm joint jog",
            "{} from the current robot joint state?\n\nContinue?".format(label),
        ):
            return

        self._run_robot_task(label, lambda: self.robot.jog_joint(joint_index, direction, step, speed, user, tool), update_state=True)

    def move_pose(self) -> None:
        try:
            pose = self._parse_six(self.pose_vars, ("X", "Y", "Z", "Rx", "Ry", "Rz"))
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        if self.args.confirm_motion and not messagebox.askyesno(
            "Confirm MovL",
            "Move CR3 with MovL to:\n{}\n\nContinue?".format(", ".join("{:.3f}".format(v) for v in pose)),
        ):
            return
        self._run_robot_task("MovL", lambda: self.robot.move_pose(pose, speed, user, tool), update_state=True)

    def jog_pose_axis(self, axis_index: int, direction: float) -> None:
        names = ("X", "Y", "Z", "Rx", "Ry", "Rz")
        try:
            step = self._read_tcp_step() if axis_index < 3 else self._read_rot_step()
            speed, user, tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        unit = "mm" if axis_index < 3 else "deg"
        label = "Jog {} {}{:.3f} {}".format(names[axis_index], "+" if direction > 0 else "-", step, unit)
        if self.args.confirm_motion and not messagebox.askyesno(
            "Confirm TCP jog",
            "{} from the current robot TCP pose?\n\nContinue?".format(label),
        ):
            return
        self._run_robot_task(label, lambda: self.robot.jog_pose_axis(axis_index, direction, step, speed, user, tool), update_state=True)

    def sample(self) -> None:
        if self.robot_busy:
            messagebox.showwarning("Robot busy", "Wait for the robot command to finish before sampling.")
            return
        if self.active_jog_axis is not None:
            messagebox.showwarning("Robot jogging", "Release the jog button before sampling.")
            return
        try:
            _speed, capture_user, capture_tool = self._read_speed_user_tool()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        pair_row = self._current_pair_row()
        if self.pair_plan is not None and pair_row is None:
            messagebox.showinfo("Pair plan complete", "All pair-plan rows have been recorded. Start a new plan before sampling again.")
            return
        frame, frame_time, error = self.camera.get_latest()
        if frame is None:
            messagebox.showwarning("No frame", "No GelSight frame is available yet.")
            return
        self.sample_button.configure(state="disabled")
        try:
            joints = pose = None
            raw_joints = raw_pose = ""
            try:
                joints, pose, raw_joints, raw_pose = self.robot.read_state()
            except Exception as exc:
                if not self.args.allow_missing_robot:
                    raise
                self.last_status = "robot read failed, saved image only: {}: {}".format(type(exc).__name__, exc)

            pair_verification_status = "unpaired"
            position_error_mm = rotation_error_deg = None
            if pair_row is not None:
                pair_verification_status, position_error_mm, rotation_error_deg = verify_real_pose(
                    pair_row,
                    pose,
                    self.args.pair_position_tolerance_mm,
                    self.args.pair_rotation_tolerance_deg,
                )

            self.sample_index += 1
            sample_id = "{:06d}".format(self.sample_index)
            image_name = "{}{}.png".format(self.args.prefix, sample_id)
            image_path = self.image_dir / image_name

            import cv2

            ok = cv2.imwrite(str(image_path), frame)
            if not ok:
                raise RuntimeError("Could not write image: {}".format(image_path))
            if self.preprocessor is not None:
                self.preprocessor.process_and_save(frame, image_path)
            if self.args.save_npy:
                np.save(str(image_path.with_suffix(".npy")), frame)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            sim2real_image_name = self._export_sample_to_sim2real(
                image_path,
                sample_id,
                timestamp,
                frame_time,
                joints,
                pose,
                pair_row=pair_row,
                pair_status=pair_verification_status,
            )
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        sample_id,
                        timestamp,
                        "{:.6f}".format(time.time() - self.started_at),
                        str(image_path.relative_to(self.output_dir)),
                        "{:.6f}".format(frame_time),
                        capture_user,
                        capture_tool,
                    ]
                    + format_values(joints)
                    + format_values(pose)
                    + [raw_joints, raw_pose]
                )
            if pair_row is not None:
                self._write_pair_record(
                    pair_row,
                    sim2real_image_name,
                    sample_id,
                    timestamp,
                    pose,
                    pair_verification_status,
                    position_error_mm,
                    rotation_error_deg,
                )
            self.last_status = "saved sample {} | joints={}".format(sample_id, ", ".join(format_values(joints)))
            if sim2real_image_name:
                self.last_status += " | sim2real={}".format(sim2real_image_name)
            if pair_row is not None:
                self.last_status += " | pair={} {}".format(pair_row.pair_id, pair_verification_status)
                if pair_verification_status in PAIR_PLAN_COMPLETED_STATUSES:
                    self.pair_plan.advance()
                    self._load_current_pair_target()
                else:
                    if position_error_mm is None:
                        detail = "The robot pose was unavailable, so this plan row remains active."
                    else:
                        detail = (
                            "Position error {:.3f} mm, rotation error {:.3f} deg. "
                            "Reposition the robot and sample this same pair again."
                        ).format(position_error_mm, rotation_error_deg)
                    self.pair_last_status = "pair {} not accepted: {}".format(pair_row.pair_id, detail)
                    messagebox.showwarning("Pair not accepted", self.pair_last_status)
            print(self.last_status, flush=True)
        except Exception as exc:
            messagebox.showerror("Sample failed", "{}: {}".format(type(exc).__name__, exc))
            self.last_status = "sample failed: {}: {}".format(type(exc).__name__, exc)
        finally:
            self.sample_button.configure(state="normal")

    def open_output(self) -> None:
        os.startfile(str(self.output_dir))

    def close(self) -> None:
        self.closed = True
        if self.state_poll_after_id is not None:
            try:
                self.root.after_cancel(self.state_poll_after_id)
            except tk.TclError:
                pass
            self.state_poll_after_id = None
        if self.active_jog_axis is not None:
            try:
                self.robot.stop_move_jog()
            except Exception:
                pass
            self.active_jog_axis = None
        try:
            self.camera.stop()
        finally:
            close = getattr(self.robot, "close", None)
            if close is not None:
                close()
            if self.preprocessor is not None:
                self.preprocessor.close()
            self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "cr3_gelsight_real"
        / "manual_live_sampling"
        / DEFAULT_SHARED_OUTPUT_NAME
    )
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--prefix", default="gelsight_")
    parser.add_argument(
        "--capture-once",
        action="store_true",
        help="Save one stationary sample after the first camera frame arrives, then close without moving the robot.",
    )
    parser.add_argument("--capture-once-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--sim2real-dataset-dir",
        default=None,
        help="Optional tactile_sim2real split directory to receive sensor_images, targets.csv, and manifest.csv.",
    )
    parser.add_argument("--sim2real-object-label", default="unlabeled")
    parser.add_argument("--reset-sim2real-dataset", action="store_true")
    parser.add_argument(
        "--pair-plan",
        default=None,
        help="CSV plan with pair_id, task-relative pose labels, and optional calibrated real_target_tcp_* columns.",
    )
    parser.add_argument(
        "--pair-plan-start",
        default=None,
        help="Optional pair_id at which to start instead of automatically resuming the first unfinished row.",
    )
    parser.add_argument("--pair-position-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--pair-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--camera-source", default="0", help="OpenCV camera index or URL/path.")
    default_camera_backend = "dshow" if sys.platform.startswith("win") else "any"
    parser.add_argument(
        "--camera-backend",
        choices=("dshow", "msmf", "any"),
        default=default_camera_backend,
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--robot-ip", default=DEFAULT_IP)
    parser.add_argument("--dashboard-port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--move-port", type=int, default=MOVE_PORT)
    parser.add_argument("--robot-timeout", type=float, default=2.0)
    parser.add_argument("--robot-speed", type=float, default=50.0)
    parser.add_argument("--robot-user", type=int, default=0)
    parser.add_argument("--robot-tool", type=int, default=0)
    parser.add_argument("--joint-step-deg", type=float, default=1.0)
    parser.add_argument("--tcp-step-mm", type=float, default=1.0)
    parser.add_argument("--rot-step-deg", type=float, default=1.0)
    parser.add_argument("--set-speed-on-motion", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true", dest="confirm_motion")
    parser.add_argument("--allow-missing-robot", action="store_true")
    parser.add_argument("--dry-run-robot", action="store_true")
    parser.add_argument("--dry-run-camera", action="store_true")
    parser.add_argument("--ui-interval-ms", type=int, default=30)
    parser.add_argument("--robot-state-poll-ms", type=int, default=500)
    add_tactip_preprocess_args(parser)
    parser.set_defaults(confirm_motion=False)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.pair_plan and not args.sim2real_dataset_dir:
        parser.error("--pair-plan requires --sim2real-dataset-dir so pair_records.csv can be written.")
    if args.pair_plan and PAIRING_IMPORT_ERROR is not None:
        parser.error("--pair-plan requires tactile_sim2real: {}".format(PAIRING_IMPORT_ERROR))
    if args.pair_position_tolerance_mm <= 0:
        parser.error("--pair-position-tolerance-mm must be positive.")
    if args.pair_rotation_tolerance_deg <= 0:
        parser.error("--pair-rotation-tolerance-deg must be positive.")
    if args.capture_once_timeout_sec <= 0:
        parser.error("--capture-once-timeout-sec must be positive.")
    if not args.dry_run_camera:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("OpenCV is required for GelSight camera capture. Install opencv-python in this environment.", file=sys.stderr)
            return 2
    root = tk.Tk()
    try:
        LiveSamplerApp(root, args)
    except Exception as exc:
        messagebox.showerror("Startup failed", "{}: {}".format(type(exc).__name__, exc))
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
