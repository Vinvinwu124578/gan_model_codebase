"""Minimal Dobot CR3 TCP client used by real data collection.

This mirrors the TCP command path used by
``C:\\isaacsim\\common_robot_interface\\tools\\dobot_cr3_gui.py`` without
pulling in PyQt.  It intentionally exposes only the commands needed by the
collection scripts.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


DEFAULT_IP = "192.168.31.88"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003


def reply_error_code(reply: str) -> Optional[int]:
    head = reply.split(",", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def parse_reply_values(reply: str, expected: int, label: str) -> Tuple[float, ...]:
    left = reply.find("{")
    right = reply.find("}", left + 1)
    if left < 0 or right < 0:
        raise RuntimeError("{} reply cannot be parsed: {}".format(label, reply))
    values = [float(x.strip()) for x in reply[left + 1 : right].split(",") if x.strip()]
    if len(values) < expected:
        raise RuntimeError("{} reply has too few values: {}".format(label, reply))
    return tuple(values[:expected])


def pose_text(values: Iterable[float]) -> str:
    return "({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f})".format(*values)


@dataclass
class RobotState:
    joints: Tuple[float, ...]
    pose: Tuple[float, ...]


class DobotCR3TcpClient:
    def __init__(self, ip: str = DEFAULT_IP, timeout: float = 5.0) -> None:
        self.ip = ip
        self.move = socket.create_connection((ip, MOVE_PORT), timeout=timeout)
        self.move.settimeout(timeout)
        self._dashboard_available = True
        try:
            self.dashboard = socket.create_connection((ip, DASHBOARD_PORT), timeout=timeout)
            self.dashboard.settimeout(timeout)
        except OSError:
            self.dashboard = None
            self._dashboard_available = False

    def _send_recv(self, sock: socket.socket, command: str, timeout: float = 10.0) -> str:
        old_timeout = sock.gettimeout()
        sock.settimeout(timeout)
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

    def dashboard_cmd(self, command: str, timeout: float = 10.0) -> str:
        if not self._dashboard_available or self.dashboard is None:
            return self._send_recv(self.move, command, timeout=timeout)
        try:
            return self._send_recv(self.dashboard, command, timeout=timeout)
        except RuntimeError as exc:
            if "No reply" not in str(exc):
                raise
            self._dashboard_available = False
            try:
                self.dashboard.close()
            except OSError:
                pass
            return self._send_recv(self.move, command, timeout=timeout)

    def move_cmd(self, command: str, timeout: float = 10.0) -> str:
        return self._send_recv(self.move, command, timeout=timeout)

    def initialize(self, speed: float, user: int, tool: int, enable: bool = True) -> None:
        self.clear_error()
        if enable:
            self.enable()
        self.dashboard_cmd("User({:d})".format(user))
        self.dashboard_cmd("Tool({:d})".format(tool))
        self.set_speed(speed)

    def set_speed(self, speed: float) -> str:
        if not 1 <= speed <= 100:
            raise ValueError("SpeedFactor must be in [1, 100].")
        return self.dashboard_cmd("SpeedFactor({:d})".format(round(speed)))

    def clear_error(self) -> str:
        return self.dashboard_cmd("ClearError()")

    def enable(self) -> str:
        return self.dashboard_cmd("EnableRobot()", timeout=20.0)

    def disable(self) -> str:
        return self.dashboard_cmd("DisableRobot()")

    def read_joints(self) -> Tuple[float, ...]:
        return parse_reply_values(self.dashboard_cmd("GetAngle()"), 6, "GetAngle")

    def read_pose(self) -> Tuple[float, ...]:
        return parse_reply_values(self.dashboard_cmd("GetPose()"), 6, "GetPose")

    def read_state(self) -> RobotState:
        return RobotState(joints=self.read_joints(), pose=self.read_pose())

    def move_pose(self, pose: Iterable[float], speed: float, sync: bool = True) -> RobotState:
        self.set_speed(speed)
        self.move_cmd("MovL{}".format(pose_text(tuple(pose))), timeout=10.0)
        if sync:
            self.move_cmd("Sync()", timeout=180.0)
        return self.read_state()

    def close(self) -> None:
        for sock in (getattr(self, "move", None), getattr(self, "dashboard", None)):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass


class DryRunDobotCR3Client:
    def __init__(self, initial_pose: Iterable[float] = (0, 0, 0, 0, 0, 0)) -> None:
        self.pose = tuple(float(v) for v in initial_pose)
        self.joints = (0.0, 30.0, 110.0, -50.0, -90.0, 0.0)
        self.speed = 10.0

    def initialize(self, speed: float, user: int, tool: int, enable: bool = True) -> None:
        self.speed = speed
        print("DRY-RUN initialize speed={} user={} tool={} enable={}".format(speed, user, tool, enable))

    def set_speed(self, speed: float) -> str:
        self.speed = speed
        return "DRY-RUN SpeedFactor({})".format(round(speed))

    def read_state(self) -> RobotState:
        return RobotState(joints=self.joints, pose=self.pose)

    def move_pose(self, pose: Iterable[float], speed: float, sync: bool = True) -> RobotState:
        self.speed = speed
        self.pose = tuple(float(v) for v in pose)
        print("DRY-RUN MovL{} SpeedFactor({})".format(pose_text(self.pose), round(speed)))
        return self.read_state()

    def close(self) -> None:
        pass
