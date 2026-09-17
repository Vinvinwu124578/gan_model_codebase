#!/usr/bin/env python3
"""Visual planner and guarded launcher for CR3 coverage-board collection.

The Jogger is intentionally a control surface around
``auto_cr3_coverage_board_sampler.py``.  It does not implement a second,
divergent CR3 motion stack.  A user can choose the installed board folder,
rotate the board plan in 90 degree increments, review or temporarily edit the
saved TacTip rest TCP, generate an interactive no-motion route preview, and
only then confirm a formal collection.

For every plan or collection the editable rest TCP is written to a *session*
copy of the selected fixture profile.  The original calibrated profile is
never overwritten.  On an execute run the existing sampler refreshes the
camera reference at the fixed crossbar and automatically lowers a TCP that is
already aligned in XY/orientation but 0.1--5.0 mm above the saved rest datum.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Sequence
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "tools" / "auto_cr3_coverage_board_sampler.py"
DEFAULT_BOARD_DIR = ROOT / "outputs" / "tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DEFAULT_RUN_ROOT = DEFAULT_BOARD_DIR / "cr3_coverage_board_runs"
DEFAULT_PROFILE = ROOT / "outputs" / "cr3_coverage_board_runs" / "profiles" / "tile_ne_331pin_hough_heightcal_v7_20260914.json"
EXTERNAL_DOCK_DESIGN = (
    Path("/Users/vincent/Downloads/tactip_experiment_tactistruct/outputs")
    / "tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
    / "tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm"
    / "v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
)
LOCAL_DOCK_DESIGN = (
    DEFAULT_BOARD_DIR
    / "tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm"
    / "v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
)
DEFAULT_PYTHON = Path("/Users/vincent/Downloads/tactip_experiment_tactistruct/.venv/bin/python")
TILES = ("tile_nw", "tile_ne", "tile_sw", "tile_se")
POSE_LABELS = ("X", "Y", "Z", "Rx", "Ry", "Rz")


def default_dock_design() -> Path:
    """Prefer the verified physical dock design used by the formal launcher."""

    for candidate in (EXTERNAL_DOCK_DESIGN, LOCAL_DOCK_DESIGN):
        if candidate.is_file():
            return candidate
    return EXTERNAL_DOCK_DESIGN


def default_python() -> Path:
    return DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable)


def absolute_path_preserving_symlink(path_value: str | Path) -> Path:
    """Make an executable path absolute without leaving its virtual environment.

    A venv's ``bin/python`` is commonly a symlink to the base interpreter.
    ``Path.resolve()`` follows that symlink, and executing the resolved target
    bypasses the venv's ``pyvenv.cfg`` and installed packages.
    """

    expanded = Path(path_value).expanduser()
    return Path(os.path.abspath(str(expanded)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("{} is not a JSON object".format(path))
    return payload


def finite_pose(values: Sequence[float | str], name: str) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError("{} must contain X/Y/Z/Rx/Ry/Rz".format(name))
    pose = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("{} must contain only finite numbers".format(name))
    return (pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])


def normalize_board_dir(path_value: str | Path) -> Path:
    """Accept a board folder, manifest, STL, or OBJ and locate its manifest."""

    selected = Path(path_value).expanduser().resolve()
    if selected.is_dir():
        board_dir = selected
    elif selected.is_file():
        board_dir = selected.parent
    else:
        raise FileNotFoundError("Board selection does not exist: {}".format(selected))
    manifests = sorted(board_dir.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            "Expected exactly one *_manifest.json beside the selected board, found {} under {}"
            .format(len(manifests), board_dir)
        )
    return board_dir


def normalise_quarter_turn(value: float) -> float:
    """Keep a repeated +/- 90 degree UI rotation readable without changing geometry."""

    result = float(value)
    while result > 180.0:
        result -= 360.0
    while result < -180.0:
        result += 360.0
    return result


def profile_rest_pose(profile: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return finite_pose(profile.get("dock_tcp", ()), "fixture profile dock_tcp")


def build_session_profile(
    source_profile: Path,
    rest_pose: Sequence[float],
    session_root: Path,
    purpose: str,
) -> Path:
    """Write a per-run profile copy with the GUI's temporary rest TCP.

    The source fixture profile remains immutable.  The rest-stop image reference
    is intentionally retained until the formal sampler refreshes it at the
    physical crossbar for the current TacTip head orientation.
    """

    source = read_json(source_profile)
    pose = finite_pose(rest_pose, "editable rest pose")
    payload = copy.deepcopy(source)
    original_pose = profile_rest_pose(source)
    payload["dock_tcp"] = list(pose)
    height_calibration = payload.get("height_calibration")
    if isinstance(height_calibration, dict):
        height_calibration["source_measured_tool_tcp_at_crossbar"] = list(
            height_calibration.get("measured_tool_tcp_at_crossbar", original_pose)
        )
        height_calibration["measured_tool_tcp_at_crossbar"] = list(pose)
    payload["jogger_session"] = {
        "schema": "coverage_board_sampling_jogger_session.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": purpose,
        "source_fixture_profile": str(source_profile.resolve()),
        "source_fixture_profile_sha256": sha256_file(source_profile),
        "source_dock_tcp": list(original_pose),
        "session_dock_tcp": list(pose),
        "source_profile_modified": False,
        "note": (
            "This is a temporary GUI session copy. The formal sampler may refresh only this copy's "
            "tactile rest reference after verifying the fixed crossbar pose."
        ),
    }
    session_root.mkdir(parents=True, exist_ok=False)
    destination = session_root / "fixture_profile_session.json"
    write_json(destination, payload)
    return destination


@dataclass(frozen=True)
class JoggerConfig:
    tile: str
    board_dir: Path
    dock_design: Path
    source_fixture_profile: Path
    source_fixture_profile_sha256: str
    rest_pose: tuple[float, float, float, float, float, float]
    saved_board_yaw_deg: float
    board_yaw_offset_deg: float
    samples_per_tile: int
    min_depth_mm: float
    max_depth_mm: float
    speed_percent: float
    camera_source: str
    width: int
    height: int
    fps: float
    robot_ip: str
    tool: int
    user: int
    continuous_board_transit: bool
    zero_tilt: bool
    python: Path
    output_root: Path

    @property
    def effective_board_yaw_deg(self) -> float:
        return self.saved_board_yaw_deg + self.board_yaw_offset_deg

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "tile": self.tile,
            "board_dir": str(self.board_dir),
            "dock_design": str(self.dock_design),
            "source_fixture_profile": str(self.source_fixture_profile),
            "source_fixture_profile_sha256": self.source_fixture_profile_sha256,
            "rest_pose": list(self.rest_pose),
            "saved_board_yaw_deg": self.saved_board_yaw_deg,
            "board_yaw_offset_deg": self.board_yaw_offset_deg,
            "samples_per_tile": self.samples_per_tile,
            "min_depth_mm": self.min_depth_mm,
            "max_depth_mm": self.max_depth_mm,
            "speed_percent": self.speed_percent,
            "camera_source": self.camera_source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "robot_ip": self.robot_ip,
            "tool": self.tool,
            "user": self.user,
            "continuous_board_transit": self.continuous_board_transit,
            "zero_tilt": self.zero_tilt,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PlanState:
    config: JoggerConfig
    fingerprint: str
    output_dir: Path
    session_profile: Path

    @property
    def contact_preview(self) -> Path:
        return self.output_dir / "planned_sampling_points_3d.html"

    @property
    def route_preview(self) -> Path:
        return self.output_dir / "full_tcp_motion_safety_preview.html"


@dataclass(frozen=True)
class ProcessContext:
    mode: str
    config: JoggerConfig
    output_dir: Path
    session_profile: Path


def make_sampler_command(
    config: JoggerConfig,
    session_profile: Path,
    output_dir: Path,
    *,
    execute: bool,
) -> list[str]:
    """Create the exact sampler command for a no-motion plan or formal run."""

    command = [
        str(config.python),
        "-u",
        str(SAMPLER_PATH),
        "--tile",
        config.tile,
        "--board-dir",
        str(config.board_dir),
        "--dock-design",
        str(config.dock_design),
        "--fixture-profile",
        str(session_profile),
        "--samples-per-tile",
        str(config.samples_per_tile),
        "--dense-spatial-layout",
        "region_grid",
        "--dense-region-anchor-count",
        "25",
        "--min-post-contact-depth-mm",
        "{:.6g}".format(config.min_depth_mm),
        "--max-post-contact-depth-mm",
        "{:.6g}".format(config.max_depth_mm),
        "--board-yaw-offset-deg",
        "{:.6g}".format(config.board_yaw_offset_deg),
        "--camera-source",
        config.camera_source,
        "--width",
        str(config.width),
        "--height",
        str(config.height),
        "--fps",
        "{:.6g}".format(config.fps),
        "--robot-ip",
        config.robot_ip,
        "--tool",
        str(config.tool),
        "--user",
        str(config.user),
        "--speed",
        "{:.6g}".format(config.speed_percent),
        "--output-dir",
        str(output_dir),
    ]
    if config.continuous_board_transit:
        command.append("--continuous-board-transit")
    if config.zero_tilt:
        command.append("--zero-tilt")
    if execute:
        command.extend(
            (
                "--refresh-dock-reference-at-start",
                "--continue-on-no-contact",
                "--continue-on-safe-sample-error",
                "--return-to-dock",
                "--skip-previews",
                "--execute",
                "--yes-i-confirm-cr3-is-safe",
            )
        )
    return command


def unique_run_dir(root: Path, tile: str, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1, 1000):
        suffix = "" if index == 1 else "_{:02d}".format(index)
        candidate = root / "{}_jogger_{}_{}{}".format(tile, label, stamp, suffix)
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not reserve a unique Jogger output directory under {}".format(root))


def open_in_default_application(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError("Output is not available yet: {}".format(path))
    if path.suffix.lower() == ".html":
        webbrowser.open(path.resolve().as_uri())
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class CoverageBoardJoggerApp:
    """Tkinter GUI that starts only deliberately confirmed physical batches."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.active_process: subprocess.Popen[str] | None = None
        self.plan_state: PlanState | None = None
        self.collection_output: Path | None = None
        self.saved_rest_pose: tuple[float, float, float, float, float, float] | None = None
        self.saved_board_yaw_deg = 0.0
        self._loading_profile = False

        self.board_path_var = tk.StringVar(value=str(args.board_dir))
        self.dock_design_var = tk.StringVar(value=str(args.dock_design))
        self.fixture_profile_var = tk.StringVar(value=str(args.fixture_profile))
        self.tile_var = tk.StringVar(value=args.tile)
        self.yaw_offset_var = tk.StringVar(value="0")
        self.effective_yaw_var = tk.StringVar(value="Effective board yaw: loading profile...")
        self.rest_vars = [tk.StringVar(value="0.000000") for _ in POSE_LABELS]
        self.samples_var = tk.StringVar(value=str(args.samples_per_tile))
        self.min_depth_var = tk.StringVar(value="{:.3f}".format(args.min_depth_mm))
        self.max_depth_var = tk.StringVar(value="{:.3f}".format(args.max_depth_mm))
        self.speed_var = tk.StringVar(value="{:.1f}".format(args.speed))
        self.camera_source_var = tk.StringVar(value=str(args.camera_source))
        self.width_var = tk.StringVar(value=str(args.width))
        self.height_var = tk.StringVar(value=str(args.height))
        self.fps_var = tk.StringVar(value="{:.1f}".format(args.fps))
        self.robot_ip_var = tk.StringVar(value=str(args.robot_ip))
        self.output_root_var = tk.StringVar(value=str(args.output_root))
        self.continuous_var = tk.BooleanVar(value=True)
        self.zero_tilt_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Load the fixture profile, set board direction, then create a no-motion route plan.")

        self._build_ui()
        self._install_change_traces()
        self.load_profile()
        self.root.after(80, self._drain_events)

    def _build_ui(self) -> None:
        self.root.title("CR3 Coverage Board Jogger")
        self.root.geometry("1320x930")
        self.root.minsize(1080, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(self.root)
        style.configure("Heading.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("Subheading.TLabel", font=("Helvetica", 11, "bold"))

        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(4, weight=1)

        ttk.Label(outer, text="CR3 Coverage Board Jogger", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                "Plan first with no robot motion. Execute only after reviewing the interactive TCP route. "
                "The source fixture profile is never overwritten."
            ),
            wraplength=1180,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))

        board_frame = ttk.LabelFrame(outer, text="Board, Fixture, and Orientation", padding=10)
        board_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        board_frame.grid_columnconfigure(1, weight=1)
        self._path_row(board_frame, 0, "Board folder / manifest / STL", self.board_path_var, self.choose_board)
        self._path_row(board_frame, 1, "Dock design JSON", self.dock_design_var, self.choose_dock_design)
        self._path_row(board_frame, 2, "Fixture profile JSON", self.fixture_profile_var, self.choose_fixture_profile)
        ttk.Button(board_frame, text="Load Profile Defaults", command=self.load_profile).grid(
            row=3, column=2, sticky="e", padx=(6, 0), pady=(2, 7)
        )
        ttk.Label(board_frame, text="Mounted tile").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(board_frame, textvariable=self.tile_var, values=TILES, state="readonly", width=14).grid(
            row=4, column=1, sticky="w", pady=3
        )
        ttk.Label(board_frame, text="Runtime board yaw offset (deg)").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(board_frame, textvariable=self.yaw_offset_var, width=16).grid(row=5, column=1, sticky="w", pady=3)
        rotations = ttk.Frame(board_frame)
        rotations.grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 2))
        ttk.Button(rotations, text="Rotate board -90°", command=lambda: self.rotate_board(-90.0)).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(rotations, text="Reset board 0°", command=lambda: self.set_yaw_offset(0.0)).grid(row=0, column=1, padx=6)
        ttk.Button(rotations, text="Rotate board +90°", command=lambda: self.rotate_board(90.0)).grid(row=0, column=2, padx=6)
        ttk.Label(board_frame, textvariable=self.effective_yaw_var, style="Subheading.TLabel").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Label(
            board_frame,
            text=(
                "Rotation is about the tile centre while the dock/rest pose stays fixed. The button changes the "
                "pending angle; click Plan Route to rebuild and automatically open a new no-motion preview."
            ),
            wraplength=560,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 0))

        rest_frame = ttk.LabelFrame(outer, text="Editable Rest Pose (Tool TCP)", padding=10)
        rest_frame.grid(row=2, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        for column in range(3):
            rest_frame.grid_columnconfigure(column * 2 + 1, weight=1)
        for index, (label, variable) in enumerate(zip(POSE_LABELS, self.rest_vars)):
            row = index // 3
            column = (index % 3) * 2
            ttk.Label(rest_frame, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=4)
            ttk.Entry(rest_frame, textvariable=variable, width=17).grid(row=row, column=column + 1, sticky="ew", padx=(0, 10), pady=4)
        ttk.Button(rest_frame, text="Restore Profile Pose", command=self.restore_profile_pose).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 3)
        )
        ttk.Label(
            rest_frame,
            text=(
                "These six values are applied only to a session copy of the selected profile. "
                "The original calibration profile is preserved."
            ),
            wraplength=520,
        ).grid(row=3, column=0, columnspan=6, sticky="w", pady=(5, 0))

        sampling_frame = ttk.LabelFrame(outer, text="Batch Settings", padding=10)
        sampling_frame.grid(row=3, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._entry_row(sampling_frame, 0, "Samples", self.samples_var, 10, 0)
        self._entry_row(sampling_frame, 0, "Depth min mm", self.min_depth_var, 10, 2)
        self._entry_row(sampling_frame, 0, "Depth max mm", self.max_depth_var, 10, 4)
        self._entry_row(sampling_frame, 1, "CR3 speed %", self.speed_var, 10, 0)
        self._entry_row(sampling_frame, 1, "Camera source", self.camera_source_var, 10, 2)
        self._entry_row(sampling_frame, 1, "Robot IP", self.robot_ip_var, 15, 4)
        self._entry_row(sampling_frame, 2, "Capture width", self.width_var, 10, 0)
        self._entry_row(sampling_frame, 2, "Capture height", self.height_var, 10, 2)
        self._entry_row(sampling_frame, 2, "FPS", self.fps_var, 10, 4)
        ttk.Checkbutton(sampling_frame, text="Continuous site-high transit", variable=self.continuous_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(7, 0)
        )
        ttk.Checkbutton(sampling_frame, text="Zero tilt only", variable=self.zero_tilt_var).grid(
            row=3, column=3, columnspan=3, sticky="w", pady=(7, 0)
        )

        output_frame = ttk.LabelFrame(outer, text="Output and Execution", padding=10)
        output_frame.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        output_frame.grid_columnconfigure(1, weight=1)
        self._path_row(output_frame, 0, "Output root", self.output_root_var, self.choose_output_root)
        guard_text = (
            "Execute guard: the formal sampler first checks the current TCP. When it is correctly aligned "
            "but 0.1-5.0 mm above the fixed rest crossbar, it lowers at 1% speed to the stored rest pose, "
            "captures a fresh camera reference, then begins continuous multi-point sampling."
        )
        ttk.Label(output_frame, text=guard_text, wraplength=560, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(5, 10)
        )
        self.plan_button = ttk.Button(output_frame, text="Plan Route (No Robot Motion)", command=self.plan_route)
        self.plan_button.grid(row=2, column=0, sticky="ew", padx=(0, 5), pady=3)
        self.open_contacts_button = ttk.Button(
            output_frame, text="Open All Contact Sites", command=self.open_contact_preview, state="disabled"
        )
        self.open_contacts_button.grid(row=2, column=1, sticky="ew", padx=5, pady=3)
        self.open_route_button = ttk.Button(
            output_frame, text="Open TCP Route", command=self.open_route_preview, state="disabled"
        )
        self.open_route_button.grid(row=2, column=2, sticky="ew", padx=(5, 0), pady=3)
        self.execute_button = ttk.Button(
            output_frame, text="Confirm and Start Automatic Sampling", command=self.confirm_and_execute, state="disabled"
        )
        self.execute_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(0, 5), pady=(8, 3))
        self.stop_button = ttk.Button(output_frame, text="Stop Active Sampler", command=self.stop_active, state="disabled")
        self.stop_button.grid(row=3, column=2, sticky="ew", padx=(5, 0), pady=(8, 3))
        self.open_results_button = ttk.Button(
            output_frame, text="Open Latest Output", command=self.open_latest_output, state="disabled"
        )
        self.open_results_button.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        log_frame = ttk.LabelFrame(outer, text="Planner / Sampler Log", padding=7)
        log_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=16, wrap="word", state="disabled", font=("Menlo", 11))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        ttk.Label(outer, textvariable=self.status_var, anchor="w", wraplength=1240).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )

    def _path_row(self, parent: ttk.Widget, row: int, label: str, variable: tk.StringVar, command: Any) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", padx=(6, 0), pady=3)

    def _entry_row(self, parent: ttk.Widget, row: int, label: str, variable: tk.StringVar, width: int, column: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=column + 1, sticky="w", padx=(0, 12), pady=3)

    def _install_change_traces(self) -> None:
        variables: list[tk.Variable] = [
            self.board_path_var,
            self.dock_design_var,
            self.fixture_profile_var,
            self.tile_var,
            self.yaw_offset_var,
            self.samples_var,
            self.min_depth_var,
            self.max_depth_var,
            self.speed_var,
            self.camera_source_var,
            self.width_var,
            self.height_var,
            self.fps_var,
            self.robot_ip_var,
            self.output_root_var,
            self.continuous_var,
            self.zero_tilt_var,
            *self.rest_vars,
        ]
        for variable in variables:
            variable.trace_add("write", self._settings_changed)

    def _settings_changed(self, *_unused: Any) -> None:
        if self._loading_profile:
            return
        self._update_orientation_label()
        if self.plan_state is not None and self.active_process is None:
            self.plan_state = None
            self.execute_button.configure(state="disabled")
            self.open_contacts_button.configure(state="disabled")
            self.open_route_button.configure(state="disabled")
            self.status_var.set(
                "Settings changed. The old HTML is now stale; click Plan Route to generate a fresh no-motion preview."
            )

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_orientation_label(self) -> None:
        try:
            offset = float(self.yaw_offset_var.get().strip())
            if not math.isfinite(offset):
                raise ValueError
            effective = self.saved_board_yaw_deg + offset
            self.effective_yaw_var.set(
                "Saved yaw {:+.3f} deg + offset {:+.3f} deg = effective {:+.3f} deg".format(
                    self.saved_board_yaw_deg, offset, effective
                )
            )
        except ValueError:
            self.effective_yaw_var.set("Runtime yaw offset must be a finite number")

    def choose_board(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a board manifest, STL, or OBJ",
            filetypes=(("Board files", "*.json *.stl *.obj"), ("All files", "*")),
        )
        if selected:
            try:
                self.board_path_var.set(str(normalize_board_dir(selected)))
            except Exception as exc:
                messagebox.showerror("Invalid board selection", str(exc))

    def choose_dock_design(self) -> None:
        selected = filedialog.askopenfilename(title="Select dock design JSON", filetypes=(("JSON", "*.json"), ("All files", "*")))
        if selected:
            self.dock_design_var.set(selected)

    def choose_fixture_profile(self) -> None:
        selected = filedialog.askopenfilename(title="Select fixture profile JSON", filetypes=(("JSON", "*.json"), ("All files", "*")))
        if selected:
            self.fixture_profile_var.set(selected)
            self.load_profile()

    def choose_output_root(self) -> None:
        selected = filedialog.askdirectory(title="Select Jogger output root")
        if selected:
            self.output_root_var.set(selected)

    def load_profile(self) -> None:
        try:
            profile_path = Path(self.fixture_profile_var.get()).expanduser().resolve()
            profile = read_json(profile_path)
            rest_pose = profile_rest_pose(profile)
            tile = str(profile.get("tile_id", ""))
            if tile not in TILES:
                raise ValueError("Fixture profile has unsupported tile_id {!r}".format(tile))
            saved_yaw = float(profile.get("board_yaw_deg", 0.0))
            if not math.isfinite(saved_yaw):
                raise ValueError("Fixture profile board_yaw_deg must be finite")
            self._loading_profile = True
            self.fixture_profile_var.set(str(profile_path))
            self.tile_var.set(tile)
            self.saved_rest_pose = rest_pose
            self.saved_board_yaw_deg = saved_yaw
            for variable, value in zip(self.rest_vars, rest_pose):
                variable.set("{:.6f}".format(value))
            self._update_orientation_label()
            self.plan_state = None
            self.execute_button.configure(state="disabled")
            self.open_contacts_button.configure(state="disabled")
            self.open_route_button.configure(state="disabled")
            self.status_var.set("Loaded saved rest TCP and yaw from {}".format(profile_path.name))
            self._append_log("Loaded fixture profile: {}".format(profile_path))
        except Exception as exc:
            messagebox.showerror("Could not load fixture profile", "{}: {}".format(type(exc).__name__, exc))
            self.status_var.set("Fixture profile could not be loaded.")
        finally:
            self._loading_profile = False

    def restore_profile_pose(self) -> None:
        if self.saved_rest_pose is None:
            self.load_profile()
            return
        self._loading_profile = True
        try:
            for variable, value in zip(self.rest_vars, self.saved_rest_pose):
                variable.set("{:.6f}".format(value))
        finally:
            self._loading_profile = False
        self._settings_changed()
        self.status_var.set("Rest pose restored from the selected fixture profile.")

    def set_yaw_offset(self, value: float) -> None:
        self.yaw_offset_var.set("{:.3f}".format(normalise_quarter_turn(value)))
        if not self._loading_profile and self.active_process is None:
            self.status_var.set(
                "Board rotation is now {:+.0f}°. The old HTML cannot update in place; click Plan Route to rebuild it."
                .format(float(self.yaw_offset_var.get()))
            )

    def rotate_board(self, amount_deg: float) -> None:
        try:
            current = float(self.yaw_offset_var.get().strip())
            if not math.isfinite(current):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid board rotation", "Enter a finite yaw offset before rotating it by 90 degrees.")
            return
        self.set_yaw_offset(current + amount_deg)

    def build_config(self) -> JoggerConfig:
        board_dir = normalize_board_dir(self.board_path_var.get())
        dock_design = Path(self.dock_design_var.get()).expanduser().resolve()
        if not dock_design.is_file():
            raise FileNotFoundError("Dock design JSON does not exist: {}".format(dock_design))
        source_profile = Path(self.fixture_profile_var.get()).expanduser().resolve()
        if not source_profile.is_file():
            raise FileNotFoundError("Fixture profile does not exist: {}".format(source_profile))
        profile = read_json(source_profile)
        tile = self.tile_var.get().strip()
        if tile not in TILES:
            raise ValueError("Mounted tile must be one of {}".format(", ".join(TILES)))
        if str(profile.get("tile_id", "")) != tile:
            raise ValueError(
                "Fixture profile belongs to {}, but the UI requests {}. Load the matching profile or change the tile."
                .format(profile.get("tile_id"), tile)
            )
        rest_pose = finite_pose([variable.get().strip() for variable in self.rest_vars], "editable rest pose")
        yaw_offset = float(self.yaw_offset_var.get().strip())
        if not math.isfinite(yaw_offset):
            raise ValueError("Runtime board yaw offset must be finite")
        saved_yaw = float(profile.get("board_yaw_deg", 0.0))
        if not math.isfinite(saved_yaw):
            raise ValueError("Fixture profile board_yaw_deg must be finite")
        samples = int(self.samples_var.get().strip())
        min_depth = float(self.min_depth_var.get().strip())
        max_depth = float(self.max_depth_var.get().strip())
        speed = float(self.speed_var.get().strip())
        width = int(self.width_var.get().strip())
        height = int(self.height_var.get().strip())
        fps = float(self.fps_var.get().strip())
        if samples < 1:
            raise ValueError("Samples must be at least 1")
        if not 1.0 <= min_depth <= max_depth <= 10.0:
            raise ValueError("Depth range must satisfy 1.0 <= min <= max <= 10.0 mm")
        if not 1.0 <= speed <= 5.0:
            raise ValueError("CR3 speed must be in [1, 5] percent")
        if width < 32 or height < 32 or fps <= 0.0:
            raise ValueError("Capture width/height must be >= 32 and FPS must be positive")
        output_root = Path(self.output_root_var.get()).expanduser().resolve()
        python = absolute_path_preserving_symlink(self.args.python)
        if not python.is_file():
            raise FileNotFoundError("Python executable does not exist: {}".format(python))
        if not SAMPLER_PATH.is_file():
            raise FileNotFoundError("Sampler is missing: {}".format(SAMPLER_PATH))
        return JoggerConfig(
            tile=tile,
            board_dir=board_dir,
            dock_design=dock_design,
            source_fixture_profile=source_profile,
            source_fixture_profile_sha256=sha256_file(source_profile),
            rest_pose=rest_pose,
            saved_board_yaw_deg=saved_yaw,
            board_yaw_offset_deg=yaw_offset,
            samples_per_tile=samples,
            min_depth_mm=min_depth,
            max_depth_mm=max_depth,
            speed_percent=speed,
            camera_source=self.camera_source_var.get().strip(),
            width=width,
            height=height,
            fps=fps,
            robot_ip=self.robot_ip_var.get().strip(),
            tool=int(profile.get("tool", 2)),
            user=int(profile.get("user", 0)),
            continuous_board_transit=bool(self.continuous_var.get()),
            zero_tilt=bool(self.zero_tilt_var.get()),
            python=python,
            output_root=output_root,
        )

    def _prepare_context(self, config: JoggerConfig, mode: str) -> ProcessContext:
        output_dir = unique_run_dir(config.output_root, config.tile, "jogger_{}".format(mode))
        session_root = config.output_root / "jogger_sessions" / output_dir.name
        session_profile = build_session_profile(config.source_fixture_profile, config.rest_pose, session_root, mode)
        write_json(
            session_root / "jogger_config.json",
            {
                "schema": "coverage_board_sampling_jogger_config.v1",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "config": config.canonical_payload(),
                "fingerprint": config.fingerprint(),
                "output_dir": str(output_dir),
                "session_fixture_profile": str(session_profile),
            },
        )
        return ProcessContext(mode=mode, config=config, output_dir=output_dir, session_profile=session_profile)

    def plan_route(self) -> None:
        if self.active_process is not None:
            return
        try:
            config = self.build_config()
            context = self._prepare_context(config, "plan")
            command = make_sampler_command(config, context.session_profile, context.output_dir, execute=False)
        except Exception as exc:
            messagebox.showerror("Cannot plan route", "{}: {}".format(type(exc).__name__, exc))
            return
        self.plan_state = None
        self.execute_button.configure(state="disabled")
        self.open_contacts_button.configure(state="disabled")
        self.open_route_button.configure(state="disabled")
        self.status_var.set("Building a no-motion 3D contact plan and TCP route preview...")
        self._start_process(context, command)

    def confirm_and_execute(self) -> None:
        if self.active_process is not None:
            return
        if self.plan_state is None:
            messagebox.showwarning("Plan required", "Create and review a no-motion TCP route plan before starting collection.")
            return
        try:
            current = self.build_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", "{}: {}".format(type(exc).__name__, exc))
            return
        if current.fingerprint() != self.plan_state.fingerprint:
            messagebox.showwarning(
                "Plan is stale",
                "Board, fixture, rest pose, orientation, or sampling settings changed after planning. Create a new route preview first.",
            )
            return
        confirmation = (
            "Start automatic sampling now?\n\n"
            "Tile: {tile}\n"
            "Samples: {samples}\n"
            "Yaw: saved {saved:+.3f} + offset {offset:+.3f} = {effective:+.3f} deg\n"
            "Rest TCP: {pose}\n"
            "Depth: {minimum:.1f}-{maximum:.1f} mm\n"
            "Speed: {speed:.1f}%\n\n"
            "At startup the sampler accepts only the fixed rest pose or an aligned TCP 0.1-5.0 mm above it. "
            "In the latter case it lowers at 1% speed, refreshes the camera reference, then starts the batch."
        ).format(
            tile=current.tile,
            samples=current.samples_per_tile,
            saved=current.saved_board_yaw_deg,
            offset=current.board_yaw_offset_deg,
            effective=current.effective_board_yaw_deg,
            pose=" ".join("{:.3f}".format(value) for value in current.rest_pose),
            minimum=current.min_depth_mm,
            maximum=current.max_depth_mm,
            speed=current.speed_percent,
        )
        if not messagebox.askyesno("Confirm CR3 automatic collection", confirmation, icon="warning"):
            return
        try:
            context = self._prepare_context(current, "collection")
            command = make_sampler_command(current, context.session_profile, context.output_dir, execute=True)
        except Exception as exc:
            messagebox.showerror("Cannot start collection", "{}: {}".format(type(exc).__name__, exc))
            return
        self.status_var.set("Formal collection started. The log records automatic rest-height adjustment, contact checks, skips, and saved output.")
        self._start_process(context, command)

    def _start_process(self, context: ProcessContext, command: list[str]) -> None:
        if self.active_process is not None:
            raise RuntimeError("A planner or sampler process is already active")
        self._append_log("\n[{}] {}".format(context.mode.upper(), shlex.join(command)))
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
        except Exception as exc:
            self._append_log("Could not start {}: {}: {}".format(context.mode, type(exc).__name__, exc))
            self.status_var.set("Could not start {}.".format(context.mode))
            return
        self.active_process = process
        self._set_busy(True)

        def worker() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("log", line))
            return_code = process.wait()
            self.events.put(("finished", context, return_code))

        threading.Thread(target=worker, name="coverage-board-{}".format(context.mode), daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self.plan_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.execute_button.configure(state="disabled")
        elif self.plan_state is not None:
            self.execute_button.configure(state="normal")

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(str(event[1]))
                elif kind == "finished":
                    self._finish_process(event[1], int(event[2]))
        except queue.Empty:
            pass
        finally:
            try:
                self.root.after(80, self._drain_events)
            except tk.TclError:
                pass

    def _finish_process(self, context: ProcessContext, return_code: int) -> None:
        self.active_process = None
        self._set_busy(False)
        if return_code != 0:
            self.status_var.set("{} stopped with exit code {}. Review the log; no further motion is sent by this GUI.".format(context.mode.title(), return_code))
            self._append_log("[{}] finished with exit code {}".format(context.mode.upper(), return_code))
            return
        self._append_log("[{}] completed successfully: {}".format(context.mode.upper(), context.output_dir))
        self.collection_output = context.output_dir
        self.open_results_button.configure(state="normal")
        if context.mode == "plan":
            self.plan_state = PlanState(
                config=context.config,
                fingerprint=context.config.fingerprint(),
                output_dir=context.output_dir,
                session_profile=context.session_profile,
            )
            self.open_contacts_button.configure(state="normal")
            self.open_route_button.configure(state="normal")
            self.execute_button.configure(state="normal")
            self.status_var.set("Route plan is ready. Review the TCP route, then use Confirm and Start Automatic Sampling.")
            try:
                open_in_default_application(self.plan_state.route_preview)
            except Exception as exc:
                self._append_log("Could not automatically open route preview: {}".format(exc))
        else:
            report = context.output_dir / "collection_report.html"
            self.status_var.set(
                "Collection completed and the sampler returned to the dock. Output saved under {}{}".format(
                    context.output_dir,
                    " (collection_report.html is ready)" if report.exists() else "",
                )
            )
            if report.exists():
                try:
                    open_in_default_application(report)
                except Exception as exc:
                    self._append_log("Could not automatically open collection report: {}".format(exc))

    def open_contact_preview(self) -> None:
        if self.plan_state is None:
            return
        try:
            open_in_default_application(self.plan_state.contact_preview)
        except Exception as exc:
            messagebox.showerror("Cannot open contact plan", str(exc))

    def open_route_preview(self) -> None:
        if self.plan_state is None:
            return
        try:
            open_in_default_application(self.plan_state.route_preview)
        except Exception as exc:
            messagebox.showerror("Cannot open TCP route", str(exc))

    def open_latest_output(self) -> None:
        if self.collection_output is None:
            return
        try:
            report = self.collection_output / "collection_report.html"
            open_in_default_application(report if report.exists() else self.collection_output)
        except Exception as exc:
            messagebox.showerror("Cannot open latest output", str(exc))

    def stop_active(self) -> None:
        process = self.active_process
        if process is None:
            return
        if not messagebox.askyesno("Stop sampler", "Send Ctrl-C to the active sampler? It will run its existing cleanup path.", icon="warning"):
            return
        try:
            process.send_signal(signal.SIGINT)
            self.status_var.set("Stop requested. Waiting for the sampler to clean up its camera/CR3 connection...")
        except Exception as exc:
            messagebox.showerror("Could not stop sampler", "{}: {}".format(type(exc).__name__, exc))

    def close(self) -> None:
        if self.active_process is not None:
            if not messagebox.askyesno(
                "Sampler is active", "Stop the active sampler and close the Jogger?", icon="warning"
            ):
                return
            try:
                self.active_process.send_signal(signal.SIGINT)
            except Exception:
                pass
        self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument("--dock-design", type=Path, default=default_dock_design())
    parser.add_argument("--fixture-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--tile", choices=TILES, default="tile_ne")
    parser.add_argument("--samples-per-tile", type=int, default=2000)
    parser.add_argument("--min-depth-mm", type=float, default=1.0)
    parser.add_argument("--max-depth-mm", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robot-ip", default="192.168.31.88")
    parser.add_argument("--python", type=Path, default=default_python())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = tk.Tk()
    try:
        CoverageBoardJoggerApp(root, args)
    except Exception as exc:
        messagebox.showerror("Jogger startup failed", "{}: {}".format(type(exc).__name__, exc))
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
