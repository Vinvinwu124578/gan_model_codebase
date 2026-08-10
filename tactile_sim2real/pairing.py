"""Shared plan and record helpers for calibrated tactile image pairs.

The pair plan describes a contact pose relative to a zero-contact reference in
the task frame. It can optionally include an expected absolute CR3 TCP pose;
that is what allows the real collector to reject off-plan captures.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from scipy.spatial.transform import Rotation


POSE_COLUMNS = ["pose_x", "pose_y", "pose_z", "pose_Rx", "pose_Ry", "pose_Rz"]
SHEAR_COLUMNS = ["shear_x", "shear_y", "shear_z", "shear_Rx", "shear_Ry", "shear_Rz"]
OBJECT_COLUMNS = ["object_x", "object_y", "object_z", "object_Rx", "object_Ry", "object_Rz"]
REAL_TARGET_TCP_COLUMNS = [
    "real_target_tcp_x",
    "real_target_tcp_y",
    "real_target_tcp_z",
    "real_target_tcp_Rx",
    "real_target_tcp_Ry",
    "real_target_tcp_Rz",
]
ACTUAL_TCP_COLUMNS = [
    "actual_tcp_x",
    "actual_tcp_y",
    "actual_tcp_z",
    "actual_tcp_Rx",
    "actual_tcp_Ry",
    "actual_tcp_Rz",
]
PAIR_PLAN_COLUMNS = [
    "pair_id",
    "object_label",
    *POSE_COLUMNS,
    *SHEAR_COLUMNS,
    *OBJECT_COLUMNS,
    *REAL_TARGET_TCP_COLUMNS,
    "sim_seed",
    "calibration_note",
]
PAIR_RECORD_COLUMNS = [
    "pair_id",
    "plan_index",
    "collector",
    "sensor_image",
    "source_sample_id",
    "timestamp",
    "object_label",
    *POSE_COLUMNS,
    *SHEAR_COLUMNS,
    *OBJECT_COLUMNS,
    *ACTUAL_TCP_COLUMNS,
    *REAL_TARGET_TCP_COLUMNS,
    "position_error_mm",
    "rotation_error_deg",
    "verification_status",
    "plan_path",
]


def format_values(values) -> list[str]:
    return ["{:.8f}".format(float(value)) for value in values]


def write_header_if_missing(path: Path, columns: list[str]) -> None:
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            existing_columns = csv.DictReader(file).fieldnames
        if existing_columns != columns:
            raise RuntimeError("{} has an incompatible header.".format(path))
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()


def parse_optional_float(value: str, column: str, pair_id: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError("pair_id {!r} has a non-numeric {} value {!r}".format(pair_id, column, text)) from exc
    if not math.isfinite(number):
        raise ValueError("pair_id {!r} has a non-finite {} value".format(pair_id, column))
    return number


def angular_difference_deg(actual: float, expected: float) -> float:
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def pose_error(actual: tuple[float, ...], expected: tuple[float, ...]) -> tuple[float, float]:
    position_error = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual[:3], expected[:3])))
    rotation_error = max(angular_difference_deg(a, b) for a, b in zip(actual[3:], expected[3:]))
    return position_error, rotation_error


def translation_basis_from_values(values) -> tuple[tuple[float, float, float], ...]:
    """Validate a row-major task-translation basis expressed in CR3 base axes."""
    raw_values = tuple(values)
    if len(raw_values) == 3 and all(isinstance(row, (tuple, list)) for row in raw_values):
        flattened = tuple(float(value) for row in raw_values for value in row)
    else:
        flattened = tuple(float(value) for value in raw_values)
    if len(flattened) != 9:
        raise ValueError("translation basis must contain nine row-major values")
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError("translation basis must contain only finite values")
    matrix = tuple(tuple(flattened[row * 3 : row * 3 + 3]) for row in range(3))
    columns = tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))
    norms = tuple(math.sqrt(sum(value * value for value in column)) for column in columns)
    if any(abs(norm - 1.0) > 1e-3 for norm in norms):
        raise ValueError("translation-basis columns must be unit vectors")
    for first in range(3):
        for second in range(first + 1, 3):
            dot_product = sum(columns[first][index] * columns[second][index] for index in range(3))
            if abs(dot_product) > 1e-3:
                raise ValueError("translation-basis columns must be orthogonal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ValueError("translation-basis must be right-handed with determinant +1")
    return matrix


def task_offset_to_real_tcp(
    reference_tcp: tuple[float, ...] | list[float],
    task_offset: tuple[float, ...] | list[float],
    translation_basis_base: tuple[tuple[float, float, float], ...] | list[float] | None = None,
    real_rx_sign: float = 1.0,
    real_ry_sign: float = 1.0,
    real_rz_sign: float = 1.0,
) -> tuple[float, ...]:
    """Map a simulated task-frame offset onto a calibrated real CR3 TCP pose.

    Tactile Gym plan translations are expressed in the zero-contact work frame.
    The real task frame is therefore defined by the calibrated TCP orientation at
    that same contact. Per-axis real rotation signs handle cameras whose
    observed rotation convention is reversed from the simulator while leaving
    the plan's sim rotation values intact.
    """
    reference = tuple(float(value) for value in reference_tcp)
    offset = tuple(float(value) for value in task_offset)
    if len(reference) != 6 or len(offset) != 6:
        raise ValueError("reference_tcp and task_offset must each contain six values")
    if not all(math.isfinite(value) for value in (*reference, *offset)):
        raise ValueError("reference_tcp and task_offset must contain only finite values")
    if any(sign not in (-1.0, 1.0) for sign in (real_rx_sign, real_ry_sign, real_rz_sign)):
        raise ValueError("real rotation signs must each be either -1 or 1")
    if translation_basis_base is None:
        position_delta = Rotation.from_euler("XYZ", reference[3:], degrees=True).apply(offset[:3])
    else:
        basis = translation_basis_from_values(translation_basis_base)
        position_delta = tuple(sum(basis[row][column] * offset[column] for column in range(3)) for row in range(3))
    rotation_delta = (
        real_rx_sign * offset[3],
        real_ry_sign * offset[4],
        real_rz_sign * offset[5],
    )
    return tuple(float(reference[index] + position_delta[index]) for index in range(3)) + tuple(
        float(reference[index + 3] + rotation_delta[index]) for index in range(3)
    )


@dataclass(frozen=True)
class PairPlanRow:
    index: int
    values: dict[str, str]

    @property
    def pair_id(self) -> str:
        return self.values["pair_id"]

    @property
    def object_label(self) -> str:
        return self.values.get("object_label", "")

    def value(self, column: str) -> str:
        return self.values.get(column, "")

    def values_for(self, columns: list[str]) -> list[str]:
        return [self.value(column) for column in columns]

    def required_pose(self) -> tuple[float, ...]:
        values = [parse_optional_float(self.value(column), column, self.pair_id) for column in POSE_COLUMNS]
        if any(value is None for value in values):
            missing = [column for column, value in zip(POSE_COLUMNS, values) if value is None]
            raise ValueError("pair_id {!r} is missing plan pose columns: {}".format(self.pair_id, ", ".join(missing)))
        return tuple(float(value) for value in values)

    def expected_real_tcp(self) -> tuple[float, ...] | None:
        values = [
            parse_optional_float(self.value(column), column, self.pair_id)
            for column in REAL_TARGET_TCP_COLUMNS
        ]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            missing = [column for column, value in zip(REAL_TARGET_TCP_COLUMNS, values) if value is None]
            raise ValueError(
                "pair_id {!r} has a partial expected CR3 TCP pose; missing {}".format(
                    self.pair_id, ", ".join(missing)
                )
            )
        return tuple(float(value) for value in values)

    def sim_seed(self, default: int) -> int:
        value = parse_optional_float(self.value("sim_seed"), "sim_seed", self.pair_id)
        if value is None:
            return int(default)
        if not float(value).is_integer():
            raise ValueError("pair_id {!r} has a non-integer sim_seed".format(self.pair_id))
        return int(value)


class PairPlan:
    def __init__(self, path: Path, rows: list[PairPlanRow]) -> None:
        self.path = path.resolve()
        self.rows = rows
        self.cursor = 0

    @classmethod
    def load(cls, path: Path) -> "PairPlan":
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Could not find pair plan: {}".format(path))
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "pair_id" not in reader.fieldnames:
                raise ValueError("Pair plan {} must include a pair_id column.".format(path))
            rows = []
            seen_pair_ids = set()
            for index, raw_row in enumerate(reader, start=1):
                values = {column: str(raw_row.get(column, "") or "").strip() for column in PAIR_PLAN_COLUMNS}
                pair_id = values["pair_id"]
                if not pair_id:
                    raise ValueError("Pair plan {} row {} has an empty pair_id.".format(path, index + 1))
                if pair_id in seen_pair_ids:
                    raise ValueError("Pair plan {} contains duplicate pair_id {!r}.".format(path, pair_id))
                seen_pair_ids.add(pair_id)
                for column in [*POSE_COLUMNS, *SHEAR_COLUMNS, *OBJECT_COLUMNS, *REAL_TARGET_TCP_COLUMNS]:
                    parse_optional_float(values[column], column, pair_id)
                rows.append(PairPlanRow(index=index, values=values))
        if not rows:
            raise ValueError("Pair plan {} has no data rows.".format(path))
        return cls(path, rows)

    @property
    def current(self) -> PairPlanRow | None:
        if self.cursor >= len(self.rows):
            return None
        return self.rows[self.cursor]

    def advance(self) -> PairPlanRow | None:
        if self.current is not None:
            self.cursor += 1
        return self.current

    def start_at(self, pair_id: str | None) -> None:
        if not pair_id:
            return
        for index, row in enumerate(self.rows):
            if row.pair_id == pair_id:
                self.cursor = index
                return
        raise ValueError("pair_id {!r} does not exist in {}.".format(pair_id, self.path))

    def resume_after(self, completed_pair_ids: set[str]) -> None:
        for index, row in enumerate(self.rows):
            if row.pair_id not in completed_pair_ids:
                self.cursor = index
                return
        self.cursor = len(self.rows)


def verify_real_pose(
    row: PairPlanRow,
    actual_pose: tuple[float, ...] | None,
    position_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> tuple[str, float | None, float | None]:
    expected_pose = row.expected_real_tcp()
    if actual_pose is None:
        return "pose_missing", None, None
    if expected_pose is None:
        return "pose_unverified", None, None
    position_error, rotation_error = pose_error(actual_pose, expected_pose)
    if position_error <= position_tolerance_mm and rotation_error <= rotation_tolerance_deg:
        return "pose_verified", position_error, rotation_error
    return "pose_mismatch", position_error, rotation_error
