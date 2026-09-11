"""Fit a bounded board-height correction from independently checked measurements.

This is an offline utility: it never connects to a robot or camera.  The input
brackets enclose a *visual detector threshold*, not necessarily physical zero
contact.  A separately measured uncertainty is mandatory.  Acceptance means
the supplied observations satisfy the recorded budget, not zero physical error.
Only zero-tilt samples inside the measured support polygon are supported.
Use broad flat-reference patches with the same detector settings.  Complex
textures have feature-dependent threshold bias and cannot establish one plane.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "cr3_board_height_calibration.v1"
MODEL_KIND = "visual_threshold_residual_plane"
DEFAULT_LIMITS = {
    "max_interval_width_mm": 0.04,
    "max_repeat_spread_mm": 0.03,
    "max_validation_error_mm": 0.03,
    "max_error_budget_mm": 0.10,
    "max_condition_number": 100.0,
    "max_correction_mm": 10.0,
    "max_slope_deg": 2.0,
}
MEASUREMENT_FIELDS = (
    "point_id", "role", "repeat_id", "local_x_mm", "local_y_mm",
    "nominal_surface_z_mm", "contact_z_mm", "no_contact_z_mm",
)
BINDING_FIELDS = ("fixture_profile_sha256", "dock_design_sha256", "board_manifest_sha256", "tile_id", "user", "tool")
LIMITATIONS = [
    "The bracket measures visual threshold crossing, not geometric zero contact.",
    "Only broad flat_reference patches are valid calibration references; one texture's contact threshold cannot calibrate another feature's height.",
    "Recorded robot decimal places express readout resolution, not independently established absolute accuracy or measurement uncertainty.",
    "The independent reference uncertainty is an operator-supplied bound; software cannot verify the physical measurement.",
    "The additive error budget is an acceptance bound for the supplied observations, not a statistical confidence interval or guarantee elsewhere.",
    "Correction supports zero tilt only; TCP, surface shape, lateral registration, warping and force-dependent deformation require separate checks.",
    "No extrapolation or manual height-offset stacking is permitted. Recalibrate after any fixture, Tool/User, sensor or board change.",
]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bindings(
    fixture_profile_path: str | Path,
    dock_design_path: str | Path,
    board_manifest_path: str | Path,
    tile_id: str,
    user: int,
    tool: int,
) -> dict[str, Any]:
    if not str(tile_id).strip():
        raise ValueError("tile_id must not be empty")
    return {
        "fixture_profile_sha256": _sha256(fixture_profile_path),
        "dock_design_sha256": _sha256(dock_design_path),
        "board_manifest_sha256": _sha256(board_manifest_path),
        "tile_id": str(tile_id), "user": int(user), "tool": int(tool),
    }


def _normalise_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_FIELDS):
        raise ValueError("Complete fixture/dock/manifest/tile/User/Tool bindings are required")
    result = dict(bindings)
    for key in BINDING_FIELDS[:3]:
        digest = str(bindings[key]).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid {key}")
        result[key] = digest
    if not isinstance(bindings["tile_id"], str) or not bindings["tile_id"].strip():
        raise ValueError("Invalid tile_id binding")
    for key in ("user", "tool"):
        value = _finite(bindings[key], key)
        if value < 0 or value != int(value):
            raise ValueError(f"Invalid integer {key} binding")
        result[key] = int(value)
    return result


def _audit_equal(saved: Any, rebuilt: Any) -> bool:
    """Allow floating-point roundoff across NumPy/BLAS builds, not model edits."""
    if isinstance(rebuilt, dict):
        return isinstance(saved, dict) and set(saved) == set(rebuilt) and all(
            _audit_equal(saved[key], value) for key, value in rebuilt.items())
    if isinstance(rebuilt, list):
        return isinstance(saved, list) and len(saved) == len(rebuilt) and all(
            _audit_equal(left, right) for left, right in zip(saved, rebuilt))
    if isinstance(rebuilt, float):
        return (isinstance(saved, (int, float)) and not isinstance(saved, bool)
                and math.isfinite(saved) and math.isclose(saved, rebuilt, abs_tol=1e-10, rel_tol=1e-10))
    return saved == rebuilt


def _cross(origin: Iterable[float], a: Iterable[float], b: Iterable[float]) -> float:
    ox, oy = origin
    ax, ay = a
    bx, by = b
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _convex_hull(points: Iterable[Iterable[float]]) -> list[list[float]]:
    ordered = sorted(set(tuple(map(float, point)) for point in points))
    if len(ordered) < 3:
        raise ValueError("At least three distinct fit XY locations are required")
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Fit points must be non-collinear")
    return [list(point) for point in hull]


def _inside_hull(hull: list[list[float]], x: float, y: float) -> bool:
    if len(hull) < 3:
        return False
    # Numerical tolerance only: no millimetre-sized extension of support.
    return all(_cross(a, hull[(index + 1) % len(hull)], (x, y)) >= -1e-9
               for index, a in enumerate(hull))


def _normalise_rows(rows: Iterable[Mapping[str, Any]], bindings: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows, 1):
        missing = set(MEASUREMENT_FIELDS + BINDING_FIELDS) - set(row)
        if missing:
            raise ValueError(f"Row {index} lacks fields: {sorted(missing)}")
        clean: dict[str, Any] = {}
        for name in MEASUREMENT_FIELDS[:3]:
            value = row[name]
            if value is None or not str(value).strip():
                raise ValueError(f"Row {index}: {name} must not be empty")
            clean[name] = str(value).strip()
        if clean["role"] not in {"fit", "validate"}:
            raise ValueError(f"Row {index}: role must be fit or validate")
        for name in MEASUREMENT_FIELDS[3:]:
            clean[name] = _finite(row[name], f"row {index} {name}")
        measured_bindings = _normalise_bindings({key: row[key] for key in BINDING_FIELDS})
        if measured_bindings != bindings:
            raise ValueError(f"Row {index} measurement bindings differ from the active fixture/dock/board/Tool/User")
        clean.update(measured_bindings)
        result.append(clean)
    if not result:
        raise ValueError("No measurement rows supplied")
    return result


def fit_height_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    measurement_uncertainty_mm: float,
    measurement_reference: str,
    bindings: Mapping[str, Any],
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a profile only if every input, support and quality gate passes."""
    uncertainty = _finite(measurement_uncertainty_mm, "measurement_uncertainty_mm")
    if uncertainty <= 0:
        raise ValueError("A positive independently measured uncertainty is required")
    if not isinstance(measurement_reference, str) or not measurement_reference.strip():
        raise ValueError("Describe the independent measurement reference; do not invent uncertainty")
    actual_limits = dict(DEFAULT_LIMITS)
    if limits is not None:
        if set(limits) - set(DEFAULT_LIMITS):
            raise ValueError("Unknown quality limit")
        actual_limits.update(limits)
    for key, value in actual_limits.items():
        actual_limits[key] = _finite(value, key)
        if actual_limits[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if actual_limits["max_slope_deg"] >= 90:
        raise ValueError("max_slope_deg must be below 90 degrees")
    active_bindings = _normalise_bindings(bindings)
    measurements = _normalise_rows(rows, active_bindings)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in measurements:
        groups.setdefault(row["point_id"], []).append(row)
    points = []
    occupied_xy: list[tuple[float, float]] = []
    for point_id, group in groups.items():
        if len(group) < 3 or len({row["repeat_id"] for row in group}) != len(group):
            raise ValueError(f"{point_id}: at least three independent, uniquely numbered repeats are required")
        first = group[0]
        xy = (first["local_x_mm"], first["local_y_mm"])
        if any(math.hypot(xy[0] - old[0], xy[1] - old[1]) <= 1e-6 for old in occupied_xy):
            raise ValueError("Different point IDs must represent distinct XY locations; validation must be independent")
        occupied_xy.append(xy)
        midpoints = []
        half_widths = []
        for row in group:
            if row["role"] != first["role"] or any(abs(row[key] - first[key]) > 1e-6 for key in
                                                     ("local_x_mm", "local_y_mm", "nominal_surface_z_mm")):
                raise ValueError(f"{point_id}: repeated point geometry or role is inconsistent")
            width = row["no_contact_z_mm"] - row["contact_z_mm"]
            if width <= 0 or width > actual_limits["max_interval_width_mm"] + 1e-9:
                raise ValueError(f"{point_id}: contact bracket width must be positive and within limit")
            midpoints.append((row["no_contact_z_mm"] + row["contact_z_mm"]) / 2)
            half_widths.append(width / 2)
        spread = max(midpoints) - min(midpoints)
        if spread > actual_limits["max_repeat_spread_mm"] + 1e-9:
            raise ValueError(f"{point_id}: repeat spread exceeds limit")
        measured_z = float(np.median(midpoints))
        points.append({
            "point_id": point_id, "role": first["role"], "local_x_mm": xy[0], "local_y_mm": xy[1],
            "nominal_surface_z_mm": first["nominal_surface_z_mm"], "measured_threshold_z_mm": measured_z,
            "residual_mm": measured_z - first["nominal_surface_z_mm"], "repeat_count": len(group),
            "repeat_spread_mm": spread, "interval_half_width_mm": max(half_widths),
        })
    fit = [point for point in points if point["role"] == "fit"]
    validation = [point for point in points if point["role"] == "validate"]
    if len(fit) < 3 or len(validation) < 2:
        raise ValueError("Need at least three fit points and two independent validation points")
    fit_xy = np.asarray([[point["local_x_mm"], point["local_y_mm"]] for point in fit])
    hull = _convex_hull(fit_xy)
    centre = np.mean(fit_xy, axis=0)
    scale = float(np.max(np.ptp(fit_xy, axis=0)))
    matrix = np.column_stack(((fit_xy - centre) / scale, np.ones(len(fit))))
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > actual_limits["max_condition_number"]:
        raise ValueError("Fit point geometry is ill-conditioned; spread measurements across the board")
    coef_scaled, _, rank, _ = np.linalg.lstsq(matrix, np.asarray([point["residual_mm"] for point in fit]), rcond=None)
    if rank != 3:
        raise ValueError("Height plane is underdetermined")
    slopes = coef_scaled[:2] / scale
    a, b = map(float, slopes)
    c = float(coef_scaled[2] - np.dot(slopes, centre))
    slope_deg = math.degrees(math.atan(math.hypot(a, b)))
    max_correction = max(abs(a * x + b * y + c) for x, y in hull)
    if slope_deg > actual_limits["max_slope_deg"] + 1e-9:
        raise ValueError("Correction slope exceeds limit; check fixture/TCP")
    if max_correction > actual_limits["max_correction_mm"] + 1e-9:
        raise ValueError("Correction magnitude exceeds limit; check fixture/TCP")
    for point in points:
        x, y = point["local_x_mm"], point["local_y_mm"]
        if not _inside_hull(hull, x, y):
            raise ValueError(f"{point['point_id']}: validation lies outside fit support; extrapolation is forbidden")
        point["predicted_correction_mm"] = a * x + b * y + c
        point["prediction_error_mm"] = point["residual_mm"] - point["predicted_correction_mm"]
    validation_error = max(abs(point["prediction_error_mm"]) for point in validation)
    fit_error = max(abs(point["prediction_error_mm"]) for point in fit)
    if max(validation_error, fit_error) > actual_limits["max_validation_error_mm"] + 1e-9:
        raise ValueError("Fit or independent validation residual exceeds limit; a plane does not explain the measurements")
    interval_half_width = max(point["interval_half_width_mm"] for point in points)
    repeat_spread = max(point["repeat_spread_mm"] for point in points)
    budget = max(validation_error, fit_error) + interval_half_width + repeat_spread + uncertainty
    if budget > actual_limits["max_error_budget_mm"] + 1e-9:
        raise ValueError(f"Additive error budget {budget:.6f} mm exceeds limit {actual_limits['max_error_budget_mm']:.6f} mm")
    return {
        "schema": SCHEMA, "model_kind": MODEL_KIND, "status": "accepted",
        "bindings": active_bindings, "coefficients_mm": [a, b, c], "support_hull_xy_mm": hull,
        "supported_tilt_deg": 0.0, "limits": actual_limits,
        "measurement_uncertainty_mm": uncertainty, "measurement_reference": measurement_reference.strip(),
        "measurements": measurements, "points": points,
        "quality": {"fit_point_count": len(fit), "validation_point_count": len(validation),
                    "fit_max_error_mm": fit_error, "validation_max_error_mm": validation_error,
                    "interval_half_width_mm": interval_half_width, "repeat_spread_mm": repeat_spread,
                    "additive_error_budget_mm": budget, "condition_number": condition,
                    "max_correction_mm": max_correction, "slope_deg": slope_deg},
        "limitations": list(LIMITATIONS),
    }


def load_height_calibration(
    path: str | Path,
    fixture_profile_path: str | Path,
    dock_design_path: str | Path,
    board_manifest_path: str | Path,
    tile_id: str,
    user: int,
    tool: int,
) -> dict[str, Any]:
    """Bind to the active setup and independently repeat all quality gates."""
    with Path(path).open(encoding="utf-8") as stream:
        profile = json.load(stream)
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA or profile.get("model_kind") != MODEL_KIND:
        raise ValueError("Unsupported height-calibration schema/model")
    expected = make_bindings(fixture_profile_path, dock_design_path, board_manifest_path, tile_id, user, tool)
    if profile.get("bindings") != expected:
        raise ValueError("Height calibration belongs to another fixture/dock/board/Tool/User; measure again")
    if profile.get("status") != "accepted" or profile.get("supported_tilt_deg") != 0.0:
        raise ValueError("Height calibration was not accepted for zero-tilt use")
    rebuilt = fit_height_calibration(
        profile.get("measurements", []), measurement_uncertainty_mm=profile.get("measurement_uncertainty_mm"),
        measurement_reference=profile.get("measurement_reference"), bindings=expected, limits=profile.get("limits"),
    )
    # Detect editing of the model, support, quality report or declared limits.
    for key in ("coefficients_mm", "support_hull_xy_mm", "quality", "points", "limits", "limitations"):
        if not _audit_equal(profile.get(key), rebuilt[key]):
            raise ValueError(f"Height calibration {key} does not match its measurement audit")
    return rebuilt


def height_correction_at(profile: Mapping[str, Any], x: float, y: float) -> float:
    """Evaluate an already loaded profile; reject positions outside support."""
    x, y = _finite(x, "x"), _finite(y, "y")
    if (profile.get("schema") != SCHEMA or profile.get("status") != "accepted"
            or profile.get("model_kind") != MODEL_KIND or profile.get("supported_tilt_deg") != 0.0):
        raise ValueError("An accepted zero-tilt height-calibration profile is required")
    hull = profile.get("support_hull_xy_mm", [])
    if not _inside_hull(hull, x, y):
        raise ValueError(f"Sample ({x:.3f}, {y:.3f}) is outside measured height-calibration support; extrapolation forbidden")
    a, b, c = [_finite(value, "coefficient") for value in profile["coefficients_mm"]]
    correction = a * x + b * y + c
    if not math.isfinite(correction) or abs(correction) > profile["limits"]["max_correction_mm"] + 1e-9:
        raise ValueError("Height correction exceeds the accepted bound")
    return correction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Measured threshold brackets in tile-local coordinates, millimetres")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-profile", type=Path, required=True)
    parser.add_argument("--dock-design", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--tool", type=int, required=True)
    parser.add_argument("--measurement-uncertainty-mm", type=float, required=True,
                        help="Positive uncertainty bound from independent reference measurement, not a guessed step size")
    parser.add_argument("--measurement-reference", required=True,
                        help="Describe instrument/reference, measurement method, date and uncertainty evidence")
    for key, default in DEFAULT_LIMITS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=default)
    args = parser.parse_args(argv)
    try:
        with args.csv.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        profile = fit_height_calibration(
            rows, measurement_uncertainty_mm=args.measurement_uncertainty_mm,
            measurement_reference=args.measurement_reference,
            bindings=make_bindings(args.fixture_profile, args.dock_design, args.board_manifest, args.tile, args.user, args.tool),
            limits={key: getattr(args, key) for key in DEFAULT_LIMITS},
        )
        # Refuse to replace an existing accepted calibration accidentally.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Height calibration rejected: {exc}\n")
    print(f"Accepted observation error budget: {profile['quality']['additive_error_budget_mm']:.4f} mm")
    print(f"Saved {args.output}; zero tilt, inside measured support only. Physical zero error is not guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
