#!/usr/bin/env python3
"""Export measured visual-contact brackets for board-height calibration.

Only new, uncorrected zero-tilt flat-reference collection logs containing
actual feedback brackets are accepted. This command neither connects to nor
moves a robot. Edge and curved features cannot establish a global height
residual because the finite TacTip footprint can touch before its apex.
Each input collection represents one independent repeat at its sampled points.
The output is bound to the exact fixture, dock and board files by SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


FIELDNAMES = (
    "point_id", "role", "repeat_id", "local_x_mm", "local_y_mm",
    "nominal_surface_z_mm", "contact_z_mm", "no_contact_z_mm",
    "fixture_profile_sha256", "dock_design_sha256", "board_manifest_sha256",
    "tile_id", "user", "tool", "detection", "collection_sha256",
)
PROFILE_SCHEMAS = {
    "coverage_board_tactip_fixture_profile.v1",
    "coverage_board_tactip_fixture_profile.v2",
    "coverage_board_tactip_fixture_profile.v3",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number")
    return number


def vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label}: expected {length} finite coordinates")
    return tuple(finite_number(v, label) for v in value)


def frame_index(value: Any, label: str) -> int:
    number = finite_number(value, label)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{label}: expected a nonnegative integer")
    return int(number)


def matrix_product(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation_matrix(axis: str, degrees: float) -> list[list[float]]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "X":
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == "Y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def fixture_transform(profile: dict[str, Any], design: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...], list[list[float]]]:
    """Same intrinsic XYZ convention as scipy Rotation.from_euler('XYZ')."""
    if profile.get("schema") not in PROFILE_SCHEMAS:
        raise ValueError("Unsupported fixture profile schema")
    dock = vector(profile.get("dock_tcp"), 6, "fixture dock_tcp")
    reference = design.get("tactip_reference")
    if not isinstance(reference, dict):
        raise ValueError("Dock design lacks tactip_reference")
    nominal = vector(reference.get("nominal_seated_tool_tcp_local_mm"), 3, "dock nominal seated TCP")
    if "rest_pose_stop" in design:
        rest = design["rest_pose_stop"]
        if not isinstance(rest, dict):
            raise ValueError("Invalid rest_pose_stop")
        rest_local = vector(rest.get("contact_centre_tile_local_mm"), 3, "rest-stop contact")
        top = finite_number(rest.get("top_surface_z_mm"), "rest-stop top")
        recorded = profile.get("height_calibration")
        if not isinstance(recorded, dict):
            raise ValueError("Rest-stop dock requires a height-calibrated fixture profile")
        recorded_local = vector(recorded.get("rest_stop_contact_tile_local_mm"), 3, "recorded rest-stop contact")
        if abs(rest_local[2] - top) > .01 or any(abs(a - b) > .01 for a, b in zip(rest_local, nominal)) or any(abs(a - b) > .01 for a, b in zip(recorded_local, rest_local)):
            raise ValueError("Fixture rest-stop datum differs from the dock design")
    rotation = matrix_product(matrix_product(rotation_matrix("X", dock[3]), rotation_matrix("Y", dock[4])), rotation_matrix("Z", dock[5]))
    adaptor = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    yaw = finite_number(profile.get("board_yaw_deg", 0.0), "board_yaw_deg")
    tile_to_base = matrix_product(matrix_product(rotation, adaptor), rotation_matrix("Z", yaw))
    return dock, nominal, tile_to_base


def feedback_to_local(tcp: Any, transform: tuple, label: str) -> tuple[float, ...]:
    pose = vector(tcp, 6, label)
    dock, nominal, rotation = transform
    delta = [pose[i] - dock[i] for i in range(3)]
    return tuple(nominal[i] + sum(rotation[j][i] * delta[j] for j in range(3)) for i in range(3))


def export_measurements(
    fit_collections: Sequence[Path],
    validate_collections: Sequence[Path],
    fixture_profile: Path,
    dock_design: Path,
    board_manifest: Path,
    output: Path,
    max_lateral_error_mm: float = 0.05,
) -> list[dict[str, Any]]:
    """Validate all inputs before atomically writing the combined CSV."""
    lateral_limit = finite_number(max_lateral_error_mm, "max_lateral_error_mm")
    if lateral_limit <= 0:
        raise ValueError("max_lateral_error_mm must be positive")
    if not fit_collections and not validate_collections:
        raise ValueError("Supply at least one fit or validation collection")
    profile, design, manifest = (read_object(Path(p)) for p in (fixture_profile, dock_design, board_manifest))
    bindings = {
        "fixture_profile_sha256": sha256_file(Path(fixture_profile)),
        "dock_design_sha256": sha256_file(Path(dock_design)),
        "board_manifest_sha256": sha256_file(Path(board_manifest)),
    }
    if profile.get("dock_design_sha256") != bindings["dock_design_sha256"]:
        raise ValueError("Fixture profile is bound to a different dock design")
    tile_id = profile.get("tile_id")
    if not isinstance(tile_id, str) or not tile_id:
        raise ValueError("Fixture profile lacks tile_id")
    if not any(isinstance(tile, dict) and tile.get("tile_id") == tile_id for tile in manifest.get("tiles", [])):
        raise ValueError("Fixture tile is absent from the board manifest")
    user, tool = (frame_index(profile.get(key), f"fixture {key}") for key in ("user", "tool"))
    design_tool = design.get("tactip_reference", {}).get("tool")
    if design_tool is not None and frame_index(design_tool, "dock tool") != tool:
        raise ValueError("Fixture tool differs from the dock design")
    transform = fixture_transform(profile, design)
    seen_hashes: set[str] = set()
    coordinates: dict[str, tuple[float, ...]] = {}
    point_roles: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for role, paths in (("fit", fit_collections), ("validate", validate_collections)):
        for raw_path in paths:
            path = Path(raw_path)
            collection_hash = sha256_file(path)
            if collection_hash in seen_hashes:
                raise ValueError(f"{path}: duplicate collection content cannot count as another repeat")
            seen_hashes.add(collection_hash)
            collection = read_object(path)
            if collection.get("schema") != "cr3_coverage_board_collection.v1":
                raise ValueError(f"{path}: unsupported collection schema")
            for key, expected in bindings.items():
                if collection.get(key) != expected:
                    raise ValueError(f"{path}: missing or mismatched {key}; collect new bound feedback logs")
            if collection.get("tile_id") != tile_id:
                raise ValueError(f"{path}: collection tile differs from fixture")
            settings = collection.get("settings")
            if not isinstance(settings, dict):
                raise ValueError(f"{path}: collection lacks settings")
            for key, expected in (("user", user), ("tool", tool)):
                if frame_index(settings.get(key), f"{path}: {key}") != expected:
                    raise ValueError(f"{path}: collection {key} differs from fixture")
            if finite_number(settings.get("board_height_offset_mm"), "board_height_offset_mm") != 0:
                raise ValueError(f"{path}: calibration requires board_height_offset_mm == 0")
            if "height_calibration" not in settings or settings["height_calibration"] is not None:
                raise ValueError(f"{path}: collection must explicitly record height_calibration: null")
            if settings.get("reference_pad_check_enabled") is not False:
                raise ValueError(f"{path}: collection must explicitly record reference_pad_check_enabled: false; disable reference-pad correction when measuring the uncorrected board height")
            samples = collection.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{path}: collection has no measurements")
            seen_points: set[str] = set()
            for index, item in enumerate(samples):
                label = f"{path}: sample {index + 1}"
                if not isinstance(item, dict) or not isinstance(item.get("sample"), dict) or not isinstance(item.get("result"), dict):
                    raise ValueError(f"{label}: missing sample/result object")
                sample, result = item["sample"], item["result"]
                if result.get("status") not in {"captured", "contact_found"}:
                    raise ValueError(f"{label}: unsuccessful measurement cannot enter calibration")
                if sample.get("tile_id") != tile_id:
                    raise ValueError(f"{label}: sample tile differs from fixture")
                if sample.get("stimulus") != "flat_reference":
                    raise ValueError(f"{label}: height calibration requires flat_reference points; edge/curved-feature contact cannot establish a global board height residual")
                if any(finite_number(sample.get(key), f"{label}: {key}") != 0 for key in ("tilt_x_deg", "tilt_y_deg")):
                    raise ValueError(f"{label}: calibration measurements must have zero tilt")
                local = vector(sample.get("local_contact_mm"), 3, f"{label}: local_contact_mm")
                nominal_z = finite_number(sample.get("expected_surface_z_mm"), f"{label}: expected_surface_z_mm")
                if nominal_z != local[2]:
                    raise ValueError(f"{label}: sample nominal surface differs from uncorrected local contact")
                point_id = sample.get("source_seed_site_id")
                if not isinstance(point_id, str) or not point_id.strip():
                    raise ValueError(f"{label}: missing source_seed_site_id")
                if point_id in seen_points:
                    raise ValueError(f"{label}: repeated point within one collection; use independent collection files")
                seen_points.add(point_id)
                if point_id in coordinates and coordinates[point_id] != local:
                    raise ValueError(f"{label}: point_id {point_id} changed local coordinates; disable jitter")
                coordinates[point_id] = local
                if point_id in point_roles and point_roles[point_id] != role:
                    raise ValueError(f"{label}: fit and validation points must be distinct")
                point_roles[point_id] = role
                bracket = result.get("first_contact_bracket")
                if not isinstance(bracket, dict) or bracket.get("detection") != "visual_marker_threshold":
                    raise ValueError(f"{label}: missing actual visual first-contact bracket; planned poses cannot substitute")
                contact = feedback_to_local(bracket.get("contact_tcp"), transform, f"{label}: contact_tcp feedback")
                no_contact = feedback_to_local(bracket.get("no_contact_tcp"), transform, f"{label}: no_contact_tcp feedback")
                if contact[2] >= no_contact[2]:
                    raise ValueError(f"{label}: no-contact feedback must lie strictly above contact feedback")
                for name, actual in (("contact", contact), ("no_contact", no_contact)):
                    lateral_error = math.hypot(actual[0] - local[0], actual[1] - local[1])
                    if lateral_error > lateral_limit:
                        raise ValueError(f"{label}: {name} feedback lateral error {lateral_error:.6g} mm exceeds {lateral_limit:g} mm")
                rows.append({
                    "point_id": point_id, "role": role, "repeat_id": collection_hash,
                    "local_x_mm": local[0], "local_y_mm": local[1], "nominal_surface_z_mm": nominal_z,
                    "contact_z_mm": contact[2], "no_contact_z_mm": no_contact[2],
                    **bindings, "tile_id": tile_id, "user": user, "tool": tool,
                    "detection": bracket["detection"], "collection_sha256": collection_hash,
                })
    output = Path(output)
    source_paths = [Path(p).resolve() for p in (*fit_collections, *validate_collections, fixture_profile, dock_design, board_manifest)]
    if output.resolve() in source_paths:
        raise ValueError("Output must not overwrite a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-collection", type=Path, action="append", default=[], help="Independent fit collection.json; repeat for each run")
    parser.add_argument("--validate-collection", type=Path, action="append", default=[], help="Independent held-out collection.json; repeat for each run")
    parser.add_argument("--fixture-profile", type=Path, required=True)
    parser.add_argument("--dock-design", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lateral-error-mm", type=float, default=.05, help="Reject feedback outside this lateral distance from the intended point (default: 0.05 mm; a software acceptance limit, not certified hardware accuracy)")
    args = parser.parse_args(argv)
    try:
        rows = export_measurements(args.fit_collection, args.validate_collection, args.fixture_profile, args.dock_design, args.board_manifest, args.output, args.max_lateral_error_mm)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Exported {len(rows)} measured brackets to {args.output}")
    print("These bracket visual detection; physical first touch still includes detector bias and robot/TCP uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
