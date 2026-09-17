#!/usr/bin/env python3
"""Build and validate a reusable region-aware board-contact datum.

The coverage board has no broad flat reference on every physical tile.  A
runtime datum therefore measures repeated visual first contacts at known model
locations on each installed tile region.  It keeps a separate tile-local Z
offset per region instead of forcing edges and curved features into one global
height correction.  The common median is retained only as an audit value.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "cr3_coverage_board_runtime_height_datum.v2"
MODEL_KIND = "robust_region_visual_contact_height_offsets"
# Kept as a deterministic synthetic default for callers/tests which do not
# supply a tile-derived site set.  Real collection derives one centre site per
# installed region at runtime.
DEFAULT_SITE_IDS = ("R01_S01", "R01_S03", "R01_S05", "R01_S07", "R01_S09")
DEFAULT_REPEATS_PER_SITE = 3
# A normal one-off height measurement uses the fine 0.02 mm bracket produced
# by the refinement scan.  The automatic board datum deliberately accepts a
# coarser 0.5 mm first-contact interval: it is repeated three times at each
# known region centre and avoids a second, fragile vision sweep while the skin
# is already deforming.  Its midpoint still bounds the surface-height error to
# +/-0.25 mm, below the formal sampler's 1 mm minimum indentation.
DEFAULT_MAX_BRACKET_WIDTH_MM = 0.10
DEFAULT_MAX_COARSE_BRACKET_WIDTH_MM = 0.50
DEFAULT_MAX_REPEAT_SPREAD_MM = 0.50
DEFAULT_MAX_ABS_OFFSET_MM = 10.0
DEFAULT_MAX_LATERAL_ERROR_MM = 0.50
BINDING_FIELDS = (
    "fixture_profile_sha256",
    "dock_design_sha256",
    "board_manifest_sha256",
    "tile_id",
    "user",
    "tool",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_bindings(
    fixture_profile_path: str | Path,
    dock_design_path: str | Path,
    board_manifest_path: str | Path,
    tile_id: str,
    user: int,
    tool: int,
) -> dict[str, Any]:
    return {
        "fixture_profile_sha256": sha256_file(fixture_profile_path),
        "dock_design_sha256": sha256_file(dock_design_path),
        "board_manifest_sha256": sha256_file(board_manifest_path),
        "tile_id": str(tile_id),
        "user": int(user),
        "tool": int(tool),
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    return result


def _point3(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError("{} must contain at least three coordinates".format(name))
    result = np.asarray([_finite(component, "{}[{}]".format(name, index)) for index, component in enumerate(value[:3])])
    return result


def _audit_equal(saved: Any, rebuilt: Any) -> bool:
    if isinstance(rebuilt, dict):
        return isinstance(saved, dict) and set(saved) == set(rebuilt) and all(
            _audit_equal(saved[key], value) for key, value in rebuilt.items()
        )
    if isinstance(rebuilt, list):
        return isinstance(saved, list) and len(saved) == len(rebuilt) and all(
            _audit_equal(left, right) for left, right in zip(saved, rebuilt)
        )
    if isinstance(rebuilt, float):
        return (
            isinstance(saved, (float, int))
            and not isinstance(saved, bool)
            and math.isfinite(float(saved))
            and math.isclose(float(saved), rebuilt, rel_tol=1e-10, abs_tol=1e-10)
        )
    return saved == rebuilt


def _normalise_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_FIELDS):
        raise ValueError("Complete fixture/dock/manifest/tile/User/Tool bindings are required")
    result = dict(bindings)
    for key in BINDING_FIELDS[:3]:
        digest = str(bindings[key]).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Invalid {}".format(key))
        result[key] = digest
    if not isinstance(bindings["tile_id"], str) or not bindings["tile_id"].strip():
        raise ValueError("Invalid tile_id binding")
    for key in ("user", "tool"):
        value = _finite(bindings[key], key)
        if value < 0 or value != int(value):
            raise ValueError("Invalid integer {} binding".format(key))
        result[key] = int(value)
    return result


def normalise_measurements(
    measurements: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    site_ids: Sequence[str] = DEFAULT_SITE_IDS,
    repeats_per_site: int = DEFAULT_REPEATS_PER_SITE,
) -> list[dict[str, Any]]:
    """Validate known-site visual-contact brackets and return clean rows."""

    active_bindings = _normalise_bindings(bindings)
    expected_sites = tuple(str(value) for value in site_ids)
    if len(expected_sites) < 3 or len(set(expected_sites)) != len(expected_sites):
        raise ValueError("Runtime height calibration needs at least three distinct reference sites")
    if int(repeats_per_site) < 2:
        raise ValueError("Runtime height calibration needs at least two repeats per reference site")
    result: list[dict[str, Any]] = []
    for index, source in enumerate(measurements, 1):
        if not isinstance(source, Mapping):
            raise ValueError("Measurement {} is not an object".format(index))
        site_id = str(source.get("site_id", ""))
        if site_id not in expected_sites:
            raise ValueError("Measurement {} uses unexpected site {}".format(index, site_id))
        repeat = int(_finite(source.get("repeat"), "measurement {} repeat".format(index)))
        if repeat < 1:
            raise ValueError("Measurement {} repeat must be positive".format(index))
        stimulus = str(source.get("stimulus", "")).strip()
        category = str(source.get("category", "")).strip()
        region_id = str(source.get("region_id", "")).strip()
        if not stimulus or not category or not region_id:
            raise ValueError("{} lacks its known region/category/stimulus metadata".format(site_id))
        local = _point3(source.get("local_contact_mm"), "measurement {} local_contact_mm".format(index))
        contact = _point3(source.get("contact_tile_local_mm"), "measurement {} contact tile local".format(index))
        unloaded = _point3(source.get("no_contact_tile_local_mm"), "measurement {} no-contact tile local".format(index))
        bracket = float(unloaded[2] - contact[2])
        bracket_mode = str(source.get("bracket_mode", "refined")).strip().lower()
        if bracket_mode not in {"refined", "coarse"}:
            raise ValueError("{} repeat {} has unknown bracket_mode {}".format(site_id, repeat, bracket_mode))
        maximum_bracket = (
            DEFAULT_MAX_COARSE_BRACKET_WIDTH_MM if bracket_mode == "coarse" else DEFAULT_MAX_BRACKET_WIDTH_MM
        )
        if bracket <= 0.0 or bracket > maximum_bracket + 1e-9:
            raise ValueError(
                "{} repeat {} has a {:.4f} mm visual-contact bracket outside (0, {:.3f}] mm"
                .format(site_id, repeat, bracket, maximum_bracket)
            )
        lateral_contact = float(np.linalg.norm(contact[:2] - local[:2]))
        lateral_unloaded = float(np.linalg.norm(unloaded[:2] - local[:2]))
        lateral = max(lateral_contact, lateral_unloaded)
        if lateral > DEFAULT_MAX_LATERAL_ERROR_MM + 1e-9:
            raise ValueError(
                "{} repeat {} drifted {:.3f} mm laterally from its planned flat reference"
                .format(site_id, repeat, lateral)
            )
        midpoint = float((contact[2] + unloaded[2]) / 2.0)
        result.append(
            {
                "site_id": site_id,
                "repeat": repeat,
                "sample_id": str(source.get("sample_id", "")),
                "region_id": region_id,
                "category": category,
                "stimulus": stimulus,
                "local_contact_mm": [float(value) for value in local],
                "nominal_surface_z_mm": float(local[2]),
                "contact_tile_local_mm": [float(value) for value in contact],
                "no_contact_tile_local_mm": [float(value) for value in unloaded],
                "bracket_width_mm": bracket,
                "bracket_mode": bracket_mode,
                "measured_threshold_z_mm": midpoint,
                "residual_tile_z_mm": float(midpoint - local[2]),
                "max_lateral_error_mm": lateral,
                "bindings": active_bindings,
            }
        )
    expected_count = len(expected_sites) * int(repeats_per_site)
    if len(result) != expected_count:
        raise ValueError("Expected {} calibration measurements, received {}".format(expected_count, len(result)))
    for site_id in expected_sites:
        group = [row for row in result if row["site_id"] == site_id]
        repeats = sorted(row["repeat"] for row in group)
        if repeats != list(range(1, int(repeats_per_site) + 1)):
            raise ValueError("{} must contain repeats 1..{} exactly once".format(site_id, repeats_per_site))
        first = group[0]
        for row in group[1:]:
            if not np.allclose(row["local_contact_mm"], first["local_contact_mm"], atol=1e-6):
                raise ValueError("{} repeat locations differ".format(site_id))
            for key in ("region_id", "category", "stimulus"):
                if row[key] != first[key]:
                    raise ValueError("{} repeat metadata differs".format(site_id))
    return sorted(result, key=lambda row: (expected_sites.index(row["site_id"]), row["repeat"]))


def build_runtime_height_datum(
    measurements: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    *,
    site_ids: Sequence[str] = DEFAULT_SITE_IDS,
    repeats_per_site: int = DEFAULT_REPEATS_PER_SITE,
) -> dict[str, Any]:
    """Create robust per-region tile-local Z corrections from real contacts."""

    active_bindings = _normalise_bindings(bindings)
    cleaned = normalise_measurements(measurements, active_bindings, site_ids, repeats_per_site)
    expected_sites = tuple(str(value) for value in site_ids)
    points: list[dict[str, Any]] = []
    for site_id in expected_sites:
        group = [row for row in cleaned if row["site_id"] == site_id]
        residuals = [float(row["residual_tile_z_mm"]) for row in group]
        repeat_spread = float(max(residuals) - min(residuals))
        if repeat_spread > DEFAULT_MAX_REPEAT_SPREAD_MM + 1e-9:
            raise ValueError(
                "{} repeat spread {:.3f} mm exceeds {:.3f} mm; re-seat and repeat calibration"
                .format(site_id, repeat_spread, DEFAULT_MAX_REPEAT_SPREAD_MM)
            )
        points.append(
            {
                "site_id": site_id,
                "region_id": str(group[0]["region_id"]),
                "category": str(group[0]["category"]),
                "stimulus": str(group[0]["stimulus"]),
                "local_contact_mm": list(group[0]["local_contact_mm"]),
                "nominal_surface_z_mm": float(group[0]["nominal_surface_z_mm"]),
                "repeat_count": len(group),
                "median_residual_tile_z_mm": float(np.median(residuals)),
                "repeat_spread_mm": repeat_spread,
                "max_bracket_width_mm": float(max(row["bracket_width_mm"] for row in group)),
                "bracket_modes": sorted(set(str(row["bracket_mode"]) for row in group)),
                "max_lateral_error_mm": float(max(row["max_lateral_error_mm"] for row in group)),
            }
        )
    region_points: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        region_points.setdefault(str(point["region_id"]), []).append(point)
    region_offsets = {
        region_id: float(np.median([float(point["median_residual_tile_z_mm"]) for point in region_rows]))
        for region_id, region_rows in sorted(region_points.items())
    }
    region_site_spreads = {
        region_id: float(
            max(float(point["median_residual_tile_z_mm"]) for point in region_rows)
            - min(float(point["median_residual_tile_z_mm"]) for point in region_rows)
        )
        for region_id, region_rows in sorted(region_points.items())
    }
    offset = float(np.median([float(row["residual_tile_z_mm"]) for row in cleaned]))
    if max(abs(value) for value in region_offsets.values()) > DEFAULT_MAX_ABS_OFFSET_MM + 1e-9:
        raise ValueError(
            "Runtime region height offset exceeds the {:.1f} mm safety limit: {}"
            .format(DEFAULT_MAX_ABS_OFFSET_MM, ", ".join("{}={:+.3f}".format(key, value) for key, value in region_offsets.items()))
        )
    quality = {
        "measurement_count": len(cleaned),
        "site_count": len(points),
        "repeats_per_site": int(repeats_per_site),
        "max_bracket_width_mm": float(max(row["bracket_width_mm"] for row in cleaned)),
        "max_repeat_spread_mm": float(max(point["repeat_spread_mm"] for point in points)),
        "region_count": len(region_offsets),
        "region_offsets_tile_z_mm": region_offsets,
        "region_site_median_spread_mm": region_site_spreads,
        "max_lateral_error_mm": float(max(row["max_lateral_error_mm"] for row in cleaned)),
        "max_abs_offset_mm": DEFAULT_MAX_ABS_OFFSET_MM,
    }
    return {
        "schema": SCHEMA,
        "model_kind": MODEL_KIND,
        "status": "accepted",
        "bindings": active_bindings,
        "offset_tile_z_mm": offset,
        "region_offsets_tile_z_mm": region_offsets,
        "sampling_protocol": {
            "site_ids": list(expected_sites),
            "repeats_per_site": int(repeats_per_site),
            "capture_mode": "visual_first_contact_bracket_midpoint",
            "post_contact_depth_mm": 0.0,
            "tilt_x_deg": 0.0,
            "tilt_y_deg": 0.0,
        },
        "measurements": cleaned,
        "points": points,
        "quality": quality,
        "limitations": [
            "The datum changes the planned tile-local Z per calibrated region; it does not change Tool TCP or learn new feature geometry.",
            "The visual-contact bracket is a detector threshold, not an independently measured force-zero point.",
            "Every capture still uses visual first contact before its requested post-contact indentation, so local shape variation remains checked at every site.",
            "Recalibrate after replacing the board, dock, TacTip, Tool/User frame, or changing the keyed mounting orientation.",
        ],
    }


def load_runtime_height_datum(
    path: str | Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate binding and recompute the robust datum before authorising use."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("Unsupported runtime-height datum schema/model")
    active_bindings = _normalise_bindings(bindings)
    if payload.get("bindings") != active_bindings:
        raise ValueError("Runtime-height datum belongs to another fixture/dock/board/Tool/User; recalibrate")
    if payload.get("status") != "accepted":
        raise ValueError("Runtime-height datum was not accepted")
    protocol = dict(payload.get("sampling_protocol", {}))
    rebuilt = build_runtime_height_datum(
        payload.get("measurements", []),
        active_bindings,
        site_ids=protocol.get("site_ids", ()),
        repeats_per_site=protocol.get("repeats_per_site"),
    )
    for key in ("offset_tile_z_mm", "region_offsets_tile_z_mm", "sampling_protocol", "points", "quality", "limitations"):
        if not _audit_equal(payload.get(key), rebuilt[key]):
            raise ValueError("Runtime-height datum {} does not match its measurement audit".format(key))
    return payload
