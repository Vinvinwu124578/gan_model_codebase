"""Offline integrity checks for feedback-based height measurement export."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_board_height_measurements.py"
spec = importlib.util.spec_from_file_location("height_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class HeightExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("height_export_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        self.design = self.root / "dock.json"
        self.profile = self.root / "fixture.json"
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "measured.csv"
        self.write(self.design, {"tactip_reference": {"tool": 2, "nominal_seated_tool_tcp_local_mm": [0, -139, 12]}})
        self.write(self.manifest, {"tiles": [{"tile_id": "tile_nw"}]})
        self.profile_data = {
            "schema": "coverage_board_tactip_fixture_profile.v1", "tile_id": "tile_nw", "user": 0, "tool": 2,
            "dock_design_sha256": exporter.sha256_file(self.design), "dock_tcp": [300, 40, 200, 180, 0, 0], "board_yaw_deg": 0,
        }
        self.write(self.profile, self.profile_data)

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def cleanup(self):
        resolved = self.root.resolve()
        if resolved.parent != Path(__file__).resolve().parent or not resolved.name.startswith("height_export_test_"):
            raise RuntimeError("Temporary test directory escaped tests directory")
        shutil.rmtree(resolved)

    def measurement(self, point="P1", local=(10, 20, 6)):
        # Rx(180) * diag(1,-1,-1) = identity: no transform helper is used
        # to manufacture these expected base-frame feedback values.
        x, y, z = local
        return {
            "sample": {"tile_id": "tile_nw", "source_seed_site_id": point, "stimulus": "flat_reference", "local_contact_mm": list(local),
                       "expected_surface_z_mm": z, "tilt_x_deg": 0, "tilt_y_deg": 0},
            "result": {"status": "contact_found", "first_contact_bracket": {
                "contact_tcp": [300 + x, 40 + y + 139, 200 + z - 12 + .9, 180, 0, 0],
                "no_contact_tcp": [300 + x, 40 + y + 139, 200 + z - 12 + 1.0, 180, 0, 0],
                "detection": "visual_marker_threshold",
            }},
        }

    def collection(self, name="run1.json", samples=None, **changes):
        value = {
            "schema": "cr3_coverage_board_collection.v1", "created_at": name,
            "fixture_profile_sha256": exporter.sha256_file(self.profile),
            "dock_design_sha256": exporter.sha256_file(self.design),
            "board_manifest_sha256": exporter.sha256_file(self.manifest), "tile_id": "tile_nw",
            "settings": {"tool": 2, "user": 0, "board_height_offset_mm": 0, "height_calibration": None, "reference_pad_check_enabled": False},
            "samples": samples if samples is not None else [self.measurement()],
        }
        value.update(changes)
        return self.write(self.root / name, value)

    def run_export(self, fit, validate=(), **kwargs):
        return exporter.export_measurements(fit, validate, self.profile, self.design, self.manifest, self.output, **kwargs)

    def test_actual_feedback_and_binding_written(self):
        p = self.collection()
        rows = self.run_export([p])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["contact_z_mm"], 6.9)
        self.assertAlmostEqual(rows[0]["no_contact_z_mm"], 7.0)
        self.assertEqual(rows[0]["repeat_id"], exporter.sha256_file(p))
        with self.output.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["fixture_profile_sha256"], exporter.sha256_file(self.profile))
        self.assertEqual(row["detection"], "visual_marker_threshold")

    def test_independent_repeat_files_and_heldout_points(self):
        fit = [self.collection(f"fit{i}.json") for i in range(3)]
        val = [self.collection(f"val{i}.json", [self.measurement("V1", (-5, 11, 6))]) for i in range(3)]
        rows = self.run_export(fit, val)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["role"] for r in rows}, {"fit", "validate"})
        self.assertEqual(len({r["repeat_id"] for r in rows}), 6)

    def test_same_content_copy_is_not_independent_repeat(self):
        p = self.collection()
        q = self.root / "copy.json"
        q.write_bytes(p.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate collection"):
            self.run_export([p, q])

    def test_missing_real_bracket_rejected_and_output_preserved(self):
        self.output.write_text("previous valid output", encoding="utf-8")
        m = self.measurement()
        del m["result"]["first_contact_bracket"]
        m["result"]["visual_contact_tcp"] = [300, 40, 200, 180, 0, 0]
        with self.assertRaisesRegex(ValueError, "missing actual visual"):
            self.run_export([self.collection(samples=[m])])
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous valid output")

    def test_all_reference_hashes_required(self):
        for key in ("fixture_profile_sha256", "dock_design_sha256", "board_manifest_sha256"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self.run_export([self.collection(**{key: "wrong"})])

    def test_tool_user_tile_mismatch(self):
        cases = [
            {"settings": {"user": 0, "tool": 3, "board_height_offset_mm": 0, "height_calibration": None, "reference_pad_check_enabled": False}},
            {"settings": {"user": 1, "tool": 2, "board_height_offset_mm": 0, "height_calibration": None, "reference_pad_check_enabled": False}},
            {"tile_id": "tile_ne"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "differs from fixture"):
                    self.run_export([self.collection(**changes)])

    def test_no_offset_or_preexisting_plane_calibration(self):
        for change in ({"board_height_offset_mm": 1}, {"height_calibration": "plane.json"}, {"height_calibration": False}):
            settings = {"user": 0, "tool": 2, "board_height_offset_mm": 0, "height_calibration": None, "reference_pad_check_enabled": False, **change}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.run_export([self.collection(settings=settings)])
        with self.assertRaisesRegex(ValueError, "explicitly record"):
            self.run_export([self.collection(settings={"user": 0, "tool": 2, "board_height_offset_mm": 0, "reference_pad_check_enabled": False})])

    def test_reference_pad_correction_must_be_explicitly_disabled(self):
        for value in (True, None, 0, "false"):
            settings = {"user": 0, "tool": 2, "board_height_offset_mm": 0, "height_calibration": None}
            if value is not None:
                settings["reference_pad_check_enabled"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reference_pad_check_enabled: false"):
                    self.run_export([self.collection(settings=settings)])

    def test_nonzero_tilt_rejected(self):
        m = self.measurement()
        m["sample"]["tilt_y_deg"] = 1
        with self.assertRaisesRegex(ValueError, "zero tilt"):
            self.run_export([self.collection(samples=[m])])

    def test_only_flat_reference_stimulus_accepted(self):
        for stimulus in (None, "flat_step_6mm", "sharp_edge", "broad_curvature", "small_feature"):
            m = self.measurement()
            m["sample"]["stimulus"] = stimulus
            with self.subTest(stimulus=stimulus):
                with self.assertRaisesRegex(ValueError, "requires flat_reference points"):
                    self.run_export([self.collection(samples=[m])])

    def test_missing_or_nonfinite_feedback_rejected(self):
        for bad in (None, [1, 2, 3], [1, 2, float("nan"), 180, 0, 0]):
            m = self.measurement()
            m["result"]["first_contact_bracket"]["contact_tcp"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.run_export([self.collection(samples=[m])])

    def test_lateral_error_rejected(self):
        m = self.measurement()
        m["result"]["first_contact_bracket"]["contact_tcp"][0] += 1
        with self.assertRaisesRegex(ValueError, "lateral error"):
            self.run_export([self.collection(samples=[m])])

    def test_default_lateral_limit_is_005_mm(self):
        m = self.measurement()
        m["result"]["first_contact_bracket"]["contact_tcp"][0] += .06
        path = self.collection(samples=[m])
        with self.assertRaisesRegex(ValueError, "exceeds 0.05 mm"):
            self.run_export([path])
        self.assertEqual(len(self.run_export([path], max_lateral_error_mm=.1)), 1)

    def test_bracket_order_and_zero_width_rejected(self):
        for delta in (0, -1):
            m = self.measurement()
            b = m["result"]["first_contact_bracket"]
            b["no_contact_tcp"][2] = b["contact_tcp"][2] + delta
            with self.subTest(delta=delta):
                with self.assertRaisesRegex(ValueError, "strictly above"):
                    self.run_export([self.collection(samples=[m])])

    def test_changed_point_location_rejected(self):
        first = self.collection()
        second = self.collection("run2.json", [self.measurement(local=(10.001, 20, 6))])
        with self.assertRaisesRegex(ValueError, "changed local coordinates"):
            self.run_export([first, second])

    def test_fit_validation_may_not_share_points(self):
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self.run_export([self.collection()], [self.collection("run2.json")])

    def test_duplicate_point_in_one_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeated point within"):
            self.run_export([self.collection(samples=[self.measurement(), self.measurement()])])

    def test_unsuccessful_sample_rejected(self):
        m = self.measurement()
        m["result"]["status"] = "no_contact"
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            self.run_export([self.collection(samples=[m])])

    def test_inverse_coordinate_convention_and_yaw(self):
        # Zero dock Euler, yaw +90: local +X -> base -Y, +Y -> base -X,
        # and local +Z -> base -Z. This checks handedness independently.
        profile = {**self.profile_data, "dock_tcp": [10, 20, 30, 0, 0, 0], "board_yaw_deg": 90}
        design = {"tactip_reference": {"nominal_seated_tool_tcp_local_mm": [1, 2, 3]}}
        transform = exporter.fixture_transform(profile, design)
        local = exporter.feedback_to_local([5, 16, 24, 0, 0, 0], transform, "test")
        for actual, expected in zip(local, [5, 7, 9]):
            self.assertAlmostEqual(actual, expected)

    def test_intrinsic_xyz_order(self):
        # Rx(90)*Ry(90) maps adapted local X -> base +Y, local Y ->
        # base -Z, local Z -> base -X; extrinsic XYZ would differ.
        profile = {**self.profile_data, "dock_tcp": [0, 0, 0, 90, 90, 0]}
        design = {"tactip_reference": {"nominal_seated_tool_tcp_local_mm": [0, 0, 0]}}
        local = exporter.feedback_to_local([-3, 1, -2, 90, 90, 0], exporter.fixture_transform(profile, design), "test")
        for actual, expected in zip(local, [1, 2, 3]):
            self.assertAlmostEqual(actual, expected)

    def test_rest_stop_datum_mismatch_rejected(self):
        design = {"tactip_reference": {"nominal_seated_tool_tcp_local_mm": [0, 0, 6]},
                  "rest_pose_stop": {"contact_centre_tile_local_mm": [0, 0, 6], "top_surface_z_mm": 6}}
        profile = {**self.profile_data, "height_calibration": {"rest_stop_contact_tile_local_mm": [0, 0, 7]}}
        with self.assertRaisesRegex(ValueError, "datum differs"):
            exporter.fixture_transform(profile, design)


if __name__ == "__main__":
    unittest.main()
