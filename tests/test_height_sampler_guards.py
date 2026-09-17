"""Exercise real argument, dock and route guards without importing robot modules."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BOARD = ROOT / "outputs" / "tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DOCK = BOARD / "tactip_calibration_dock_lightweight_v1" / "v4_150mm_tactip_calibration_dock_lightweight_design.json"
if os.name == "nt":
    # The repository checkout plus the published dock filename exceeds the
    # classic Windows 260-character path limit; exercise guards, not that limit.
    BOARD = Path("\\\\?\\" + str(BOARD))
    DOCK = Path("\\\\?\\" + str(DOCK))
SAMPLER = TOOLS / "auto_cr3_coverage_board_sampler.py"


def guard_namespace():
    tree = ast.parse(SAMPLER.read_text(encoding="utf-8"))
    names = {
        "parse_args",
        "base_to_tile_local",
        "base_to_fixed_fixture_local",
        "board_surface_correction_mm",
        "make_route",
        "load_dock_design",
        "rest_stop_contract",
        "dock_alignment_decision",
        "low_pose_recovery_route",
    }
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    shared = ast.parse((TOOLS / "auto_cr3_visual_contact_search.py").read_text(encoding="utf-8"))
    functions.extend(node for node in shared.body if isinstance(node, ast.FunctionDef) and node.name == "finite_pose")
    def add_preprocess_args(parser):
        parser.add_argument("--no-tactip-preprocess", action="store_true")
    def list_pose(pose):
        return [float(value) for value in pose]

    def position_error_mm(expected, actual):
        return float(np.linalg.norm(np.asarray(expected[:3], dtype=float) - np.asarray(actual[:3], dtype=float)))

    def rotation_error_deg(expected, actual):
        return float(np.linalg.norm(np.asarray(expected[3:], dtype=float) - np.asarray(actual[3:], dtype=float)))

    namespace = {
        "argparse": argparse,
        "Path": Path,
        "math": math,
        "np": np,
        "__doc__": "Isolated sampler guard tests",
        "add_tactip_preprocess_args": add_preprocess_args,
        "list_pose": list_pose,
        "position_error_mm": position_error_mm,
        "rotation_error_deg": rotation_error_deg,
    }
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *constants, *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<isolated sampler guards>", "exec"), namespace)
    return namespace


class HeightCliGuardsTests(unittest.TestCase):
    def setUp(self):
        self.namespace = guard_namespace()
        self.assertTrue(BOARD.is_dir())
        self.assertTrue(DOCK.is_file())
        self.base = ["sampler", "--tile", "tile_nw", "--board-dir", str(BOARD), "--dock-design", str(DOCK)]
        self.measurement = ["--height-measurement", "--first-contact-test", "--max-samples", "1", "--site", "R01_S05"]

    def parse(self, extra):
        with mock.patch.object(sys, "argv", self.base + extra), contextlib.redirect_stderr(io.StringIO()):
            return self.namespace["parse_args"]()

    def rejects(self, extra):
        with self.assertRaises(SystemExit) as caught:
            self.parse(extra)
        self.assertEqual(caught.exception.code, 2)

    def test_measurement_requires_no_motion_confirmation_for_offline_plan(self):
        args = self.parse(self.measurement)
        self.assertFalse(args.execute)
        self.assertTrue(args.save_search_frames)
        self.assertLessEqual(args.motion_position_tolerance_mm, 0.1)
        self.assertLessEqual(args.motion_rotation_tolerance_deg, 0.2)
        self.assertGreaterEqual(args.consecutive_hits, 3)

    def test_execute_cannot_bypass_confirmation(self):
        self.rejects(self.measurement + ["--execute"])

    def test_full_route_preflight_is_opt_in_for_execute_runs(self):
        default_args = self.parse([])
        self.assertFalse(default_args.preflight_all_routes)
        self.assertFalse(default_args.disable_auto_dock_reseat)
        self.assertTrue(default_args.return_to_dock)
        stay_high_args = self.parse(["--leave-at-site-high"])
        self.assertFalse(stay_high_args.return_to_dock)
        explicit_args = self.parse(["--execute", "--yes-i-confirm-cr3-is-safe", "--preflight-all-routes"])
        self.assertTrue(explicit_args.preflight_all_routes)

    def test_full_route_preflight_requires_motion_mode_and_immediate_ik(self):
        self.rejects(["--preflight-all-routes"])
        self.rejects([
            "--execute", "--yes-i-confirm-cr3-is-safe", "--preflight-all-routes", "--disable-ik-preflight",
        ])

    def test_measurement_requires_exactly_one_repeatable_site(self):
        for extra in (["--height-measurement"],
                      ["--height-measurement", "--first-contact-test", "--max-samples", "1"],
                      self.measurement + ["--site", "R01_S06"],
                      self.measurement + ["--profile", "dense"],
                      self.measurement + ["--max-samples", "2"]):
            with self.subTest(extra=extra):
                self.rejects(extra)

    def test_measurement_rejects_existing_height_correction(self):
        for extra in (["--height-calibration", "unused.json"], ["--runtime-height-datum", "unused.json"], ["--board-height-offset-mm", "0.2"], ["--use-reference-pad-check"]):
            with self.subTest(extra=extra):
                self.rejects(self.measurement + extra)

    def test_runtime_height_calibration_owns_the_fixed_multi_contact_protocol(self):
        args = self.parse(["--calibrate-runtime-height", "--execute", "--yes-i-confirm-cr3-is-safe"])
        self.assertTrue(args.height_measurement)
        self.assertTrue(args.zero_tilt)
        self.assertTrue(args.continuous_board_transit)
        self.assertTrue(args.return_to_dock)
        self.assertEqual(args.max_extra_below_planned_contact_mm, 4.0)
        self.assertEqual(args.contact_search_margin_mm, 4.0)
        self.assertIsNotNone(args.runtime_height_datum)

    def test_runtime_height_calibration_rejects_manual_plan_selection_and_unsafe_bypass(self):
        for extra in (
            ["--calibrate-runtime-height", "--site", "R01_S05", "--execute", "--yes-i-confirm-cr3-is-safe"],
            ["--calibrate-runtime-height", "--skip-dock-tactile-reference-check", "--execute", "--yes-i-confirm-cr3-is-safe"],
            ["--calibrate-runtime-height"],
        ):
            with self.subTest(extra=extra):
                self.rejects(extra)

    def test_measurement_rejects_other_operating_modes(self):
        for flag in ("--teach-dock-from-current", "--calibrate-height-from-rest-stop", "--capture-camera-only",
                     "--fixture-local-preview", "--reseat-dock-only", "--recover-reference-to-dock",
                     "--recover-low-pose-to-dock", "--verify-dock-tactile-reference-only"):
            with self.subTest(flag=flag):
                self.rejects(self.measurement + [flag])

    def test_geometry_preview_rejects_every_hardware_entry(self):
        for flag in ("--execute", "--ik-preflight-only", "--teach-dock-from-current", "--calibrate-height-from-rest-stop",
                     "--capture-camera-only", "--verify-dock-tactile-reference-only", "--reseat-dock-only",
                     "--recover-reference-to-dock", "--recover-low-pose-to-dock"):
            with self.subTest(flag=flag):
                self.rejects(["--fixture-local-preview", flag, "--yes-i-confirm-cr3-is-safe"])

    def test_calibration_rejects_offset_and_reference_correction_stacking(self):
        for extra in (["--board-height-offset-mm", "-0.1"], ["--use-reference-pad-check"], ["--teach-dock-from-current"]):
            with self.subTest(extra=extra):
                self.rejects(["--height-calibration", "unused.json", *extra])

    def test_invalid_measurement_motion_tolerance_is_not_silently_accepted(self):
        for value in ("0", "-1", "nan"):
            with self.subTest(value=value):
                self.rejects(self.measurement + ["--motion-position-tolerance-mm", value])


class DockPreviewGuardsTests(unittest.TestCase):
    def setUp(self):
        self.namespace = guard_namespace()
        self.design = {"schema": "coverage_board_dock_geometry_preview.v1", "geometry_preview_only": True,
                       "tactip_reference": {"tool": 2, "nominal_seated_tool_tcp_local_mm": [0, -139, 27]},
                       "rest_pose_stop": {"contact_centre_tile_local_mm": [0, -139, 27], "top_surface_z_mm": 27}}
        self.namespace["read_json"] = lambda path: self.design

    def test_derived_design_is_accepted_only_for_fixture_local_preview(self):
        args = SimpleNamespace(dock_design=Path("preview.json"), tool=2, fixture_local_preview=True)
        self.assertIs(self.namespace["load_dock_design"](args), self.design)
        args.fixture_local_preview = False
        with self.assertRaisesRegex(ValueError, "geometry-preview only"):
            self.namespace["load_dock_design"](args)

    def test_legacy_schema_cannot_hide_preview_only_flag(self):
        self.design["schema"] = "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2"
        with self.assertRaisesRegex(ValueError, "geometry-preview only"):
            self.namespace["load_dock_design"](SimpleNamespace(dock_design=Path("preview.json"), tool=2, fixture_local_preview=False))

    def test_preview_schema_cannot_hide_by_removing_flag(self):
        del self.design["geometry_preview_only"]
        with self.assertRaisesRegex(ValueError, "geometry-preview only"):
            self.namespace["load_dock_design"](SimpleNamespace(dock_design=Path("preview.json"), tool=2, fixture_local_preview=False))

    def test_derived_crossbar_never_becomes_a_physical_height_datum(self):
        for flag, schema in ((True, "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2"),
                             (False, "coverage_board_dock_geometry_preview.v1")):
            with self.subTest(flag=flag, schema=schema):
                self.design.update({"geometry_preview_only": flag, "schema": schema})
                with self.assertRaisesRegex(ValueError, "physical height datum"):
                    self.namespace["rest_stop_contract"](self.design)

    def test_real_rest_stop_still_requires_consistent_contact_height(self):
        self.design.update({"geometry_preview_only": False, "schema": "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2"})
        self.assertEqual(self.namespace["rest_stop_contract"](self.design)["top_surface_z_mm"], 27)
        self.design["rest_pose_stop"]["top_surface_z_mm"] = 28
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.namespace["rest_stop_contract"](self.design)


class LocalFixture:
    """Known local identity map and downward press axis, with no hardware APIs."""
    dock_tcp = (0.0, -139.0, 12.0, 180.0, 0.0, 0.0)
    dock_tcp_local_mm = (0.0, -139.0, 12.0)
    tile_to_base = np.eye(3)

    def pose(self, xyz, tilt_x=0, tilt_y=0):
        return (*xyz, 180.0 + tilt_x, tilt_y, 0.0)

    def press_axis(self, tilt_x, tilt_y):
        return np.array([0.0, 0.0, -1.0])

    def local_vector(self, base_vector_mm):
        return np.asarray(base_vector_mm, dtype=float)


class DockAutoReseatDecisionTests(unittest.TestCase):
    def setUp(self):
        self.namespace = guard_namespace()
        self.fixture = LocalFixture()
        self.args = SimpleNamespace(
            dock_position_tolerance_mm=1.5,
            dock_rotation_tolerance_deg=2.0,
        )

    def decision(self, pose):
        return self.namespace["dock_alignment_decision"](self.fixture, pose, self.args)

    def test_aligned_tactip_above_crossbar_is_safe_for_low_speed_reseat(self):
        decision = self.decision((0.0, -139.0, 15.2085, 180.0, 0.0, 0.0))
        self.assertEqual(decision["status"], "aligned_above_dock")
        self.assertAlmostEqual(decision["above_dock_mm"], 3.2085)
        self.assertAlmostEqual(decision["lateral_error_mm"], 0.0)

    def test_small_repeatable_lateral_offset_above_crossbar_is_auto_reseated(self):
        decision = self.decision((0.30, -139.0, 15.2085, 180.0, 0.0, 0.0))
        self.assertEqual(decision["status"], "aligned_above_dock")
        self.assertLessEqual(decision["lateral_error_mm"], 0.50)

    def test_large_lateral_offset_above_crossbar_is_never_auto_reseated(self):
        decision = self.decision((0.51, -139.0, 15.2085, 180.0, 0.0, 0.0))
        self.assertEqual(decision["status"], "unsafe_start_pose")
        self.assertGreater(decision["lateral_error_mm"], 0.50)

    def test_current_fixture_repeatability_envelope_is_auto_reseated(self):
        decision = self.decision((0.285, -139.0, 13.823, 180.382, 0.0, 0.0))
        self.assertEqual(decision["status"], "aligned_above_dock")

    def test_saved_dock_datum_is_already_seated(self):
        decision = self.decision(self.fixture.dock_tcp)
        self.assertEqual(decision["status"], "already_seated")


class LowPoseRecoveryRouteTests(unittest.TestCase):
    def setUp(self):
        self.namespace = guard_namespace()
        self.fixture = LocalFixture()
        self.args = SimpleNamespace(
            safe_height_mm=100.0,
            dock_exit_lift_mm=80.0,
            motion_position_tolerance_mm=0.75,
        )

    def route(self, current_pose=(40.0, -80.0, 15.0, 180.0, 0.0, 0.0)):
        return self.namespace["low_pose_recovery_route"](
            self.fixture,
            current_pose,
            self.args,
        )

    def test_recovery_lifts_without_lateral_motion_before_returning_to_dock(self):
        plan, route = self.route()
        labels = [label for label, _pose in route]
        self.assertEqual(
            labels,
            ["vertical_retract_to_safe_height", "dock_high", "dock_exit", "reseat_dock"],
        )
        first_target = route[0][1]
        self.assertEqual(first_target[:2], (40.0, -80.0))
        self.assertEqual(first_target[2], 100.0)
        self.assertEqual(route[-1][1], self.fixture.dock_tcp)
        self.assertEqual(plan["route"][0]["target_tile_local_mm"], [40.0, -80.0, 100.0])

    def test_recovery_refuses_unknown_or_misoriented_start_pose(self):
        unsafe_poses = (
            (0.0, -139.0, -0.01, 180.0, 0.0, 0.0),
            (200.0, -139.0, 15.0, 180.0, 0.0, 0.0),
            (0.0, -139.0, 15.0, 180.0, 0.0, 2.01),
        )
        for pose in unsafe_poses:
            with self.subTest(pose=pose), self.assertRaises(RuntimeError):
                self.route(pose)


class CalibratedRouteGuardsTests(unittest.TestCase):
    def setUp(self):
        self.namespace = guard_namespace()
        spec = importlib.util.spec_from_file_location("board_height_calibration", TOOLS / "board_height_calibration.py")
        self.calibration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.calibration_module)
        patcher = mock.patch.dict(sys.modules, {"board_height_calibration": self.calibration_module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calibration = {"schema": self.calibration_module.SCHEMA, "status": "accepted",
                            "model_kind": self.calibration_module.MODEL_KIND, "supported_tilt_deg": 0.0,
                            "coefficients_mm": [0.01, -0.005, 1.0],
                            "support_hull_xy_mm": [[-50, -50], [50, -50], [50, 50], [-50, 50]],
                            "limits": {"max_correction_mm": 10.0}}
        self.args = SimpleNamespace(_height_calibration=self.calibration, board_height_offset_mm=0.0,
                                    approach_clearance_mm=5.0, dock_exit_lift_mm=30.0, safe_height_mm=100.0)
        self.sample = SimpleNamespace(local_contact_mm=(10.0, 20.0, 10.0), tilt_x_deg=0.0, tilt_y_deg=0.0)
        self.fixture = LocalFixture()

    def route(self):
        return self.namespace["make_route"](self.sample, self.fixture, self.args)

    def test_real_height_evaluator_corrects_contact_and_preserves_safe_transit(self):
        result = self.route()
        self.assertAlmostEqual(result["contact_tcp"][2], 11.0)
        self.assertAlmostEqual(result["approach_tcp"][2], 16.0)
        self.assertAlmostEqual(result["calibrated_height_correction_mm"], 1.0)
        self.assertEqual(result["site_high_tcp"][2], 100.0)
        self.assertEqual(dict(result["outbound"])["dock_high"][2], 100.0)

    def test_negative_correction_never_lowers_safe_transit(self):
        self.calibration["coefficients_mm"] = [0.0, 0.0, -3.0]
        result = self.route()
        self.assertEqual(result["contact_tcp"][2], 7.0)
        self.assertEqual(result["site_high_tcp"][2], 100.0)

    def test_runtime_global_datum_applies_to_tilted_routes_without_plane_extrapolation(self):
        self.args._height_calibration = None
        self.args._runtime_height_datum = {"offset_tile_z_mm": -1.25}
        self.sample.tilt_x_deg = 3.0
        result = self.route()
        self.assertAlmostEqual(result["contact_tcp"][2], 8.75)
        self.assertAlmostEqual(result["runtime_height_correction_mm"], -1.25)

    def test_zero_tilt_calibration_cannot_authorize_tilted_route(self):
        self.sample.tilt_x_deg = 0.5
        with self.assertRaisesRegex(ValueError, "zero tilt"):
            self.route()

    def test_calibrated_route_cannot_extrapolate_outside_measured_hull(self):
        self.sample.local_contact_mm = (50.001, 0.0, 10.0)
        with self.assertRaisesRegex(ValueError, "outside measured"):
            self.route()

    def test_safe_height_must_exceed_corrected_approach(self):
        self.args.safe_height_mm = 15.0
        with self.assertRaisesRegex(ValueError, "Safe height"):
            self.route()


if __name__ == "__main__":
    unittest.main()
