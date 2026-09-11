"""Run contact-bracket logic with in-memory motion/camera doubles only."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SAMPLER = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"


def isolated_namespace():
    tree = ast.parse(SAMPLER.read_text(encoding="utf-8"))
    names = {"bounded_search_depths", "refine_visual_contact_bracket"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    shared = ast.parse((SAMPLER.parent / "auto_cr3_visual_contact_search.py").read_text(encoding="utf-8"))
    selected.extend(node for node in shared.body if isinstance(node, ast.FunctionDef) and node.name in {"finite_pose", "list_pose", "position_error_mm"})
    namespace = {
        "argparse": argparse, "np": np, "math": math,
        "time": SimpleNamespace(sleep=lambda seconds: None),
        "rotation_error_deg": lambda a, b: float(np.linalg.norm(np.asarray(a[3:]) - np.asarray(b[3:]))),
    }
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<isolated contact refinement>", "exec"), namespace)
    return namespace


class ContactRefinementTests(unittest.TestCase):
    def setUp(self):
        self.namespace = isolated_namespace()
        self.args = argparse.Namespace(motion_position_tolerance_mm=0.8,
                                       motion_rotation_tolerance_deg=0.5,
                                       settle_sec=0.0, probe_frames=2, consecutive_hits=2)
        self.baseline = object()
        self.approach = (0.0,) * 6
        self.lower = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
        self.upper = (0.0, 0.0, 10.5, 0.0, 0.0, 0.0)
        self.axis = np.array([0.0, 0.0, 1.0])
        self.commands = []
        self.captures = []
        self.current = self.lower
        self.contact_depth = 10.17
        self.frame_values = None
        self.move_override = None
        self.capture_error = None
        self.namespace["move_and_verify"] = self.fake_move
        self.namespace["capture_record"] = self.fake_capture

    def fake_move(self, robot, label, target, args):
        self.assertIsNot(args, self.args)
        self.assertLessEqual(args.motion_position_tolerance_mm, 0.01)
        self.commands.append(tuple(target))
        actual = self.move_override(target, len(self.commands)) if self.move_override else tuple(target)
        self.current = tuple(actual)
        return {"label": label, "target_tcp": list(target), "actual_tcp": list(actual)}

    def fake_capture(self, camera, args, output, preprocessor, label, frames, previous_time, baseline):
        self.assertIs(baseline, self.baseline)
        self.assertGreaterEqual(frames, 2)
        self.captures.append((label, tuple(self.current), previous_time))
        if self.capture_error:
            raise self.capture_error
        value = self.frame_values.pop(0) if self.frame_values is not None else (2.0 if self.current[2] >= self.contact_depth else 0.0)
        record = {"label": label, "marker_motion": {"mean": value, "p95": value}}
        return record, object(), previous_time + 1.0

    def run_refinement(self):
        return self.namespace["refine_visual_contact_bracket"](
            object(), object(), object(), self.args, Path("unused"), "sample_001",
            self.approach, self.axis, self.baseline, {"mean": 1.0, "p95": 1.0},
            self.lower, self.upper, 0.0,
        )

    def test_refines_only_explored_bracket_and_confirms_stationary(self):
        result = self.run_refinement()
        self.assertEqual(result["status"], "refined")
        self.assertAlmostEqual(result["no_contact_tcp"][2], 10.16)
        self.assertAlmostEqual(result["contact_tcp"][2], 10.18)
        self.assertAlmostEqual(result["actual_bracket_width_mm"], 0.02)
        self.assertEqual(self.commands[0], self.lower)
        self.assertEqual(self.captures[0][1], self.captures[1][1])
        self.assertEqual(self.captures[-1][1], self.captures[-2][1])
        self.assertTrue(all(self.lower[2] <= target[2] <= self.upper[2] for target in self.commands))
        self.assertTrue(all(b[2] - a[2] <= 0.02000001 for a, b in zip(self.commands, self.commands[1:])))
        self.assertEqual(self.args.motion_position_tolerance_mm, 0.8)
        self.assertEqual(len(result["motion"]), len(self.commands))
        self.assertEqual(len(result["frames"]), len(self.captures))

    def test_exact_endpoint_is_visited_without_rounding_overshoot(self):
        self.upper = (0.0, 0.0, 10.055, 0.0, 0.0, 0.0)
        self.contact_depth = self.upper[2]
        result = self.run_refinement()
        self.assertEqual(result["status"], "refined")
        self.assertAlmostEqual(self.commands[-1][2], 10.055)
        self.assertAlmostEqual(result["actual_bracket_width_mm"], 0.015)
        self.assertLessEqual(max(target[2] for target in self.commands), self.upper[2])

    def test_release_requires_two_nonhit_observations(self):
        self.frame_values = [0.0, 2.0]
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIn("did not release", result["reason"])
        self.assertIsNone(result["contact_tcp"])
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(len(result["frames"]), 2)

    def test_first_release_hit_stops_immediately(self):
        self.contact_depth = 0.0
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(len(self.captures), 1)

    def test_unstable_contact_does_not_continue_descending(self):
        self.frame_values = [0.0, 0.0, 2.0, 0.0]
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIn("not confirmed", result["reason"])
        self.assertIsNone(result["contact_tcp"])
        self.assertEqual(len(self.commands), 2)

    def test_no_hit_never_guesses_a_contact_or_extends_coarse_bound(self):
        self.contact_depth = 100.0
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIsNone(result["contact_tcp"])
        self.assertAlmostEqual(self.commands[-1][2], self.upper[2])
        self.assertLessEqual(max(target[2] for target in self.commands), self.upper[2])
        self.assertEqual(len(self.commands), 26)

    def test_invalid_coarse_geometry_sends_no_commands(self):
        variants = [
            (np.zeros(3), self.lower, self.upper),
            (self.axis, self.upper, self.lower),
            (self.axis, self.lower, (0, 0, 12, 0, 0, 0)),
            (self.axis, (0.1, 0, 10, 0, 0, 0), self.upper),
            (self.axis, (0, 0, 10, 1, 0, 0), self.upper),
        ]
        for axis, lower, upper in variants:
            with self.subTest(axis=axis, lower=lower, upper=upper):
                self.axis, self.lower, self.upper = axis, lower, upper
                result = self.run_refinement()
                self.assertEqual(result["status"], "unusable")
                self.assertEqual(self.commands, [])

    def test_feedback_outside_contact_end_is_rejected_before_more_capture(self):
        self.contact_depth = 100.0
        self.move_override = lambda target, count: (*target[:2], target[2] + 0.005, *target[3:]) if target[2] >= self.upper[2] else target
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIn("previously explored", result["reason"])
        self.assertIsNone(result["contact_tcp"])
        self.assertLess(self.captures[-1][1][2], self.upper[2])

    def test_nonadvancing_actual_motion_is_rejected(self):
        self.args.contact_refine_step_mm = 0.005
        self.move_override = lambda target, count: self.lower
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIn("did not advance", result["reason"])
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(len(self.captures), 2)

    def test_actual_position_error_over_0_01_stops(self):
        self.move_override = lambda target, count: (*target[:2], target[2] + 0.011, *target[3:])
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertIn("tightened motion tolerance", result["reason"])
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(len(self.captures), 0)

    def test_actual_bracket_must_satisfy_requested_stricter_limit(self):
        self.args.contact_refine_max_bracket_mm = 0.015
        self.contact_depth = 10.01
        result = self.run_refinement()
        self.assertEqual(result["status"], "unusable")
        self.assertAlmostEqual(result["actual_bracket_width_mm"], 0.02)
        self.assertIsNone(result["contact_tcp"])

    def test_requested_coarser_resolution_cannot_relax_hard_caps(self):
        self.args.contact_refine_step_mm = 2.0
        self.args.contact_refine_max_bracket_mm = 1.0
        result = self.run_refinement()
        self.assertEqual(result["fine_step_mm"], 0.02)
        self.assertEqual(result["max_bracket_mm"], 0.04)
        self.assertEqual(result["status"], "refined")

    def test_motion_error_propagates_with_audit_and_no_recovery(self):
        def fail_move(robot, label, target, args):
            self.commands.append(tuple(target))
            raise RuntimeError("feedback lost")
        self.namespace["move_and_verify"] = fail_move
        with self.assertRaisesRegex(RuntimeError, "feedback lost") as caught:
            self.run_refinement()
        record = caught.exception.refinement_record
        self.assertEqual(record["motion"][0]["status"], "failed")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.captures, [])

    def test_capture_error_propagates_with_audit_and_no_more_motion(self):
        self.capture_error = RuntimeError("camera lost")
        with self.assertRaisesRegex(RuntimeError, "camera lost") as caught:
            self.run_refinement()
        self.assertEqual(caught.exception.refinement_record["frames"][0]["status"], "failed")
        self.assertEqual(len(self.commands), 1)

    def test_nonfinite_optical_flow_is_not_accepted_as_release(self):
        self.frame_values = [float("nan")]
        with self.assertRaisesRegex(RuntimeError, "invalid marker-motion") as caught:
            self.run_refinement()
        self.assertEqual(len(caught.exception.refinement_record["frames"]), 1)
        self.assertEqual(len(self.commands), 1)


if __name__ == "__main__":
    unittest.main()
