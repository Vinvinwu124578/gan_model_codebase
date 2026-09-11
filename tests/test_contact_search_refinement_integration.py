"""Exercise coarse search -> refinement together with in-memory device doubles."""
from __future__ import annotations

import argparse
import unittest
from unittest.mock import Mock

import numpy as np

from test_contact_search_offline import SearchHarness


def measurement_harness(surface_depth=.673):
    harness = SearchHarness(surface_depth, indentation=0.0)
    harness.args.height_measurement = True
    harness.args.motion_rotation_tolerance_deg = .5
    harness.namespace["argparse"] = argparse
    harness.namespace["position_error_mm"] = lambda a, b: float(np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3])))
    harness.namespace["rotation_error_deg"] = lambda a, b: float(np.linalg.norm(np.asarray(a[3:]) - np.asarray(b[3:])))
    return harness


class SearchRefinementIntegrationTests(unittest.TestCase):
    def assert_no_usable_contact(self, result):
        self.assertEqual(result["status"], "measurement_unusable")
        self.assertNotIn("visual_contact_tcp", result)
        self.assertNotIn("actual_capture_tcp", result)
        self.assertIsNone(result.get("first_contact_bracket"))

    def test_actual_refined_endpoint_replaces_coarse_and_commanded_contact(self):
        harness = measurement_harness()
        original_move = harness.move

        def move_with_fine_feedback_error(robot, label, target, args):
            motion = original_move(robot, label, target, args)
            if "refine_" in label:
                self.assertLessEqual(args.motion_position_tolerance_mm, .01)
                self.assertLessEqual(args.motion_rotation_tolerance_deg, .1)
            if "refine_step_" in label:
                harness.depth = float(target[2]) + .004
                motion["actual_tcp"][2] = harness.depth
            return motion

        harness.namespace["move_and_verify"] = move_with_fine_feedback_error
        result = harness.run()
        self.assertEqual(result["status"], "contact_found")
        self.assertAlmostEqual(result["coarse_contact_depth_from_approach_mm"], 1.0)
        self.assertAlmostEqual(result["commanded_contact_depth_from_approach_mm"], .68)
        self.assertAlmostEqual(result["visual_contact_depth_from_approach_mm"], .684)
        self.assertAlmostEqual(result["capture_depth_from_approach_mm"], .684)
        self.assertAlmostEqual(result["visual_contact_tcp"][2], .684)
        self.assertAlmostEqual(result["actual_capture_tcp"][2], .684)
        self.assertAlmostEqual(result["first_contact_bracket"]["no_contact_tcp"][2], .664)
        self.assertAlmostEqual(result["first_contact_bracket"]["contact_tcp"][2], .684)
        self.assertEqual(result["contact_refinement"]["status"], "refined")
        self.assertEqual(len(result["motion"]), len(harness.moves))
        self.assertEqual(len(result["frames"]), len(harness.frames))
        self.assertFalse(any("final capture" in label for label, _ in harness.moves))
        self.assertEqual(harness.args.motion_position_tolerance_mm, .75)
        self.assertEqual(harness.args.motion_rotation_tolerance_deg, .5)

    def test_unreleased_skin_returns_unusable_without_coarse_contact_fallback(self):
        harness = measurement_harness()
        original_capture = harness.capture

        def sticky_release(*args, **kwargs):
            frame, feature, camera_time = original_capture(*args, **kwargs)
            if "_refine_release_" in args[4]:
                frame["marker_motion"] = {"mean": 1.0, "p95": 2.0}
            return frame, feature, camera_time

        harness.namespace["capture_record"] = sticky_release
        result = harness.run()
        self.assert_no_usable_contact(result)
        self.assertIn("did not release", result["reason"])
        self.assertEqual(result["contact_refinement"]["status"], "unusable")
        self.assertIsNone(result["contact_refinement"]["contact_tcp"])
        self.assertTrue(result["contact_refinement"]["frames"])
        self.assertIn("refine_release", harness.moves[-1][0])
        self.assertAlmostEqual(harness.moves[-1][1], .5)

    def test_contact_without_observed_unloaded_endpoint_is_not_refined(self):
        harness = measurement_harness(surface_depth=.1)
        refinement = Mock(side_effect=AssertionError("No bracket may be invented from the baseline"))
        harness.namespace["refine_visual_contact_bracket"] = refinement
        result = harness.run()
        self.assert_no_usable_contact(result)
        self.assertIn("No observed unloaded endpoint", result["reason"])
        refinement.assert_not_called()
        self.assertEqual(len(harness.moves), 1)

    def test_refinement_camera_exception_retains_nested_search_diagnostics(self):
        harness = measurement_harness()
        original_capture = harness.capture
        failure = RuntimeError("synthetic fine camera failure")

        def failed_fine_capture(*args, **kwargs):
            if "_refine_probe_" in args[4]:
                raise failure
            return original_capture(*args, **kwargs)

        harness.namespace["capture_record"] = failed_fine_capture
        with self.assertRaises(RuntimeError) as caught:
            harness.run()
        self.assertIs(caught.exception, failure)
        record = caught.exception.contact_search_record
        self.assertEqual(record["status"], "failed")
        self.assertIs(record["contact_refinement"], caught.exception.refinement_record)
        self.assertTrue(record["frames"])
        self.assertTrue(record["motion"])
        refinement = record["contact_refinement"]
        self.assertIn("synthetic fine camera failure", refinement["reason"])
        self.assertEqual(refinement["frames"][-1]["status"], "failed")
        self.assertIn("refine_probe", refinement["frames"][-1]["refinement_stage"])
        self.assertAlmostEqual(refinement["motion"][-1]["actual_tcp"][2], .52)
        self.assertAlmostEqual(harness.moves[-1][1], .52)
        self.assertIsNone(record.get("first_contact_bracket"))
        self.assertNotIn("visual_contact_tcp", record)

    def test_normal_sampling_keeps_coarse_contact_and_requested_indentation(self):
        harness = SearchHarness(.673, indentation=1.0)
        harness.args.height_measurement = False
        refinement = Mock(side_effect=AssertionError("Normal collection must not refine"))
        harness.namespace["refine_visual_contact_bracket"] = refinement
        result = harness.run()
        refinement.assert_not_called()
        self.assertEqual(result["status"], "captured")
        self.assertAlmostEqual(result["visual_contact_tcp"][2], 1.0)
        self.assertAlmostEqual(result["actual_capture_tcp"][2], 2.0)
        self.assertAlmostEqual(result["capture_depth_from_approach_mm"], 2.0)
        self.assertNotIn("contact_refinement", result)
        self.assertTrue("final capture" in harness.moves[-1][0])
        self.assertEqual(harness.moves[-1][1], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
