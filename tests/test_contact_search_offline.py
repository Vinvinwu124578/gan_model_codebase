"""Offline contact-search regressions; never imports a robot/camera driver.

Run with Python and NumPy installed:
    python test_contact_search_offline.py

Only function definitions are loaded from the sampler's syntax tree. Robot
movement, camera captures, baselines, and sleeps are replaced with fakes.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
from functools import lru_cache
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"


def load_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "math": math,
        "lru_cache": lru_cache,
        "Path": Path,
        "time": SimpleNamespace(sleep=lambda _: None, monotonic=time.monotonic, perf_counter=time.perf_counter),
    }
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), "exec"), namespace)
    namespace["finite_pose"] = lambda pose, label: tuple(float(value) for value in pose)
    namespace["list_pose"] = lambda pose: list(pose)
    return namespace


class SearchHarness:
    def __init__(self, surface_depth, *, clearance=1.0, margin=1.3, indentation=1.0,
                 confirmations=3, transient_depth=None, actual_depth_error=0.0):
        self.namespace = load_functions()
        self.depth = 0.0
        self.surface_depth = surface_depth
        self.clearance = clearance
        self.indentation = indentation
        self.transient_depth = transient_depth
        self.transient_emitted = False
        self.actual_depth_error = actual_depth_error
        self.moves = []
        self.frames = []
        self.events = []
        self.args = SimpleNamespace(
            max_extra_below_planned_contact_mm=indentation + margin,
            contact_search_margin_mm=margin,
            step_mm=0.5,
            settle_sec=0.0,
            probe_frames=3,
            capture_frames=9,
            save_search_frames=True,
            consecutive_hits=confirmations,
            motion_position_tolerance_mm=0.75,
            board_height_offset_mm=0.0,
        )
        self.namespace["prepare_site_baseline"] = lambda *args: (
            object(), 0.0, {"mean": 0.1, "p95": 0.4}, []
        )
        self.namespace["move_and_verify"] = self.move
        self.namespace["capture_record"] = self.capture
        self.namespace["discard_intermediate_capture"] = lambda *args: None

    def move(self, robot, label, target, args):
        commanded_depth = float(target[2])
        self.depth = commanded_depth + (0.0 if "final capture" in label else self.actual_depth_error)
        self.moves.append((label, commanded_depth))
        self.events.append(("move", commanded_depth))
        actual = list(target)
        actual[2] = self.depth
        return {"actual_tcp": actual, "target_tcp": list(target)}

    def capture(self, camera, args, output_dir, preprocessor, label, frame_count,
                camera_time, baseline=None):
        hit = self.depth >= self.surface_depth - 1e-9
        if (self.transient_depth is not None and not self.transient_emitted
                and abs(self.depth - self.transient_depth) < 1e-9):
            hit = True
            self.transient_emitted = True
        record = {"marker_motion": {"mean": 1.0 if hit else 0.0, "p95": 2.0 if hit else 0.0}}
        self.frames.append((label, self.depth, hit))
        self.events.append(("frame_hit" if hit else "frame_miss", self.depth))
        return record, object(), camera_time + 1.0

    def run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.namespace["search_visual_contact"](
                object(), object(), object(), self.args, SOURCE.parent, "offline_site",
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, self.clearance, 0.0, 0.0, 0.0),
                np.asarray((0.0, 0.0, 1.0)), self.indentation,
            )


class ContactSearchTests(unittest.TestCase):
    def test_no_contact_stops_exactly_at_nonintegral_limit(self):
        fake = SearchHarness(math.inf)
        result = fake.run()
        self.assertEqual(result["status"], "no_contact")
        self.assertAlmostEqual(fake.moves[-1][1], 2.3)
        self.assertTrue(all(depth <= 2.3 + 1e-9 for _, depth in fake.moves))

    def test_contact_at_last_endpoint_can_be_confirmed_three_times(self):
        fake = SearchHarness(2.3, confirmations=3)
        result = fake.run()
        self.assertEqual(result["status"], "captured")
        self.assertAlmostEqual(result["visual_contact_depth_from_approach_mm"], 2.3)
        self.assertAlmostEqual(result["capture_depth_from_approach_mm"], 3.3)
        self.assertGreaterEqual(sum(abs(depth - 2.3) < 1e-9 and hit
                                    for _, depth, hit in fake.frames), 3)
        self.assertTrue(all(depth <= 3.3 + 1e-9 for _, depth in fake.moves))

    def test_first_hit_confirmation_does_not_move_deeper(self):
        fake = SearchHarness(1.0, confirmations=3)
        result = fake.run()
        self.assertEqual(result["status"], "captured")
        first_hit = next(index for index, event in enumerate(fake.events) if event[0] == "frame_hit")
        first_events = fake.events[first_hit:first_hit + 3]
        self.assertEqual(first_events, [("frame_hit", 1.0)] * 3)

    def test_transient_first_hit_does_not_become_contact(self):
        fake = SearchHarness(2.3, transient_depth=1.0, confirmations=3)
        result = fake.run()
        self.assertEqual(result["status"], "captured")
        self.assertAlmostEqual(result["visual_contact_depth_from_approach_mm"], 2.3)
        self.assertIn(("frame_miss", 1.0), fake.events)

    def test_zero_indentation_localization_confirms_at_same_tcp(self):
        fake = SearchHarness(2.3, indentation=0.0, confirmations=3)
        result = fake.run()
        self.assertEqual(result["status"], "contact_found")
        self.assertAlmostEqual(fake.moves[-1][1], 2.3)
        self.assertAlmostEqual(result["visual_contact_tcp"][2], 2.3)

    def test_search_can_finish_before_first_full_step(self):
        fake = SearchHarness(math.inf, clearance=0.1, margin=0.2)
        result = fake.run()
        self.assertEqual(result["status"], "no_contact")
        self.assertEqual(len(fake.moves), 1)
        self.assertAlmostEqual(fake.moves[0][1], 0.3)

    def test_actual_first_contact_sets_capture_position(self):
        fake = SearchHarness(1.2, actual_depth_error=0.2)
        result = fake.run()
        self.assertEqual(result["status"], "captured")
        self.assertAlmostEqual(result["commanded_contact_depth_from_approach_mm"], 1.0)
        self.assertAlmostEqual(result["visual_contact_depth_from_approach_mm"], 1.2)
        self.assertAlmostEqual(fake.moves[-1][1], 2.2)

    def test_actual_contact_beyond_budget_rejects_capture(self):
        fake = SearchHarness(2.5, actual_depth_error=0.2)
        result = fake.run()
        self.assertEqual(result["status"], "capture_limit_reached")
        self.assertAlmostEqual(fake.moves[-1][1], 2.3)
        self.assertFalse(any("final capture" in label for label, _ in fake.moves))

    def test_no_contact_diagnostics_report_actual_endpoint(self):
        fake = SearchHarness(math.inf, actual_depth_error=-0.2)
        result = fake.run()
        self.assertEqual(result["status"], "no_contact")
        self.assertAlmostEqual(result["last_commanded_search_depth_mm"], 2.3)
        self.assertAlmostEqual(result["last_actual_depth_from_approach_mm"], 2.1)
        self.assertAlmostEqual(result["last_actual_relative_to_nominal_mm"], 1.1)


class BoundedSearchDepthTests(unittest.TestCase):
    def test_limits_are_included_without_overshoot_or_large_steps(self):
        bounded = load_functions()["bounded_search_depths"]
        for limit in (0.01, 0.3, 0.5, 0.75, 1.0, 2.3, 25.001, 35.0):
            for step in (0.1, 0.3, 0.5):
                with self.subTest(limit=limit, step=step):
                    depths = bounded(limit, step)
                    self.assertEqual(depths[-1], limit)
                    self.assertTrue(all(0.0 < value <= limit for value in depths))
                    differences = np.diff([0.0, *depths])
                    self.assertTrue(np.all(differences > 0.0))
                    self.assertTrue(np.all(differences <= step + 1e-9))

    def test_invalid_limits_and_steps_are_rejected(self):
        bounded = load_functions()["bounded_search_depths"]
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    bounded(bad, 0.5)
                with self.assertRaises(ValueError):
                    bounded(1.0, bad)


class BoardHeightTests(unittest.TestCase):
    def route(self, *, offset=-3.0, correction=(0.0, 0.0, 0.0), safe_height=100.0, rotation=None):
        namespace = load_functions()
        rotation = np.eye(3) if rotation is None else rotation
        fixture = SimpleNamespace(
            dock_tcp_local_mm=(0.0, 0.0, 0.0),
            tile_to_base=rotation,
            pose=lambda xyz, *tilt: (*map(float, rotation @ np.asarray(xyz)), 180.0, 0.0, 0.0),
            press_axis=lambda *tilt: -rotation[:, 2],
            local_vector=lambda vector: rotation.T @ np.asarray(vector),
        )
        sample = SimpleNamespace(local_contact_mm=(20.0, 30.0, 5.0), tilt_x_deg=0.0, tilt_y_deg=0.0)
        args = SimpleNamespace(
            approach_clearance_mm=2.0, dock_exit_lift_mm=65.0,
            safe_height_mm=safe_height, board_height_offset_mm=offset,
        )
        return namespace["make_route"](sample, fixture, args, correction)

    def test_height_offset_moves_contact_and_approach_but_keeps_high(self):
        route = self.route()
        self.assertAlmostEqual(route["contact_tcp"][2], 2.0)
        self.assertAlmostEqual(route["approach_tcp"][2], 4.0)
        self.assertAlmostEqual(route["site_high_tcp"][2], 100.0)

    def test_negative_reference_correction_does_not_lower_high(self):
        route = self.route(correction=(0.0, 0.0, -2.0))
        self.assertAlmostEqual(route["contact_tcp"][2], 0.0)
        self.assertAlmostEqual(route["approach_tcp"][2], 2.0)
        self.assertAlmostEqual(route["site_high_tcp"][2], 100.0)

    def test_high_below_corrected_approach_is_rejected(self):
        with self.assertRaises(ValueError):
            self.route(offset=10.0, safe_height=10.0)

    def test_height_offset_follows_board_normal(self):
        cosine, sine = math.cos(0.4), math.sin(0.4)
        rotation = np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))
        base = self.route(offset=0.0, rotation=rotation)
        shifted = self.route(offset=-3.0, rotation=rotation)
        np.testing.assert_allclose(
            np.asarray(shifted["contact_tcp"][:3]) - np.asarray(base["contact_tcp"][:3]),
            -3.0 * rotation[:, 2],
        )
        np.testing.assert_allclose(shifted["site_high_tcp"], base["site_high_tcp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
