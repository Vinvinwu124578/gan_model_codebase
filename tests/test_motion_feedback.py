"""Offline regressions for real TCP feedback; no client, socket or camera is opened."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def extracted_functions():
    """Load only the functions under test, avoiding GUI/camera module imports."""
    live_tree = ast.parse((TOOLS / "live_cr3_gelsight_sampler.py").read_text(encoding="utf-8"))
    exception_class = next(node for node in live_tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionFeedbackError")
    client_class = next(node for node in live_tree.body if isinstance(node, ast.ClassDef) and node.name == "DobotCR3LiveClient")
    move_pose = next(node for node in client_class.body if isinstance(node, ast.FunctionDef) and node.name == "move_pose")
    search_tree = ast.parse((TOOLS / "auto_cr3_visual_contact_search.py").read_text(encoding="utf-8"))
    shared = [node for node in search_tree.body if isinstance(node, ast.FunctionDef) and node.name in {"finite_pose", "list_pose", "position_error_mm", "move_and_verify"}]
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = {"np": np, "rotation_error_deg": lambda expected, actual: 0.0}
    module = ast.Module(body=[future, exception_class, move_pose, *shared], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<isolated motion feedback functions>", "exec"), namespace)
    return namespace


FUNCTIONS = extracted_functions()
MotionFeedbackError = FUNCTIONS["MotionFeedbackError"]
MOVE_POSE = FUNCTIONS["move_pose"]
MOVE_AND_VERIFY = FUNCTIONS["move_and_verify"]
TARGET = (100.0, 200.0, 300.0, 180.0, 0.0, 0.0)
JOINTS = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


class FeedbackHarness:
    """In-memory client double: every would-be command is only appended to a list."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.commands = []
        self.read_count = 0

    def prepare_motion(self, speed, user, tool):
        pass

    def require_motion_ready(self):
        pass

    def move_cmd(self, command, timeout):
        self.commands.append(command)
        return "ack " + command

    def read_state(self):
        self.read_count += 1
        if self.error:
            raise self.error
        return self.result


class VerificationHarness:
    def __init__(self, result):
        self.result = result
        self.move_count = 0
        self.mode_count = 0

    def read_robot_mode(self):
        self.mode_count += 1
        return 5, "0,{5},RobotMode();"

    def move_pose(self, expected, speed, user, tool):
        self.move_count += 1
        return self.result


class MotionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(disable_ik_preflight=True, speed=1, user=0, tool=2,
                                    motion_position_tolerance_mm=0.2, motion_rotation_tolerance_deg=0.2)

    def test_success_returns_real_pose_and_raw_feedback_in_four_tuple(self):
        actual = (100.1, *TARGET[1:])
        result = (JOINTS, actual, "raw GetAngle", "raw GetPose")
        client = FeedbackHarness(result=result)
        self.assertEqual(MOVE_POSE(client, TARGET, 1, 0, 2), result)
        self.assertEqual(client.read_count, 1)
        self.assertEqual(len(client.commands), 2)
        self.assertEqual(client.commands[-1], "Sync()")

    def test_lost_feedback_raises_and_never_returns_command_as_measurement(self):
        client = FeedbackHarness(error=TimeoutError("GetPose timeout"))
        with self.assertRaises(MotionFeedbackError) as caught:
            MOVE_POSE(client, TARGET, 1, 0, 2)
        self.assertTrue(caught.exception.motion_feedback_unverified)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertIn("Do not attempt automatic recovery motion", str(caught.exception))
        self.assertEqual(client.read_count, 1)
        self.assertEqual(len(client.commands), 2)

    def test_move_pose_rejects_missing_short_and_nonfinite_feedback(self):
        invalid = [None, (), (0.0,) * 5, (0.0,) * 7,
                   (float("nan"),) + (0.0,) * 5, (float("inf"),) + (0.0,) * 5]
        for bad in invalid:
            for field in (0, 1):
                with self.subTest(feedback=bad, field=field):
                    result = [JOINTS, TARGET, "raw GetAngle", "raw GetPose"]
                    result[field] = bad
                    client = FeedbackHarness(result=result)
                    with self.assertRaises(MotionFeedbackError):
                        MOVE_POSE(client, TARGET, 1, 0, 2)
                    self.assertEqual(client.read_count, 1)
                    self.assertEqual(len(client.commands), 2)

    def test_shared_verifier_rejects_legacy_fabricated_target(self):
        client = VerificationHarness((None, TARGET, "MovL acknowledgement", "Sync acknowledgement"))
        with self.assertRaises(MotionFeedbackError):
            MOVE_AND_VERIFY(client, "probe", TARGET, self.args)
        self.assertEqual(client.move_count, 1)
        self.assertEqual(client.mode_count, 1)

    def test_shared_verifier_rejects_invalid_joint_and_tcp_values(self):
        for joints, actual in [(JOINTS, None), ((), TARGET), (JOINTS, TARGET[:5]),
                               ((float("nan"),) * 6, TARGET), (JOINTS, (float("inf"),) * 6)]:
            with self.subTest(joints=joints, actual=actual):
                client = VerificationHarness((joints, actual, "raw GetAngle", "raw GetPose"))
                with self.assertRaises(MotionFeedbackError):
                    MOVE_AND_VERIFY(client, "probe", TARGET, self.args)
                self.assertEqual(client.move_count, 1)

    def test_shared_verifier_reports_real_position_error(self):
        actual = (100.1, *TARGET[1:])
        client = VerificationHarness((JOINTS, actual, "raw GetAngle", "raw GetPose"))
        record = MOVE_AND_VERIFY(client, "probe", TARGET, self.args)
        self.assertEqual(record["actual_tcp"], list(actual))
        self.assertAlmostEqual(record["position_error_mm"], 0.1)
        self.assertEqual(record["joint_count"], 6)
        self.assertEqual(record["raw_get_pose"], "raw GetPose")

    def test_real_out_of_tolerance_pose_still_stops(self):
        actual = (101.0, *TARGET[1:])
        client = VerificationHarness((JOINTS, actual, "raw GetAngle", "raw GetPose"))
        with self.assertRaisesRegex(RuntimeError, "unexpected TCP"):
            MOVE_AND_VERIFY(client, "probe", TARGET, self.args)
        self.assertEqual(client.move_count, 1)


if __name__ == "__main__":
    unittest.main()
