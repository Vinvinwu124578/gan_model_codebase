"""Focused offline checks for safe per-sample skip classification."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"


def load_classifiers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
    names = {"_exception_messages", "safe_sample_failure_kind"}
    selected = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    found = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    if found != names:
        raise AssertionError("Missing classifiers: {}".format(sorted(names - found)))
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class SafeSampleSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classify = staticmethod(load_classifiers()["safe_sample_failure_kind"])

    def test_ik_rejection_is_skippable_only_when_movl_was_not_sent(self):
        error = RuntimeError("site_high has no CR3 inverse-kinematics solution; MovL was not sent. Reply: -1")
        self.assertEqual(self.classify(error), "ik_preflight_rejected")

    def test_camera_timeout_is_recoverable_after_high_route_recovery(self):
        error = RuntimeError("Timed out waiting for a GelSight frame: Could not read frame from camera source 0")
        self.assertEqual(self.classify(error), "tactile_capture_error")

    def test_unverified_post_motion_error_is_never_skippable(self):
        error = RuntimeError("site_high reached an unexpected TCP: position 4.0 mm")
        self.assertIsNone(self.classify(error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
