"""Offline regression checks; load only pure IK orchestration, never robot drivers."""

import ast
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"
NAMES = {
    "check_ik_target", "check_ik_sequence", "preflight_sample_route",
    "select_ik_reachable_samples", "route_ik_targets", "reindex_samples",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
selected.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES)


@dataclass(frozen=True)
class Sample:
    index: int
    sample_id: str
    region_id: str = "R01"
    post_contact_depth_mm: float = 1.0
    route_offset: float = 0.0


def pose(value):
    return (float(value), 0.0, 0.0, 0.0, 0.0, 0.0)


def make_route(sample, fixture, args, correction):
    return {
        "site_high_tcp": pose(1 + sample.route_offset),
        "approach_tcp": pose(2 + sample.route_offset),
        "contact_tcp": pose(3 + sample.route_offset),
        "deepest": pose(4 + sample.route_offset),
    }


ns = {
    "time": time, "Counter": Counter, "asdict": asdict, "replace": replace,
    "finite_pose": lambda target, label: tuple(float(x) for x in target),
    "list_pose": list, "make_route": make_route,
    "maximum_capture_tcp": lambda route, args, depth: route["deepest"],
    "np": SimpleNamespace(asarray=lambda values, dtype: tuple(values)),
}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), "exec"), ns)


class Controller:
    def __init__(self, failure=None, fallback_succeeds=False):
        self.calls = []
        self.failure = failure
        self.fallback_succeeds = fallback_succeeds

    def inverse_solution(self, target, user, tool, joint_near):
        self.calls.append((target, joint_near, user, tool))
        if self.failure is not None and (joint_near is not None or not self.fallback_succeeds):
            raise self.failure
        return tuple(target), "mock reply"


def context(cache=True):
    return {"cache": {} if cache else None, "target_checks": 0, "controller_calls": 0,
            "diagnostic_calls": 0, "cache_hits": 0}


def settings(**overrides):
    result = dict(user=0, tool=2, ik_diagnose_failures=False, disable_ik_cache=False,
                  disable_ik_preflight=False, tile="tile_nw", safe_height_mm=100.0)
    result.update(overrides)
    return SimpleNamespace(**result)


class IKPreflightTests(unittest.TestCase):
    def test_cache_requires_exact_pose_branch_user_and_tool(self):
        robot, args, state = Controller(), settings(), context()
        check = ns["check_ik_target"]
        first, _ = check(robot, "first", pose(1), args, pose(0), state)
        again, record = check(robot, "second", pose(1), args, pose(0), state)
        self.assertEqual(first, again)
        self.assertEqual(record["label"], "second")
        self.assertEqual(record["ik_source"], "exact_query_cache")
        self.assertEqual(len(robot.calls), 1)
        check(robot, "different hint", pose(1), args, pose(4), state)
        check(robot, "tiny pose difference", pose(1 + 1e-10), args, pose(0), state)
        check(robot, "different user", pose(1), settings(user=1), pose(0), state)
        check(robot, "different tool", pose(1), settings(tool=1), pose(0), state)
        self.assertEqual(len(robot.calls), 5)
        self.assertEqual(state["cache_hits"], 1)

    def test_cache_can_be_disabled(self):
        robot, state = Controller(), context(cache=False)
        for label in ("a", "b"):
            ns["check_ik_target"](robot, label, pose(1), settings(), pose(0), state)
        self.assertEqual(len(robot.calls), 2)
        self.assertEqual(state["cache_hits"], 0)

    def test_failures_are_not_cached_and_no_diagnostic_is_default(self):
        robot, state = Controller(RuntimeError("no solution")), context()
        for label in ("a", "b"):
            solution, record = ns["check_ik_target"](robot, label, pose(1), settings(), pose(0), state)
            self.assertIsNone(solution)
            self.assertEqual(record["failure_kind"], "near_hint_failed_unclassified")
        self.assertEqual(len(robot.calls), 2)
        self.assertEqual(state["cache"], {})

    def test_unhinted_success_never_accepts_failed_branch(self):
        robot, state = Controller(RuntimeError("branch rejected"), fallback_succeeds=True), context()
        solution, record = ns["check_ik_target"](
            robot, "a", pose(1), settings(ik_diagnose_failures=True), pose(0), state)
        self.assertIsNone(solution)
        self.assertEqual(record["failure_kind"], "near_joint_branch")
        self.assertEqual(state["controller_calls"], 2)
        self.assertEqual(state["diagnostic_calls"], 1)
        self.assertEqual(state["cache"], {})

    def test_communication_failures_stop_without_fallback(self):
        for error in (TimeoutError("timeout"), ConnectionError("closed"), OSError("socket error"),
                      RuntimeError("No reply for command: InverseSolution(...)")):
            with self.subTest(error=type(error).__name__):
                robot = Controller(error)
                with self.assertRaises(type(error)):
                    ns["check_ik_target"](
                        robot, "a", pose(1), settings(ik_diagnose_failures=True), pose(0), context())
                self.assertEqual(len(robot.calls), 1)

    def test_sequence_stops_after_rejection(self):
        robot = Controller(RuntimeError("no solution"))
        solution, records = ns["check_ik_sequence"](
            robot, (("first", pose(1)), ("second", pose(2))), settings(), pose(0), context())
        self.assertIsNone(solution)
        self.assertEqual(len(robot.calls), 1)
        self.assertEqual(records[1]["status"], "not_checked_after_upstream_failure")

    def test_full_filter_cache_scope_and_retreat_branch(self):
        robot, args = Controller(), settings()
        fixture = SimpleNamespace(dock_tcp_local_mm=(0, 0, 0), pose=lambda xyz: pose(100))
        samples = [Sample(1, "a"), Sample(2, "b")]
        for run in (1, 2):
            selected_samples, report = ns["select_ik_reachable_samples"](
                robot, samples, samples, fixture, args, (0, 0, 0), pose(0))
            self.assertEqual(len(selected_samples), 2)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["performance"]["controller_ik_call_count"], 6)
            self.assertEqual(report["performance"]["cache_hit_count"], 5)
            self.assertEqual(len(robot.calls), run * 6)
            for candidate in report["candidates"]:
                self.assertEqual([r["label"] for r in candidate["checks"]],
                                 ["site_high", "approach", "planned_contact", "deepest_capture_limit", "site_retract"])
            first_checks = report["candidates"][0]["checks"]
            self.assertNotEqual(first_checks[0]["joint_near_deg"], first_checks[4]["joint_near_deg"])
            self.assertEqual(first_checks[4]["ik_source"], "controller")


if __name__ == "__main__":
    unittest.main(verbosity=2)
