"""Offline regression checks for sampler configuration and frame validation.

Run with ``python -m unittest test_sampler_config_offline -v``. Only selected
AST definitions are loaded, so importing the robot, camera, mesh and SciPy
dependencies is neither necessary nor possible through this test loader.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import math
from pathlib import Path
import sys
from typing import Any
import unittest
from unittest.mock import Mock, patch
import uuid


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "auto_cr3_coverage_board_sampler.py"


def add_tactip_preprocess_args(parser: argparse.ArgumentParser) -> None:
    """Provide the only preprocessing argument referenced by parse_args."""
    parser.add_argument("--no-tactip-preprocess", action="store_true")


def load_configuration_definitions() -> dict[str, Any]:
    source = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
    names = {"parse_args", "fixture_from_profile"}
    constants = {
        "DEFAULT_BOARD_DIR",
        "DEFAULT_DOCK_DIR",
        "DEFAULT_DOCK_DESIGN",
        "DEFAULT_RUN_ROOT",
        "VALID_TILES",
        "SUPPORTED_FIXTURE_PROFILE_SCHEMAS",
        "DEFAULT_POST_CONTACT_DEPTH_MIN_MM",
        "DEFAULT_POST_CONTACT_DEPTH_MAX_MM",
        "DEFAULT_FIRST_CONTACT_MAX_EXTRA_BELOW_NOMINAL_MM",
        "DEFAULT_MAX_EXTRA_BELOW_PLANNED_CONTACT_MM",
    }
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    found = set()
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            body.append(node)
    if found != names:
        raise AssertionError(f"Missing source functions: {sorted(names - found)}")
    namespace = {
        "__doc__": "Offline sampler argument validation",
        "argparse": argparse,
        "math": math,
        "Path": Path,
        "Any": Any,
        "add_tactip_preprocess_args": add_tactip_preprocess_args,
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class SamplerConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definitions = load_configuration_definitions()

    def setUp(self) -> None:
        workspace = SOURCE.parent.resolve()
        # tempfile's Windows mode=0o700 ACL can exclude a sandbox identity.
        # A unique ordinary directory inherits the writable workspace ACL.
        root = workspace / f"cr3_config_offline_{uuid.uuid4().hex}"
        self.assertEqual(root.resolve().parent, workspace)
        root.mkdir()
        self.addCleanup(root.rmdir)
        self.board_dir = root / "board"
        self.board_dir.mkdir()
        self.addCleanup(self.board_dir.rmdir)
        self.dock_design = root / "dock.json"
        self.dock_design.write_text("{}\n", encoding="utf-8")
        self.addCleanup(self.dock_design.unlink)
        self.base_argv = [
            str(SOURCE),
            "--tile", "tile_nw",
            "--board-dir", str(self.board_dir),
            "--dock-design", str(self.dock_design),
        ]

    def parse(self, *extra: str) -> argparse.Namespace:
        with patch.object(sys, "argv", self.base_argv + list(extra)):
            return self.definitions["parse_args"]()

    def assert_rejected(self, *extra: str, containing: str) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as rejected:
                self.parse(*extra)
        self.assertEqual(rejected.exception.code, 2)
        self.assertIn(containing, stderr.getvalue())

    @staticmethod
    def first_contact_args() -> tuple[str, ...]:
        return ("--first-contact-test", "--max-samples", "1", "--site", "R01_S01")

    def test_normal_defaults_preserve_requested_depth_range_and_global_cap(self) -> None:
        args = self.parse()
        self.assertEqual(args.min_post_contact_depth_mm, 1.0)
        self.assertEqual(args.max_post_contact_depth_mm, 10.0)
        self.assertEqual(args.contact_search_margin_mm, 1.0)
        self.assertEqual(args.max_extra_below_planned_contact_mm, 11.0)
        self.assertFalse(args.execute)
        self.assertFalse(args.no_tactip_preprocess)

    def test_first_contact_default_cap_is_two_mm(self) -> None:
        args = self.parse(*self.first_contact_args())
        self.assertEqual(args.min_post_contact_depth_mm, 1.0)
        self.assertEqual(args.max_extra_below_planned_contact_mm, 2.0)

    def test_first_contact_margin_three_automatically_sets_four_mm_cap(self) -> None:
        args = self.parse(*self.first_contact_args(), "--contact-search-margin-mm", "3")
        self.assertEqual(args.max_extra_below_planned_contact_mm, 4.0)

    def test_deep_localization_margin_ten_automatically_sets_eleven_mm_cap(self) -> None:
        args = self.parse(
            *self.first_contact_args(),
            "--allow-deep-contact-localization",
            "--contact-search-margin-mm", "10",
        )
        self.assertEqual(args.max_extra_below_planned_contact_mm, 11.0)

    def test_deep_localization_requires_first_contact_mode(self) -> None:
        self.assert_rejected(
            "--allow-deep-contact-localization",
            containing="only valid with --first-contact-test",
        )

    def test_deep_localization_requires_explicit_opt_in(self) -> None:
        self.assert_rejected(
            *self.first_contact_args(), "--contact-search-margin-mm", "10",
            containing="--contact-search-margin-mm must be in",
        )

    def test_deep_localization_cannot_expand_dock_reference_pad_search(self) -> None:
        self.assert_rejected(
            *self.first_contact_args(),
            "--allow-deep-contact-localization",
            "--contact-search-margin-mm", "10",
            "--use-reference-pad-check",
            containing="omit --use-reference-pad-check",
        )

    def test_explicit_caps_that_cannot_cover_depth_and_margin_are_rejected(self) -> None:
        cases = [
            ((), "10"),
            (self.first_contact_args(), "1"),
            ((*self.first_contact_args(), "--contact-search-margin-mm", "3"), "3"),
            ((
                *self.first_contact_args(),
                "--allow-deep-contact-localization", "--contact-search-margin-mm", "10",
            ), "10"),
        ]
        for options, cap in cases:
            with self.subTest(options=options, cap=cap):
                self.assert_rejected(
                    *options, "--max-extra-below-planned-contact-mm", cap,
                    containing="must cover the deepest requested indentation PLUS",
                )

    def test_explicit_sufficient_cap_is_preserved(self) -> None:
        args = self.parse(
            *self.first_contact_args(),
            "--contact-search-margin-mm", "3",
            "--max-extra-below-planned-contact-mm", "5",
        )
        self.assertEqual(args.max_extra_below_planned_contact_mm, 5.0)

    def test_nonfinite_critical_numeric_parameters_are_rejected(self) -> None:
        options = (
            "robot-timeout-sec", "speed", "camera-read-timeout-sec", "settle-sec",
            "step-mm", "min-post-contact-depth-mm", "max-post-contact-depth-mm",
            "approach-clearance-mm", "safe-height-mm", "dock-exit-lift-mm",
            "max-extra-below-planned-contact-mm", "noise-multiplier",
            "min-contact-mean", "min-contact-p95", "dock-position-tolerance-mm",
            "dock-rotation-tolerance-deg", "motion-position-tolerance-mm",
            "motion-rotation-tolerance-deg", "reference-max-lateral-correction-mm",
            "reference-max-vertical-correction-mm", "rest-stop-stability-tolerance-mm",
            "rest-stop-stability-tolerance-deg", "board-height-offset-mm", "board-yaw-deg",
            "contact-search-margin-mm", "ik-candidate-multiplier",
        )
        for option in options:
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    self.assert_rejected(f"--{option}={value}", containing=f"--{option}")

    def test_fixture_user_mismatch_is_rejected_before_hash_or_geometry(self) -> None:
        args = self.parse("--user", "1")
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v3",
            "tile_id": args.tile,
            "tool": args.tool,
            "user": 0,
            "dock_design_sha256": "same-hash",
        }
        hash_stub = Mock(return_value="same-hash")
        with patch.dict(self.definitions, {"sha256_file": hash_stub}):
            with self.assertRaisesRegex(ValueError, r"User\(0\).*User\(1\)"):
                self.definitions["fixture_from_profile"](profile, {}, args)
        hash_stub.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
