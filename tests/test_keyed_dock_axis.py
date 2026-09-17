"""Regression tests for the mechanically keyed v2 board/dock frame."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import auto_cr3_coverage_board_sampler as sampler  # noqa: E402


class KeyedDockAxisTests(unittest.TestCase):
    def test_saved_fixed_profile_yaw_is_applied_to_keyed_v2_dock(self):
        args = type("Args", (), {"tile": "tile_ne", "tool": 2, "user": 0, "dock_design": Path("dock.json")})()
        dock = {
            "schema": "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2",
            "board_interface": {"shared_board_hole_pitch_mm": 150.0},
            "tactip_reference": {"nominal_seated_tool_tcp_local_mm": [0.0, -139.0, 27.0]},
        }
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v3",
            "tile_id": "tile_ne",
            "tool": 2,
            "user": 0,
            "dock_design_sha256": "dock-hash",
            "dock_tcp": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            "board_yaw_deg": 134.38132,
        }
        with mock.patch.object(sampler, "sha256_file", return_value="dock-hash"):
            fixture = sampler.fixture_from_profile(profile, dock, args)
        expected = (
            np.diag((1.0, -1.0, -1.0))
            @ sampler.Rotation.from_euler("Z", 134.38132, degrees=True).as_matrix()
        )
        self.assertAlmostEqual(fixture.board_yaw_deg, 134.38132)
        np.testing.assert_allclose(fixture.tile_to_base, expected, atol=1e-12)

    def test_runtime_yaw_offset_rotates_plan_without_mutating_saved_profile(self):
        args = type(
            "Args",
            (),
            {
                "tile": "tile_ne",
                "tool": 2,
                "user": 0,
                "dock_design": Path("dock.json"),
                "board_yaw_offset_deg": -90.0,
            },
        )()
        dock = {
            "schema": "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2",
            "board_interface": {"shared_board_hole_pitch_mm": 150.0},
            "tactip_reference": {"nominal_seated_tool_tcp_local_mm": [0.0, -139.0, 27.0]},
        }
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v3",
            "tile_id": "tile_ne",
            "tool": 2,
            "user": 0,
            "dock_design_sha256": "dock-hash",
            "dock_tcp": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            "board_yaw_deg": 134.38132,
        }
        with mock.patch.object(sampler, "sha256_file", return_value="dock-hash"):
            fixture = sampler.fixture_from_profile(profile, dock, args)

        expected_yaw = 44.38132
        expected = (
            np.diag((1.0, -1.0, -1.0))
            @ sampler.Rotation.from_euler("Z", expected_yaw, degrees=True).as_matrix()
        )
        self.assertAlmostEqual(fixture.board_yaw_deg, expected_yaw)
        self.assertEqual(profile["board_yaw_deg"], 134.38132)
        np.testing.assert_allclose(fixture.tile_to_base, expected, atol=1e-12)

        metadata = sampler.board_orientation_metadata(fixture, args, profile)
        self.assertEqual(metadata["saved_profile_yaw_deg"], 134.38132)
        self.assertEqual(metadata["runtime_yaw_offset_deg"], -90.0)
        self.assertAlmostEqual(metadata["effective_yaw_deg"], expected_yaw)
        self.assertFalse(metadata["mutates_fixture_profile"])
        self.assertEqual(metadata["transform_version"], "board-centre-pivot.v1")
        self.assertEqual(metadata["rotation_pivot_tile_local_mm"], [0.0, 0.0, 0.0])

    def test_runtime_quarter_turn_keeps_board_centre_and_dock_fixed(self):
        dock = {
            "schema": "tactile_gan_coverage_board.v4.mountpitch150.lightweight_dock.v2",
            "board_interface": {"shared_board_hole_pitch_mm": 150.0},
            "tactip_reference": {"nominal_seated_tool_tcp_local_mm": [0.0, -139.0, 27.0]},
        }
        profile = {
            "schema": "coverage_board_tactip_fixture_profile.v3",
            "tile_id": "tile_ne",
            "tool": 2,
            "user": 0,
            "dock_design_sha256": "dock-hash",
            "dock_tcp": [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            "board_yaw_deg": 15.0,
        }
        base_args = {
            "tile": "tile_ne",
            "tool": 2,
            "user": 0,
            "dock_design": Path("dock.json"),
        }
        with mock.patch.object(sampler, "sha256_file", return_value="dock-hash"):
            zero = sampler.fixture_from_profile(
                profile, dock, type("Args", (), {**base_args, "board_yaw_offset_deg": 0.0})()
            )
            quarter = sampler.fixture_from_profile(
                profile, dock, type("Args", (), {**base_args, "board_yaw_offset_deg": 90.0})()
            )

        centre = (0.0, 0.0, 0.0)
        np.testing.assert_allclose(quarter.position(centre), zero.position(centre), atol=1e-10)
        np.testing.assert_allclose(
            quarter.dock_pose(quarter.dock_tcp_local_mm)[:3],
            profile["dock_tcp"][:3],
            atol=1e-10,
        )
        centre_base = quarter.position(centre)
        np.testing.assert_allclose(
            quarter.position((20.0, 0.0, 0.0)) - centre_base,
            quarter.tile_to_base @ np.asarray((20.0, 0.0, 0.0)),
            atol=1e-10,
        )

    def test_fixed_preview_rotates_points_about_board_centre(self):
        rotated = sampler.rotate_board_points_for_fixed_preview(
            np.asarray(((10.0, 0.0, 2.0), (0.0, 0.0, 0.0))),
            90.0,
        )
        np.testing.assert_allclose(rotated[0], (0.0, 10.0, 2.0), atol=1e-10)
        np.testing.assert_allclose(rotated[1], (0.0, 0.0, 0.0), atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
