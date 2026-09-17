"""Offline tests for the coverage-board Jogger's non-GUI safety wiring."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import coverage_board_sampling_jogger as jogger  # noqa: E402


class CoverageBoardSamplingJoggerTests(unittest.TestCase):
    def make_profile(self, root: Path) -> Path:
        profile = root / "fixture.json"
        profile.write_text(
            json.dumps(
                {
                    "schema": "coverage_board_tactip_fixture_profile.v3",
                    "tile_id": "tile_ne",
                    "tool": 2,
                    "user": 0,
                    "dock_tcp": [-467.0, -133.0, 12.0, -179.0, 0.0, 155.0],
                    "board_yaw_deg": 134.38132,
                    "height_calibration": {
                        "measured_tool_tcp_at_crossbar": [-467.0, -133.0, 12.0, -179.0, 0.0, 155.0]
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return profile

    def make_config(self, root: Path, profile: Path) -> jogger.JoggerConfig:
        board = root / "board"
        board.mkdir()
        (board / "test_manifest.json").write_text("{}\n", encoding="utf-8")
        dock = root / "dock.json"
        dock.write_text("{}\n", encoding="utf-8")
        return jogger.JoggerConfig(
            tile="tile_ne",
            board_dir=board,
            dock_design=dock,
            source_fixture_profile=profile,
            source_fixture_profile_sha256=jogger.sha256_file(profile),
            rest_pose=(-467.5, -133.5, 15.0, -179.0, 0.0, 155.0),
            saved_board_yaw_deg=134.38132,
            board_yaw_offset_deg=90.0,
            samples_per_tile=2000,
            min_depth_mm=1.0,
            max_depth_mm=10.0,
            speed_percent=5.0,
            camera_source="0",
            width=640,
            height=480,
            fps=30.0,
            robot_ip="192.168.31.88",
            tool=2,
            user=0,
            continuous_board_transit=True,
            zero_tilt=False,
            python=Path(sys.executable),
            output_root=root / "runs",
        )

    def test_session_profile_preserves_source_and_overrides_only_the_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_profile(root)
            original = json.loads(source.read_text(encoding="utf-8"))
            session = jogger.build_session_profile(
                source,
                (-467.5, -133.5, 15.0, -179.0, 0.0, 155.0),
                root / "session",
                "plan",
            )
            copied = json.loads(session.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), original)
            self.assertEqual(copied["dock_tcp"], [-467.5, -133.5, 15.0, -179.0, 0.0, 155.0])
            self.assertEqual(copied["jogger_session"]["source_profile_modified"], False)
            self.assertEqual(copied["height_calibration"]["measured_tool_tcp_at_crossbar"], copied["dock_tcp"])

    def test_plan_and_execute_commands_share_geometry_but_only_execute_moves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = self.make_profile(root)
            config = self.make_config(root, profile)
            session = jogger.build_session_profile(profile, config.rest_pose, root / "session", "plan")
            plan = jogger.make_sampler_command(config, session, root / "plan", execute=False)
            execute = jogger.make_sampler_command(config, session, root / "run", execute=True)
            self.assertIn("--board-yaw-offset-deg", plan)
            self.assertEqual(plan[plan.index("--board-yaw-offset-deg") + 1], "90")
            self.assertIn("--continuous-board-transit", plan)
            self.assertNotIn("--execute", plan)
            self.assertIn("--execute", execute)
            self.assertIn("--refresh-dock-reference-at-start", execute)
            self.assertIn("--continue-on-safe-sample-error", execute)
            self.assertIn("--return-to-dock", execute)
            self.assertIn("--skip-previews", execute)

    def test_quarter_turn_normalisation_preserves_expected_cycle(self):
        self.assertEqual(jogger.normalise_quarter_turn(270.0), -90.0)
        self.assertEqual(jogger.normalise_quarter_turn(-270.0), 90.0)
        self.assertEqual(jogger.normalise_quarter_turn(180.0), 180.0)

    def test_python_path_keeps_virtual_environment_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_python = root / "base-python"
            base_python.write_text("placeholder\n", encoding="utf-8")
            venv_python = root / "venv-python"
            venv_python.symlink_to(base_python)

            selected = jogger.absolute_path_preserving_symlink(venv_python)

            self.assertEqual(selected, venv_python)
            self.assertNotEqual(selected, base_python.resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
