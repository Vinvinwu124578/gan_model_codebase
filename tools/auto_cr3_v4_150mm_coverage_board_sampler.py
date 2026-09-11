#!/usr/bin/env python3
"""Run the CR3 coverage-board sampler with v4 150 mm-pitch defaults.

This wrapper pins the public hardware bundle to the 170 mm v4 tile and
the raised-crossbar dock used by this installation. Supply --dock-design
explicitly until the matching real dock design is installed. The legacy
v1 ring dock uses a different height datum. All options remain those of
``auto_cr3_coverage_board_sampler.py``.
"""

from __future__ import annotations

from pathlib import Path

import auto_cr3_coverage_board_sampler as sampler


V4_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
V4_DOCK_DIR = V4_BOARD_DIR / "tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm"


def main() -> int:
    sampler.DEFAULT_BOARD_DIR = V4_BOARD_DIR
    sampler.DEFAULT_DOCK_DIR = V4_DOCK_DIR
    sampler.DEFAULT_DOCK_DESIGN = V4_DOCK_DIR / "v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
    sampler.DEFAULT_RUN_ROOT = V4_BOARD_DIR / "cr3_coverage_board_runs"
    return sampler.main()


if __name__ == "__main__":
    raise SystemExit(main())
