#!/usr/bin/env python3
"""Run the CR3 coverage-board sampler with v4 150 mm-pitch defaults.

This wrapper pins the public hardware bundle to the 170 mm v4 tile, its
nominal 6.0 mm M6 holes, and the lightweight TacTip dock used by the CR3
workflow. All command-line options remain those of
``auto_cr3_coverage_board_sampler.py``.
"""

from __future__ import annotations

from pathlib import Path

import auto_cr3_coverage_board_sampler as sampler


V4_BOARD_DIR = Path("outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150")
V4_DOCK_DIR = V4_BOARD_DIR / "tactip_calibration_dock_lightweight_v1"


def main() -> int:
    sampler.DEFAULT_BOARD_DIR = V4_BOARD_DIR
    sampler.DEFAULT_DOCK_DIR = V4_DOCK_DIR
    sampler.DEFAULT_DOCK_DESIGN = V4_DOCK_DIR / "v4_150mm_tactip_calibration_dock_lightweight_design.json"
    sampler.DEFAULT_RUN_ROOT = V4_BOARD_DIR / "cr3_coverage_board_runs"
    return sampler.main()


if __name__ == "__main__":
    raise SystemExit(main())
