#!/bin/zsh
# Fixed, hardware-specific 2,000-sample launcher for the installed NE board.
# The sampler automatically re-seats a TacTip that is only 0.1-5.0 mm above
# the keyed dock crossbar, then validates the saved tactile crossbar image.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/vincent/Downloads/tactip_experiment_tactistruct/.venv/bin/python"
BOARD_DIR="$ROOT/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DOCK_DESIGN="/Users/vincent/Downloads/tactip_experiment_tactistruct/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150/tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm/v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
FIXTURE_PROFILE="$ROOT/outputs/cr3_coverage_board_runs/profiles/tile_ne_331pin_hough_heightcal_v7_20260914.json"

exec "$PYTHON" "$ROOT/tools/auto_cr3_coverage_board_sampler.py" \
  --tile tile_ne \
  --board-dir "$BOARD_DIR" \
  --dock-design "$DOCK_DESIGN" \
  --fixture-profile "$FIXTURE_PROFILE" \
  --samples-per-tile 2000 \
  --dense-spatial-layout region_grid \
  --dense-region-anchor-count 25 \
  --min-post-contact-depth-mm 1 \
  --max-post-contact-depth-mm 10 \
  --camera-source 0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --robot-ip 192.168.31.88 \
  --tool 2 \
  --user 0 \
  --speed 5 \
  --continuous-board-transit \
  --return-to-dock \
  --skip-previews \
  --execute \
  --yes-i-confirm-cr3-is-safe \
  "$@"
