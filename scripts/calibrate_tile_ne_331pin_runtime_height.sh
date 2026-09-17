#!/bin/zsh
# Compatibility entry point for the keyed NE tile. This tile contains only
# edge/curvature features, so it must never use them as a board-height datum.
# The physical crossbar under the TacTip seat is the sole fixed-height datum.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/vincent/Downloads/tactip_experiment_tactistruct/.venv/bin/python"
BOARD_DIR="$ROOT/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DOCK_DESIGN="/Users/vincent/Downloads/tactip_experiment_tactistruct/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150/tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm/v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
FIXTURE_PROFILE="$ROOT/outputs/cr3_coverage_board_runs/profiles/tile_ne_331pin_hough_heightcal_v7_20260914.json"
BOARD_YAW_DEG="$(jq -r '.board_yaw_deg // 0' "$FIXTURE_PROFILE")"
echo "tile_ne has no broad flat-reference patch; refreshing the fixed crossbar datum instead."
exec "$PYTHON" "$ROOT/tools/auto_cr3_coverage_board_sampler.py" \
  --tile tile_ne \
  --board-dir "$BOARD_DIR" \
  --dock-design "$DOCK_DESIGN" \
  --fixture-profile "$FIXTURE_PROFILE" \
  --calibrate-height-from-rest-stop \
  --camera-source 0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --robot-ip 192.168.31.88 \
  --tool 2 \
  --user 0 \
  --speed 3 \
  --board-yaw-deg "$BOARD_YAW_DEG" \
  "$@"
