#!/bin/zsh
# Fixed, hardware-specific 2,000-sample launcher for the installed NE board.
# Start with TacTip physically seated on the raised dock crossbar. Before every
# formal run, the sampler automatically lowers a safely aligned TacTip onto
# the saved fixed crossbar TCP at 1% speed, then records a fresh two-frame
# tactile reference. The prior run's reference image is deliberately ignored
# so a replaced/rotated TacTip head cannot reject a valid new run. This step
# never rewrites the TCP, crossbar height, or board coordinate frame.
# To use a different physical board mounting direction for only this run,
# append e.g. `--board-yaw-offset-deg 90` after the script name. First inspect
# the matching no-motion HTML route from preview_tile_ne_331pin_board_yaw.sh.

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
  --refresh-dock-reference-at-start \
  --continuous-board-transit \
  --continue-on-no-contact \
  --continue-on-safe-sample-error \
  --return-to-dock \
  --skip-previews \
  --execute \
  --yes-i-confirm-cr3-is-safe \
  "$@"
