#!/bin/zsh
# Make a no-motion route preview for a per-run physical board orientation.
# Usage:
#   ./scripts/preview_tile_ne_331pin_board_yaw.sh <signed_degrees> [output_dir]
# Example:
#   ./scripts/preview_tile_ne_331pin_board_yaw.sh -90

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  print -u2 "Usage: $0 <signed_degrees> [output_dir]"
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/vincent/Downloads/tactip_experiment_tactistruct/.venv/bin/python"
BOARD_DIR="$ROOT/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
DOCK_DESIGN="/Users/vincent/Downloads/tactip_experiment_tactistruct/outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150/tactip_calibration_dock_lightweight_v4_camera_style_rest_stop_raised15mm/v4_150mm_tactip_calibration_dock_camera_style_rest_stop_design.json"
FIXTURE_PROFILE="$ROOT/outputs/cr3_coverage_board_runs/profiles/tile_ne_331pin_hough_heightcal_v7_20260914.json"
OFFSET_DEG="$1"
SAFE_OFFSET="${OFFSET_DEG//[^0-9A-Za-z_.-]/_}"
OUTPUT_DIR="${2:-$ROOT/outputs/diagnostics/tile_ne_board_yaw_offset_${SAFE_OFFSET}_preview_$(date +%Y%m%d_%H%M%S)}"

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
  --continuous-board-transit \
  --board-yaw-offset-deg "$OFFSET_DEG" \
  --output-dir "$OUTPUT_DIR"
