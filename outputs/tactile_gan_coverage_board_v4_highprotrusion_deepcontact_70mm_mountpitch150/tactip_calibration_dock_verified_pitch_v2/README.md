# Verified TacTip Calibration Dock v2 (150 mm)

This optional rigid dock is matched to the public v4 board variant whose south mounting-hole pair is exactly 150 mm apart.

## Included STL

Use `v4_tactip_calibration_dock_v2_exact_150.stl` only when the two south board-hole centres measure **150 mm**. The former 156 mm and dual-pitch development variants are intentionally not part of this repository.

## Mechanical check before collecting

1. Put the tile with its tactile surface facing up.
2. Put the dock on the south edge: the straight rear rail must touch the tile's straight edge.
3. Slide the board into the continuous C-shaped rail: its underside sits on the lower lip and its south edge reaches the hard stop.
4. Insert both M6 screws without force. Both must drop through the dock and into the board together.
5. Tighten both screws. The dock must not translate or yaw by hand.
6. Only then seat the TacTip and record a **new** Tool(2) fixture profile.

The board's 6.0 mm holes remain unchanged. The dock's 6.6 mm holes are deliberate M6 clearance, not an error.

Key files:

- `v4_tactip_calibration_dock_v2_exact_150.stl`
- `v4_tactip_calibration_dock_v2_exact_150_design.json`
- `v4_tactip_calibration_dock_v2_hole_alignment.png`
- `v4_tactip_calibration_dock_v2_clip_section.png`
