# Lightweight TacTip calibration dock for v4 150 mm board

This one-piece dock is for the non-destructive v4 mounting-hole variant. It shares the two south-edge M6 tile holes at local `(-75,-75)` and `(75,-75)` mm, so the mounting pitch is exactly `150.0 mm`. The board must remain bolted flat through all four of its own M6 holes; the dock's two feet rest on that same table.

## Installation

1. Print the dock with its two feet down.
2. Bolt a v4 170 mm tile to the table. Keep the dock at tile-local `-Y`.
3. Loosen the south pair of board screws, slide the dock U-slots in from local `-Y`, and retighten the two M6 screws.
4. Seat the rigid TacTip flange in the 50.8 mm guide. The compliant tip remains clear through the 44 mm opening.
5. Record Tool(2) while seated. This is the repeatable rest datum for automatic collection.

The `*_design.json` file records the complete local geometry, including the nominal seated Tool(2) TCP and reference-pad height.
