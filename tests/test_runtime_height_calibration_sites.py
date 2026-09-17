"""Ensure runtime height calibration uses only the broad flat reference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import auto_cr3_coverage_board_sampler as sampler  # noqa: E402


class RuntimeHeightCalibrationSiteTests(unittest.TestCase):
    def test_tile_nw_uses_four_outer_flat_reference_sites(self):
        board = ROOT / "outputs" / "tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
        with (board / "tactile_gan_coverage_board_340mm_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        tile = next(row for row in manifest["tiles"] if row["tile_id"] == "tile_nw")
        with (board / "tactile_gan_coverage_board_340mm_sampling_sites.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        samples = sampler.build_runtime_height_calibration_samples(
            tile, rows, SimpleNamespace(tile="tile_nw")
        )
        site_ids = list(dict.fromkeys(sample.source_seed_site_id for sample in samples))
        self.assertEqual(site_ids, ["R01_S01", "R01_S03", "R01_S07", "R01_S09"])
        self.assertEqual(len(samples), 12)
        self.assertTrue(all(sample.category == "flat" and sample.stimulus == "flat_reference" for sample in samples))
        self.assertTrue(all(sample.region_id == "R01" for sample in samples))

    def test_tile_ne_rejects_edges_and_curves_as_a_height_datum(self):
        board = ROOT / "outputs" / "tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
        with (board / "tactile_gan_coverage_board_340mm_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        tile = next(row for row in manifest["tiles"] if row["tile_id"] == "tile_ne")
        with (board / "tactile_gan_coverage_board_340mm_sampling_sites.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with self.assertRaisesRegex(ValueError, "flat_reference"):
            sampler.build_runtime_height_calibration_samples(
                tile, rows, SimpleNamespace(tile="tile_ne")
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
