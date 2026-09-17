"""Audit the repeated region-aware runtime board-contact datum."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime_board_height_datum import (
    DEFAULT_REPEATS_PER_SITE,
    DEFAULT_SITE_IDS,
    build_runtime_height_datum,
    load_runtime_height_datum,
    make_bindings,
)


class RuntimeBoardHeightDatumTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="runtime_height_datum_"))
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.fixture = self.root / "fixture.json"
        self.dock = self.root / "dock.json"
        self.manifest = self.root / "manifest.json"
        for path, payload in (
            (self.fixture, {"fixture": "v1"}),
            (self.dock, {"dock": "v1"}),
            (self.manifest, {"board": "v1"}),
        ):
            path.write_text(json.dumps(payload), encoding="utf-8")
        self.bindings = make_bindings(self.fixture, self.dock, self.manifest, "tile_ne", 0, 2)

    def measurements(self, offset=-1.25):
        rows = []
        for site_index, site_id in enumerate(DEFAULT_SITE_IDS):
            local = [-10.0 + site_index * 5.0, 12.0 + site_index, 10.0]
            region_id = "R01" if site_index < 2 else "R02"
            region_offset = offset if region_id == "R01" else offset + 1.0
            for repeat in range(1, DEFAULT_REPEATS_PER_SITE + 1):
                midpoint = local[2] + region_offset + (repeat - 2) * 0.02
                rows.append(
                    {
                        "site_id": site_id,
                        "repeat": repeat,
                        "sample_id": "{}_r{:02d}".format(site_id, repeat),
                        "region_id": region_id,
                        "category": "flat",
                        "stimulus": "flat_reference",
                        "local_contact_mm": local,
                        "contact_tile_local_mm": [local[0], local[1], midpoint - 0.01],
                        "no_contact_tile_local_mm": [local[0], local[1], midpoint + 0.01],
                    }
                )
        return rows

    def test_repeated_contacts_produce_region_offsets_and_revalidate(self):
        datum = build_runtime_height_datum(self.measurements(), self.bindings)
        self.assertAlmostEqual(datum["region_offsets_tile_z_mm"]["R01"], -1.25)
        self.assertAlmostEqual(datum["region_offsets_tile_z_mm"]["R02"], -0.25)
        self.assertEqual(datum["quality"]["measurement_count"], 15)
        self.assertEqual(datum["quality"]["site_count"], 5)
        self.assertEqual(datum["quality"]["region_count"], 2)
        path = self.root / "datum.json"
        path.write_text(json.dumps(datum), encoding="utf-8")
        loaded = load_runtime_height_datum(path, self.bindings)
        self.assertAlmostEqual(loaded["region_offsets_tile_z_mm"]["R01"], -1.25)
        self.assertAlmostEqual(loaded["region_offsets_tile_z_mm"]["R02"], -0.25)

    def test_one_unstable_repeat_is_not_silently_averaged_away(self):
        rows = self.measurements()
        for row in rows:
            if row["site_id"] == DEFAULT_SITE_IDS[-1] and row["repeat"] == 1:
                row["contact_tile_local_mm"][2] += 1.0
                row["no_contact_tile_local_mm"][2] += 1.0
        with self.assertRaisesRegex(ValueError, "repeat spread"):
            build_runtime_height_datum(rows, self.bindings)

    def test_runtime_datum_is_bound_to_the_exact_fixture(self):
        datum = build_runtime_height_datum(self.measurements(), self.bindings)
        path = self.root / "datum.json"
        path.write_text(json.dumps(datum), encoding="utf-8")
        other = dict(self.bindings)
        other["tool"] = 3
        with self.assertRaisesRegex(ValueError, "another fixture"):
            load_runtime_height_datum(path, other)

    def test_coarse_half_millimetre_brackets_are_accepted_for_runtime_contacts(self):
        rows = self.measurements()
        for row in rows:
            row["bracket_mode"] = "coarse"
            midpoint = (row["contact_tile_local_mm"][2] + row["no_contact_tile_local_mm"][2]) / 2.0
            row["contact_tile_local_mm"][2] = midpoint - 0.25
            row["no_contact_tile_local_mm"][2] = midpoint + 0.25
        datum = build_runtime_height_datum(rows, self.bindings)
        self.assertEqual(datum["points"][0]["bracket_modes"], ["coarse"])

    def test_coarse_bracket_wider_than_half_millimetre_is_rejected(self):
        rows = self.measurements()
        row = rows[0]
        row["bracket_mode"] = "coarse"
        midpoint = (row["contact_tile_local_mm"][2] + row["no_contact_tile_local_mm"][2]) / 2.0
        row["contact_tile_local_mm"][2] = midpoint - 0.251
        row["no_contact_tile_local_mm"][2] = midpoint + 0.251
        with self.assertRaisesRegex(ValueError, "visual-contact bracket"):
            build_runtime_height_datum(rows, self.bindings)


if __name__ == "__main__":
    unittest.main()
