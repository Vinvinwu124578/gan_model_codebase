"""Tests for same-frame geometric repair of a single Hough glare outlier."""

from __future__ import annotations

import unittest

from tactip_hough_chromatic import _replace_single_isolated_lattice_marker


def marker(x: float, y: float, score: float = 40.0):
    return {
        "x_px": x,
        "y_px": y,
        "hough_radius_px": 5.0,
        "blue_yellow_score": score,
    }


class LatticeRepairTests(unittest.TestCase):
    def test_replaces_one_isolated_glare_circle_with_supported_lattice_circle(self):
        accepted = [
            marker(x, y)
            for y in (0.0, 10.0, 20.0, 30.0)
            for x in (0.0, 10.0, 20.0, 30.0)
            if (x, y) != (10.0, 10.0)
        ]
        accepted.append(marker(80.0, 80.0, 80.0))
        repaired, diagnostics = _replace_single_isolated_lattice_marker(
            accepted, [marker(10.0, 10.0, 35.0)]
        )

        self.assertEqual(diagnostics["status"], "repaired")
        self.assertEqual(diagnostics["removed_marker"]["x_px"], 80.0)
        self.assertEqual(diagnostics["replacement_marker"]["x_px"], 10.0)
        self.assertEqual(diagnostics["replacement_marker"]["y_px"], 10.0)
        self.assertFalse(any(item["x_px"] == 80.0 and item["y_px"] == 80.0 for item in repaired))
        self.assertTrue(any(item["x_px"] == 10.0 and item["y_px"] == 10.0 for item in repaired))

    def test_keeps_valid_lattice_when_no_circle_isolated(self):
        accepted = [marker(x, y) for y in (0.0, 10.0, 20.0, 30.0) for x in (0.0, 10.0, 20.0)]
        repaired, diagnostics = _replace_single_isolated_lattice_marker(
            accepted, [marker(40.0, 10.0)]
        )

        self.assertIs(repaired, accepted)
        self.assertEqual(diagnostics["status"], "not_needed")


if __name__ == "__main__":
    unittest.main()
