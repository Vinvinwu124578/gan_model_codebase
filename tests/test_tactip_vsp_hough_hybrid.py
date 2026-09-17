from __future__ import annotations

import unittest

import numpy as np

from tactip_vsp_hough_hybrid import HybridConfig, fuse_detections


def _hough_marker(point: np.ndarray) -> dict[str, float]:
    return {
        "x_px": float(point[0]),
        "y_px": float(point[1]),
        "hough_radius_px": 3.0,
        "blue_yellow_score": 40.0,
    }


class HybridFusionTests(unittest.TestCase):
    def test_vsp_remains_primary_and_hough_only_fills_missing_marker(self) -> None:
        true_points = np.asarray(
            [(10.0 * column, 10.0 * row) for row in range(5) for column in range(5)]
        )
        missing = np.asarray((20.0, 20.0))
        vsp_points = np.asarray(
            [point for point in true_points if not np.array_equal(point, missing)]
        )
        hough_points = np.vstack((true_points, np.asarray((100.0, 100.0))))
        config = HybridConfig(expected_markers=25, match_radius_px=2.0)

        result = fuse_detections(
            vsp_points,
            np.full(len(vsp_points), 6.0),
            [_hough_marker(point) for point in hough_points],
            config,
        )

        self.assertEqual(len(result["final_markers"]), 25)
        self.assertEqual(len(result["selected_supplements"]), 1)
        self.assertEqual(result["final_markers"][-1]["source"], "hough_supplement")
        self.assertTrue(
            np.allclose(
                [
                    result["final_markers"][-1]["x_px"],
                    result["final_markers"][-1]["y_px"],
                ],
                missing,
            )
        )
        final_vsp = np.asarray(
            [
                (marker["x_px"], marker["y_px"])
                for marker in result["final_markers"]
                if marker["source"] == "vsp_primary"
            ]
        )
        self.assertTrue(np.allclose(final_vsp, vsp_points))

    def test_hybrid_refuses_to_infer_when_no_measured_fill_exists(self) -> None:
        true_points = np.asarray(
            [(10.0 * column, 10.0 * row) for row in range(5) for column in range(5)]
        )
        vsp_points = true_points[:-1]
        config = HybridConfig(expected_markers=25, match_radius_px=2.0)

        with self.assertRaisesRegex(ValueError, "output was not fabricated"):
            fuse_detections(
                vsp_points,
                np.full(len(vsp_points), 6.0),
                [_hough_marker(point) for point in vsp_points],
                config,
            )

    def test_unmistakable_vsp_duplicate_is_merged_without_hough(self) -> None:
        true_points = np.asarray(
            [(10.0 * column, 10.0 * row) for row in range(5) for column in range(5)]
        )
        duplicated = np.vstack((true_points, true_points[12] + np.asarray((1.0, 0.5))))
        result = fuse_detections(
            duplicated,
            np.full(len(duplicated), 6.0),
            [],
            HybridConfig(expected_markers=25, match_radius_px=2.0),
        )

        self.assertEqual(result["raw_vsp_count"], 26)
        self.assertEqual(result["clean_vsp_count"], 25)
        self.assertEqual(len(result["vsp_duplicate_groups"]), 1)
        self.assertEqual(len(result["selected_supplements"]), 0)
        self.assertTrue(
            all(marker["source"] == "vsp_primary" for marker in result["final_markers"])
        )


if __name__ == "__main__":
    unittest.main()
