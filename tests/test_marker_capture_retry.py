"""Regression tests for stationary camera retry after strict Hough rejection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import numpy as np

import auto_cr3_visual_contact_search as contact_search


class MarkerCaptureRetryTests(unittest.TestCase):
    def test_capture_record_retries_fresh_frame_after_marker_rejection(self):
        args = SimpleNamespace(capture_retry_count=2, capture_retry_sec=0.0)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        feature = contact_search.ImageFeature(
            gray=np.zeros((8, 8), dtype=np.uint8),
            allowed_mask=np.ones((8, 8), dtype=bool),
            marker_support=np.ones((8, 8), dtype=bool),
            normalized_texture=np.zeros((8, 8), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            paths = (
                output_dir / "frames" / "reference.png",
                output_dir / "tactip_preprocessed" / "gray" / "reference.png",
                output_dir / "tactip_preprocessed" / "model_roi" / "reference.png",
            )
            with patch.object(
                contact_search,
                "median_capture",
                side_effect=[(frame, 10.0, 120.0), (frame, 11.0, 121.0)],
            ) as median, patch.object(
                contact_search,
                "write_frame",
                side_effect=[
                    contact_search.TacTipPreprocessRejected("331 markers not found"),
                    paths,
                ],
            ) as write, patch.object(contact_search, "load_feature", return_value=feature):
                record, returned_feature, timestamp = contact_search.capture_record(
                    object(), args, output_dir, object(), "reference", 3, 0.0
                )

        self.assertIs(returned_feature, feature)
        self.assertEqual(timestamp, 11.0)
        self.assertEqual(record["preprocess_attempts"], 2)
        self.assertEqual(record["preprocess_rejections_before_accept"], ["331 markers not found"])
        self.assertEqual(median.call_args_list[0].args[-1], 0.0)
        self.assertEqual(median.call_args_list[1].args[-1], 10.0)
        self.assertEqual(write.call_count, 2)

    def test_capture_record_stops_after_configured_retries_without_moving(self):
        args = SimpleNamespace(capture_retry_count=1, capture_retry_sec=0.0)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            contact_search,
            "median_capture",
            side_effect=[(frame, 10.0, 120.0), (frame, 11.0, 121.0)],
        ) as median, patch.object(
            contact_search,
            "write_frame",
            side_effect=contact_search.TacTipPreprocessRejected("331 markers not found"),
        ) as write:
            with self.assertRaisesRegex(RuntimeError, "rejected 2 fresh capture"):
                contact_search.capture_record(
                    object(), args, Path(temporary), object(), "reference", 3, 0.0
                )

        self.assertEqual(median.call_count, 2)
        self.assertEqual(write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
