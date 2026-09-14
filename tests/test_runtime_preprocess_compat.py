import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tactip_hough_chromatic import HoughChromaticConfig
from tactip_runtime_preprocess import TacTipRuntimePreprocessor


class RuntimePreprocessCompatibilityTests(unittest.TestCase):
    def test_writes_native_motion_maps_for_visual_contact(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        native_binary = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(native_binary, (30, 30), 12, 255, -1)
        cv2.circle(native_binary, (70, 50), 12, 255, -1)
        model_input = cv2.resize(native_binary, (256, 256), interpolation=cv2.INTER_NEAREST)
        images = {
            "contact_binary": native_binary,
            "raw_roi": frame,
            "gray_roi": native_binary,
            "blue_yellow_score_roi": native_binary,
            "overlay_roi": frame,
            "binary_roi": native_binary,
            "model_input_256": model_input,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "frames" / "capture.png"
            raw.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(raw), frame))
            preprocessor = TacTipRuntimePreprocessor(root / "tactip_preprocessed", HoughChromaticConfig())
            with patch("tactip_runtime_preprocess.process_frame", return_value=({}, images, [])):
                output = preprocessor.process_and_save(frame, raw)
            preprocessor.close()

            self.assertTrue(output.is_file())
            motion_gray = cv2.imread(str(root / "tactip_preprocessed" / "gray" / "capture.png"), cv2.IMREAD_GRAYSCALE)
            motion_roi = cv2.imread(str(root / "tactip_preprocessed" / "model_roi" / "capture.png"), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.array_equal(motion_gray, native_binary))
            self.assertTrue(np.array_equal(motion_roi, native_binary))
            self.assertTrue((root / "tactip_preprocessed" / "contact_binary" / "capture.png").is_file())
            self.assertTrue((root / "tactip_preprocessed" / "model_input" / "capture.png").is_file())


if __name__ == "__main__":
    unittest.main()
