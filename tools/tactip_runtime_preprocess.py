#!/usr/bin/env python3
"""Shared online TacTip marker preprocessing for real-camera samplers.

The profile in this module is the fixed 640x480 optical geometry validated on
the camera_manual_tool2_20 batch.  Raw captures are never overwritten: each
saved tactile image gets a matching marker-preserving 256x256 model input in a
neighbouring ``tactip_preprocessed`` directory.
"""

from __future__ import annotations

import argparse
import atexit
import html
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from preprocess_tactip_markers import model_input_mask, process_color_image, processing_mask


PROFILE_NAME = "tactip_outer_ring_lattice_protected_v1"
REFERENCE_WIDTH_PX = 640
REFERENCE_HEIGHT_PX = 480
DEFAULT_CENTER_X_PX = 289.55555555555554
DEFAULT_CENTER_Y_PX = 254.63636363636363
DEFAULT_CROP_RADIUS_PX = 175
DEFAULT_MARKER_OUTER_RADIUS_PX = 136.5452879415405


@dataclass(frozen=True)
class TacTipPreprocessConfig:
    center_x_px: float = DEFAULT_CENTER_X_PX
    center_y_px: float = DEFAULT_CENTER_Y_PX
    crop_radius_px: float = DEFAULT_CROP_RADIUS_PX
    marker_outer_radius_px: float = DEFAULT_MARKER_OUTER_RADIUS_PX
    output_size: int = 256
    # ``live_cr3_gelsight_sampler`` can rotate and mirror frames before it
    # writes them. Keep the optical centre in that same final image frame.
    mirror: bool = False
    rotate: int = 0


def add_tactip_preprocess_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("TacTip marker preprocessing")
    group.add_argument(
        "--no-tactip-preprocess",
        action="store_true",
        help="Keep saving raw tactile images, but do not write the marker-focused preprocessing outputs.",
    )
    group.add_argument(
        "--tactip-preprocess-center",
        default="{:.8f},{:.8f}".format(DEFAULT_CENTER_X_PX, DEFAULT_CENTER_Y_PX),
        help="Reference optical center in 640x480 pixels as X,Y.",
    )
    group.add_argument(
        "--tactip-preprocess-crop-radius-px",
        type=float,
        default=DEFAULT_CROP_RADIUS_PX,
        help="Reference circular crop radius in pixels at 640x480.",
    )
    group.add_argument(
        "--tactip-preprocess-marker-outer-radius-px",
        type=float,
        default=DEFAULT_MARKER_OUTER_RADIUS_PX,
        help="Reference outer-marker radius in pixels at 640x480.",
    )
    group.add_argument(
        "--tactip-preprocess-output-size",
        type=int,
        default=256,
        help="Square resolution saved under tactip_preprocessed/model_input_256/.",
    )


def _parse_center(text: str) -> tuple[float, float]:
    values = [float(value) for value in str(text).replace(",", " ").split() if value]
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError("--tactip-preprocess-center must contain finite X,Y values")
    return float(values[0]), float(values[1])


def config_from_args(args: argparse.Namespace | Any) -> TacTipPreprocessConfig:
    default_center = "{:.8f},{:.8f}".format(DEFAULT_CENTER_X_PX, DEFAULT_CENTER_Y_PX)
    center_x, center_y = _parse_center(getattr(args, "tactip_preprocess_center", default_center))
    crop_radius = float(getattr(args, "tactip_preprocess_crop_radius_px", DEFAULT_CROP_RADIUS_PX))
    marker_outer = float(
        getattr(args, "tactip_preprocess_marker_outer_radius_px", DEFAULT_MARKER_OUTER_RADIUS_PX)
    )
    output_size = int(getattr(args, "tactip_preprocess_output_size", 256))
    mirror = bool(getattr(args, "mirror", False))
    rotate = int(getattr(args, "rotate", 0))
    if crop_radius < 32.0:
        raise ValueError("--tactip-preprocess-crop-radius-px must be at least 32")
    if marker_outer <= 0.0 or marker_outer >= crop_radius:
        raise ValueError("The marker outer radius must be positive and smaller than the crop radius")
    if output_size <= 0:
        raise ValueError("--tactip-preprocess-output-size must be positive")
    if rotate not in (0, 90, 180, 270):
        raise ValueError("Camera rotation must be one of 0, 90, 180, or 270 degrees")
    return TacTipPreprocessConfig(
        center_x,
        center_y,
        crop_radius,
        marker_outer,
        output_size,
        mirror,
        rotate,
    )


def _processing_args(output_size: int) -> argparse.Namespace:
    """Arguments matching the marker-preserving batch profile used on 20260729."""
    return SimpleNamespace(
        marker_percentile=96.0,
        background_sigma=5.0,
        clahe_clip_limit=2.5,
        min_marker_area=5,
        max_marker_area=180,
        ring_suppression="gray-clusters",
        ring_clusters=4,
        ring_marker_guard_px=8.0,
        ring_min_component_area=100,
        ring_dilate_px=5,
        ring_inpaint_radius=5.0,
        outer_rim_treatment="none",
        model_input_mask="outer-ring-protected",
        model_marker_neighborhood_radius_px=20.0,
        model_component_neighbor_min_px=6.0,
        model_component_neighbor_max_px=30.0,
        model_component_dilate_px=6,
        model_outer_ring_start_margin_px=3.5,
        model_outer_marker_protection_dilate_px=6,
        outer_rim_fade_start_margin_px=17.0,
        outer_rim_fade_end_margin_px=1.0,
        outer_rim_cap_percentile=90.0,
        outer_rim_roi_margin_px=30.0,
        flat_field_sigma=20.0,
        flat_field_level=82.0,
        output_size=int(output_size),
    )


class TacTipRuntimePreprocessor:
    """Write marker-preserving images alongside a sampler's raw captures.

    Preprocessing errors are recorded and reported, but never interrupt a
    physical sampling run.  The raw image remains the source of truth.
    """

    def __init__(self, output_dir: Path, config: TacTipPreprocessConfig) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.config = config
        self.args = _processing_args(config.output_size)
        self._lock = threading.Lock()
        self._closed = False
        self._records: dict[str, dict[str, object]] = {}
        self._geometry_cache: dict[tuple[int, int], tuple[np.ndarray, int, float, np.ndarray, np.ndarray]] = {}
        for name in ("gray", "ring_suppressed", "model_input", "model_input_256", "model_roi", "overlay"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self._load_existing_records()
        self._write_metadata()
        atexit.register(self.close)

    def _load_existing_records(self) -> None:
        manifest = self.output_dir / "samples.jsonl"
        if not manifest.is_file():
            return
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(record.get("name", ""))
            if name and record.get("status") == "ok":
                self._records[name] = record

    def _write_metadata(self) -> None:
        payload = {
            "schema": "tactip_runtime_preprocess.v1",
            "profile": PROFILE_NAME,
            "reference_frame_size_px": [REFERENCE_WIDTH_PX, REFERENCE_HEIGHT_PX],
            "configuration": asdict(self.config),
            "description": (
                "Gray-cluster outer-ring suppression followed by a marker-lattice-protected "
                "model ROI. Raw source images are intentionally kept unchanged."
            ),
        }
        (self.output_dir / "preprocess_metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _geometry_for(self, frame: np.ndarray) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        key = (height, width)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        # The live sampler applies its transform as resize -> rotation ->
        # horizontal mirror. Recover the resized frame dimensions so the fixed
        # 640x480 profile can be projected before applying that same transform.
        rotated_quarter_turn = self.config.rotate in (90, 270)
        source_width = int(height) if rotated_quarter_turn else int(width)
        source_height = int(width) if rotated_quarter_turn else int(height)
        scale_x = float(source_width) / float(REFERENCE_WIDTH_PX)
        scale_y = float(source_height) / float(REFERENCE_HEIGHT_PX)
        scale = min(scale_x, scale_y)
        center = np.asarray(
            [self.config.center_x_px * scale_x, self.config.center_y_px * scale_y], dtype=np.float64
        )
        if self.config.rotate == 90:
            center = np.asarray([float(source_height - 1) - center[1], center[0]], dtype=np.float64)
        elif self.config.rotate == 180:
            center = np.asarray(
                [float(source_width - 1) - center[0], float(source_height - 1) - center[1]],
                dtype=np.float64,
            )
        elif self.config.rotate == 270:
            center = np.asarray([center[1], float(source_width - 1) - center[0]], dtype=np.float64)
        if self.config.mirror:
            center[0] = float(width - 1) - center[0]
        radius = int(round(self.config.crop_radius_px * scale))
        radius = max(32, min(radius, max(32, int(min(height, width) / 2) - 1)))
        marker_outer_radius = float(self.config.marker_outer_radius_px * scale)
        marker_outer_radius = min(marker_outer_radius, float(radius) - 2.0)
        # The active profile uses a circle and derives the per-frame outer ROI
        # from detected marker components, so no separately estimated reference
        # marker positions are needed at capture time.
        empty_lattice = np.empty((0, 2), dtype=np.float64)
        support_mask = processing_mask(center, radius, empty_lattice, "circle", 0)
        fixed_model_mask = model_input_mask(
            center,
            radius,
            empty_lattice,
            support_mask,
            "outer-ring-protected",
            float(self.args.model_marker_neighborhood_radius_px),
        )
        cached = center, radius, marker_outer_radius, support_mask, fixed_model_mask
        self._geometry_cache[key] = cached
        return cached

    def _append_record(self, record: dict[str, object]) -> None:
        with (self.output_dir / "samples.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_error(self, name: str, raw_image: Path, error: Exception) -> None:
        record = {
            "status": "error",
            "name": name,
            "raw_image": os.path.relpath(raw_image, self.output_dir),
            "error": "{}: {}".format(type(error).__name__, error),
        }
        self._append_record(record)

    def process_and_save(self, frame: np.ndarray, raw_image: Path) -> Path | None:
        """Process a just-saved raw BGR frame and return its 256px output path."""
        raw_image = Path(raw_image).expanduser().resolve()
        name = raw_image.name
        try:
            with self._lock:
                center, radius, marker_outer_radius, support_mask, fixed_model_mask = self._geometry_for(frame)
                record, images = process_color_image(
                    frame,
                    center,
                    radius,
                    support_mask,
                    fixed_model_mask,
                    marker_outer_radius,
                    None,
                    self.args,
                    source=str(raw_image),
                )
                outputs = {
                    "gray": images["gray"],
                    "ring_suppressed": images["ring_suppressed"],
                    "model_input": images["model_input"],
                    "model_input_256": images["model_input_256"],
                    "model_roi": images["model_roi"],
                    "overlay": images["overlay"],
                }
                for directory, image in outputs.items():
                    path = self.output_dir / directory / name
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError("Could not write {}".format(path))
                record.update(
                    {
                        "status": "ok",
                        "name": name,
                        "raw_image": os.path.relpath(raw_image, self.output_dir),
                        "frame_size_px": [int(frame.shape[1]), int(frame.shape[0])],
                        "scaled_center_px": [float(center[0]), float(center[1])],
                        "scaled_crop_radius_px": int(radius),
                        "scaled_marker_outer_radius_px": float(marker_outer_radius),
                        "model_input_256": "model_input_256/{}".format(name),
                    }
                )
                self._records[name] = record
                self._append_record(record)
                return self.output_dir / "model_input_256" / name
        except Exception as exc:
            try:
                self._append_error(name, raw_image, exc)
            except Exception:
                # The raw capture has already been written. A reporting error
                # must not turn preprocessing into a physical-run failure.
                pass
            print(
                "WARNING: TacTip preprocessing skipped for {}: {}: {}".format(name, type(exc).__name__, exc),
                flush=True,
            )
            return None

    def _write_collection(self) -> Path:
        cards: list[str] = []
        for name in sorted(self._records):
            record = self._records[name]
            raw_image = html.escape(str(record["raw_image"]))
            safe_name = html.escape(name)
            marker_count = int(record.get("marker_count", 0))
            cards.append(
                """
<article class="card"><h2>{}</h2><p>{} detected markers</p><div class="images">
<figure><figcaption>Raw capture</figcaption><img src="{}"></figure>
<figure><figcaption>Ring-suppressed grayscale</figcaption><img src="ring_suppressed/{}"></figure>
<figure><figcaption>Final 256x256 model input</figcaption><img src="model_input_256/{}"></figure>
</div></article>""".format(safe_name, marker_count, raw_image, safe_name, safe_name)
            )
        page = """<!doctype html><html><head><meta charset="utf-8"><title>TacTip runtime preprocessing collection</title>
<style>body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232;position:sticky;top:0;z-index:1}h1{margin:0;font-size:22px}header p{margin:8px 0 0;color:#b7c5d6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:16px;padding:16px}.card{background:#182232;border:1px solid #33475f;border-radius:7px;padding:12px}h2{font-size:15px;margin:0}p{color:#b7c5d6;font-size:12px;margin:6px 0 10px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0;background:#0d141e;padding:6px}figcaption{font-size:11px;color:#cbd7e5;margin-bottom:5px;min-height:25px}img{width:100%;display:block;image-rendering:pixelated;background:#000}</style></head>
<body><header><h1>TacTip runtime preprocessing collection</h1><p>Raw captures are kept unchanged. The final column is the shared 256x256 marker-preserving model input.</p></header><main>""" + "\n".join(cards) + """</main></body></html>"""
        path = self.output_dir / "collection.html"
        path.write_text(page, encoding="utf-8")
        return path

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._records:
                self._write_collection()


def create_tactip_preprocessor(
    args: argparse.Namespace | Any,
    output_root: Path,
    directory_name: str = "tactip_preprocessed",
) -> TacTipRuntimePreprocessor | None:
    """Create the default online preprocessor unless the caller opted out."""
    if bool(getattr(args, "no_tactip_preprocess", False)):
        return None
    return TacTipRuntimePreprocessor(Path(output_root) / directory_name, config_from_args(args))
