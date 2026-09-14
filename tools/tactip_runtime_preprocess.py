#!/usr/bin/env python3
"""Shared online Hough/chromatic preprocessing for TacTip samplers.

Raw captures remain untouched. Each accepted frame gets a strict 0/255 marker
image in ``tactip_preprocessed/model_input_256``. A native-resolution marker
map is also written for the visual-contact optical-flow detector, whose
coordinates must not change when the model crop moves between frames.
"""

from __future__ import annotations

import argparse
import atexit
import html
import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2

from tactip_hough_chromatic import (
    EXPECTED_MARKERS,
    HoughChromaticConfig,
    MarkerDetectionError,
    config_dict,
    process_frame,
)


PROFILE_NAME = "tactip_hough_chromatic_331_v2"


def add_tactip_preprocess_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("TacTip marker preprocessing")
    group.add_argument(
        "--no-tactip-preprocess",
        action="store_true",
        help="Save raw tactile images without Hough/chromatic marker preprocessing.",
    )
    group.add_argument(
        "--tactip-preprocess-output-size",
        type=int,
        default=256,
        help="Output resolution saved under tactip_preprocessed/model_input_256/.",
    )
    group.add_argument(
        "--tactip-preprocess-expected-markers",
        type=int,
        default=EXPECTED_MARKERS,
        help="Exact connected-marker count required for accepting a frame.",
    )
    group.add_argument(
        "--tactip-preprocess-hough-vote-threshold",
        type=float,
        default=13.0,
        help="Starting Hough accumulator threshold; nearby values are tried automatically.",
    )
    group.add_argument(
        "--tactip-preprocess-blue-yellow-threshold",
        type=float,
        default=20.0,
        help="Minimum median 2B-G-R score used to reject yellow/orange glare circles.",
    )
    group.add_argument(
        "--tactip-preprocess-render-radius-scale",
        type=float,
        default=0.90,
        help="Scale applied to each measured Hough radius before binary rendering.",
    )
    # Accept old commands without silently reusing their obsolete geometry.
    group.add_argument("--tactip-preprocess-center", default="", help=argparse.SUPPRESS)
    group.add_argument("--tactip-preprocess-crop-radius-px", type=float, default=0.0, help=argparse.SUPPRESS)
    group.add_argument(
        "--tactip-preprocess-marker-outer-radius-px",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )


def config_from_args(args: argparse.Namespace | Any) -> HoughChromaticConfig:
    config = HoughChromaticConfig(
        expected_markers=int(
            getattr(args, "tactip_preprocess_expected_markers", EXPECTED_MARKERS)
        ),
        hough_vote_threshold=float(
            getattr(args, "tactip_preprocess_hough_vote_threshold", 13.0)
        ),
        minimum_blue_yellow_score=float(
            getattr(args, "tactip_preprocess_blue_yellow_threshold", 20.0)
        ),
        render_radius_scale=float(
            getattr(args, "tactip_preprocess_render_radius_scale", 0.90)
        ),
        output_size=int(getattr(args, "tactip_preprocess_output_size", 256)),
    )
    config.validate()
    return config


class TacTipRuntimePreprocessor:
    """Write validated marker binaries beside a sampler's raw captures."""

    def __init__(self, output_dir: Path, config: HoughChromaticConfig) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.config = config
        self._lock = threading.Lock()
        self._closed = False
        self._records: dict[str, dict[str, Any]] = {}
        for name in (
            "contact_binary",
            # Compatibility outputs consumed by auto_cr3_visual_contact_search.
            # They deliberately use the native-resolution marker map instead of
            # the variable Hough crop, so LK optical flow stays in camera pixels.
            "gray",
            "model_roi",
            "model_input",
            "raw_roi",
            "gray_roi",
            "blue_yellow_score_roi",
            "overlay",
            "binary",
            "model_input_256",
        ):
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
            "schema": "tactip_runtime_preprocess.v2",
            "profile": PROFILE_NAME,
            "configuration": config_dict(self.config),
            "description": (
                "Hough circular-boundary detection followed by median 2B-G-R glare rejection. "
                "Raw captures remain unchanged; accepted outputs contain only values 0 and 255. "
                "Native grayscale frames and fixed-coordinate Hough contact masks are provided "
                "for visual-contact flow and rest-stop texture verification."
            ),
        }
        (self.output_dir / "preprocess_metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _append_record(self, record: dict[str, Any]) -> None:
        with (self.output_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def process_and_save(self, frame: Any, raw_image: Path) -> Path | None:
        """Process a just-saved BGR frame and return its validated model input."""
        raw_image = Path(raw_image).expanduser().resolve()
        name = raw_image.stem + ".png"
        try:
            with self._lock:
                record, images, _markers = process_frame(frame, self.config)
                contact_binary = images["contact_binary"]
                native_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                outputs = {
                    "contact_binary": contact_binary,
                    # Keep intensity for optical flow and texture checks.  The matching
                    # model_roi stays binary and fixes the marker-safe support region.
                    "gray": native_gray,
                    "model_roi": contact_binary,
                    "model_input": images["model_input_256"],
                    "raw_roi": images["raw_roi"],
                    "gray_roi": images["gray_roi"],
                    "blue_yellow_score_roi": images["blue_yellow_score_roi"],
                    "overlay": images["overlay_roi"],
                    "binary": images["binary_roi"],
                    "model_input_256": images["model_input_256"],
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
                        "contact_binary": "contact_binary/{}".format(name),
                        "motion_gray": "gray/{}".format(name),
                        "model_roi": "model_roi/{}".format(name),
                        "model_input": "model_input/{}".format(name),
                        "model_input_256": "model_input_256/{}".format(name),
                    }
                )
                self._records[name] = record
                self._append_record(record)
                return self.output_dir / "model_input_256" / name
        except Exception as exc:
            error_record: dict[str, Any] = {
                "status": "error",
                "name": name,
                "raw_image": os.path.relpath(raw_image, self.output_dir),
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            if isinstance(exc, MarkerDetectionError):
                error_record["diagnostics"] = exc.diagnostics
            try:
                self._append_record(error_record)
            except Exception:
                pass
            print(
                "WARNING: TacTip preprocessing rejected {}: {}".format(name, error_record["error"]),
                flush=True,
            )
            return None

    def _write_collection(self) -> Path:
        cards: list[str] = []
        for name in sorted(self._records):
            record = self._records[name]
            raw_image = html.escape(str(record["raw_image"]))
            safe_name = html.escape(name)
            cards.append(
                """<article class="card"><h2>{}</h2><p>{} markers; {} glare circles rejected</p><div class="images">
<figure><figcaption>Raw capture</figcaption><img src="{}"></figure>
<figure><figcaption>Detection overlay</figcaption><img src="overlay/{}"></figure>
<figure><figcaption>Strict 256x256 model input</figcaption><img src="model_input_256/{}"></figure>
</div></article>""".format(
                    safe_name,
                    int(record.get("marker_count", 0)),
                    int(record.get("rejected_glare_count", 0)),
                    raw_image,
                    safe_name,
                    safe_name,
                )
            )
        page = """<!doctype html><html><head><meta charset="utf-8"><title>TacTip runtime preprocessing</title><style>body{margin:0;background:#111722;color:#eef3f8;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182232;position:sticky;top:0}h1{margin:0;font-size:22px}header p,p{color:#b7c5d6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px;padding:16px}.card{background:#182232;border:1px solid #33475f;border-radius:7px;padding:12px}h2{font-size:15px;margin:0}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0;background:#0d141e;padding:6px}figcaption{font-size:11px;color:#cbd7e5;margin-bottom:5px}img{width:100%;display:block;background:#000}</style></head><body><header><h1>Hough + chromatic runtime preprocessing</h1><p>Raw captures remain unchanged. Green circles are accepted marker boundaries; gold-reflection circles are rejected by colour.</p></header><main>__CARDS__</main></body></html>""".replace(
            "__CARDS__", "\n".join(cards)
        )
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
    if bool(getattr(args, "no_tactip_preprocess", False)):
        return None
    return TacTipRuntimePreprocessor(Path(output_root) / directory_name, config_from_args(args))
