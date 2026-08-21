#!/usr/bin/env python3
"""Batch TacTip marker preprocessing using Hough boundaries and colour rejection.

This replaces the legacy illumination-ring/percentile pipeline.  Circular
boundaries provide measured marker centres and radii; the median 2B-G-R score
rejects yellow/orange reflections.  Accepted markers are rendered as a strict
0/255 image and validated before and after conversion to model resolution.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="One image or a directory of images.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.png", help="Glob used when --input is a directory.")
    parser.add_argument("--expected-markers", type=int, default=EXPECTED_MARKERS)
    parser.add_argument("--hough-dp", type=float, default=1.0)
    parser.add_argument("--minimum-centre-distance-px", type=float, default=14.0)
    parser.add_argument("--canny-high-threshold", type=float, default=80.0)
    parser.add_argument("--hough-vote-threshold", type=float, default=13.0)
    parser.add_argument("--hough-vote-search-radius", type=int, default=3)
    parser.add_argument("--minimum-radius-px", type=float, default=4.0)
    parser.add_argument("--maximum-radius-px", type=float, default=11.0)
    parser.add_argument("--minimum-blue-yellow-score", type=float, default=20.0)
    parser.add_argument("--render-radius-scale", type=float, default=0.90)
    parser.add_argument("--render-radius-offset-px", type=float, default=0.0)
    parser.add_argument("--roi-padding-px", type=float, default=24.0)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--downsample-threshold", type=int, default=80)
    parser.add_argument("--preview-limit", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(input_path: Path, pattern: str) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    if not paths:
        raise FileNotFoundError("No images matched {} in {}".format(pattern, path))
    return paths


def config_from_args(args: argparse.Namespace) -> HoughChromaticConfig:
    return HoughChromaticConfig(
        expected_markers=int(args.expected_markers),
        hough_dp=float(args.hough_dp),
        minimum_centre_distance_px=float(args.minimum_centre_distance_px),
        canny_high_threshold=float(args.canny_high_threshold),
        hough_vote_threshold=float(args.hough_vote_threshold),
        hough_vote_search_radius=int(args.hough_vote_search_radius),
        minimum_radius_px=float(args.minimum_radius_px),
        maximum_radius_px=float(args.maximum_radius_px),
        minimum_blue_yellow_score=float(args.minimum_blue_yellow_score),
        render_radius_scale=float(args.render_radius_scale),
        render_radius_offset_px=float(args.render_radius_offset_px),
        roi_padding_px=float(args.roi_padding_px),
        output_size=int(args.output_size),
        downsample_threshold=int(args.downsample_threshold),
    )


def _write_preview(output_dir: Path, records: list[dict[str, Any]], limit: int) -> Path:
    cards: list[str] = []
    for record in records[: max(0, int(limit))]:
        if record["status"] != "ok":
            cards.append(
                '<article class="card error"><h2>{}</h2><p>{}</p></article>'.format(
                    html.escape(str(record["source_name"])),
                    html.escape(str(record["error"])),
                )
            )
            continue
        name = html.escape(str(record["output_name"]))
        cards.append(
            """<article class="card"><h2>{}</h2><p>{} markers; {} glare candidates rejected</p><div class="images">
<figure><figcaption>Raw ROI</figcaption><img src="raw_roi/{}"></figure>
<figure><figcaption>Green accepted / red rejected</figcaption><img src="overlay/{}"></figure>
<figure><figcaption>Strict 256x256 binary</figcaption><img src="model_input_256/{}"></figure>
</div></article>""".format(
                html.escape(str(record["source_name"])),
                int(record["marker_count"]),
                int(record["rejected_glare_count"]),
                name,
                name,
                name,
            )
        )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>TacTip preprocessing</title><style>body{margin:0;background:#111722;color:#edf3fa;font-family:Arial,sans-serif}header{padding:20px 28px;background:#182333;position:sticky;top:0}h1{margin:0;font-size:22px}header p,p{color:#b8c7d8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px;padding:16px}.card{background:#182333;border:1px solid #39506a;border-radius:7px;padding:12px}.card h2{font-size:15px;margin:0}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0;background:#0d141e;padding:6px}figcaption{font-size:11px;color:#cbd7e5;margin-bottom:5px}img{width:100%;display:block;background:#000}.error{border-color:#a85858}</style></head><body><header><h1>Hough + chromatic TacTip preprocessing</h1><p>Marker boundaries are measured by Hough voting. Median 2B-G-R rejects gold glare; brightness peaks are not used.</p></header><main class="grid">__CARDS__</main></body></html>""".replace(
        "__CARDS__", "\n".join(cards)
    )
    path = output_dir / "preview.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    config.validate()
    inputs = image_paths(args.input, args.glob)
    output_dir = args.output_dir.expanduser().resolve()
    directories = {
        "raw_roi": output_dir / "raw_roi",
        "gray_roi": output_dir / "gray_roi",
        "blue_yellow_score_roi": output_dir / "blue_yellow_score_roi",
        "overlay_roi": output_dir / "overlay",
        "binary_roi": output_dir / "binary",
        "model_input_256": output_dir / "model_input_256",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    for source in inputs:
        output_name = source.stem + ".png"
        existing = directories["model_input_256"] / output_name
        if existing.exists() and not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite or choose a new output directory".format(existing))
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            records.append(
                {
                    "status": "error",
                    "source_name": source.name,
                    "output_name": output_name,
                    "error": "OpenCV could not read the image",
                }
            )
            continue
        try:
            record, images, markers = process_frame(image, config)
            for key, directory in directories.items():
                if not cv2.imwrite(str(directory / output_name), images[key]):
                    raise RuntimeError("Could not write {}".format(directory / output_name))
            record.update(
                {
                    "status": "ok",
                    "source": str(source),
                    "source_name": source.name,
                    "output_name": output_name,
                }
            )
            records.append(record)
            for marker in markers:
                marker_rows.append({"source_name": source.name, **marker})
        except MarkerDetectionError as exc:
            records.append(
                {
                    "status": "error",
                    "source": str(source),
                    "source_name": source.name,
                    "output_name": output_name,
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                }
            )

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    marker_fields = (
        "source_name",
        "marker_id",
        "x_px",
        "y_px",
        "hough_radius_px",
        "render_radius_px",
        "blue_yellow_score",
    )
    with (output_dir / "detected_markers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=marker_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(marker_rows)
    ok_count = sum(record["status"] == "ok" for record in records)
    failure_count = len(records) - ok_count
    metadata = {
        "schema": "tactip_hough_chromatic_batch.v1",
        "configuration": config_dict(config),
        "input_count": len(inputs),
        "accepted_count": ok_count,
        "failed_count": failure_count,
        "raw_images_overwritten": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    preview = _write_preview(output_dir, records, args.preview_limit)
    print("Accepted {}/{} images; preview={}".format(ok_count, len(inputs), preview))
    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
