#!/usr/bin/env python3
"""Run the unmodified VSP CvBlobDetector on saved TacTip frames."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vsp-root", type=Path, default=Path("/Users/vincent/Downloads/vsp")
    )
    return parser.parse_args()


def add_heading(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (12, 14, 17), -1)
    cv2.putText(
        output,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def depth_from_name(path: Path) -> float | None:
    match = re.search(r"_capture_([0-9]+(?:\.[0-9]+)?)mm", path.stem)
    return float(match.group(1)) if match else None


def build_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = []
    for row in rows:
        cards.append(
            "<article>"
            f"<h3>{row['index']:02d}. {html.escape(row['frame'])}</h3>"
            f"<p>Depth: {row['depth_mm']:.2f} mm | detected: "
            f"<b>{row['marker_count']}</b> / 331</p>"
            f"<a href=\"{html.escape(row['overlay'])}\">"
            f"<img loading=\"lazy\" src=\"{html.escape(row['overlay'])}\"></a>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Official VSP detector results</title>
<style>
body {{ margin:0; background:#0c1118; color:#eef3f8; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1400px,96vw); margin:auto; padding:24px 0 60px; }}
h1 {{ margin:0; }} p {{ color:#b8c4d1; }} code {{ color:#8fc8ff; }}
article {{ background:#141c27; border:1px solid #2c3949; border-radius:6px; padding:12px; margin:18px 0; }}
article h3 {{ margin:0; }} img {{ width:100%; height:auto; display:block; margin-top:10px; }}
</style></head><body><main>
<h1>Official VSP CvBlobDetector, afternoon captures</h1>
<p>Unmodified detector class and official example parameters. Input conversion is only BGR to grayscale. No tuning, marker repair, forced count, lattice constraint, or other postprocessing was applied.</p>
<p>VSP commit: <code>{summary['vsp_commit']}</code>. Frames: {summary['frames']}.
Detected range: {summary['count_min']}-{summary['count_max']}; mean: {summary['count_mean']:.2f};
mean absolute count error from 331: {summary['count_mae']:.2f}.</p>
<a href="official_vsp_overview.png"><img src="official_vsp_overview.png"></a>
{''.join(cards)}
</main></body></html>"""


def main() -> int:
    args = parse_args()
    detector_source = args.vsp_root.resolve() / "vsp" / "detector.py"
    if not detector_source.is_file():
        raise FileNotFoundError(detector_source)
    sys.path.insert(0, str(args.vsp_root.resolve()))
    from vsp.detector import CvBlobDetector  # type: ignore[import-not-found]

    # Copied verbatim from the repository's examples/processor_test.py.
    detector = CvBlobDetector(
        min_threshold=31.23,
        max_threshold=207.05,
        filter_by_color=True,
        blob_color=255,
        filter_by_area=True,
        min_area=17.05,
        max_area=135.46,
        filter_by_circularity=True,
        min_circularity=0.62,
        filter_by_inertia=True,
        min_inertia_ratio=0.27,
        filter_by_convexity=True,
        min_convexity=0.60,
    )

    output_dir = args.output_dir.resolve()
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[str, Path]] = []
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        frames.extend(
            (run_dir.name, path)
            for path in sorted((run_dir / "frames").glob("*_capture_*.png"))
        )
    if not frames:
        raise FileNotFoundError("No *_capture_*.png frames found")

    rows: list[dict[str, Any]] = []
    previews: list[tuple[float, np.ndarray, str, int]] = []
    for index, (run_name, path) in enumerate(frames, start=1):
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"Could not read {path}")
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        points = np.asarray([item.point for item in keypoints], dtype=float).reshape(-1, 2)
        sizes = np.asarray([item.size for item in keypoints], dtype=float)
        left = add_heading(raw, "Raw saved frame")
        right = add_heading(raw, f"Official VSP CvBlobDetector: {len(keypoints)} markers")
        for (x, y), size in zip(points, sizes):
            radius = max(3, min(9, round(float(size) / 2.0)))
            cv2.circle(right, (round(x), round(y)), radius, (0, 0, 255), 2, cv2.LINE_AA)
        comparison = np.hstack((left, right))
        relative_overlay = Path("overlays") / f"{run_name}__{path.stem}.jpg"
        cv2.imwrite(
            str(output_dir / relative_overlay),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        depth_mm = depth_from_name(path)
        rows.append(
            {
                "index": index,
                "run": run_name,
                "frame": path.name,
                "source": str(path),
                "depth_mm": depth_mm,
                "marker_count": len(keypoints),
                "points_xy": points.tolist(),
                "sizes": sizes.tolist(),
                "overlay": str(relative_overlay),
            }
        )
        if depth_mm is not None:
            previews.append((depth_mm, comparison, path.name, len(keypoints)))

    counts = np.asarray([row["marker_count"] for row in rows])
    commit = subprocess.check_output(
        ["git", "-C", str(args.vsp_root.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "implementation": "unmodified vsp.detector.CvBlobDetector",
        "parameters_source": "vsp/examples/processor_test.py",
        "postprocessing": "none",
        "expected_markers": 331,
        "frames": len(rows),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_mean": float(counts.mean()),
        "count_median": float(np.median(counts)),
        "count_mae": float(np.abs(counts - 331).mean()),
        "exact_331_frames": int(np.count_nonzero(counts == 331)),
        "vsp_commit": commit,
        "run_dirs": [str(path.resolve()) for path in args.run_dir],
    }
    (output_dir / "official_vsp_detections.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=2), encoding="utf-8"
    )

    targets = (1.1, 2.1, 3.1, 4.3, 5.3, 5.9)
    chosen: list[tuple[float, np.ndarray, str, int]] = []
    used: set[str] = set()
    for target in targets:
        choice = min(
            (item for item in previews if item[2] not in used),
            key=lambda item: abs(item[0] - target),
        )
        chosen.append(choice)
        used.add(choice[2])
    tiles = []
    for depth_mm, comparison, name, count in chosen:
        tile = cv2.resize(comparison, (640, 240), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 205), (640, 240), (12, 14, 17), -1)
        cv2.putText(
            tile,
            f"{name} | depth={depth_mm:.2f} mm | detected={count}",
            (8, 229),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    overview = np.vstack(
        (np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6]))
    )
    cv2.imwrite(str(output_dir / "official_vsp_overview.png"), overview)
    (output_dir / "official_vsp_results.html").write_text(
        build_html(rows, summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(output_dir / "official_vsp_results.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
