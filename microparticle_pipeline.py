#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microparticle Counting & Tracking pipeline.

This module powers the command-line interface advertised in the README.  The
original implementation depended on NumPy and pandas primarily for type hints
and convenience wrappers around CSV/JSON handling.  Those imports made the
script unusable in lightweight environments where only OpenCV was available, as
simply executing ``python microparticle_pipeline.py --help`` would crash before
``argparse`` had a chance to show usage information.  To make the thesis code
usable out-of-the-box we now defer the heavy OpenCV import and rely solely on
standard-library helpers for bookkeeping.

Examples
--------
# From a video with Hough circles detector + annotated video out
python microparticle_pipeline.py   --input path/to/video.mp4   --detector hough   --output_dir outputs/exp1   --save_video   --hough_dp 1.2 --hough_min_dist 12 --hough_param1 120 --hough_param2 15 --hough_min_radius 3 --hough_max_radius 20

# From an image folder with contour detector
python microparticle_pipeline.py   --input path/to/images   --detector contour   --output_dir outputs/exp2   --contour_thresh 25 --morph_kernel 3

# Webcam stream (index 0) and live display (no disk writes)
python microparticle_pipeline.py --input 0 --detector hough --display
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - exercised indirectly via run_pipeline
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]
    CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - handled at runtime
    CV2_IMPORT_ERROR = None

# -------------------------
# I/O readers
# -------------------------

def require_cv2() -> Any:
    """Return the OpenCV module or raise a friendly error if missing."""

    if cv2 is None:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "OpenCV (cv2) is required to run this pipeline. Install it with "
            "`pip install opencv-python` or `pip install opencv-python-headless`."
        ) from CV2_IMPORT_ERROR
    return cv2


class FrameSource:
    def __iter__(self) -> Iterable[Tuple[int, Any]]:
        raise NotImplementedError

class VideoReader(FrameSource):
    def __init__(self, path_or_index):
        self.path_or_index = path_or_index

    def __iter__(self):
        cv2_local = require_cv2()
        if isinstance(self.path_or_index, int) or str(self.path_or_index).isdigit():
            cap = cv2_local.VideoCapture(int(self.path_or_index))
        else:
            cap = cv2_local.VideoCapture(str(self.path_or_index))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video/camera: {self.path_or_index}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        cap.release()

class ImageFolderReader(FrameSource):
    def __init__(self, folder: Path):
        self.paths = sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])
        if not self.paths:
            raise RuntimeError(f"No images found in folder: {folder}")

    def __iter__(self):
        cv2_local = require_cv2()
        for i, p in enumerate(self.paths):
            img = cv2_local.imread(str(p), cv2_local.IMREAD_COLOR)
            if img is None:
                continue
            yield i, img

def infer_source(input_arg: str) -> FrameSource:
    # numeric -> webcam index
    if input_arg.isdigit():
        return VideoReader(int(input_arg))
    p = Path(input_arg)
    if p.is_dir():
        return ImageFolderReader(p)
    return VideoReader(input_arg)

# -------------------------
# Detection modules
# -------------------------

@dataclass
class Detection:
    cx: float
    cy: float
    r: float
    score: float = 1.0

class HoughCircleDetector:
    def __init__(self, dp=1.2, min_dist=12, param1=120, param2=15, min_radius=3, max_radius=20, blur_ksize=5):
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.blur_ksize = blur_ksize

    def __call__(self, frame: Any) -> List[Detection]:
        cv2_local = require_cv2()
        gray = cv2_local.cvtColor(frame, cv2_local.COLOR_BGR2GRAY)
        if self.blur_ksize > 0:
            gray = cv2_local.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        circles = cv2_local.HoughCircles(gray, cv2_local.HOUGH_GRADIENT, dp=self.dp, minDist=self.min_dist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        dets: List[Detection] = []
        if circles is not None:
            for (x, y, r) in circles[0, :]:
                dets.append(Detection(cx=float(x), cy=float(y), r=float(r), score=1.0))
        return dets

class ContourDetector:
    def __init__(self, thresh=25, morph_kernel=3, min_area=8, max_area=1e6, adaptive=False):
        self.thresh = thresh
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.max_area = max_area
        self.adaptive = adaptive

    def __call__(self, frame: Any) -> List[Detection]:
        cv2_local = require_cv2()
        gray = cv2_local.cvtColor(frame, cv2_local.COLOR_BGR2GRAY)
        if self.adaptive:
            bw = cv2_local.adaptiveThreshold(gray, 255, cv2_local.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2_local.THRESH_BINARY_INV, 21, 2)
        else:
            _, bw = cv2_local.threshold(gray, self.thresh, 255, cv2_local.THRESH_BINARY_INV)
        if self.morph_kernel > 0:
            k = cv2_local.getStructuringElement(cv2_local.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            bw = cv2_local.morphologyEx(bw, cv2_local.MORPH_OPEN, k, iterations=1)
            bw = cv2_local.morphologyEx(bw, cv2_local.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2_local.findContours(bw, cv2_local.RETR_EXTERNAL, cv2_local.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        for c in cnts:
            a = cv2_local.contourArea(c)
            if a < self.min_area or a > self.max_area:
                continue
            (x, y), r = cv2_local.minEnclosingCircle(c)
            dets.append(Detection(cx=float(x), cy=float(y), r=float(r), score=1.0))
        return dets

def build_detector(args):
    if args.detector == "hough":
        return HoughCircleDetector(dp=args.hough_dp, min_dist=args.hough_min_dist, param1=args.hough_param1,
                                   param2=args.hough_param2, min_radius=args.hough_min_radius,
                                   max_radius=args.hough_max_radius, blur_ksize=args.hough_blur)
    elif args.detector == "contour":
        return ContourDetector(thresh=args.contour_thresh, morph_kernel=args.morph_kernel,
                               min_area=args.min_area, max_area=args.max_area, adaptive=args.adaptive_thresh)
    else:
        raise ValueError("Unknown detector")

# -------------------------
# Simple tracker (centroid-based with aging)
# -------------------------

@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    r: float
    age: int = 0          # frames since last seen
    hits: int = 1         # how many times matched
    alive: bool = True

class CentroidTracker:
    def __init__(self, max_age=10, dist_thresh=25.0):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _distance(self, a: Track, b: Detection) -> float:
        return float(math.hypot(a.cx - b.cx, a.cy - b.cy))

    def update(self, detections: List[Detection]) -> List[Track]:
        # Aging
        for tr in self.tracks.values():
            tr.age += 1

        # Greedy matching by nearest distance
        unmatched_dets = list(range(len(detections)))
        pairs = []
        for tid, tr in self.tracks.items():
            for j in unmatched_dets:
                d = self._distance(tr, detections[j])
                pairs.append((d, tid, j))
        pairs.sort(key=lambda x: x[0])

        assigned_tr, assigned_det = set(), set()
        for d, tid, j in pairs:
            if tid in assigned_tr or j in assigned_det:
                continue
            if d <= self.dist_thresh:
                tr = self.tracks[tid]
                det = detections[j]
                tr.cx, tr.cy, tr.r = det.cx, det.cy, det.r
                tr.age = 0
                tr.hits += 1
                assigned_tr.add(tid)
                assigned_det.add(j)

        # New tracks for unmatched detections
        for j in range(len(detections)):
            if j in assigned_det:
                continue
            det = detections[j]
            tr = Track(track_id=self.next_id, cx=det.cx, cy=det.cy, r=det.r)
            self.tracks[self.next_id] = tr
            self.next_id += 1

        # Remove old tracks
        to_del = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return list(self.tracks.values())

# -------------------------
# Drawing helpers
# -------------------------

def draw_annotations(frame: Any, tracks: List[Track], color=(0, 255, 0)) -> Any:
    cv2_local = require_cv2()
    vis = frame.copy()
    for tr in tracks:
        cv2_local.circle(vis, (int(tr.cx), int(tr.cy)), int(max(2, tr.r)), color, 2)
        cv2_local.putText(vis, f"ID {tr.track_id}", (int(tr.cx)+5, int(tr.cy)-5),
                          cv2_local.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2_local.LINE_AA)
    cv2_local.putText(vis, f"Count (active): {len(tracks)}", (10, 24),
                      cv2_local.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2_local.LINE_AA)
    return vis

# -------------------------
# Persistence helpers (standard library only)
# -------------------------

def _resolve_resize(frame: Any, width: int, height: int) -> Tuple[int, int]:
    """Return a target (width, height) keeping aspect ratio when one dim is zero."""

    h, w = frame.shape[:2]
    if width > 0 and height > 0:
        return width, height
    if width > 0:
        scale = width / float(w)
        return width, max(1, int(round(h * scale)))
    if height > 0:
        scale = height / float(h)
        return max(1, int(round(w * scale))), height
    return w, h


def _write_counts(csv_path: Path, rows: List[Dict[str, int]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["frame", "count_active"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_summary(rows: List[Dict[str, int]], unique_ids_seen, args) -> Dict[str, Any]:
    frames_processed = len(rows)
    avg_active = (
        sum(row["count_active"] for row in rows) / frames_processed if frames_processed else 0.0
    )
    return {
        "frames_processed": int(frames_processed),
        "total_unique_ids": int(len(unique_ids_seen)),
        "avg_active_per_frame": float(avg_active),
        "detector": args.detector,
        "params": vars(args),
    }

# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(args):
    cv2_local = require_cv2()
    source = infer_source(args.input)
    detector = build_detector(args)
    tracker = CentroidTracker(max_age=args.max_age, dist_thresh=args.track_dist)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "counts.csv"
    summary_path = out_dir / "summary.json"

    video_writer: Optional[Dict[str, Any]] = None
    if args.save_video:
        fourcc = cv2_local.VideoWriter_fourcc(*"mp4v")
        video_writer = {"fourcc": fourcc, "obj": None}

    rows: List[Dict[str, int]] = []
    unique_ids_seen = set()
    first_size: Optional[Tuple[int, int]] = None

    for idx, frame in source:
        if args.width > 0 or args.height > 0:
            target_w, target_h = _resolve_resize(frame, args.width, args.height)
            frame = cv2_local.resize(frame, (target_w, target_h), interpolation=cv2_local.INTER_LINEAR)

        if first_size is None:
            first_size = (frame.shape[1], frame.shape[0])
            if args.save_video and video_writer is not None:
                w, h = first_size
                fps = args.fps if args.fps > 0 else 25
                vw = cv2_local.VideoWriter(str(out_dir / "annotated.mp4"), video_writer["fourcc"], fps, (w, h))
                video_writer["obj"] = vw

        dets = detector(frame)
        tracks = tracker.update(dets)

        for tr in tracks:
            unique_ids_seen.add(tr.track_id)

        count_active = len(tracks)
        rows.append({"frame": idx, "count_active": count_active})

        vis = draw_annotations(frame, tracks)
        if args.display:
            cv2_local.imshow("Microparticle Counting", vis)
            if cv2_local.waitKey(1) & 0xFF == 27:  # ESC
                break
        if args.save_video and video_writer is not None and video_writer["obj"] is not None:
            video_writer["obj"].write(vis)
        if args.save_frames:
            cv2_local.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), vis)

    if args.save_video and video_writer is not None and video_writer["obj"] is not None:
        video_writer["obj"].release()
    if cv2 is not None:  # pragma: no cover - depends on environment
        cv2_local.destroyAllWindows()

    _write_counts(csv_path, rows)
    summary = _build_summary(rows, unique_ids_seen, args)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-frame counts to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    if args.save_video:
        print(f"Annotated video: {out_dir/'annotated.mp4'}")
    if args.save_frames:
        print(f"Annotated frames in: {out_dir}")

# -------------------------
# CLI
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Microparticle Counting & Tracking Pipeline (OpenCV)")

    # Input / Output
    p.add_argument("--input", required=True, help="Path to video, image folder, or webcam index (e.g., 0)")
    p.add_argument("--output_dir", default="outputs/run1", help="Directory to save outputs")
    p.add_argument("--display", action="store_true", help="Show live annotated window")
    p.add_argument("--save_video", action="store_true", help="Save annotated MP4 video")
    p.add_argument("--save_frames", action="store_true", help="Save annotated frames as images")
    p.add_argument("--width", type=int, default=0, help="Resize output width (0 to keep original)")
    p.add_argument("--height", type=int, default=0, help="Resize output height (0 to keep original)")
    p.add_argument("--fps", type=int, default=0, help="Override FPS for saved video (0=auto)")

    # Detector
    p.add_argument("--detector", choices=["hough", "contour"], default="hough", help="Detection method")

    # Hough params
    p.add_argument("--hough_dp", type=float, default=1.2)
    p.add_argument("--hough_min_dist", type=float, default=12)
    p.add_argument("--hough_param1", type=float, default=120)
    p.add_argument("--hough_param2", type=float, default=15)
    p.add_argument("--hough_min_radius", type=int, default=3)
    p.add_argument("--hough_max_radius", type=int, default=20)
    p.add_argument("--hough_blur", type=int, default=5, help="Gaussian blur kernel size (odd number)")

    # Contour params
    p.add_argument("--contour_thresh", type=int, default=25, help="Binary threshold for THRESH_BINARY_INV")
    p.add_argument("--morph_kernel", type=int, default=3, help="Morphology kernel size (ellipse)")
    p.add_argument("--min_area", type=float, default=8.0, help="Min contour area")
    p.add_argument("--max_area", type=float, default=1e6, help="Max contour area")
    p.add_argument("--adaptive_thresh", action="store_true", help="Use adaptive thresholding")

    # Tracker
    p.add_argument("--max_age", type=int, default=10, help="Frames a track can be unseen before deletion")
    p.add_argument("--track_dist", type=float, default=25.0, help="Max centroid distance to match detection->track")

    return p

def main():
    args = build_argparser().parse_args()
    try:
        run_pipeline(args)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - CLI convenience
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
