#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microparticle Counting & Tracking - Modern Pipeline (OpenCV + Python)
Author: Martin Badrous (repo modernization)

Features
- Clean CLI with argparse
- Flexible inputs: video file, webcam, or image folder
- Pluggable detectors: Hough circles or contour-based
- Simple online tracker with stable IDs (centroid tracking with aging)
- CSV logging of per-frame counts + summary JSON
- Optional annotated video export (MP4/H264) and frame snapshots
- Reproducible, dependency-light (opencv-python, numpy, pandas)

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
import json
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict

import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass

# -------------------------
# I/O readers
# -------------------------

class FrameSource:
    def __iter__(self) -> Iterable[Tuple[int, np.ndarray]]:
        raise NotImplementedError

class VideoReader(FrameSource):
    def __init__(self, path_or_index):
        self.path_or_index = path_or_index

    def __iter__(self):
        if isinstance(self.path_or_index, int) or str(self.path_or_index).isdigit():
            cap = cv2.VideoCapture(int(self.path_or_index))
        else:
            cap = cv2.VideoCapture(str(self.path_or_index))
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
        for i, p in enumerate(self.paths):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
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

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 0:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=self.dp, minDist=self.min_dist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        dets: List[Detection] = []
        if circles is not None:
            circles = np.uint16(np.around(circles))[0, :]
            for (x, y, r) in circles:
                dets.append(Detection(cx=float(x), cy=float(y), r=float(r), score=1.0))
        return dets

class ContourDetector:
    def __init__(self, thresh=25, morph_kernel=3, min_area=8, max_area=1e6, adaptive=False):
        self.thresh = thresh
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.max_area = max_area
        self.adaptive = adaptive

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.adaptive:
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 2)
        else:
            _, bw = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)
        if self.morph_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
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
        return float(np.hypot(a.cx - b.cx, a.cy - b.cy))

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

def draw_annotations(frame: np.ndarray, tracks: List[Track], color=(0, 255, 0)) -> np.ndarray:
    vis = frame.copy()
    for tr in tracks:
        cv2.circle(vis, (int(tr.cx), int(tr.cy)), int(max(2, tr.r)), color, 2)
        cv2.putText(vis, f"ID {tr.track_id}", (int(tr.cx)+5, int(tr.cy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Count (active): {len(tracks)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return vis

# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(args):
    source = infer_source(args.input)
    detector = build_detector(args)
    tracker = CentroidTracker(max_age=args.max_age, dist_thresh=args.track_dist)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "counts.csv"
    summary_path = out_dir / "summary.json"

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = None
        # we'll determine size dynamically from first frame if not provided
        writer = {"fourcc": fourcc, "obj": None}

    rows = []
    unique_ids_seen = set()
    first_size = None

    for idx, frame in source:
        if args.width > 0 and args.height > 0:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        if first_size is None:
            first_size = (frame.shape[1], frame.shape[0])
            if args.save_video:
                w, h = first_size
                vw = cv2.VideoWriter(str(out_dir / "annotated.mp4"), writer["fourcc"], args.fps if args.fps>0 else 25, (w, h))
                writer["obj"] = vw

        dets = detector(frame)
        tracks = tracker.update(dets)

        for tr in tracks:
            unique_ids_seen.add(tr.track_id)

        count_active = len(tracks)
        rows.append({"frame": idx, "count_active": count_active})

        vis = draw_annotations(frame, tracks)
        if args.display:
            cv2.imshow("Microparticle Counting", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        if args.save_video and writer["obj"] is not None:
            writer["obj"].write(vis)
        if args.save_frames:
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), vis)

    if args.save_video and writer["obj"] is not None:
        writer["obj"].release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    summary = {
        "frames_processed": int(len(rows)),
        "total_unique_ids": int(len(unique_ids_seen)),
        "avg_active_per_frame": float(df["count_active"].mean() if len(df) else 0.0),
        "detector": args.detector,
        "params": vars(args),
    }
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
    run_pipeline(args)

if __name__ == "__main__":
    main()
