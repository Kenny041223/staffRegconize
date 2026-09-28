"""Sample a video for the reference tag and report raw model scores.

This diagnostic scans full frames. Use identify_staff.py for person crops,
evidence aggregation and staff frame exports. High scores need visual review.

Usage:
    python src/scan_for_tag.py [video_path] [reference_tag_path] [--stride N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tag_match import DEFAULT_TAG_MODEL, TagMatcher

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT.parent / "sample.mp4"
DEFAULT_REFERENCE = ROOT / "assets" / "reference_1.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", nargs="?", default=str(DEFAULT_VIDEO))
    parser.add_argument("reference_path", nargs="?", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--stride", type=int, default=25, help="Check every Nth frame; default 25. Use 1 for an exhaustive scan.")
    parser.add_argument("--tag-model", default=DEFAULT_TAG_MODEL)
    parser.add_argument("--reference-box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--candidate-threshold", type=float, default=0.65)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15, help="How many top matches to report.")
    args = parser.parse_args()
    if args.stride < 1 or args.top_k < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--stride, --top-k and --max-frames must be positive")
    if not 0 <= args.candidate_threshold <= 1:
        parser.error("--candidate-threshold must be in [0, 1]")
    return args


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {args.video_path}")
    matcher = TagMatcher(args.reference_path, device=args.device, model_name=args.tag_model,
                         reference_box=args.reference_box, min_similarity=args.candidate_threshold)
    print(f"Tag model: {matcher.model_name}; device: {matcher.device}; scores: raw sigmoid")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    results = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            score = matcher.best_score(frame)
            results.append((frame_idx, score))
            if frame_idx % 100 == 0:
                print(f"  ...frame {frame_idx}/{total}", flush=True)
        frame_idx += 1
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
    cap.release()

    results.sort(key=lambda r: -r[1])
    print(f"\nTop {args.top_k} frames by tag-match score:")
    for idx, score in results[: args.top_k]:
        print(f"  frame {idx:5d} (t={idx / fps:5.1f}s)  score={score:.4f}")


if __name__ == "__main__":
    main()
