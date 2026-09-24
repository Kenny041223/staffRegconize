"""Detect + track every person in a video and save an annotated output video.

No staff/nametag logic here — this only proves out person detection + tracking
(the tracker attempts to keep each person's ID across frames, with boxes and trails).

Usage:
    python src/track_people.py [video_path] [output_path]

Defaults to ../sample.mp4 (the brief's sample clip, kept outside the repo).
Every run is saved under output/runs/ as run_001.mp4, run_002.mp4, ... (the
number always increases, so the highest number is always the newest run),
unless an explicit output_path is given.
"""
from __future__ import annotations

import argparse
import re
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from detect_track import PersonTracker, TRACKER_CONFIG

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT.parent / "sample.mp4"
RUNS_DIR = ROOT / "output" / "runs"
RUN_NAME_RE = re.compile(r"run_(\d+)\.mp4$")


def next_run_path() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for f in RUNS_DIR.glob("run_*.mp4") if (m := RUN_NAME_RE.match(f.name))]
    next_n = max(existing, default=0) + 1
    return RUNS_DIR / f"run_{next_n:03d}.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", nargs="?", default=str(DEFAULT_VIDEO))
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--conf", type=float, default=0.10, help="Detector cutoff; keep at/below track_low_thresh (0.10).")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None, help="e.g. 0 for CUDA, or cpu")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="Show estimated boxes during short gaps; 0 disables.")
    parser.add_argument("--track-buffer-seconds", type=float, default=6.0, help="Keep lost identities available for matching.")
    parser.add_argument("--tracker", default=TRACKER_CONFIG, help="BoT-SORT YAML config path.")
    parser.add_argument("--max-frames", type=int, help="Stop early for a short validation run.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main():
    args = parse_args()
    video_path = args.video_path
    output_path = args.output_path or str(next_run_path())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracker = PersonTracker(
        model_name=str(ROOT / "yolo26x.pt"), device=args.device, conf=args.conf,
        imgsz=args.imgsz, hold_seconds=args.hold_seconds,
        track_buffer_seconds=args.track_buffer_seconds, tracker_config=args.tracker,
    )
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(trace_length=60, color_lookup=sv.ColorLookup.TRACK)

    frame_count = 0
    unique_track_ids: set[int] = set()
    video_info = sv.VideoInfo(width=w, height=h, fps=fps)

    with closing(tracker.track(video_path)) as results, sv.VideoSink(output_path, video_info=video_info) as sink:
        for frame_idx, frame, all_tracks in results:
            detections = [d for d in all_tracks if not d["predicted"]]
            predictions = [d for d in all_tracks if d["predicted"]]
            if detections:
                xyxy = np.array([d["bbox"] for d in detections], dtype=float)
                tracker_ids = np.array([d["track_id"] for d in detections], dtype=int)
                unique_track_ids.update(d["track_id"] for d in detections)
            else:
                xyxy = np.zeros((0, 4), dtype=float)
                tracker_ids = np.zeros((0,), dtype=int)

            dets = sv.Detections(xyxy=xyxy, tracker_id=tracker_ids)
            labels = [f"person #{tid}" for tid in tracker_ids]

            annotated = trace_annotator.annotate(frame.copy(), dets)
            annotated = box_annotator.annotate(annotated, dets)
            annotated = label_annotator.annotate(annotated, dets, labels=labels)
            for prediction in predictions:
                x1, y1, x2, y2 = (int(v) for v in prediction["bbox"])
                label = f"person #{prediction['track_id']} estimated {prediction['missed_seconds']:.1f}s"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)
                cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            sink.write_frame(annotated)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...", flush=True)
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    print(f"Processed {frame_count} frames.")
    print(f"Track IDs observed (not a unique-person count): {sorted(unique_track_ids)}")
    print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    main()
