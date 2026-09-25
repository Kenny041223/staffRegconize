"""Render saved scans with the same staff rules used by CSV export.

No model inference is run. Staff overrides produce new CSV/decision sidecars;
the original scan is preserved. All people participate in ownership checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from staff_identity import StaffPolicy, build_staff_decisions, export_decisions, track_key
from tag_overlay import build_tag_display_map

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT.parent / "sample.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("video_path", nargs="?", default=None)
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--display-seconds", type=float, default=0.6)
    parser.add_argument("--min-display-score", type=float, default=0.65)
    parser.add_argument("--only-candidates", action="store_true")
    parser.add_argument("--staff-min-hits", type=int, default=None,
                        help="Re-evaluate staff with 1-5 remembered owned hits and show staff only, unless --show-all is supplied.")
    parser.add_argument("--staff-score", type=float, default=None)
    parser.add_argument("--show-all", action="store_true", help="Show unknown/uncertain people as well as staff.")
    parser.add_argument("--staff-mode", choices=("track", "continuity"), default=None,
                        help="Override the run's staff policy (default: the run's saved mode, else track).")
    parser.add_argument("--crossing-iou", type=float, default=None)
    parser.add_argument("--evidence-window", type=float, default=None)
    parser.add_argument("--evidence-ttl", type=float, default=None)
    parser.add_argument("--backfill-seconds", type=float, default=None)
    parser.add_argument("--overlap-grace", type=float, default=None)
    args = parser.parse_args(argv)
    if not np.isfinite(args.display_seconds) or args.display_seconds < 0:
        parser.error("Display seconds must be finite and non-negative.")
    if not 0 <= args.min_display_score <= 1:
        parser.error("Display score must be in [0, 1].")
    if args.staff_score is not None and args.staff_min_hits is None:
        parser.error("--staff-score requires --staff-min-hits.")
    if args.staff_min_hits is not None and not 1 <= args.staff_min_hits <= 5:
        parser.error("Staff hits must be 1-5.")
    if args.staff_score is not None and not 0 <= args.staff_score <= 1:
        parser.error("Staff score must be in [0, 1].")
    # Validate combined policy after reading the saved run's settings.
    return args


def policy_for_report(report, args):
    settings = report.get("settings", {})
    saved = report.get("staff_decisions", {}).get("policy")
    policy = dict(saved) if saved else StaffPolicy(
        min_score=settings.get("tag_threshold") if report.get("confirmation_enabled") else None,
        min_hits=settings.get("confirmations", 3), gap_seconds=settings.get("reverify_gap", 1.0)).metadata()
    if args.staff_min_hits is not None:
        policy.update(min_hits=args.staff_min_hits,
                      min_score=args.staff_score if args.staff_score is not None else 0.9)
    for argument, field in (("staff_mode", "mode"), ("crossing_iou", "crossing_iou"), ("evidence_window", "evidence_window_seconds"),
                            ("evidence_ttl", "evidence_ttl_seconds"), ("backfill_seconds", "backfill_seconds"),
                            ("overlap_grace", "overlap_grace_seconds")):
        value = getattr(args, argument)
        if value is not None:
            policy[field] = value
    return StaffPolicy(**policy)


def load_observations(path):
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frames[int(row["frame_idx"])].append({
                "track_id": int(row["track_id"]), "segment": int(row.get("segment", 0)),
                "bbox": tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))})
    return frames


def visible_detections(decisions, checks, args):
    """Visibility never changes the inputs used for identity decisions."""
    candidate_keys = {track_key(c) for c in checks
                      if c.get("tag_bbox") is not None and c["raw_score"] >= args.min_display_score}
    staff_only = args.staff_min_hits is not None and not args.show_all
    return {frame_idx: [d for d in detections
                        if (not args.only_candidates or track_key(d) in candidate_keys)
                        and (not staff_only or d["status"] == "confirmed_staff")]
            for frame_idx, detections in decisions.frames.items()}


def render(args):
    run_dir = Path(args.run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    fps = report["fps"]
    policy = policy_for_report(report, args)
    observations = load_observations(run_dir / "observations.csv")
    decisions = build_staff_decisions(report["checks"], observations, fps, policy)
    visible = visible_detections(decisions, report["checks"], args)
    allowed = {(f, track_key(d)) for f, detections in visible.items() for d in detections}
    total_frames = report["processed_frames"]
    display_map = build_tag_display_map(report["checks"], decisions.frames, total_frames, fps,
                                        args.display_seconds, args.min_display_score,
                                        policy=policy, allowed_owners=allowed)
    video_path = args.video_path or report.get("video") or str(DEFAULT_VIDEO)
    output = Path(args.output_path or ROOT / "output" / "runs" / "staff_visualization.mp4")
    if output.resolve() == Path(video_path).resolve():
        raise ValueError("Output video must differ from the source video.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    labels = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    trails = sv.TraceAnnotator(trace_length=60, color_lookup=sv.ColorLookup.TRACK)
    rendered = 0
    try:
        with sv.VideoSink(str(output), video_info=sv.VideoInfo(width=w, height=h, fps=fps)) as sink:
            for frame_idx in range(total_frames):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Source ended at frame {frame_idx}; scan expects {total_frames} frames.")
                dets = visible.get(frame_idx, [])
                xyxy = np.array([d["bbox"] for d in dets], dtype=float).reshape(-1, 4)
                # Drawing boundaries reset trails without erasing remembered badge evidence.
                sv_dets = sv.Detections(xyxy=xyxy, tracker_id=np.array([d["display_id"] for d in dets], dtype=int))
                captions = [f"STAFF #{d['track_id']}" if d["status"] == "confirmed_staff"
                            else f"person #{d['track_id']}" + (" (uncertain)" if d["status"] == "uncertain" else "") for d in dets]
                frame = trails.annotate(frame, sv_dets)
                frame = boxes.annotate(frame, sv_dets)
                frame = labels.annotate(frame, sv_dets, labels=captions)
                for box, score, estimated in display_map.get(frame_idx, []):
                    x1, y1, x2, y2 = (int(v) for v in box)
                    color = (0, 180, 180) if estimated else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if estimated else 2)
                    caption = "tag estimate" if estimated else "tag candidate"
                    cv2.putText(frame, f"{caption} {score:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                sink.write_frame(frame)
                rendered += 1
                if frame_idx % 200 == 0:
                    print(f"Rendered frame {frame_idx}", flush=True)
    finally:
        cap.release()
    staff_frames = export_decisions(decisions, fps, output.with_suffix(".observations.csv"), output.with_suffix(".staff.csv"))
    metadata = dict(decisions.metadata(), source_run=str(run_dir.resolve()), source_video=str(Path(video_path).resolve()),
                    render_settings=vars(args), rendered_frames=rendered, confirmed_staff_frame_count=staff_frames)
    output.with_suffix(".decisions.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output}; {staff_frames} observed frames with confirmed staff in {len(decisions.intervals)} display intervals.")
    if policy.min_score is not None and not decisions.intervals:
        print("No remembered identity has enough owned badge evidence at a trusted location. Inspect candidates with --show-all; no IDs were forcibly joined.")
    return metadata


if __name__ == "__main__":
    render(parse_args())
