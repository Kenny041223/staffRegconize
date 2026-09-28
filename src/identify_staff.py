"""YOLO person tracking + periodic OWLv2 nametag verification, without training.

Save raw-score evidence, per-person frame observations, and confirmed staff frames.
Confirmation thresholds are starting settings: validate them on labelled footage.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import cv2

from badge_prefilter import DEFAULT_PREFILTER, PrefilteredMatcher
from detect_track import PersonTracker
from staff_scan import ScanScheduler, make_candidate, owned_tag_match
from staff_identity import StaffPolicy, ContinuityGuard, build_staff_decisions, export_decisions
from tag_match import DEFAULT_TAG_MODEL, TagMatcher

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT.parent / "sample.mp4"
DEFAULT_REFERENCE = ROOT / "assets" / "reference_1.jpg"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", nargs="?", default=str(DEFAULT_VIDEO))
    parser.add_argument("reference_path", nargs="?", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--tag-model", default=DEFAULT_TAG_MODEL, help="OWLv2 model ID or local path.")
    parser.add_argument("--reference-box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Mark the badge in the reference; keep surrounding context. Defaults to its .tag.json annotation.")
    parser.add_argument("--second-look", action=argparse.BooleanOptionalAction, default=False,
                        help="After a badge sighting on an unconfirmed person, scan another recent crop of them "
                             "at once (at least the hit spacing apart). Off by default.")
    parser.add_argument("--reference-rotations", type=int, nargs="+", default=[0],
                        choices=(0, 90, 180, 270),
                        help="Match the reference badge at these rotations (degrees). Default: upright only. "
                             "0 90 180 270 finds more rotated badges but also more look-alikes.")
    parser.add_argument("--badge-prefilter", nargs="?", const=str(DEFAULT_PREFILTER), default=str(DEFAULT_PREFILTER),
                        metavar="WEIGHTS",
                        help="Scan every crop with a small trained YOLO badge detector first; only crops it flags go "
                             f"to OWLv2. Default: yolo_folder/{DEFAULT_PREFILTER.name}, skipped with a warning if missing.")
    parser.add_argument("--no-badge-prefilter", dest="badge_prefilter", action="store_const", const=None,
                        help="Send every crop to OWLv2 (same result, slower).")
    parser.add_argument("--prefilter-conf", type=float, default=0.01,
                        help="Pre-filter confidence at which a crop is passed to OWLv2 (low = safer, slower).")
    parser.add_argument("--sam2-follow", action="store_true",
                        help="Follow each badge-sighted person with SAM 2.1 video segmentation instead of tracker IDs; "
                             "a follow is staff when --confirmations sightings fall inside its outline.")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large", help="SAM 2.1 model ID for --sam2-follow.")
    parser.add_argument("--candidate-threshold", type=float, default=0.65,
                        help="Minimum image similarity after foreground/background checks; not a staff-confirmation threshold.")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, or 0")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO person-detector input size.")
    parser.add_argument("--reid-tracker", action=argparse.BooleanOptionalAction, default=True,
                        help="Track with BoxMOT BoT-SORT + an OSNet person re-identification model (default; needs "
                             "boxmot). --no-reid-tracker uses the Ultralytics BoT-SORT tracker instead.")
    parser.add_argument("--scan-interval", type=float, default=0.75, help="Seconds per unconfirmed person's scan window.")
    parser.add_argument("--sample-every", type=int, default=None, help="Legacy override: scan window in VIDEO frames.")
    parser.add_argument("--confirmed-interval", type=float, default=None,
                        help="Seconds between checks of confirmed tracks. Default: --scan-interval in track mode "
                             "(every extra hit extends the label), 5 in continuity mode.")
    parser.add_argument("--reverify-gap", type=float, default=1.0, help="Reconfirm after this many seconds without observations.")
    parser.add_argument("--tag-threshold", type=float, default=0.9,
                        help="OWLv2 score at which a crop counts as a badge sighting (default 0.9).")
    parser.add_argument("--confirmations", type=int, default=2, help="Badge sightings needed to confirm a person as staff.")
    parser.add_argument("--batch-size", type=int, default=4, help="OWLv2 crop batch size; lower it if GPU memory runs out.")
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--min-sharpness", type=float, default=10.0, help="Minimum Laplacian variance; 0 disables blur filtering.")
    parser.add_argument("--max-tag-area", type=float, default=0.08, help="Largest tag/person box area ratio.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Number of evidence images to save.")
    parser.add_argument("--output-dir", default=None, help="An empty directory; otherwise create a unique run directory.")
    parser.add_argument("--staff-mode", choices=("track", "continuity"), default="track",
                        help="track: a track's hits combine however far apart; the label spreads from each hit "
                             "until a crossing or a brief disappearance. continuity: strict, any overlap resets evidence.")
    parser.add_argument("--crossing-iou", type=float, default=0.3)
    parser.add_argument("--evidence-window", type=float, default=30.0,
                        help="Continuity mode: seconds within which hits must fall. Both modes: scan-rate bookkeeping.")
    parser.add_argument("--evidence-ttl", type=float, default=30.0,
                        help="Continuity mode: expire staff status without a fresh hit. Both modes: scan-rate bookkeeping.")
    parser.add_argument("--backfill-seconds", type=float, default=1.0)
    parser.add_argument("--overlap-grace", type=float, default=0.2,
                        help="Resume after brief overlap; longer ambiguity requires a fresh owned badge, preserving history.")
    parser.add_argument("--max-step-diagonals", type=float, default=0.75,
                        help="Break continuity above this displacement per video frame, in person-box diagonals.")
    return parser


def parse_args(argv=None, parser=None):
    parser = parser or make_parser()
    args = parser.parse_args(argv)
    if args.confirmed_interval is None:
        args.confirmed_interval = args.scan_interval if args.staff_mode == "track" else 5.0
    try:
        staff_policy(args)
    except ValueError as error:
        parser.error(str(error))
    for name in ("scan_interval", "confirmed_interval", "reverify_gap"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("batch_size", "min_crop_size", "top_k", "imgsz"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("sample_every", "max_frames"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (args.tag_threshold is not None and not 0 < args.tag_threshold <= 1) or not 0 < args.max_tag_area <= 1:
        parser.error("--tag-threshold and --max-tag-area must be in (0, 1]")
    if not 0 <= args.candidate_threshold <= 1:
        parser.error("--candidate-threshold must be in [0, 1]")
    if args.tag_threshold is not None and args.tag_threshold < args.candidate_threshold:
        parser.error("--tag-threshold must be at least --candidate-threshold")
    if not 1 <= args.confirmations <= 5:
        parser.error("--confirmations must be between 1 and 5")
    if not math.isfinite(args.min_sharpness) or args.min_sharpness < 0:
        parser.error("--min-sharpness must be finite and non-negative")
    return args


def staff_policy(args):
    return StaffPolicy(min_score=args.tag_threshold, min_hits=args.confirmations,
                       crossing_iou=args.crossing_iou, gap_seconds=args.reverify_gap,
                       max_step_diagonals=args.max_step_diagonals,
                       evidence_window_seconds=args.evidence_window,
                       evidence_ttl_seconds=args.evidence_ttl, backfill_seconds=args.backfill_seconds,
                       overlap_grace_seconds=args.overlap_grace, mode=args.staff_mode)


def save_evidence_images(video_path, out_dir, ranked, top_k):
    cap = cv2.VideoCapture(video_path)
    try:
        candidates = (track for track in ranked if track.best_match is not None and track.best_match["tag_bbox"] is not None)
        for track in list(candidates)[:top_k]:
            match = track.best_match
            if match is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, match["frame_idx"])
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(v) for v in match["person_bbox"])
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2].copy()
            if not crop.size:
                continue
            if match["tag_bbox"] is not None:
                a, b, c, d = (int(v) for v in match["tag_bbox"])
                cv2.rectangle(crop, (a-x1, b-y1), (c-x1, d-y1), (0, 255, 255), 1)
            name = f"track{track.track_id}_segment{track.segment}_frame{match['frame_idx']}_score{track.best_score:.4f}.jpg"
            cv2.imwrite(str(out_dir / name), crop)
    finally:
        cap.release()


def run(args, tracker=None, matcher=None):
    """Dependency injection permits end-to-end tests without downloading weights."""
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    interval = args.sample_every or max(1, round(args.scan_interval * fps))
    policy = staff_policy(args)
    guard = ContinuityGuard(fps, policy)
    scheduler = ScanScheduler(interval, round(args.confirmed_interval * fps), round(args.reverify_gap * fps),
                              round(policy.evidence_window_seconds * fps), round(policy.evidence_ttl_seconds * fps),
                              round(policy.min_hit_spacing_seconds * fps))
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        parent = ROOT / "output" / "tag_scan"
        parent.mkdir(parents=True, exist_ok=True)
        out_dir = parent / f"run_{uuid4().hex[:12]}"
        out_dir.mkdir()

    load_started = time.perf_counter()
    if tracker is None and args.reid_tracker:
        from reid_track import ReidPersonTracker

        tracker = ReidPersonTracker(model_name=str(ROOT / "yolo_folder" / "yolo26x.pt"), device=args.device, imgsz=args.imgsz)
    tracker = tracker or PersonTracker(model_name=str(ROOT / "yolo_folder" / "yolo26x.pt"), device=args.device, imgsz=args.imgsz)
    if matcher is None:
        matcher = TagMatcher(args.reference_path, device=args.device, model_name=args.tag_model,
                             reference_box=args.reference_box, min_similarity=args.candidate_threshold,
                             max_area_ratio=args.max_tag_area, rotations=args.reference_rotations)
        if args.badge_prefilter and Path(args.badge_prefilter).exists():
            matcher = PrefilteredMatcher.load(matcher, args.badge_prefilter, args.prefilter_conf, device=args.device)
        elif args.badge_prefilter:
            print(f"Warning: badge pre-filter {args.badge_prefilter} not found; every crop goes to OWLv2 "
                  "(same result, slower).", flush=True)
    initial_forward_calls = matcher.forward_calls
    model_load_seconds = time.perf_counter() - load_started
    print(f"Tag model: {matcher.model_name}; device: {matcher.device}; batch size: {args.batch_size}", flush=True)
    if isinstance(matcher, PrefilteredMatcher):
        print(f"Badge pre-filter: {Path(args.badge_prefilter).name} at conf >= {args.prefilter_conf}; "
              "OWLv2 checks only the crops it flags", flush=True)
    within = "on one track" if args.staff_mode == "track" else f"in {args.evidence_window:g}s"
    confirmation = (f"{args.confirmations} owned hits {within} at raw score >= {args.tag_threshold} ({args.staff_mode} mode)"
                    if args.tag_threshold is not None else "candidate review only (no validated threshold supplied)")
    print(f"Scan window: {interval / fps:.2f}s; confirmation: {confirmation}", flush=True)
    observations, checks = defaultdict(list), []
    observed_counts = Counter()
    tag_seconds, scanned, rejected_quality = 0.0, 0, 0
    started = time.perf_counter()
    frame_count = 0

    def scan_due(frame_idx, flush=False):
        positives = run_checks(scheduler.take_due(frame_idx, flush=flush), frame_idx)
        if args.second_look:
            # A person seen with the badge once gets an immediate second look at another
            # recent crop, instead of waiting a full window the badge may not survive.
            follow_ups = [(track, other) for track, hit in positives
                          if track.confirmed_at is None and (other := scheduler.second_look(track, hit)) is not None]
            run_checks(follow_ups, frame_idx, second_look=True)

    def run_checks(pending, frame_idx, second_look=False):
        nonlocal tag_seconds, scanned
        positives = []
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            tick = time.perf_counter()
            results = matcher.detect_batch([candidate.crop for _, candidate in batch])
            tag_seconds += time.perf_counter() - tick
            for (track, candidate), result in zip(batch, results):
                score, box = owned_tag_match(result, candidate, args.max_tag_area)
                evidence = {}
                if box is not None:
                    local_box = [box[i] - candidate.offset[i % 2] for i in range(4)]
                    selected = next((i for i, item in enumerate(result["boxes"])
                                     if all(abs(a - b) < 1e-4 for a, b in zip(item, local_box))), None)
                    if selected is not None:
                        for field in ("objectness", "text_margins"):
                            if field in result:
                                evidence[field] = result[field][selected]
                track.record(candidate, score, box, frame_idx, args.tag_threshold, args.confirmations)
                if track.confirmed_at is not None:
                    track.due_frame = max(track.due_frame, frame_idx + scheduler.confirmed_interval)
                checks.append({"track_id": track.track_id, "segment": track.segment,
                               "frame_idx": candidate.frame_idx, "processed_at_frame": frame_idx,
                               "raw_score": score,
                               "filter_stats": result.get("filter_stats", {}),
                               "badge_evidence": evidence,
                               "positive": args.tag_threshold is not None and score >= args.tag_threshold and box is not None,
                               "second_look": second_look,
                               "tag_bbox": box, "person_bbox": list(candidate.bbox)})
                if checks[-1]["positive"]:
                    positives.append((track, candidate))
                scanned += 1
        return positives

    with closing(tracker.track(args.video_path)) as results:
        for frame_idx, frame, detections in results:
            guarded = guard.update(frame_idx, detections)
            for det in guarded:
                tid = det["track_id"]
                observed_counts[tid] += 1
                track = scheduler.observe(frame_idx, tid, reset=det["identity_boundary"])
                observations[frame_idx].append(dict(det, segment=track.segment))
                if scheduler.wants_crop(track, frame_idx):
                    candidate = make_candidate(frame_idx, frame, det["bbox"], args.min_crop_size, args.min_sharpness)
                    if candidate is None:
                        rejected_quality += 1
                    else:
                        candidate.other_people = tuple(d["bbox"] for d in guarded if d["track_id"] != tid)
                    scheduler.consider(track, candidate)
            scan_due(frame_idx)
            frame_count += 1
            if frame_count % 25 == 0:
                print(f"Frames: {frame_count}; tag crops scanned: {scanned}; confirmed segments: "
                      f"{sum(t.confirmed_at is not None for t in scheduler.tracks)}", flush=True)
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    if frame_count:
        scan_due(frame_idx, flush=True)
    processing_seconds = time.perf_counter() - started
    decisions = build_staff_decisions(checks, observations, fps, policy)
    sam2_report = None
    if args.sam2_follow and args.tag_threshold is not None:
        from sam_follow import SamFollower, apply_follows, follow_staff, sightings
        from staff_identity import exclusive_owner, track_key

        tick = time.perf_counter()
        follower = SamFollower(args.sam2_model, device=args.device)
        lookup = {(f, track_key(d)): d for f, dets in decisions.frames.items() for d in dets}
        seen = sightings(checks, decisions.frames, lookup, policy, track_key, exclusive_owner)
        follows = follow_staff(seen, lambda f, box, probes: follower.follow(args.video_path, frame_count, fps, f, box, probes),
                               fps, policy)
        apply_follows(decisions, follows)
        (out_dir / "sam2_follow.json").write_text(json.dumps({"model": args.sam2_model,
                                                              "follows": [f.to_json() for f in follows]}))
        sam2_report = {"model": args.sam2_model, "seconds": time.perf_counter() - tick, "sightings": len(seen),
                       "follows": len(follows), "confirmed": sum(f.confirmed_at is not None for f in follows)}
        print(f"SAM 2 follow: {sam2_report['confirmed']} of {sam2_report['follows']} follows confirmed "
              f"from {len(seen)} badge sightings in {sam2_report['seconds']:.1f}s", flush=True)
    staff_frame_count = export_decisions(decisions, fps, out_dir / "observations.csv", out_dir / "staff_frames.csv")
    confirmed_keys = {(i["track_id"], i["segment"]): i["confirmed_at_frame"]
                      for i in reversed(decisions.intervals) if "track_id" in i}
    ranked = sorted(scheduler.tracks, key=lambda t: t.best_score, reverse=True)
    save_evidence_images(args.video_path, out_dir, ranked, args.top_k)
    report = {
        "video": str(Path(args.video_path).resolve()), "reference": str(Path(args.reference_path).resolve()),
        "output_dir": str(out_dir.resolve()), "staff_decisions": decisions.metadata(),
        "tag_model": matcher.model_name, "settings": vars(args), "fps": fps, "processed_frames": frame_count,
        "matcher": getattr(matcher, "metadata", {}),
        "confirmation_enabled": args.tag_threshold is not None,
        "model_load_seconds": model_load_seconds, "processing_seconds": processing_seconds, "sam2": sam2_report,
        "tag_inference_seconds": tag_seconds, "tag_crops_scanned": scanned,
        "tag_forward_calls": matcher.forward_calls - initial_forward_calls,
        "legacy_every_3_observations_scan_count": sum(n // 3 for n in observed_counts.values()),
        "quality_rejected_observations": rejected_quality, "confirmed_staff_frame_count": staff_frame_count,
        "score_type": "raw sigmoid model score; not a calibrated probability",
        "frame_policy": ("track mode: a track's owned hits combine; the label spreads from each hit until a "
                         "crossing or a disappearance longer than overlap_grace"
                         if args.staff_mode == "track" else
                         "continuity mode: any overlap, gap or jump resets evidence; labels expire after evidence_ttl"),
        "tracks": [{"track_id": t.track_id, "segment": t.segment, "first_frame": t.first_frame,
                    "last_frame": t.last_seen, "checks": t.checks, "confirmed_at_frame": confirmed_keys.get(t.key),
                    "status": "confirmed_staff" if t.key in confirmed_keys else "unknown",
                    "best_score": t.best_score, "best_match": t.best_match} for t in ranked],
        "checks": checks,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Finished {frame_count} frames in {processing_seconds:.1f}s (tag inference {tag_seconds:.1f}s).")
    print(f"Tag crops: {scanned}; old sampling would request {report['legacy_every_3_observations_scan_count']} "
          "on these same observations.")
    if isinstance(matcher, PrefilteredMatcher):
        print(f"Pre-filter: OWLv2 checked {matcher.passed} of {matcher.passed + matcher.skipped} crops.")
    if args.tag_threshold is None:
        print("Candidate review mode: staff_frames.csv contains no automatic labels. Review report.json and the evidence JPGs.")
    else:
        print(f"Confirmed staff observed in {staff_frame_count} frames. Validate labels against the saved evidence.")
    print(f"Results: {out_dir}")
    return report


if __name__ == "__main__":
    run(parse_args())
