"""Attach short-lived badge estimates to the observed owner, never screen space."""
from collections import defaultdict
import math

from staff_identity import ContinuityGuard, StaffPolicy, exclusive_owner, track_key


def build_tag_display_map(checks, frame_dets, total_frames, fps, display_seconds=0.6, min_score=0.65,
                          policy=None, allowed_owners=None):
    if not math.isfinite(display_seconds) or display_seconds < 0:
        raise ValueError("display_seconds must be finite and non-negative.")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1].")
    lookup, guarded_frames = {}, {}
    guard = ContinuityGuard(fps, policy or StaffPolicy())
    for frame_idx in range(total_frames):
        detections = frame_dets.get(frame_idx, []) if isinstance(frame_dets, dict) else frame_dets[frame_idx]
        guarded_frames[frame_idx] = guard.update(frame_idx, detections)
        lookup[frame_idx] = {track_key(d): d for d in guarded_frames[frame_idx]}
    display = defaultdict(dict)
    duration = max(1, round(display_seconds * fps))
    for check in sorted(checks, key=lambda c: c["frame_idx"]):
        tag = check.get("tag_bbox")
        if tag is None or check["raw_score"] < min_score:
            continue
        key = (check["track_id"], check.get("segment", 0))
        x1, y1, x2, y2 = check["person_bbox"]
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        relative = ((tag[0] - x1) / width, (tag[1] - y1) / height,
                    (tag[2] - x1) / width, (tag[3] - y1) / height)
        start = check["frame_idx"]
        source = lookup.get(start, {}).get(key)
        if source is None or not exclusive_owner(tag, key, guarded_frames[start]):
            continue
        for frame_idx in range(start, min(start + duration, total_frames)):
            owner = lookup[frame_idx].get(key)
            if (owner is None or (frame_idx != start and owner["identity_state"] != "clear")
                    or owner["display_id"] != source["display_id"]):
                break
            a, b, c, d = owner["bbox"]
            box = [a + relative[0] * (c - a), b + relative[1] * (d - b),
                   a + relative[2] * (c - a), b + relative[3] * (d - b)]
            if not exclusive_owner(box, key, guarded_frames[frame_idx]):
                break
            if allowed_owners is None or (frame_idx, key) in allowed_owners:
                display[frame_idx][key] = (box, check["raw_score"], frame_idx != start)
    return {frame_idx: list(items.values()) for frame_idx, items in display.items()}
