"""Decide which observed person boxes are staff, from sparse nametag hits.

Two policies (StaffPolicy.mode):
  track (default)  hits on one track combine however far apart they are; the
                   label spreads outward from each hit until a sign of a
                   possible ID handoff (the box crossing another box, or the
                   person briefly undetected). Suits crowded ceiling views where
                   the badge is visible only now and then.
  continuity       strict: any overlap, gap or jump starts a new section and
                   discards earlier evidence; hits must be recent; labels expire.
Neither policy links different tracker IDs. Works on cached scan results.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass


def track_key(detection):
    return detection["track_id"], detection.get("segment", 0)


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0., a[2] - a[0]) * max(0., a[3] - a[1]) + max(0., b[2] - b[0]) * max(0., b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def exclusive_owner(tag_box, owner_key, detections):
    """Use every real person, including people hidden by display filters."""
    cx, cy = (tag_box[0] + tag_box[2]) / 2, (tag_box[1] + tag_box[3]) / 2
    owners = [track_key(d) for d in detections if not d.get("predicted")
              and d["bbox"][0] <= cx <= d["bbox"][2] and d["bbox"][1] <= cy <= d["bbox"][3]]
    return owners == [owner_key]


@dataclass(frozen=True)
class StaffPolicy:
    min_score: float | None = None
    min_hits: int = 3
    crossing_iou: float = 0.3
    gap_seconds: float = 1.0
    max_step_diagonals: float = 0.75
    evidence_window_seconds: float = 30.0
    evidence_ttl_seconds: float = 30.0
    overlap_grace_seconds: float = 0.2
    backfill_seconds: float = 1.0
    min_hit_spacing_seconds: float = 0.2
    mode: str = "track"

    def __post_init__(self):
        if self.mode not in ("track", "continuity"):
            raise ValueError("Staff mode must be 'track' or 'continuity'.")
        if self.min_score is not None and not 0 <= self.min_score <= 1:
            raise ValueError("Staff score must be in [0, 1].")
        if not 1 <= self.min_hits <= 5 or not 0 < self.crossing_iou <= 1:
            raise ValueError("Staff hits must be 1-5 and crossing IoU must be in (0, 1].")
        for name in ("gap_seconds", "max_step_diagonals", "evidence_window_seconds", "evidence_ttl_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("backfill_seconds", "min_hit_spacing_seconds", "overlap_grace_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.evidence_ttl_seconds < self.evidence_window_seconds:
            raise ValueError("Evidence lifetime must cover the confirmation window.")

    def metadata(self):
        return asdict(self)


class ContinuityGuard:
    """Keep identity memory through overlap; track drawing confidence separately.

    Identity boundaries require a new source segment, a long gap, or a large
    motion discontinuity. An overlap by itself never creates such a boundary.
    The display boundary is stricter: a prolonged overlap or a shorter gap can
    require fresh location evidence without discarding previous badge hits.
    """
    def __init__(self, fps, policy=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive.")
        self.fps = fps
        self.policy = policy or StaffPolicy()
        self.previous = {}
        self.next_id = 1
        self.next_display_id = 1

    def update(self, frame_idx, detections):
        observed = [dict(d) for d in detections if not d.get("predicted")]
        ambiguous = set()
        for i, a in enumerate(observed):
            for b in observed[i + 1:]:
                if box_iou(a["bbox"], b["bbox"]) >= self.policy.crossing_iou:
                    ambiguous.update((a["track_id"], b["track_id"]))
        grace_frames = max(1, round(self.policy.overlap_grace_seconds * self.fps))
        for det in observed:
            tid, bbox = det["track_id"], det["bbox"]
            old = self.previous.get(tid)
            overlaps = tid in ambiguous
            dt = frame_idx - old["frame_idx"] if old else 1
            reason = "overlap" if overlaps else "clear"
            jump = False
            if old:
                a = old["bbox"]
                distance = math.hypot((bbox[0] + bbox[2] - a[0] - a[2]) / 2,
                                      (bbox[1] + bbox[3] - a[1] - a[3]) / 2)
                scale = max(1., (math.hypot(a[2] - a[0], a[3] - a[1])
                                 + math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])) / 2)
                jump = distance / scale > self.policy.max_step_diagonals * max(1, dt)
            long_gap = dt > self.policy.gap_seconds * self.fps
            boundary = old is None or track_key(old) != track_key(det) or long_gap or jump
            if long_gap:
                reason = "gap"
            elif jump:
                reason = "motion_jump"
            if boundary:
                continuity_id = self.next_id
                self.next_id += 1
                overlap_start = frame_idx if overlaps else None
                was_blocked = False
            else:
                continuity_id = old["continuity_id"]
                overlap_start = (old["overlap_start"] if old["overlap_start"] is not None else frame_idx) if overlaps else None
                was_blocked = old["overlap_blocked"] if overlaps else False
            blocked = overlaps and frame_idx - overlap_start + 1 > grace_frames
            display_boundary = boundary or dt > grace_frames or (blocked and not was_blocked)
            if display_boundary:
                display_id = self.next_display_id
                self.next_display_id += 1
                requires_anchor = not boundary or jump
            else:
                display_id = old["display_id"]
                requires_anchor = old["requires_anchor"]
            det.update(continuity_id=continuity_id, display_id=display_id,
                       identity_boundary=boundary, requires_anchor=requires_anchor,
                       identity_state="uncertain" if overlaps or jump else "clear", identity_reason=reason)
            self.previous[tid] = dict(det, frame_idx=frame_idx, overlap_start=overlap_start,
                                      overlap_blocked=blocked)
        return observed


@dataclass
class StaffDecisions:
    frames: dict
    intervals: list
    policy: StaffPolicy
    def metadata(self):
        return {"version": 3, "policy": self.policy.metadata(), "intervals": self.intervals}


def build_staff_decisions(checks, frame_dets, fps, policy=None):
    """Label observed person boxes as confirmed staff from owned nametag hits.

    All detections must be supplied before any visibility filter: every real
    person takes part in ownership and crossing checks.
    """
    policy = policy or StaffPolicy()
    guard = ContinuityGuard(fps, policy)
    items = sorted(frame_dets.items()) if isinstance(frame_dets, dict) else enumerate(frame_dets)
    frames, lookup = {}, {}
    for frame_idx, detections in items:
        frames[frame_idx] = guard.update(frame_idx, detections)
        for det in frames[frame_idx]:
            det.update(status="uncertain" if det["identity_state"] == "uncertain" else "unknown",
                       confirmed_at_frame=None, label_source="unconfirmed")
            lookup[(frame_idx, track_key(det))] = det
    if policy.min_score is None:
        return StaffDecisions(frames, [], policy)
    decide = _track_decisions if policy.mode == "track" else _continuity_decisions
    return StaffDecisions(frames, decide(checks, frames, lookup, fps, policy), policy)


def _mark(det, frame_idx, confirmed_at):
    det.update(status="confirmed_staff", confirmed_at_frame=confirmed_at,
               label_source="track_evidence_backfill" if frame_idx < confirmed_at else "track_evidence")


def _crosses(frame_idx, det, frames, policy):
    return any(other is not det and box_iou(det["bbox"], other["bbox"]) >= policy.crossing_iou
               for other in frames[frame_idx])


def _track_decisions(checks, frames, lookup, fps, policy):
    """A track is one tracker ID up to a long gap or a large jump (continuity_id).

    Brief overlaps neither erase nor split its evidence, and a badge missing
    from a scan is not evidence against it. Once it has min_hits owned hits,
    the label spreads outward from each hit until the first sign of a possible
    ID handoff: its box crossing another person's box, or the person going
    undetected for longer than overlap_grace_seconds. Two hits are joined only
    if no such sign lies between them.
    """
    hits = defaultdict(dict)  # continuity_id -> {hit frame: frame the decision was available}
    for check in checks:
        frame_idx, key = check["frame_idx"], track_key(check)
        det, tag = lookup.get((frame_idx, key)), check.get("tag_bbox")
        if (det is None or tag is None or not math.isfinite(check["raw_score"])
                or check["raw_score"] < policy.min_score or not exclusive_owner(tag, key, frames[frame_idx])):
            continue
        decision = max(frame_idx, check.get("processed_at_frame", frame_idx))
        hits[det["continuity_id"]][frame_idx] = min(decision, hits[det["continuity_id"]].get(frame_idx, decision))

    sections = defaultdict(list)
    for frame_idx in sorted(frames):
        for det in frames[frame_idx]:
            sections[det["continuity_id"]].append((frame_idx, det))

    spacing = max(1, round(policy.min_hit_spacing_seconds * fps))
    grace = max(1, round(policy.overlap_grace_seconds * fps))
    intervals = []
    for continuity_id, found in hits.items():
        independent = []
        for frame_idx in sorted(found):
            if not independent or frame_idx - independent[-1] >= spacing:
                independent.append(frame_idx)
        if len(independent) < policy.min_hits:
            continue
        section = sections[continuity_id]
        position = {frame_idx: i for i, (frame_idx, _) in enumerate(section)}

        def handoff(i, j):
            """Possible ID handoff between neighbouring observations i and j (j is entered)."""
            return abs(section[j][0] - section[i][0]) > grace or _crosses(*section[j], frames, policy)

        spans = []
        for hit_frame in independent:
            lo = hi = position[hit_frame]
            while lo > 0 and not handoff(lo, lo - 1):
                lo -= 1
            while hi < len(section) - 1 and not handoff(hi, hi + 1):
                hi += 1
            if spans and lo <= spans[-1][1] + 1:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
        confirmed_at = max(found[f] for f in independent[:policy.min_hits])
        for lo, hi in spans:
            for frame_idx, det in section[lo:hi + 1]:
                _mark(det, frame_idx, confirmed_at)
            head = section[lo][1]
            intervals.append({"track_id": head["track_id"], "segment": head.get("segment", 0),
                              "continuity_id": continuity_id, "start_frame": section[lo][0],
                              "end_frame": section[hi][0], "confirmed_at_frame": confirmed_at,
                              "hit_frames": [f for f in independent if section[lo][0] <= f <= section[hi][0]],
                              "track_hit_frames": independent})
    return sorted(intervals, key=lambda interval: interval["start_frame"])


def _continuity_decisions(checks, frames, lookup, fps, policy):
    """Strict policy: sections also break at every overlap, and evidence resets.

    Hits must be within evidence_window_seconds, labels expire
    evidence_ttl_seconds after the last hit and backfill at most
    backfill_seconds.
    """
    sections, previous, next_id = defaultdict(list), {}, 0
    for frame_idx in sorted(frames):
        for det in frames[frame_idx]:
            uncertain = det["identity_state"] == "uncertain"
            old = previous.get(det["track_id"])
            if (old is None or old["key"] != track_key(det) or old["uncertain"] != uncertain
                    or det["identity_reason"] in ("gap", "motion_jump")):
                next_id += 1
                section_id = next_id
            else:
                section_id = old["section"]
            previous[det["track_id"]] = {"key": track_key(det), "uncertain": uncertain, "section": section_id}
            det["continuity_id"] = section_id
            if not uncertain:
                sections[section_id].append((frame_idx, det))

    votes = defaultdict(dict)
    for check in checks:
        frame_idx, key = check["frame_idx"], track_key(check)
        det = lookup.get((frame_idx, key))
        if det is None or det["identity_state"] != "clear":
            continue
        tag = check.get("tag_bbox")
        positive = (tag is not None and math.isfinite(check["raw_score"])
                    and check["raw_score"] >= policy.min_score
                    and exclusive_owner(tag, key, frames[frame_idx]))
        decision_frame = max(frame_idx, check.get("processed_at_frame", frame_idx))
        previous_vote = votes[det["continuity_id"]].get(frame_idx)
        if previous_vote is None or (positive and not previous_vote[0]):
            votes[det["continuity_id"]][frame_idx] = (positive, decision_frame)

    window = max(1, round(policy.evidence_window_seconds * fps))
    ttl = max(1, round(policy.evidence_ttl_seconds * fps))
    spacing = max(1, round(policy.min_hit_spacing_seconds * fps))
    backfill = round(policy.backfill_seconds * fps)
    intervals = []
    for section_id, observations in sections.items():
        recent, active, last_counted, found = deque(maxlen=5), None, -10**12, []
        for frame_idx, (positive, decision_frame) in sorted(votes[section_id].items()):
            if frame_idx - last_counted < spacing:
                continue
            last_counted = frame_idx
            if active is not None and frame_idx > active["last_evidence_frame"] + ttl:
                active = None
                recent.clear()
            while recent and frame_idx - recent[0][0] > window:
                recent.popleft()
            recent.append((frame_idx, positive, decision_frame))
            if active is None and sum(v[1] for v in recent) >= policy.min_hits:
                first_hit = next(v[0] for v in recent if v[1])
                active = {"track_id": observations[0][1]["track_id"],
                          "segment": observations[0][1].get("segment", 0), "continuity_id": section_id,
                          "start_frame": max(observations[0][0], first_hit - backfill), "end_frame": frame_idx,
                          "confirmed_at_frame": max(v[2] for v in recent if v[1]),
                          "last_evidence_frame": frame_idx, "hit_frames": [v[0] for v in recent if v[1]]}
                found.append(active)
            elif active is not None and positive:
                active["last_evidence_frame"] = frame_idx
                active["hit_frames"].append(frame_idx)
            if active is not None:
                active["end_frame"] = min(observations[-1][0], active["last_evidence_frame"] + ttl)
        for interval in found:
            for frame_idx, det in observations:
                if interval["start_frame"] <= frame_idx <= interval["end_frame"]:
                    _mark(det, frame_idx, interval["confirmed_at_frame"])
        intervals.extend(found)
    return sorted(intervals, key=lambda interval: interval["start_frame"])


CSV_FIELDS = ["frame_idx", "time_seconds", "track_id", "segment", "continuity_id", "display_id", "status",
              "x1", "y1", "x2", "y2", "center_x", "center_y", "confirmed_at_frame", "label_source",
              "identity_reason"]


def export_decisions(decisions, fps, observations_path, staff_path):
    """Both scanner and renderer export exactly the decisions they display."""
    staff_frames = set()
    with observations_path.open("w", newline="", encoding="utf-8") as all_file, staff_path.open("w", newline="", encoding="utf-8") as staff_file:
        all_writer, staff_writer = csv.DictWriter(all_file, CSV_FIELDS), csv.DictWriter(staff_file, CSV_FIELDS)
        all_writer.writeheader()
        staff_writer.writeheader()
        for frame_idx, detections in sorted(decisions.frames.items()):
            for det in detections:
                x1, y1, x2, y2 = map(float, det["bbox"])
                row = {k: det.get(k, 0 if k == "segment" else None) for k in CSV_FIELDS}
                row.update(frame_idx=frame_idx, time_seconds=round(frame_idx / fps, 6),
                           x1=x1, y1=y1, x2=x2, y2=y2, center_x=(x1+x2)/2, center_y=(y1+y2)/2)
                all_writer.writerow(row)
                if det["status"] == "confirmed_staff":
                    staff_writer.writerow(row)
                    staff_frames.add(frame_idx)
    return len(staff_frames)
