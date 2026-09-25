"""Select clear crops and accumulate tag evidence within continuous tracks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ScanCandidate:
    frame_idx: int
    bbox: tuple
    offset: tuple[int, int]
    crop: np.ndarray
    quality: float
    other_people: tuple = ()


@dataclass
class StaffTrack:
    track_id: int
    segment: int
    first_frame: int
    last_seen: int
    due_frame: int
    hits: deque = field(default_factory=lambda: deque(maxlen=5))
    candidate: ScanCandidate | None = None
    confirmed_at: int | None = None
    checks: int = 0
    best_score: float = 0.0
    best_match: dict | None = None
    last_evidence_frame: int = -10**9
    evidence_window_frames: int = 750
    evidence_ttl_frames: int = 750
    min_hit_spacing_frames: int = 5
    last_positive_frame: int = -10**9
    recent: deque = field(default_factory=deque)  # recent usable crops, kept for a second look
    scanned_frames: set = field(default_factory=set)

    @property
    def key(self):
        return self.track_id, self.segment

    def record(self, candidate, score, tag_box, decision_frame, threshold, confirmations):
        # This vote only controls scan frequency. staff_identity owns exported labels.
        self.checks += 1
        self.scanned_frames.add(candidate.frame_idx)
        self.last_evidence_frame = candidate.frame_idx
        while self.hits and candidate.frame_idx - self.hits[0][0] > self.evidence_window_frames:
            self.hits.popleft()
        positive = threshold is not None and score >= threshold and tag_box is not None
        if positive and all(abs(candidate.frame_idx - f) >= self.min_hit_spacing_frames for f, _ in self.hits):
            self.hits.append((candidate.frame_idx, True))
            self.last_positive_frame = candidate.frame_idx
        if self.confirmed_at is None and sum(hit for _, hit in self.hits) >= confirmations:
            self.confirmed_at = decision_frame
        if self.best_match is None or score > self.best_score:
            self.best_score = score
            self.best_match = {"frame_idx": candidate.frame_idx, "person_bbox": list(candidate.bbox),
                               "tag_bbox": tag_box, "score": score}


def make_candidate(frame_idx, frame, bbox, min_size=32, min_sharpness=10.0):
    """Measure crop quality cheaply; retain original pixels for inference."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * 0.05, (y2 - y1) * 0.05
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(np.ceil(x2 + pad_x))), min(height, int(np.ceil(y2 + pad_y)))
    if min(right - left, bottom - top) < min_size:
        return None
    crop = frame[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 160 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < min_sharpness:
        return None
    quality = np.log1p(sharpness) * np.sqrt(crop.shape[0] * crop.shape[1])
    return ScanCandidate(frame_idx, tuple(bbox), (left, top), crop, float(quality))


def owned_tag_match(result, candidate, max_area_ratio=0.08):
    """Reject background/padding and whole-person matches using box geometry.

    Rectangle containment cannot fully resolve ownership of overlapping people.
    """
    px1, py1, px2, py2 = candidate.bbox
    person_area = max(0, px2 - px1) * max(0, py2 - py1)
    if person_area <= 0:
        return 0.0, None
    for score, box in sorted(zip(result["scores"], result["boxes"]), reverse=True, key=lambda item: item[0]):
        x1, y1, x2, y2 = np.asarray(box) + np.array([*candidate.offset, *candidate.offset])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        intersection = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
        if (area > 0 and area <= max_area_ratio * person_area and intersection / area >= 0.8
                and px1 <= cx <= px2 and py1 <= cy <= py2
                and not any(a <= cx <= c and b <= cy <= d for a, b, c, d in candidate.other_people)):
            return float(score), [float(v) for v in (x1, y1, x2, y2)]
    return 0.0, None


class ScanScheduler:
    def __init__(self, interval_frames, confirmed_interval_frames, reverify_gap_frames,
                 evidence_window_frames=750, evidence_ttl_frames=750, min_hit_spacing_frames=5):
        self.interval = max(1, interval_frames)
        self.confirmed_interval = max(self.interval, confirmed_interval_frames)
        self.reverify_gap = max(1, reverify_gap_frames)
        self.evidence_window_frames = evidence_window_frames
        self.evidence_ttl_frames = evidence_ttl_frames
        self.min_hit_spacing_frames = max(1, min_hit_spacing_frames)
        self.current: dict[int, StaffTrack] = {}
        self.tracks: list[StaffTrack] = []

    def observe(self, frame_idx, track_id, reset=False):
        track = self.current.get(track_id)
        if track is None or reset or frame_idx - track.last_seen > self.reverify_gap:
            if track is not None:
                track.candidate = None  # Do not process a queued crop after an unsafe handoff.
            segment = track.segment + 1 if track is not None else 0
            track = StaffTrack(track_id, segment, frame_idx, frame_idx, frame_idx + self.interval - 1,
                               evidence_window_frames=self.evidence_window_frames,
                               evidence_ttl_frames=self.evidence_ttl_frames,
                               min_hit_spacing_frames=self.min_hit_spacing_frames)
            self.current[track_id] = track
            self.tracks.append(track)
        if track.confirmed_at is not None and frame_idx - track.last_positive_frame > track.evidence_ttl_frames:
            track.confirmed_at = None
            track.hits.clear()
            track.due_frame = frame_idx
        track.last_seen = frame_idx
        return track

    def wants_crop(self, track, frame_idx):
        return (frame_idx >= track.due_frame - self.interval + 1
                and frame_idx - track.last_evidence_frame >= max(1, self.interval // 2))

    def consider(self, track, candidate):
        if candidate is None:
            return
        candidate.crop = candidate.crop.copy()
        track.recent.append(candidate)
        if track.candidate is None or candidate.quality > track.candidate.quality:
            track.candidate = candidate

    def second_look(self, track, hit):
        """A kept crop of the same person at least the hit spacing away from `hit`, nearest in time.

        Scans are processed at the end of a window, after those frames have passed,
        so a second look reuses crops kept from the window instead of waiting.
        """
        options = [c for c in track.recent if abs(c.frame_idx - hit.frame_idx) >= self.min_hit_spacing_frames
                   and c.frame_idx not in track.scanned_frames]
        if not options:
            return None
        return min(options, key=lambda c: (abs(c.frame_idx - hit.frame_idx), -c.quality))

    def take_due(self, frame_idx, flush=False):
        due = []
        for track in self.tracks:
            while track.recent and track.recent[0].frame_idx < frame_idx - 2 * self.interval:
                track.recent.popleft()
            if track.candidate is not None and (flush or frame_idx >= track.due_frame):
                due.append((track, track.candidate))
                track.candidate = None
            if frame_idx >= track.due_frame or flush:
                interval = self.confirmed_interval if track.confirmed_at is not None else self.interval
                track.due_frame = frame_idx + interval
        return due
