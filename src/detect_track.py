"""Person detection + multi-object tracking using YOLO + BoT-SORT.

BoT-SORT (vs. plain ByteTrack) adds appearance-based re-identification, so a
person who is briefly occluded/missed is more likely to be re-matched to their
existing track ID instead of being assigned a brand new one on reappearance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.basetrack import TrackState

PERSON_CLASS_ID = 0
TRACKER_CONFIG = str(Path(__file__).resolve().parent / "trackers" / "botsort_reid.yaml")


class PersonTracker:
    def __init__(
        self,
        model_name: str = "yolo26x.pt",
        device: Optional[str] = None,
        conf: float = 0.10,
        imgsz: int = 1280,
        hold_seconds: float = 1.0,
        track_buffer_seconds: float = 6.0,
        tracker_config: str = TRACKER_CONFIG,
    ):
        if not 0 < conf <= 1:
            raise ValueError("conf must be between 0 (exclusive) and 1.")
        if not np.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError("hold_seconds must be finite and non-negative.")
        if not np.isfinite(track_buffer_seconds) or track_buffer_seconds <= 0:
            raise ValueError("track_buffer_seconds must be finite and positive.")
        if hold_seconds > track_buffer_seconds:
            raise ValueError("hold_seconds cannot exceed track_buffer_seconds.")
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.hold_seconds = hold_seconds
        self.track_buffer_seconds = track_buffer_seconds
        self.tracker_config = tracker_config

    def track(self, video_path: str) -> Iterator[tuple[int, np.ndarray, list[dict]]]:
        """Yields (frame_idx, frame_bgr, detections).

        Entries include track_id, bbox, conf, predicted, and missed_seconds.
        Predicted boxes are temporary Kalman estimates, not new detections;
        their conf is None. Only BoT-SORT assigns or recovers identities.
        Each call starts a new video; state persists across its frames only.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        hold_frames = int(self.hold_seconds * fps)

        results = self.model.track(
            source=video_path,
            classes=[PERSON_CLASS_ID],
            conf=self.conf,
            imgsz=self.imgsz,
            persist=False,
            stream=True,
            batch=1,
            vid_stride=1,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
        )
        try:
            for frame_idx, result in enumerate(results):
                native_tracker = self.model.predictor.trackers[0]
                # Ultralytics counts processed frames, so use the source FPS.
                native_tracker.max_frames_lost = max(1, round(self.track_buffer_seconds * fps))
                frame = result.orig_img
                detections = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    for box, tid, c in zip(xyxy, ids, confs):
                        detections.append({
                            "track_id": int(tid),
                            "bbox": tuple(float(v) for v in box.tolist()),
                            "conf": float(c),
                            "predicted": False,
                            "missed_seconds": 0.0,
                        })
                detections.extend(lost_track_predictions(native_tracker, detections, frame.shape, hold_frames, fps))
                yield frame_idx, frame, detections
        finally:
            results.close()


def lost_track_predictions(tracker, observed: list[dict], frame_shape, hold_frames: int, fps: float) -> list[dict]:
    """Expose briefly lost, previously confirmed tracks without inventing IDs.

    The native tracker advances these Kalman states even on empty frames.
    Suppress an estimate covering a current observation to avoid double boxes;
    this affects display only, never association or identity assignment.
    """
    predictions = []
    height, width = frame_shape[:2]
    observed_ids = {d["track_id"] for d in observed}
    for track in tracker.lost_stracks:
        missed = tracker.frame_id - track.end_frame
        if (track.state != TrackState.Lost or not track.is_activated
                or track.track_id in observed_ids or not 0 < missed <= hold_frames):
            continue
        box = np.asarray(track.xyxy, dtype=float).copy()
        if not np.isfinite(box).all():
            continue
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        overlaps_observed = False
        for detection in observed:
            other = np.asarray(detection["bbox"])
            intersection_wh = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
            intersection = float(np.prod(intersection_wh))
            other_area = max(0, other[2] - other[0]) * max(0, other[3] - other[1])
            if intersection / max(area + other_area - intersection, 1e-9) >= 0.5:
                overlaps_observed = True
                break
        if not overlaps_observed:
            predictions.append({
                "track_id": int(track.track_id),
                "bbox": tuple(float(v) for v in box),
                "conf": None,
                "predicted": True,
                "missed_seconds": missed / fps,
            })
    return predictions
