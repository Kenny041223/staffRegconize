"""Person tracking with a person re-identification model: YOLO detections fed
to BoxMOT's BoT-SORT, which matches people by an OSNet appearance embedding as
well as by position.

Same interface as detect_track.PersonTracker.track(). Optional dependency:
`boxmot` (install with --no-deps; it pins opencv-python<5 and pulls lapx,
which clashes with the `lap` package the default tracker uses).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS_ID = 0
DEFAULT_REID_WEIGHTS = Path(__file__).resolve().parent.parent / "osnet_x0_25_msmt17.pt"


class ReidPersonTracker:
    def __init__(self, model_name: str = "yolo26x.pt", device: Optional[str] = None, conf: float = 0.10,
                 imgsz: int = 1280, track_buffer_seconds: float = 6.0, reid_weights=DEFAULT_REID_WEIGHTS):
        if not np.isfinite(track_buffer_seconds) or track_buffer_seconds <= 0:
            raise ValueError("track_buffer_seconds must be finite and positive.")
        from boxmot import BotSort

        self._tracker_cls = BotSort
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.track_buffer_seconds = track_buffer_seconds
        self.reid_weights = Path(reid_weights)

    def track(self, video_path: str) -> Iterator[tuple[int, np.ndarray, list[dict]]]:
        """Yields (frame_idx, frame_bgr, detections) like PersonTracker.track().

        Only real observations are returned (no Kalman-only predictions).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        device = self.device if self.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device).isdigit():
            device = f"cuda:{device}"
        tracker = self._tracker_cls(
            reid_weights=self.reid_weights, device=device, half=False,
            track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.35,
            # BoxMOT expresses this input at 30 fps, then scales by frame_rate/30.
            track_buffer=max(1, round(self.track_buffer_seconds * 30)),
            frame_rate=max(1, round(fps)), use_cmc=False)  # fixed camera: no motion compensation
        # Use the exact source FPS, including fractional rates, for the lifetime.
        tracker.buffer_size = max(1, round(self.track_buffer_seconds * fps))
        tracker.max_time_lost = tracker.buffer_size
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                boxes = self.model.predict(frame, classes=[PERSON_CLASS_ID], conf=self.conf, imgsz=self.imgsz,
                                           device=device, verbose=False)[0].boxes
                if len(boxes):
                    dets = np.concatenate([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None],
                                           boxes.cls.cpu().numpy()[:, None]], axis=1)
                else:
                    dets = np.empty((0, 6), dtype=np.float32)
                tracks = np.asarray(tracker.update(dets, frame))
                detections = [{"track_id": int(t[4]), "bbox": tuple(float(v) for v in t[:4]),
                               "conf": float(t[5]), "predicted": False, "missed_seconds": 0.0}
                              for t in tracks]
                yield frame_idx, frame, detections
                frame_idx += 1
        finally:
            cap.release()
