"""Fast YOLO badge pre-filter in front of the OWLv2 matcher.

A small YOLO model trained on synthetic badges (yolo_folder/badge_yolo26s.pt) scans every
person crop in a few milliseconds. Only crops where it sees something badge-like
go on to OWLv2, which still makes every decision. On sample.mp4 a 0.01 pre-filter
kept exactly the same staff frames while OWLv2 checked about 19% of the crops.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PREFILTER = Path(__file__).resolve().parent.parent / "yolo_folder" / "badge_yolo26s.pt"


def _skipped():
    return {"scores": [], "boxes": [], "objectness": [], "text_margins": [],
            "filter_stats": {"prefilter": "skipped"}}


class PrefilteredMatcher:
    """Same interface as TagMatcher; OWLv2 only sees crops the detector flags."""

    def __init__(self, matcher, detector, conf=0.01, imgsz=416, weights=None, device=None):
        if not 0 <= conf <= 1:
            raise ValueError("Pre-filter confidence must be in [0, 1].")
        self.matcher, self.detector = matcher, detector
        self.conf, self.imgsz, self.weights, self.predict_device = conf, imgsz, weights, device
        self.passed = self.skipped = 0

    @classmethod
    def load(cls, matcher, weights, conf=0.01, imgsz=416, device=None):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Badge pre-filter weights not found: {weights}")
        from ultralytics import YOLO

        return cls(matcher, YOLO(str(weights)), conf, imgsz, str(weights.resolve()), device)

    model_name = property(lambda self: self.matcher.model_name)
    device = property(lambda self: self.matcher.device)
    forward_calls = property(lambda self: self.matcher.forward_calls)

    @property
    def metadata(self):
        return {**getattr(self.matcher, "metadata", {}),
                "prefilter": {"weights": self.weights, "conf": self.conf, "imgsz": self.imgsz,
                              "crops_to_owlv2": self.passed, "crops_skipped": self.skipped}}

    def flags(self, images):
        kwargs = {"device": self.predict_device} if self.predict_device is not None else {}
        predictions = self.detector.predict(images, imgsz=self.imgsz, conf=self.conf, verbose=False, **kwargs)
        return [len(p.boxes) > 0 for p in predictions]

    def detect_batch(self, images_bgr, **kwargs):
        results = [_skipped() for _ in images_bgr]
        valid = [(i, image) for i, image in enumerate(images_bgr) if image is not None and image.size]
        if not valid:
            return results
        chosen = [(i, image) for (i, image), flagged in zip(valid, self.flags([im for _, im in valid])) if flagged]
        self.passed += len(chosen)
        self.skipped += len(valid) - len(chosen)
        if chosen:
            for (i, _), result in zip(chosen, self.matcher.detect_batch([im for _, im in chosen], **kwargs)):
                results[i] = result
        return results
