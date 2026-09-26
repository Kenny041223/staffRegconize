"""Follow badge-confirmed people with SAM 2.1 video segmentation.

The badge pipeline decides *who* is staff; SAM 2 decides *where* that person is,
instead of tracker ID numbers, which break when people pass close to each other.
A confident badge sighting prompts SAM 2 with that person's box, and SAM 2
follows their pixels forwards and backwards until it loses them for
`lost_seconds`. A follow counts as staff only when at least min_hits badge
sightings fall inside its outline. The video is processed in windows (the last
outline of one window prompts the next), so memory stays bounded for long videos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_SAM2 = "facebook/sam2.1-hiera-large"


@dataclass
class Follow:
    anchor_frame: int
    anchor_box: list
    boxes: dict = field(default_factory=dict)      # frame -> [x1, y1, x2, y2] of the outline
    inside: dict = field(default_factory=dict)     # probed frame -> whether its point was inside the outline
    support: list = field(default_factory=list)    # frames of badge sightings inside the outline
    confirmed_at: int | None = None

    def to_json(self):
        return {"anchor_frame": self.anchor_frame, "anchor_box": self.anchor_box, "support_frames": self.support,
                "confirmed_at_frame": self.confirmed_at, "boxes": {str(f): b for f, b in sorted(self.boxes.items())}}

    @classmethod
    def from_json(cls, data):
        return cls(data["anchor_frame"], data["anchor_box"], {int(f): b for f, b in data["boxes"].items()},
                   support=data["support_frames"], confirmed_at=data["confirmed_at_frame"])


class SamFollower:
    def __init__(self, model_name=DEFAULT_SAM2, device=None, window_seconds=12.0, lost_seconds=0.5,
                 min_area=150, max_area_fraction=0.25):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device).isdigit():
            self.device = f"cuda:{self.device}"
        self.model_name = model_name
        self.processor = Sam2VideoProcessor.from_pretrained(model_name)
        self.model = Sam2VideoModel.from_pretrained(model_name).to(self.device, dtype=torch.float32).eval()
        self.window_seconds, self.lost_seconds = window_seconds, lost_seconds
        self.min_area, self.max_area_fraction = min_area, max_area_fraction

    @staticmethod
    def _read(video_path, lo, hi):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        frames = []
        for _ in range(lo, hi + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
        cap.release()
        return frames

    def _window(self, frames, box=None, mask=None, reverse=False):
        """Masks for one window, prompted at its first frame (or last frame when reverse)."""
        torch = self.torch
        with torch.inference_mode():
            session = self.processor.init_video_session(video=frames, inference_device=self.device,
                                                        video_storage_device="cpu", dtype=torch.float32)
            idx = len(frames) - 1 if reverse else 0
            if box is not None:
                self.processor.add_inputs_to_inference_session(session, frame_idx=idx, obj_ids=1, input_boxes=[[box]])
            else:
                self.processor.add_inputs_to_inference_session(session, frame_idx=idx, obj_ids=1, input_masks=[mask])
            self.model(inference_session=session, frame_idx=idx)
            masks = {}
            for out in self.model.propagate_in_video_iterator(session, start_frame_idx=idx, reverse=reverse):
                m = self.processor.post_process_masks([out.pred_masks], binarize=True,
                                                      original_sizes=[[session.video_height, session.video_width]])[0]
                present = float(out.object_score_logits.flatten()[0]) > 0
                masks[out.frame_idx] = (m[0, 0].cpu().numpy().astype(np.uint8), present)
        return masks

    def follow(self, video_path, n_frames, fps, frame_idx, box, probes=None):
        """Follow the person in `box` at `frame_idx` both ways. `probes` {frame: (x, y)} records
        whether each point lies inside the outline (used for badge-sighting support)."""
        probes = probes or {}
        follow = Follow(frame_idx, [float(v) for v in box])
        window = max(2, round(self.window_seconds * fps))
        lost_limit = max(1, round(self.lost_seconds * fps))
        for reverse in (False, True):
            pos, prompt_box, prompt_mask, missing = frame_idx, box, None, 0
            while (pos < n_frames - 1) if not reverse else (pos > 0):
                lo, hi = (pos, min(n_frames - 1, pos + window - 1)) if not reverse else (max(0, pos - window + 1), pos)
                frames = self._read(video_path, lo, hi)
                if len(frames) < 2:
                    break
                hi = lo + len(frames) - 1
                masks = self._window(frames, prompt_box, prompt_mask, reverse)
                order = range(len(frames)) if not reverse else range(len(frames) - 1, -1, -1)
                last_mask, stopped = None, False
                for k in order:
                    f = lo + k
                    mask, present = masks.get(k, (None, False))
                    area = int(mask.sum()) if mask is not None else 0
                    if present and self.min_area <= area <= self.max_area_fraction * mask.size:
                        ys, xs = np.nonzero(mask)
                        follow.boxes[f] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        if f in probes:
                            x, y = probes[f]
                            follow.inside[f] = bool(mask[min(mask.shape[0] - 1, max(0, int(y))),
                                                         min(mask.shape[1] - 1, max(0, int(x)))])
                        last_mask, missing = mask, 0
                    else:
                        missing += 1
                        if missing > lost_limit:
                            stopped = True
                            break
                if stopped or last_mask is None:
                    break
                # The window's last frame prompts the next window with its outline.
                pos, prompt_box, prompt_mask = (hi if not reverse else lo), None, last_mask
                if (not reverse and hi >= n_frames - 1) or (reverse and lo <= 0):
                    break
        return follow


def sightings(checks, frames, lookup, policy, track_key, exclusive_owner):
    """Confident, owned badge sightings: [(frame, tag centre, person box, decision frame)]."""
    found = []
    for check in checks:
        frame_idx, key, tag = check["frame_idx"], track_key(check), check.get("tag_bbox")
        if (tag is None or not math.isfinite(check["raw_score"]) or check["raw_score"] < policy.min_score
                or lookup.get((frame_idx, key)) is None or not exclusive_owner(tag, key, frames[frame_idx])):
            continue
        found.append((frame_idx, ((tag[0] + tag[2]) / 2, (tag[1] + tag[3]) / 2), check["person_bbox"],
                      max(frame_idx, check.get("processed_at_frame", frame_idx))))
    return sorted(found)


def follow_staff(seen, follower_fn, fps, policy, max_follows=5):
    """Start a follow from each sighting no earlier follow explains; keep follows with min_hits support."""
    probes = {f: centre for f, centre, _, _ in seen}
    follows = []
    for frame_idx, centre, person_box, _ in seen:
        if any(fol.inside.get(frame_idx) for fol in follows):
            continue
        if len(follows) >= max_follows:
            break
        follows.append(follower_fn(frame_idx, person_box, probes))
    spacing = max(1, round(policy.min_hit_spacing_seconds * fps))
    decision = {f: d for f, _, _, d in seen}
    for fol in follows:
        fol.support = []
        for f in sorted(f for f, inside in fol.inside.items() if inside):
            if not fol.support or f - fol.support[-1] >= spacing:
                fol.support.append(f)
        if len(fol.support) >= policy.min_hits:
            fol.confirmed_at = max(decision[f] for f in fol.support[:policy.min_hits])
    return follows


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def apply_follows(decisions, follows, min_iou=0.3):
    """Replace staff labels with confirmed follows. Each followed frame labels the detection that best
    matches the outline, or adds the outline's box when the person detector missed that frame."""
    for dets in decisions.frames.values():
        for det in dets:
            if det.get("status") == "confirmed_staff":
                det.update(status="unknown", confirmed_at_frame=None, label_source="unconfirmed")
    intervals = []
    for person, fol in enumerate(f for f in follows if f.confirmed_at is not None):
        for frame_idx, box in fol.boxes.items():
            dets = decisions.frames.setdefault(frame_idx, [])
            best = max(dets, key=lambda d: _iou(d["bbox"], box), default=None)
            if best is None or _iou(best["bbox"], box) < min_iou:
                best = {"track_id": -1, "segment": 0, "bbox": tuple(float(v) for v in box), "continuity_id": -1,
                        "identity_state": "clear", "identity_reason": "sam2_only"}
                dets.append(best)
                source = "sam2_follow_no_detection"
            else:
                source = "sam2_follow"
            best.update(status="confirmed_staff", confirmed_at_frame=fol.confirmed_at, label_source=source,
                        person_id=person, display_id=100000 + person)
        runs = []
        for f in sorted(fol.boxes):
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        intervals += [{"person_id": person, "start_frame": a, "end_frame": b, "confirmed_at_frame": fol.confirmed_at,
                       "hit_frames": [f for f in fol.support if a <= f <= b], "person_hit_frames": fol.support,
                       "anchor_frame": fol.anchor_frame} for a, b in runs]
    decisions.intervals = sorted(intervals, key=lambda i: i["start_frame"])
    return decisions
