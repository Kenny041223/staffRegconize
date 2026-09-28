"""OWLv2 badge matching with a marked reference and background rejection.

Image and text queries are cached; all checks share one image forward pass.
No weights are trained. Similarity scores are not employment probabilities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert, box_iou, nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DEFAULT_TAG_MODEL = "google/owlv2-base-patch16-ensemble"
MATCHER_VERSION = "marked_references_rotations_background_check_v3"
# A ceiling camera sees the badge upright, sideways or upside down. Matching the
# reference at all four rotations (0, 90, 180, 270) finds more real sightings, but
# on sample.mp4 it also let a look-alike on another person score 0.91-0.99, so
# only the upright reference is used by default.
DEFAULT_ROTATIONS = (0,)
_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def rotate_box(box, size, degrees):
    """Where box lands after rotating an image of size (width, height) counter-clockwise."""
    width, height = size
    x1, y1, x2, y2 = box
    return {0: (x1, y1, x2, y2),
            90: (y1, width - x2, y2, width - x1),
            180: (width - x2, height - y2, width - x1, height - y1),
            270: (height - y2, x1, height - y1, x2)}[degrees]


def rotate_reference(image, box, degrees):
    """Rotate the reference counter-clockwise by 0/90/180/270 degrees, with its badge box."""
    if degrees not in (0, 90, 180, 270):
        raise ValueError("Reference rotations must be 0, 90, 180 or 270 degrees.")
    rotated = image if degrees == 0 else image.transpose(_TRANSPOSE[degrees])
    return rotated, rotate_box(box, image.size, degrees)
BADGE_PROMPTS = (
    "a photo of an identification badge.", "a photo of a name tag.",
    "a photo of a black and white badge.", "a photo of a rectangular clothing logo.",
)
BACKGROUND_PROMPTS = (
    "a photo of a hand.", "a photo of a face.", "a photo of a chair.",
    "a photo of a shirt collar.", "a photo of a shirt pocket.",
    "a photo of a zipper.", "a photo of a wristwatch.", "a photo of a shoe.",
    "a photo of a table.", "a photo of a phone.",
    "a photo of a laptop.", "a photo of a sticker on a laptop.",
    "a photo of a label on furniture.",
)


def resolve_reference_box(image_path, image_size, reference_box=None):
    """Read a hash-bound annotation, or use the supplied box/tight whole image."""
    path = Path(image_path)
    source = "explicit" if reference_box is not None else "whole_image"
    sidecar = path.with_suffix(".tag.json")
    if reference_box is None and sidecar.exists():
        annotation = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != annotation["image_sha256"].lower():
            raise ValueError(f"Reference annotation does not match the image: {sidecar}. Supply --reference-box again.")
        reference_box, source = annotation["box"], str(sidecar.resolve())
    width, height = image_size
    box = tuple(reference_box) if reference_box is not None else (0, 0, width, height)
    if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("reference_box must be a nonempty box inside the reference image.")
    return box, source


def select_reference_embedding(class_embeds, boxes, objectness_logits, image_size, reference_box):
    """Select a foreground proposal covering the marked badge, in original pixels.

    The library's automatic full-image query can select clothing or square
    padding. Keep image context, but explicitly choose the badge's feature.
    """
    xyxy = box_convert(boxes[0].float(), "cxcywh", "xyxy") * max(image_size)
    roi = torch.tensor([reference_box], device=boxes.device, dtype=torch.float32)
    overlaps = box_iou(roi, xyxy)[0]
    objectness = objectness_logits[0].float().sigmoid()
    valid = torch.isfinite(xyxy).all(-1) & torch.isfinite(objectness) & (overlaps >= 0.25)
    if not valid.any():
        raise ValueError("OWLv2 found no proposal covering the marked badge. Check the reference box/image.")
    index = (overlaps * objectness).masked_fill(~valid, -1).argmax().item()
    return class_embeds[:, index:index + 1], {
        "selected_box": xyxy[index].cpu().tolist(), "iou": overlaps[index].item(),
        "objectness": objectness[index].item(),
    }


def badge_proposal_mask(boxes, image_sizes, objectness_logits, semantic_logits,
                        positive_count=len(BADGE_PROMPTS), min_objectness=0.01,
                        max_area_ratio=0.08, text_margin=0.5):
    """Reject crop edges, oversized/line-like boxes, and background categories.

    Use text-vs-text logits for the semantic comparison; their absolute scale
    differs from image-query scores. Apply this BEFORE top-k/NMS so background
    proposals cannot crowd out a genuine badge.
    """
    masks = []
    for image_boxes, (height, width), obj, semantics in zip(
            boxes, image_sizes, objectness_logits, semantic_logits):
        xyxy = box_convert(image_boxes.float(), "cxcywh", "xyxy") * max(height, width)
        wh = xyxy[:, 2:] - xyxy[:, :2]
        border = max(1.0, 0.01 * min(height, width))
        valid = (torch.isfinite(xyxy).all(-1) & (xyxy[:, 0] >= border) & (xyxy[:, 1] >= border)
                 & (xyxy[:, 2] <= width - border) & (xyxy[:, 3] <= height - border)
                 & (wh.amin(-1) >= 3) & (wh.amax(-1) <= 3.5 * wh.amin(-1))
                 & (wh.prod(-1) <= max_area_ratio * height * width)
                 & torch.isfinite(obj) & (obj.float().sigmoid() >= min_objectness))
        margin = semantics[:, :positive_count].amax(-1) - semantics[:, positive_count:].amax(-1)
        masks.append(valid & torch.isfinite(margin) & (margin > text_margin))
    return torch.stack(masks)


def postprocess_raw(logits, boxes, image_sizes, threshold=0.0, nms_threshold=0.3,
                    pre_nms_top_k=100, max_detections=20, proposal_mask=None, return_indices=False):
    """Return raw sigmoid scores and crop-pixel xyxy boxes.

    OWLv2 pads at the bottom/right to a square BEFORE resizing: all coordinates
    scale by max(height, width). Filter invalid/padded boxes and bound NMS work.
    These scores are model similarities, not calibrated employment probabilities.
    """
    if not 0 <= threshold <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError("Score and NMS thresholds must be in [0, 1].")
    if pre_nms_top_k < 1 or max_detections < 1:
        raise ValueError("Detection limits must be positive.")
    if len(logits) != len(boxes) or len(logits) != len(image_sizes):
        raise ValueError("Logits, boxes and image_sizes must have matching batches.")
    results = []
    if proposal_mask is not None and proposal_mask.shape != boxes.shape[:2]:
        raise ValueError("proposal_mask must match the batch and proposal dimensions.")
    for index, (image_logits, image_boxes, (height, width)) in enumerate(zip(logits, boxes, image_sizes)):
        scores = image_logits.float().amax(dim=-1).sigmoid()
        xyxy = box_convert(image_boxes.float(), in_fmt="cxcywh", out_fmt="xyxy") * max(height, width)
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        valid = (torch.isfinite(xyxy).all(dim=1) & torch.isfinite(scores)
                 & (scores >= threshold) & (centers[:, 0] >= 0) & (centers[:, 0] < width)
                 & (centers[:, 1] >= 0) & (centers[:, 1] < height))
        if proposal_mask is not None:
            valid &= proposal_mask[index]
        xyxy[:, 0::2].clamp_(0, width)
        xyxy[:, 1::2].clamp_(0, height)
        valid &= (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        scores, xyxy = scores[valid], xyxy[valid]
        if scores.numel() == 0:
            results.append({"scores": [], "boxes": [], **({"indices": []} if return_indices else {})})
            continue
        scores, order = scores.topk(min(pre_nms_top_k, scores.numel()))
        xyxy = xyxy[order]
        keep = nms(xyxy, scores, nms_threshold)[:max_detections]
        result = {"scores": scores[keep].cpu().tolist(), "boxes": xyxy[keep].cpu().tolist()}
        if return_indices:
            result["indices"] = valid.nonzero().flatten()[order][keep].cpu().tolist()
        results.append(result)
    return results


class TagMatcher:
    def __init__(self, reference_image_path: str, device: Optional[str] = None,
                 model_name: str = DEFAULT_TAG_MODEL, reference_box=None,
                 min_similarity=0.65, min_objectness=0.01, max_area_ratio=0.08, text_margin=0.5,
                 rotations=DEFAULT_ROTATIONS):
        if not (0 <= min_similarity <= 1 and 0 <= min_objectness <= 1 and 0 < max_area_ratio <= 1):
            raise ValueError("Invalid badge filtering thresholds.")
        if not np.isfinite(text_margin):
            raise ValueError("text_margin must be finite.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device).isdigit():
            self.device = f"cuda:{self.device}"
        self.model_name = model_name
        # One or more references; each carries its own marked badge box (.tag.json).
        paths = ([reference_image_path] if isinstance(reference_image_path, (str, Path))
                 else list(reference_image_path))
        if not paths:
            raise ValueError("At least one reference image is required.")
        if reference_box is not None and len(paths) > 1:
            raise ValueError("reference_box marks a single reference; give each reference its own .tag.json.")
        references = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
            box, source = resolve_reference_box(path, image.size, reference_box)
            references.append((str(Path(path).resolve()), image, box, source))
        self.min_similarity = min_similarity
        self.filter_settings = {"min_objectness": min_objectness, "max_area_ratio": max_area_ratio,
                                "text_margin": text_margin}
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.forward_calls = 0
        self.images_scanned = 0
        self.rotations = tuple(dict.fromkeys(rotations))
        if not self.rotations:
            raise ValueError("At least one reference rotation is required.")
        with torch.inference_mode():
            # One image query per (reference, rotation); a crop's score is its best match.
            embeddings, selected, self._query_rotations = [], [], []
            for path, reference, marked, source in references:
                for degrees in self.rotations:
                    image, box = rotate_reference(reference, marked, degrees)
                    inputs = self.processor(images=image, return_tensors="pt")
                    feature_map = self.model.image_embedder(pixel_values=inputs["pixel_values"].to(self.device))[0]
                    b, ph, pw, hd = feature_map.shape
                    features = feature_map.reshape(b, ph * pw, hd)
                    _, class_embeds = self.model.class_predictor(features)
                    embedding, info = select_reference_embedding(
                        class_embeds, self.model.box_predictor(features, feature_map),
                        self.model.objectness_predictor(features), image.size, box)
                    embeddings.append(embedding)
                    self._query_rotations.append(degrees)
                    selected.append({"reference": path, "box": list(marked), "source": source,
                                     "rotation": degrees, **info})
            self._query_embeds = torch.cat(embeddings, dim=1)
            tokens = self.processor(text=list(BADGE_PROMPTS + BACKGROUND_PROMPTS), return_tensors="pt")
            text_features = self.model.owlv2.text_model(
                input_ids=tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
            self._semantic_embeds = self.model.owlv2.text_projection(text_features).unsqueeze(0)
            self._all_queries = torch.cat([self._query_embeds, self._semantic_embeds], dim=1)
        self.metadata = {"version": MATCHER_VERSION,
                         "reference_selection": selected,
                         "filters": {"min_similarity": min_similarity, **self.filter_settings,
                                     "min_side_pixels": 3, "max_aspect_ratio": 3.5},
                         "badge_prompts": list(BADGE_PROMPTS), "background_prompts": list(BACKGROUND_PROMPTS)}


    @torch.inference_mode()
    def detect_batch(self, images_bgr: list[np.ndarray], threshold: float | None = None,
                     nms_threshold: float = 0.3, upright_recheck: bool = True) -> list[dict]:
        """Scan a small batch, preserving order including empty crops.

        Caller controls batch size; keep it small when sharing VRAM with YOLO.
        """
        results = [{"scores": [], "boxes": []} for _ in images_bgr]
        valid = [(i, im) for i, im in enumerate(images_bgr) if im is not None and im.size]
        if not valid:
            return results
        images = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1])) for _, im in valid]
        inputs = self.processor(images=images, return_tensors="pt")
        feature_map = self.model.image_embedder(pixel_values=inputs["pixel_values"].to(self.device))[0]
        b, ph, pw, hd = feature_map.shape
        image_feats = feature_map.reshape(b, ph * pw, hd)
        logits, _ = self.model.class_predictor(image_feats=image_feats, query_embeds=self._all_queries.expand(b, -1, -1))
        boxes = self.model.box_predictor(image_feats, feature_map)
        sizes = [im.shape[:2] for _, im in valid]
        objectness_logits = self.model.objectness_predictor(image_feats)
        n = self._query_embeds.shape[1]  # one image query per reference and rotation; the rest are text
        mask = badge_proposal_mask(boxes, sizes, objectness_logits,
                                   logits[:, :, n:], **self.filter_settings)
        threshold = self.min_similarity if threshold is None else threshold
        # postprocess_raw keeps the best rotation for each proposal.
        decoded = postprocess_raw(logits[:, :, :n], boxes, sizes, threshold, nms_threshold,
                                  proposal_mask=mask, return_indices=True)
        similar = logits[:, :, :n].amax(-1).sigmoid() >= threshold
        counts = torch.stack([similar.sum(-1), (similar & mask).sum(-1)], dim=-1).cpu().tolist()
        self.forward_calls += 1
        self.images_scanned += len(valid)
        for slot, ((index, _), result, (proposed, accepted)) in enumerate(zip(valid, decoded, counts)):
            indices = result.pop("indices")
            semantic = logits[slot, indices, n:]
            result["objectness"] = objectness_logits[slot, indices].sigmoid().cpu().tolist()
            result["text_margins"] = (semantic[:, :len(BADGE_PROMPTS)].amax(-1)
                                      - semantic[:, len(BADGE_PROMPTS):].amax(-1)).cpu().tolist()
            result["filter_stats"] = {"similar_proposals": proposed, "accepted_before_nms": accepted}
            results[index] = result
        if upright_recheck:
            self._recheck_upright(images_bgr, valid, results, logits, boxes, sizes, objectness_logits, mask, n,
                                  threshold, nms_threshold)
        return results

    def _recheck_upright(self, images_bgr, valid, results, logits, boxes, sizes, objectness_logits, mask, n,
                         threshold, nms_threshold):
        """The background-word check only recognises an upright badge. When a proposal
        best matches a rotated reference but only that check rejected it, rotate the
        crop so the badge is upright and scan it again with every check applied."""
        geometry = badge_proposal_mask(boxes, sizes, objectness_logits, logits[:, :, n:],
                                       **dict(self.filter_settings, text_margin=float("-inf")))
        similarity, rotation = logits[:, :, :n].sigmoid().max(-1)
        rejected_by_text = geometry & ~mask & (similarity >= threshold)
        retries = []
        for slot, (index, _) in enumerate(valid):
            if not rejected_by_text[slot].any():
                continue
            top = torch.where(rejected_by_text[slot], similarity[slot], torch.full_like(similarity[slot], -1)).argmax()
            degrees = self._query_rotations[rotation[slot, top].item()]
            accepted = results[index]["scores"][0] if results[index]["scores"] else 0.0
            if degrees and similarity[slot, top].item() > accepted:
                retries.append((index, degrees))
        if not retries:
            return
        # The reference matched after turning it `degrees` counter-clockwise, so turn the crop clockwise.
        upright = [np.rot90(images_bgr[i], k=(4 - degrees // 90) % 4) for i, degrees in retries]
        for (index, degrees), found in zip(retries, self.detect_batch(upright, threshold, nms_threshold,
                                                                      upright_recheck=False)):
            height, width = images_bgr[index].shape[:2]
            turned = (width, height) if degrees == 180 else (height, width)
            original = results[index]
            rows = list(zip(original["scores"], original["boxes"], original["objectness"], original["text_margins"]))
            rows += [(score, list(rotate_box(box, turned, degrees)), objectness, margin) for score, box, objectness, margin
                     in zip(found["scores"], found["boxes"], found["objectness"], found["text_margins"])]
            rows.sort(key=lambda row: -row[0])
            results[index] = dict(original, scores=[r[0] for r in rows], boxes=[r[1] for r in rows],
                                  objectness=[r[2] for r in rows], text_margins=[r[3] for r in rows],
                                  upright_recheck={"rotation": degrees, "found": len(found["scores"])})

    def detect(self, image_bgr: np.ndarray, threshold: float | None = None, nms_threshold: float = 0.3) -> dict:
        return self.detect_batch([image_bgr], threshold, nms_threshold)[0]

    def best_score(self, image_bgr: np.ndarray) -> float:
        result = self.detect(image_bgr)
        return result["scores"][0] if result["scores"] else 0.0
