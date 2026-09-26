"""Generate synthetic badge training data for YOLO.

A pose model finds each person's shoulders and hips; the badge is pasted on the
chest, turned so it is upright relative to the body, shrunk to camera size and
degraded (blur, contrast, colour, JPEG). Labels come from where it was pasted.
The same people without a badge are the negatives.

Tracks listed with --exclude (every track that was ever the staff member, and
look-alikes kept for testing) are left out so the real-video test stays fair.

1. Generate (on sample.mp4 the excluded tracks were his 9 IDs and look-alikes #3, #5, #9):
    python scripts/train_badge/gen_badges.py --observations output/tag_scan/RUN/observations.csv
        --exclude 48 11 62 69 75 29 107 108 115 3 5 9
2. Train (about 5 minutes on an RTX 5090):
    yolo detect train data=badge_data/data.yaml model=yolo26s.pt imgsz=416 epochs=60 batch=64
        patience=20 flipud=0.5 fliplr=0.5 degrees=0 scale=0.3 translate=0.1 close_mosaic=10
3. Copy runs/detect/train/weights/best.pt to yolo_folder/badge_yolo26s.pt.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "badge_data"
rng = random.Random(0)


def sticker(path, box, turn=0, pad=1):
    image = cv2.imread(str(path))
    x1, y1, x2, y2 = box
    crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    return np.ascontiguousarray(np.rot90(crop, k=turn // 90)) if turn else crop


# Each sticker is turned so the badge is upright relative to its wearer.
STICKERS = [sticker(ROOT / "assets/reference_1.jpg", (74, 16, 115, 44)),
            sticker(ROOT / "assets/reference_2_full.png", (88, 97, 109, 112)),
            sticker(ROOT / "assets/reference_3_full.png", (48, 107, 70, 121), turn=180)]
WEIGHTS = [0.4, 0.3, 0.3]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def paste(crop, kp):
    """Paste one badge on the chest; return its box in crop pixels, or None."""
    ls, rs, lh, rh = kp[5], kp[6], kp[11], kp[12]
    shoulders, hips = (ls + rs) / 2, (lh + rh) / 2
    torso = hips - shoulders
    length, width = np.linalg.norm(torso), np.linalg.norm(ls - rs)
    if length < 4 or width < 4:
        return None
    up = -torso / length
    side = np.array([-up[1], up[0]])
    centre = shoulders + torso * rng.uniform(0.15, 0.4) + side * width * rng.uniform(-0.3, 0.3)
    long_side = float(np.clip(width * rng.uniform(0.22, 0.45), 7, 30))
    art = rng.choices(STICKERS, WEIGHTS)[0]
    aspect = art.shape[1] / art.shape[0] * rng.uniform(0.8, 1.25)
    w, h = max(4, round(long_side)), max(3, round(long_side / aspect))
    badge = cv2.resize(art, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    badge = badge * rng.uniform(0.6, 1.05) + rng.uniform(-20, 10)            # lighting
    cx, cy = int(centre[0]), int(centre[1])
    patch = crop[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6]
    if patch.size:
        t = rng.uniform(0, 0.25)                                              # pick up local colour
        badge = badge * (1 - t) + patch.reshape(-1, 3).mean(0) * t
    theta = math.degrees(math.atan2(up[0], -up[1])) + rng.uniform(-15, 15)    # body-up, clockwise
    diag = int(math.ceil(math.hypot(w, h))) + 4
    canvas = np.zeros((diag, diag, 3), np.float32)
    mask = np.zeros((diag, diag), np.float32)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    canvas[oy:oy + h, ox:ox + w], mask[oy:oy + h, ox:ox + w] = badge, 1
    turn = cv2.getRotationMatrix2D((diag / 2, diag / 2), -theta, 1.0)
    canvas = cv2.warpAffine(canvas, turn, (diag, diag), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, turn, (diag, diag), flags=cv2.INTER_LINEAR)
    sigma = rng.uniform(0.3, 1.1)
    canvas, mask = cv2.GaussianBlur(canvas, (0, 0), sigma), cv2.GaussianBlur(mask, (0, 0), max(0.4, sigma))
    px, py = int(round(centre[0] - diag / 2)), int(round(centre[1] - diag / 2))
    height, width_px = crop.shape[:2]
    if px < 0 or py < 0 or px + diag > width_px or py + diag > height:
        return None
    region = crop[py:py + diag, px:px + diag].astype(np.float32)
    m = mask[..., None]
    crop[py:py + diag, px:px + diag] = np.clip(region * (1 - m) + canvas * m, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(mask > 0.35)
    return px + xs.min(), py + ys.min(), px + xs.max() + 1, py + ys.max() + 1


def degrade(crop):
    """Same camera-like degradation for positives and negatives."""
    if rng.random() < 0.5:
        f = rng.uniform(0.6, 1.0)
        small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        crop = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def save(crop, box, name, split):
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = crop.shape[:2]
    text = "" if box is None else "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
        (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h)
    (OUT / "labels" / split / f"{name}.txt").write_text(text)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(ROOT.parent / "sample.mp4"))
    parser.add_argument("--observations", required=True, help="observations.csv of a finished identify_staff run")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="Track IDs never used in training.")
    parser.add_argument("--pose-model", default="yolo26x-pose.pt")
    parser.add_argument("--step", type=int, default=2, help="Use every Nth frame.")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    OUT, excluded = Path(args.out), set(args.exclude)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    tracks = defaultdict(list)
    for r in csv.DictReader(open(args.observations)):
        tracks[int(r["frame_idx"])].append((int(r["track_id"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    pose = YOLO(args.pose_model)
    cap = cv2.VideoCapture(args.video)
    counts = defaultdict(int)
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        frame_idx += 1
        if not ok:
            break
        if frame_idx % args.step:
            continue
        result = pose.predict(frame, imgsz=1280, conf=0.1, verbose=False)[0]
        if not len(result.boxes):
            continue
        kps, kconf = result.keypoints.xy.cpu().numpy(), result.keypoints.conf.cpu().numpy()
        for pbox, kp, kc in zip(result.boxes.xyxy.cpu().numpy(), kps, kconf):
            match = max(tracks[frame_idx], key=lambda tb: iou(tb[1], pbox), default=None)
            if match is None or iou(match[1], pbox) < 0.5 or match[0] in excluded:
                counts["skipped (excluded / unmatched)"] += 1
                continue
            tid, (x1, y1, x2, y2) = match
            pad_x, pad_y = (x2 - x1) * 0.05, (y2 - y1) * 0.05                 # same crop as the pipeline
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right, bottom = min(frame.shape[1], int(np.ceil(x2 + pad_x))), min(frame.shape[0], int(np.ceil(y2 + pad_y)))
            if min(right - left, bottom - top) < 32:
                continue
            base = frame[top:bottom, left:right]
            split = "val" if tid % 7 == 0 else "train"
            name = f"f{frame_idx:04d}_t{tid}"
            good_pose = (kc[[5, 6, 11, 12]] > 0.3).all()
            if good_pose and rng.random() < 0.65:
                crop = base.copy()
                box = paste(crop, kp - np.array([left, top]))
                if box is not None:
                    save(degrade(crop), box, name + "_pos", split)
                    counts[f"{split} with badge"] += 1
                    if rng.random() > 0.35:
                        continue
            save(degrade(base.copy()), None, name + "_neg", split)
            counts[f"{split} without badge"] += 1
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: badge\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
