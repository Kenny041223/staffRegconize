# staffRegconize

FootfallCam AI Evaluation - staff identification from CCTV footage.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# or: .venv\Scripts\activate.bat     # Windows cmd

# PyTorch: cu128 for an RTX 5090 (Blackwell); cu124 also works on older GPUs such as a GTX 1070 Ti.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install boxmot==25.0.0 --no-deps   # ReID tracker for --reid-tracker; see requirements.txt
python scripts/download_models.py      # yolo26x.pt, osnet_x0_25_msmt17.pt, OWLv2
```

## Running on a rented GPU (vast.ai, RTX 5090)

1. Rent an RTX 5090 **on-demand** (not interruptible) instance with at least 50 GB of disk
   and SSH access, using a **PyTorch image built for CUDA 12.8 or newer**. The image already
   has PyTorch, so skip the `pip install torch` line.
2. On the instance (Linux):
   ```bash
   apt-get update && apt-get install -y libgl1 libglib2.0-0   # OpenCV needs these on server images
   git clone -b rtx5090_version https://github.com/Kenny041223/staffRegconize.git code
   cd code
   pip install -r requirements.txt
   pip install boxmot==25.0.0 --no-deps
   python scripts/download_models.py
   python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
   ```
3. Upload the video from your PC, e.g. `scp -P <port> sample.mp4 root@<host>:~/`.
   If `download_models.py` cannot fetch the ReID weights, upload
   `osnet_x0_25_msmt17.pt` into `code/` the same way.
4. Run the scan, then draw the staff-only video:
   ```bash
   python src/identify_staff.py ~/sample.mp4 --reid-tracker --tag-threshold 0.9 --confirmations 2 \
       --batch-size 4 --output-dir output/tag_scan/run_5090
   python src/render_evidence_video.py output/tag_scan/run_5090 ~/sample.mp4 output/runs/staff_5090.mp4 \
       --staff-min-hits 2 --staff-score 0.9
   ```
5. Download `output/runs/staff_5090.mp4` and `output/runs/staff_5090.staff.csv` to your PC
   (`scp -P <port> root@<host>:~/code/output/runs/staff_5090.* .`), then stop the instance.

## Person tracking

From the `code` directory, run:

```bash
python src/track_people.py
```

This uses `yolo26x.pt` and BoT-SORT with appearance matching. Each run saves a
new annotated video under `output/runs/`. Coloured boxes are observations;
thin grey boxes labelled **estimated** are temporary Kalman predictions when
the detector misses an already confirmed person. Predictions have no detection
confidence and do not add to trails or the observed track-ID count.

The detector cutoff is **0.10**, matching BoT-SORT's low-confidence threshold.
The previous cutoff of 0.35 discarded all low-confidence candidates before the
tracker could use them. A new track needs confidence **0.35** (the old effective
limit), so weaker detections can maintain an existing ID without starting a
new one by themselves.
After a completely missed frame, BoT-SORT needs a high-stage match to recover
the lost track; low-confidence recovery alone does not solve every gap.

Lost identities stay available for **6 seconds**, calculated from the source
video FPS. Estimated boxes are displayed for at most **1 second**. These are
separate limits: a long memory helps re-matching, while a short display limit
reduces stale boxes after people leave. Every new video resets tracker state.
Camera-motion compensation is disabled for this fixed CCTV view.

```bash
# Keep estimates on screen for up to two seconds.
python src/track_people.py --hold-seconds 2

# Try a larger detector input for small/partially hidden people (slower).
python src/track_people.py --imgsz 1536

# Inspect actual detections only, with identity memory still enabled.
python src/track_people.py --hold-seconds 0

# Short validation run, or specify your own input and output.
python src/track_people.py --max-frames 100
python src/track_people.py ../sample.mp4 output/custom.mp4 --device 0

python -m unittest discover -s tests -v
```

Use `--track-buffer-seconds` to tune lost-ID memory and `--tracker` to select a
custom BoT-SORT YAML. Keep `--conf` at or below `track_low_thresh` in that YAML.
For moving cameras, set `gmc_method: sparseOptFlow`. The installed/tested
Ultralytics version is 8.4.160; its tracker updates even on empty detection frames.

Holding an estimated box is not proof the person is still there, and track IDs
are not a reliable unique-person count. ReID still requires sufficient spatial
overlap, so a person returning far from their predicted position can get a new
ID. `model: auto` uses detector features where supported, otherwise a
classification encoder; it is not guaranteed to be a person-trained ReID model.

If seated people remain missed in this overhead/fisheye scene, inspect a higher
input resolution or corridor crops, and fine-tune on labelled seated/occluded
people from the same camera view. A tracker cannot reliably recover a person
who never produces a usable detection. Review crossings and exits as well as
seated people when tuning: longer memory and looser matching can also cause
incorrect ID assignments. Measure ID switches and recall on labelled frames
before claiming an accuracy improvement.

## Employee nametag scanning (no fine-tuning)

The architecture remains **YOLO26x + BoT-SORT + OWLv2 image-guided matching**.
The reference embedding and text checks are cached. Each crop uses one OWLv2
image forward pass; there is no second detector and no training. Run from the
`code` directory:

```powershell
.\.venv\Scripts\python.exe src\identify_staff.py

# Short GPU validation before processing a complete clip.
.\.venv\Scripts\python.exe src\identify_staff.py --device 0 --max-frames 60

# Keep the large OWLv2 checkpoint if that is the model you intend to use.
.\.venv\Scripts\python.exe src\identify_staff.py --tag-model google/owlv2-large-patch14-ensemble

# Optional small batches; use 1 if GPU memory is limited.
.\.venv\Scripts\python.exe src\identify_staff.py --batch-size 2

# Explicitly mark the badge inside a reference (x1 y1 x2 y2).
# These coordinates are loaded automatically for assets/reference_1.jpg.
.\.venv\Scripts\python.exe src\identify_staff.py --reference-box 74 16 115 44

# Generate a fresh video using the updated matcher.
.\.venv\Scripts\python.exe src\visualize_staff.py --batch-size 2

# Render a completed NEW scan without running the models again.
.\.venv\Scripts\python.exe src\render_evidence_video.py output\tag_scan\run_YOUR_RUN
```

The existing source default remains `google/owlv2-base-patch16-ensemble`.
Both commands print the actual model and device. Selecting a checkpoint that
is not already cached may download it; neither command trains any weights.

The pipeline selects the clearest usable person crop within each **0.75-second
window**, rather than checking every third observation. Crops smaller than 32
pixels on either side or with Laplacian sharpness below 10 are skipped. These
are adjustable with `--scan-interval`, `--min-crop-size`, and `--min-sharpness`.
The legacy `--sample-every N` overrides the window with N **video frames**;
it no longer counts N observations independently of video time.

The badge feature is now selected from the **marked badge region**, using
proposal overlap and objectness. Previously the library's automatic query
selection could encode the surrounding shirt or padded image instead of the
badge. `assets/reference_1.tag.json` contains the supplied image's badge
coordinates and SHA-256 hash. A replacement image cannot silently reuse that
annotation. `--reference-box` overrides it and keeps the full reference image
as context; it **no longer crops the reference**. For another reference without
an annotation, supply a box or an image already cropped tightly to the badge.

Candidates must also pass these checks before top-k selection and NMS:

- Raw image-reference similarity at least **0.65** (`--candidate-threshold`).
- OWLv2 objectness at least **0.01**. Higher generic cutoffs missed the small
  actual badges in the brief.
- Badge/nametag/logo text must beat background descriptions by **0.5 logits**:
  hand, face, chair, collar, pocket, zipper, watch, shoe, table, phone, laptop,
  laptop sticker, and furniture label, all on the **same features**.
  This compares text logits with text logits, not with image-query scores.
- No crop-edge/padding boxes, minimum side 3 source pixels, maximum aspect
  ratio 3.5, and maximum area 8% of the crop/person (`--max-tag-area` in the
  staff scanner). Person ownership is checked after mapping to frame pixels.

The text check uses the text encoder already inside OWLv2. It is a conservative
background filter, not proof that a logo belongs to an employee. Very blurred,
tiny, occluded, or unusual badges can still be missed. The defaults are
development settings, not thresholds validated across cameras. The existing
large checkpoint option remains available, but these changes were exercised
with the locally cached **base** checkpoint.

Tag scores use the raw image-query sigmoid logits. Older results from the image-guided
visualization postprocessor were rescaled so each crop's best result approached
1, making scores across crops misleading. Old score thresholds/rankings cannot
be reused. The postprocessor removes invalid/padded boxes, keeps at most
100 candidates, and uses compiled NMS to return up to 20 detections. Crop
coordinates account for OWLv2's square padding before resizing.

By default the pipeline runs in **candidate review mode**: it saves ranked
evidence and leaves staff labels unknown. Validation found high raw scores on
non-tag clothing/background, so an arbitrary threshold would mislabel staff.
Automatic confirmation is enabled only when `--tag-threshold` is supplied with
a value validated on positive and negative examples. It then requires **three
positive checks among the last five** (`--confirmations`), all within a
**5-second evidence window** (`--evidence-window`) and one unambiguous continuity
section. Checks use distinct frames at least 0.2 seconds apart. `--tag-threshold` must be at least `--candidate-threshold`.
Box containment does not prove ownership where person boxes overlap. Inspect the saved
evidence on labelled positive and negative examples before relying on labels.
Raw scores are not calibrated probabilities.

Both video renderers show a bright yellow **tag candidate** on the scanned
frame. During the short display window they follow the observed person's box
and show a dimmer **tag estimate**; they no longer leave a fixed yellow box at
the old screen position. Estimates stop on a missing observation, a predicted
person box, a new track segment, ambiguous overlap, or an implausible jump.
People hidden by display filters still participate in ownership checks. They assume the tag stays at the same
relative position on the body; use `--display-seconds 0` to show only actual
scanned frames. Rendering an old report cannot apply the new model checks:
run the scanner again to replace the old evidence.

Confirmed tracks are checked every **5 seconds** (`--confirmed-interval`). Staff
status expires **8 seconds** after the last accepted positive badge observation
(`--evidence-ttl`). More than **1 second** without a real observation starts a new
section (`--reverify-gap`). Overlap at IoU **0.3** (`--crossing-iou`) marks both
people uncertain; clear observations after the overlap need fresh confirmation.
Motion exceeding **0.75 person-box diagonals per frame** also breaks continuity
(`--max-step-diagonals`). These are conservative starting settings, not validated
camera-independent constants. Overlapping boxes do not prove an actual ID switch.

Queued crops are discarded at unsafe boundaries, ambiguous crops are not scanned,
and a badge whose centre lies inside another person's box cannot confirm its
owner. Predictions are never scanned or exported as observed staff. A continuity
ID identifies one safe section of a tracker ID; it is not a persistent employee ID.
This change does not stitch switched tracker IDs together or modify OSNet weights.
The BoxMOT wrapper now preserves the requested six-second identity memory at the
source frame rate rather than applying frame-rate scaling twice.

Each run creates a separate folder in `output/tag_scan/` containing:

- `report.json`: model/settings, marked reference/proposal, filter settings and
  prompts, proposal rejection counts, processing times, scan counts, raw evidence,
  and each track segment's confirmation state.
- `observations.csv`: frame index, timestamp, person ID/segment, bounding box,
  center coordinates, continuity ID, identity reason, and staff/unknown/uncertain
  status for every real observation.
- `staff_frames.csv`: confirmed staff observations only. Multiple rows may
  share a frame when multiple confirmed employees are observed. In default
  candidate review mode this file contains the header only.
- Evidence JPGs for the top tracks with accepted candidates, outlined in yellow.
  Tracks with no accepted badge do not produce misleading evidence JPGs.

For this offline video task, confirmation can be backfilled at most **1 second**
before the first supporting hit (`--backfill-seconds`), within the same safe
continuity section. The observations between that first hit and the later
confirmation are also labelled as offline backfill. No interval crosses an
ambiguous overlap, expired evidence, or a continuity boundary. CSV `confirmed_at_frame` and `label_source` disclose
when the decision was available. Frame indices are zero-based and coordinates
are source-image pixels. Missing rows in `staff_frames.csv` mean no confirmed
staff observation, not proof that no employee was present. Estimated locations
are not included. Observation metadata is retained in memory until export;
very long videos should be processed in clips or use a future streaming writer.

The standalone `scan_for_tag.py` remains available for diagnostics. Its default
stride is now 25; use `--stride 1` for exhaustive full-frame scanning. It also
accepts `--tag-model`, `--device`, and `--max-frames`, and prints raw scores.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


## Code map and identity review

The two entry points now share one pipeline:

1. `reid_track.py` / `detect_track.py`: person detection and temporary tracker IDs.
2. `staff_scan.py`: choose clear crops, schedule scans, and reject ambiguous badge ownership.
3. `staff_identity.py`: continuity boundaries, local evidence, confirmation expiry,
   bounded offline backfill, and CSV decisions. This is the source of staff labels.
4. `identify_staff.py`: run models and save evidence; `visualize_staff.py` calls it
   and then the renderer, rather than maintaining another scanning implementation.
5. `render_evidence_video.py` / `tag_overlay.py`: draw those decisions, reset trails
   at identity boundaries, and stop short badge estimates when ownership is uncertain.

For a new video, the same command supports OSNet tracking and saves the scan for
later review. Candidate review remains the default:

```powershell
.\.venv\Scripts\python.exe src\visualize_staff.py C:\videos\new_test.mp4 --reid-tracker --device 0 --batch-size 2
```

Use `--tag-threshold` and `--confirmations` only with the score/hit settings you
intend to evaluate. A threshold is not a calibrated probability. The report now
saves `staff_decisions.policy` and every confirmed interval. Video rendering uses
that policy unless explicitly overridden.

Re-render the existing run with the former two-hit/0.9 settings and the corrected
continuity rules, keeping uncertain and unknown people visible:

```powershell
.\.venv\Scripts\python.exe src\render_evidence_video.py output\tag_scan\run_reid_full ..\sample.mp4 output\runs\staff_identity_review.mp4 --staff-min-hits 2 --staff-score 0.9 --show-all
```

The output has `.observations.csv`, `.staff.csv`, and `.decisions.json` sidecars
using exactly the video decisions, including any overrides. Original scan files
are preserved. `--only-candidates` and staff-only rendering filter display after
ownership checks. Remove `--show-all` to display only confirmed staff.

On the existing `run_reid_full` cache, two local unambiguous hits at 0.9 do not
occur within any safe section, so the corrected review has **zero confirmed staff
frames**. Candidate detections remain visible. This prevents the false long-lived
label on ID 48; it is not evidence that continuous staff tracking is solved.
Further recovery of switched IDs needs appearance/motion association work and
validation on labelled footage. Re-rendering cannot add badge detections that the
original scan missed, or apply the new scan scheduling and tracker buffer fix.

The tests cover crossings between hits, noncandidate people overlapping badges,
evidence expiry, ID/gap/jump resets, queued-crop invalidation, video/CSV agreement,
and the actual BoxMOT buffer lifetime at 25 and 29.97 fps. No training is involved.
