# staffRegconize

FootfallCam AI Evaluation: find the frames in which the staff member wearing the
nametag appears in a CCTV video, and give his xy coordinates.

This branch is **step 1**: person detection and tracking, plus a two-stage nametag
check (a trained YOLO pre-filter, then OWLv2).

## File structure

```
code/
├── src/                     pipeline code (see below)
│   └── trackers/
│       └── botsort_reid.yaml
├── assets/
│   ├── reference_1.jpg      photo of the nametag (from the brief)
│   └── reference_1.tag.json where the nametag is inside that photo
├── scripts/
│   └── download_models.py   downloads the person detector, ReID model and OWLv2
├── yolo_folder/             model files (not in git)
│   ├── yolo26x.pt           person detector
│   ├── osnet_x0_25_msmt17.pt  person re-identification model
│   └── badge_yolo26s.pt     trained nametag pre-filter
├── output/                  results of each run (created automatically, not in git)
├── requirements.txt
└── README.md
```

## How to run

**1. Set up (once).** Needs Python 3.10+ and an NVIDIA GPU. Run from the `code/` folder:

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; on Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # cu124 for older GPUs such as a GTX 1070 Ti
pip install -r requirements.txt
pip install boxmot==25.0.0 --no-deps   # tracker library; installed without its own dependencies on purpose
python scripts/download_models.py      # yolo26x.pt and osnet_x0_25_msmt17.pt into yolo_folder/, plus OWLv2
```

Then copy the trained `badge_yolo26s.pt` into `yolo_folder/` (it is not in git), and put the
video one folder above `code/` (e.g. `../sample.mp4`), or give its full path.

**2. Find the staff member:**

```bash
python src/identify_staff.py ../sample.mp4 --reid-tracker --tag-threshold 0.9 --confirmations 2 --batch-size 4 --badge-prefilter --output-dir output/tag_scan/step1
```

**3. Make the video:**

```bash
python src/render_evidence_video.py output/tag_scan/step1 ../sample.mp4 output/runs/step1.mp4 --staff-min-hits 2 --staff-score 0.9
```

The answer (frames and xy coordinates) is `output/tag_scan/step1/staff_frames.csv`, and the video
is `output/runs/step1.mp4`. On `sample.mp4` this takes about 7 minutes on a GTX 1070 Ti. Without
`badge_yolo26s.pt`, leave out `--badge-prefilter`: same result, slower.

## Files in `src/`

| File | What it does |
|---|---|
| `identify_staff.py` | **Main pipeline.** Tracks every person, checks their clearest image for the nametag every 0.75 s, decides who is staff and writes the results |
| `detect_track.py` | Person detection (YOLO26x) and tracking with Ultralytics BoT-SORT; the default tracker |
| `reid_track.py` | Person detection (YOLO26x) and tracking with BoxMOT BoT-SORT plus the OSNet appearance model (`--reid-tracker`) |
| `trackers/botsort_reid.yaml` | Settings for the Ultralytics BoT-SORT tracker used by `detect_track.py` |
| `staff_scan.py` | Picks the clearest (sharpest and largest) image of each person in each 0.75 s window, and checks that a found nametag lies inside that person's box |
| `badge_prefilter.py` | Fast nametag pre-filter: the trained YOLO26s model removes images that clearly contain no nametag, so only the rest go to OWLv2 (`--badge-prefilter`) |
| `tag_match.py` | OWLv2 nametag matching: compares a person image with the marked nametag in the reference photo and returns a similarity score |
| `staff_identity.py` | Decides which boxes are staff: a person whose track has two nametag sightings is staff, and the label follows the track until they cross someone or briefly disappear. Writes the CSV files |
| `render_evidence_video.py` | Draws the result video from a saved run, without running any model |
| `tag_overlay.py` | Draws the small nametag box on the video for a moment after each sighting |
| `visualize_staff.py` | Runs `identify_staff.py` and then `render_evidence_video.py` in one command |
| `track_people.py` | Tracking-only video (every person with an ID, no nametag logic) |
| `scan_for_tag.py` | Diagnostic: scans whole frames for the nametag and prints scores; not used by the pipeline |

## Data accepted

| Input | Details |
|---|---|
| **Video** | Any video OpenCV can read (e.g. `.mp4`), given as the first argument. Default: `../sample.mp4`, one folder above `code/` |
| **Reference nametag** | `assets/reference_1.jpg` plus `assets/reference_1.tag.json` (the nametag's box `[x1, y1, x2, y2]` in the photo, and a fingerprint of the photo) |
| **Models** | The three files in `yolo_folder/`. OWLv2 (`google/owlv2-base-patch16-ensemble`) downloads automatically from Hugging Face |

## Data produced

**`identify_staff.py`** writes a run folder, `output/tag_scan/<run>/`:

| File | Contents |
|---|---|
| `staff_frames.csv` | **The answer:** one row per staff box per frame, with its frame number, time and xy coordinates |
| `observations.csv` | Every person in every frame, with their status: `confirmed_staff`, `unknown` or `uncertain` |
| `report.json` | Settings, timings, and every nametag check (score and box) |
| `track*_frame*_score*.jpg` | Pictures of the best nametag matches, as evidence |

**`render_evidence_video.py`** writes `output/runs/<name>.mp4` (people labeled STAFF) with
`<name>.staff.csv`, `<name>.observations.csv` and `<name>.decisions.json` next to it.
**`track_people.py`** writes numbered videos, `output/runs/run_001.mp4`, `run_002.mp4`, ...

Main columns of the CSV files:

| Column | Meaning |
|---|---|
| `frame_idx`, `time_seconds` | Frame number and its time in the video |
| `track_id` | The tracker's ID for that person |
| `status` | `confirmed_staff`, `unknown` (not staff) or `uncertain` (overlapping another person) |
| `x1`, `y1`, `x2`, `y2` | The person's box, in pixels (top-left corner is 0, 0) |
| `center_x`, `center_y` | Centre of the box: **the xy coordinates** |
| `confirmed_at_frame` | Frame at which this person was confirmed as staff |
| `label_source` | Why the frame is labeled: from a nametag sighting, or filled in before confirmation (`backfill`) |
