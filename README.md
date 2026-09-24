# staffRegconize

FootfallCam AI Evaluation — staff identification from CCTV footage.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# or: .venv\Scripts\activate.bat     # Windows cmd

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

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
