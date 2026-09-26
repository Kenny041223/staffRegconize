"""Download the four models the staff pipeline uses, into the paths it loads them from.

Run once after installing requirements.txt and BoxMOT:
    python scripts/download_models.py

- yolo26x.pt            person detector  -> code/yolo_folder/yolo26x.pt
- osnet_x0_25_msmt17.pt ReID (--reid-tracker) -> code/yolo_folder/osnet_x0_25_msmt17.pt
- OWLv2 base            nametag matcher  -> Hugging Face cache
- SAM 2.1 large         staff follower (--sam2-follow), about 900 MB -> Hugging Face cache
Files that already exist are skipped. If the ReID download fails (it is hosted
on Google Drive, which rate-limits), copy osnet_x0_25_msmt17.pt into yolo_folder/ instead.
"""
from pathlib import Path

from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2VideoModel, Sam2VideoProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "yolo_folder"
YOLO_WEIGHTS = MODELS / "yolo26x.pt"
REID_WEIGHTS = MODELS / "osnet_x0_25_msmt17.pt"
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
SAM2_MODEL = "facebook/sam2.1-hiera-large"   # same default as identify_staff.py --sam2-model


def main():
    MODELS.mkdir(exist_ok=True)
    print(f"YOLO: {YOLO_WEIGHTS.name} ...")
    YOLO(str(YOLO_WEIGHTS))  # Ultralytics downloads a missing official checkpoint to this path.
    print(f"  OK - {YOLO_WEIGHTS}")

    print(f"ReID: {REID_WEIGHTS.name} ...")
    if not REID_WEIGHTS.exists():
        from boxmot.reid.core.catalog import TRAINED_URLS
        from boxmot.resources.download import download_file
        download_file(TRAINED_URLS[REID_WEIGHTS.name], REID_WEIGHTS)
    print(f"  OK - {REID_WEIGHTS}")

    print(f"OWLv2: {OWLV2_MODEL} ...")
    Owlv2Processor.from_pretrained(OWLV2_MODEL)
    Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL)
    print(f"  OK - {OWLV2_MODEL}")

    print(f"SAM 2.1: {SAM2_MODEL} (about 900 MB) ...")
    Sam2VideoProcessor.from_pretrained(SAM2_MODEL)
    Sam2VideoModel.from_pretrained(SAM2_MODEL)
    print(f"  OK - {SAM2_MODEL}")


if __name__ == "__main__":
    main()
