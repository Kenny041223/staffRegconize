"""Download and cache the two models used in this project.

Run once after installing requirements.txt:
    python scripts/download_models.py
"""
from ultralytics import YOLO
from transformers import Owlv2ForObjectDetection, Owlv2Processor

YOLO_WEIGHTS = "yolo26n.pt"
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"


def main():
    print(f"Downloading {YOLO_WEIGHTS} ...")
    yolo = YOLO(YOLO_WEIGHTS)
    print(f"  OK — {YOLO_WEIGHTS} ready.")

    print(f"Downloading {OWLV2_MODEL} ...")
    Owlv2Processor.from_pretrained(OWLV2_MODEL)
    Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL)
    print(f"  OK — {OWLV2_MODEL} ready.")


if __name__ == "__main__":
    main()
