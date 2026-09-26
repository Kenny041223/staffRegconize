import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badge_prefilter import DEFAULT_PREFILTER, PrefilteredMatcher
from identify_staff import parse_args


class FakeDetector:
    """Flags an image when its first pixel is non-zero."""

    def __init__(self):
        self.calls = []

    def predict(self, images, imgsz, conf, verbose):
        self.calls.append((len(images), imgsz, conf))
        return [type("P", (), {"boxes": [1] if im[0, 0, 0] else []})() for im in images]


class FakeMatcher:
    model_name, device, forward_calls, metadata = "owl", "cpu", 3, {"version": "x"}

    def __init__(self):
        self.seen = []

    def detect_batch(self, images):
        self.seen.append([int(im[0, 0, 1]) for im in images])
        return [{"scores": [0.9], "boxes": [[1, 1, 5, 5]], "tag": int(im[0, 0, 1])} for im in images]


def crop(flagged, tag):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (255 if flagged else 0, tag, 0)
    return image


class PrefilterTests(unittest.TestCase):
    def test_only_flagged_crops_reach_owlv2_and_order_is_kept(self):
        owl, yolo = FakeMatcher(), FakeDetector()
        matcher = PrefilteredMatcher(owl, yolo, conf=0.02)
        results = matcher.detect_batch([crop(False, 1), crop(True, 2), None, crop(True, 4)])
        self.assertEqual(owl.seen, [[2, 4]])
        self.assertEqual([r.get("tag") for r in results], [None, 2, None, 4])
        self.assertEqual(results[0]["scores"], [])
        self.assertEqual(results[0]["filter_stats"], {"prefilter": "skipped"})
        self.assertEqual((matcher.passed, matcher.skipped), (2, 1))
        self.assertEqual(yolo.calls, [(3, 416, 0.02)])

    def test_nothing_flagged_skips_owlv2_entirely(self):
        owl = FakeMatcher()
        matcher = PrefilteredMatcher(owl, FakeDetector())
        self.assertEqual(len(matcher.detect_batch([crop(False, 1)] * 3)), 3)
        self.assertEqual(owl.seen, [])

    def test_passes_matcher_details_through_and_reports_counts(self):
        matcher = PrefilteredMatcher(FakeMatcher(), FakeDetector(), conf=0.05, weights="w.pt")
        matcher.detect_batch([crop(True, 1), crop(False, 2)])
        self.assertEqual((matcher.model_name, matcher.device, matcher.forward_calls), ("owl", "cpu", 3))
        self.assertEqual(matcher.metadata["version"], "x")
        self.assertEqual(matcher.metadata["prefilter"],
                         {"weights": "w.pt", "conf": 0.05, "imgsz": 416, "crops_to_owlv2": 1, "crops_skipped": 1})

    def test_invalid_confidence_and_missing_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            PrefilteredMatcher(FakeMatcher(), FakeDetector(), conf=1.5)
        with self.assertRaises(FileNotFoundError):
            PrefilteredMatcher.load(FakeMatcher(), "does_not_exist.pt")

    def test_command_line_option(self):
        self.assertIsNone(parse_args(["v.mp4"]).badge_prefilter)
        self.assertEqual(parse_args(["v.mp4", "--badge-prefilter"]).badge_prefilter, str(DEFAULT_PREFILTER))
        args = parse_args(["v.mp4", "--badge-prefilter", "x.pt", "--prefilter-conf", "0.05"])
        self.assertEqual((args.badge_prefilter, args.prefilter_conf), ("x.pt", 0.05))


if __name__ == "__main__":
    unittest.main()
