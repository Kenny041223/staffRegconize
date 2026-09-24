"""Exercise gap recovery with real BoT-SORT and synthetic detections (no weights)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detect_track import TRACKER_CONFIG, lost_track_predictions


class TrackingGapTests(unittest.TestCase):
    def setUp(self):
        config = YAML.load(TRACKER_CONFIG)
        # Geometry and lifecycle tests do not require an appearance model.
        config.update(with_reid=False, gmc_method="none", track_buffer=5)
        self.tracker = BOTSORT(SimpleNamespace(**config))
        self.shape = (300, 400, 3)

    def update(self, score=None, box=(100, 80, 150, 180)):
        data = np.empty((0, 6), dtype=np.float32) if score is None else np.array([[*box, score, 0]], dtype=np.float32)
        return self.tracker.update(Boxes(data, self.shape[:2]))

    def predictions(self, hold_frames=2, observed=None):
        return lost_track_predictions(self.tracker, observed or [], self.shape, hold_frames, 25.0)

    def test_weak_detections_keep_an_existing_id(self):
        initial = self.update(0.9)[0, 4]
        for confidence in (0.20, 0.15, 0.12, 0.9):
            tracks = self.update(confidence)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0, 4], initial)

    def test_weak_detection_alone_does_not_create_a_person(self):
        self.assertEqual(len(self.update(0.15)), 0)
        self.assertEqual(len(self.update(0.30)), 0)
        self.assertEqual(self.predictions(), [])

    def test_short_gap_is_estimated_then_recovers_same_id(self):
        initial = int(self.update(0.9)[0, 4])
        self.update()
        predicted = self.predictions()[0]
        self.assertEqual(predicted["track_id"], initial)
        self.assertTrue(predicted["predicted"])
        self.assertIsNone(predicted["conf"])
        self.assertEqual(predicted["missed_seconds"], 1 / 25)
        tracks = self.update(0.9)
        self.assertEqual(int(tracks[0, 4]), initial)
        self.assertEqual(self.predictions(), [])

    def test_display_expires_before_identity_memory(self):
        initial = int(self.update(0.9)[0, 4])
        self.update()
        self.assertEqual(len(self.predictions()), 1)
        self.update()
        self.assertEqual(len(self.predictions()), 1)
        self.update()
        self.assertEqual(self.predictions(), [])
        self.assertEqual(int(self.update(0.9)[0, 4]), initial)

    def test_expired_identity_is_not_held_forever(self):
        initial = int(self.update(0.9)[0, 4])
        for _ in range(8):
            self.update()
        self.assertEqual(self.predictions(hold_frames=100), [])
        self.update(0.9)  # new tracks after the first frame need confirmation
        tracks = self.update(0.9)
        self.assertEqual(len(tracks), 1)
        self.assertNotEqual(int(tracks[0, 4]), initial)

    def test_prediction_can_be_disabled_and_does_not_duplicate_observation(self):
        self.update(0.9)
        self.update()
        self.assertEqual(self.predictions(hold_frames=0), [])
        observed = [{"track_id": 99, "bbox": (100, 80, 150, 180)}]
        self.assertEqual(self.predictions(observed=observed), [])
        self.assertEqual(len(self.tracker.lost_stracks), 1)  # display does not change association

    def test_prediction_is_clipped_to_frame(self):
        self.update(0.9, box=(-10, 80, 40, 180))
        self.update()
        self.assertEqual(self.predictions()[0]["bbox"][0], 0)


if __name__ == "__main__":
    unittest.main()
