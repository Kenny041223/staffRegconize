import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tag_overlay import build_tag_display_map


class TagOverlayTests(unittest.TestCase):
    def setUp(self):
        self.check = {"frame_idx": 0, "track_id": 7, "segment": 0, "raw_score": .8,
                      "person_bbox": [10, 10, 50, 90], "tag_bbox": [20, 30, 30, 40]}

    def test_overlay_follows_owner_and_marks_unscanned_frames_as_estimates(self):
        frames = {0: [{"track_id": 7, "bbox": [10, 10, 50, 90]}],
                  1: [{"track_id": 7, "bbox": [50, 20, 130, 180]}]}
        result = build_tag_display_map([self.check], frames, 2, 25)
        self.assertEqual(result[0], [([20., 30., 30., 40.], .8, False)])
        self.assertEqual(result[1], [([70., 60., 90., 80.], .8, True)])

    def test_gap_prediction_and_new_segment_end_the_overlay(self):
        for interruption in ([], [{"track_id": 7, "bbox": [10, 10, 50, 90], "predicted": True}],
                             [{"track_id": 7, "bbox": [10, 10, 50, 90], "segment": 1}]):
            frames = {0: [{"track_id": 7, "bbox": [10, 10, 50, 90]}], 1: interruption,
                      2: [{"track_id": 7, "bbox": [10, 10, 50, 90]}]}
            self.assertEqual(list(build_tag_display_map([self.check], frames, 3, 25)), [0])

    def test_zero_hold_only_shows_source_frame_and_low_scores_are_skipped(self):
        frames = [[{"track_id": 7, "bbox": [10, 10, 50, 90]}]] * 3
        self.assertEqual(list(build_tag_display_map([self.check], frames, 3, 25, 0)), [0])
        self.assertEqual(build_tag_display_map([self.check], frames, 3, 25, min_score=.9), {})


if __name__ == "__main__":
    unittest.main()
