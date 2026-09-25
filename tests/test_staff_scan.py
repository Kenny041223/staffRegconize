import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from identify_staff import parse_args, run
from staff_scan import ScanCandidate, ScanScheduler, make_candidate, owned_tag_match


def candidate(frame, quality=1.0):
    return ScanCandidate(frame, (10, 10, 50, 70), (8, 8), np.zeros((64, 44, 3), dtype=np.uint8), quality)


class StaffScanTests(unittest.TestCase):
    def test_best_crop_is_selected_once_per_window_and_copied(self):
        scheduler = ScanScheduler(10, 50, 25)
        track = scheduler.observe(0, 1)
        first, best = candidate(0), candidate(5, 3)
        scheduler.consider(track, first)
        scheduler.consider(track, best)
        scheduler.consider(track, candidate(8, 2))
        self.assertEqual(scheduler.take_due(8), [])
        due = scheduler.take_due(9)
        self.assertEqual(due[0][1].frame_idx, 5)
        self.assertEqual(scheduler.take_due(10), [])

    def test_confirmation_needs_repeated_evidence_and_does_not_cross_gap_or_id(self):
        scheduler = ScanScheduler(10, 50, 25)
        track = scheduler.observe(0, 1)
        for frame, score in ((0, 0.9), (10, 0.1), (20, 0.9)):
            track.record(candidate(frame), score, [20, 20, 25, 25], frame, 0.2, 3)
        self.assertIsNone(track.confirmed_at)
        track.record(candidate(30), 0.9, [20, 20, 25, 25], 30, 0.2, 3)
        self.assertEqual(track.confirmed_at, 30)
        track.last_seen = 30
        self.assertIs(scheduler.observe(31, 1), track)
        self.assertIsNone(scheduler.observe(31, 2).confirmed_at)
        returned = scheduler.observe(60, 1)
        self.assertEqual(returned.segment, 1)
        self.assertIsNone(returned.confirmed_at)

    def test_second_look_uses_the_nearest_kept_crop_at_least_the_spacing_away(self):
        scheduler = ScanScheduler(20, 20, 25, min_hit_spacing_frames=5)
        track = scheduler.observe(0, 1)
        kept = {f: candidate(f) for f in (10, 12, 16, 21, 30)}
        for c in kept.values():
            scheduler.consider(track, c)
        self.assertIs(scheduler.second_look(track, kept[12]), kept[21])  # 10 and 16 are under 5 frames away
        self.assertIs(scheduler.second_look(track, kept[30]), kept[21])

    def test_old_kept_crops_are_dropped(self):
        scheduler = ScanScheduler(10, 10, 25)
        track = scheduler.observe(0, 1)
        scheduler.consider(track, candidate(0))
        scheduler.take_due(25)
        self.assertEqual(len(track.recent), 0)
        self.assertIsNone(scheduler.second_look(track, candidate(24)))

    def test_confirmed_tracks_keep_the_normal_scan_rate_in_track_mode(self):
        default = parse_args(["video.mp4"])
        self.assertEqual(default.confirmed_interval, default.scan_interval)
        self.assertEqual(parse_args(["video.mp4", "--staff-mode", "continuity"]).confirmed_interval, 5.0)
        self.assertEqual(parse_args(["video.mp4", "--confirmed-interval", "2"]).confirmed_interval, 2.0)

    def test_missed_badge_keeps_hits_but_old_hits_expire(self):
        track = ScanScheduler(10, 50, 25, evidence_window_frames=30).observe(0, 1)
        for frame, score in ((0, .9), (10, 0), (20, .9), (25, .9)):
            track.record(candidate(frame), score, [20, 20, 25, 25], frame, 0.2, 3)
        self.assertEqual(track.confirmed_at, 25)  # the miss at frame 10 did not erase the hit at 0
        track = ScanScheduler(10, 50, 25, evidence_window_frames=30).observe(0, 1)
        for frame in (0, 10, 70):
            track.record(candidate(frame), .9, [20, 20, 25, 25], frame, 0.2, 3)
        self.assertIsNone(track.confirmed_at)  # hits at 0 and 10 are older than the window at 70

    def test_review_mode_never_confirms_even_repeated_high_scores(self):
        track = ScanScheduler(10, 50, 25).observe(0, 1)
        for frame in range(5):
            track.record(candidate(frame * 10), 0.999, [20, 20, 25, 25], frame * 10, None, 3)
        self.assertIsNone(track.confirmed_at)
        self.assertEqual(track.best_score, 0.999)

    def test_final_flush_retains_short_tracks(self):
        scheduler = ScanScheduler(25, 100, 25)
        track = scheduler.observe(0, 1)
        scheduler.consider(track, candidate(0))
        self.assertEqual(len(scheduler.take_due(1, flush=True)), 1)
        self.assertEqual(scheduler.take_due(1, flush=True), [])

    def test_ownership_filters_padding_and_whole_person_matches(self):
        result = {"scores": [0.99, 0.95, 0.7],
                  "boxes": [[0, 0, 2, 5], [2, 2, 42, 62], [12, 12, 20, 20]]}
        score, box = owned_tag_match(result, candidate(0))
        self.assertEqual(score, 0.7)
        self.assertEqual(box, [20, 20, 28, 28])

    def test_quality_filter_rejects_blur_and_tiny_crops(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(make_candidate(0, frame, (10, 10, 50, 70)))
        self.assertIsNone(make_candidate(0, frame, (10, 10, 15, 15), min_sharpness=0))
        self.assertIsNotNone(make_candidate(0, frame, (10, 10, 50, 70), min_sharpness=0))

    def test_pipeline_exports_observations_and_backfills_only_confirmed_segment(self):
        class FakeTracker:
            def track(self, path):
                for index in range(12):
                    yield index, np.zeros((80, 80, 3), dtype=np.uint8), [
                        {"track_id": 1, "bbox": (10, 10, 50, 70), "predicted": index >= 10}]

        class FakeMatcher:
            model_name, device, forward_calls = "test-model", "cpu", 7

            def detect_batch(self, images):
                self.forward_calls += 1
                return [{"scores": [0.8], "boxes": [[15, 15, 22, 22]],
                         "objectness": [.1], "text_margins": [1.2]} for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
            for _ in range(12):
                writer.write(np.zeros((80, 80, 3), dtype=np.uint8))
            writer.release()
            output = Path(directory) / "results"
            args = parse_args([str(video), "--output-dir", str(output), "--scan-interval", "0.2",
                               "--min-sharpness", "0", "--confirmed-interval", "5", "--tag-threshold", "0.2",
                               "--candidate-threshold", "0.2", "--no-second-look"])
            report = run(args, tracker=FakeTracker(), matcher=FakeMatcher())
            self.assertEqual(report["processed_frames"], 12)
            self.assertEqual(report["tag_crops_scanned"], 3)
            self.assertEqual(report["tag_forward_calls"], 3)
            self.assertEqual(report["checks"][0]["badge_evidence"], {"objectness": .1, "text_margins": 1.2})
            self.assertEqual(report["confirmed_staff_frame_count"], 10)
            with (output / "staff_frames.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["frame_idx"]) for row in rows], list(range(10)))
            self.assertEqual(rows[0]["label_source"], "track_evidence_backfill")
            self.assertEqual(rows[-1]["label_source"], "track_evidence")


    def test_first_sighting_gets_one_immediate_second_look(self):
        class FakeTracker:
            def track(self, path):
                for index in range(20):
                    yield index, np.zeros((80, 80, 3), dtype=np.uint8), [{"track_id": 1, "bbox": (10, 10, 50, 70)}]

        class FakeMatcher:
            model_name, device, forward_calls = "test-model", "cpu", 0

            def detect_batch(self, images):
                self.forward_calls += 1
                return [{"scores": [0.8], "boxes": [[15, 15, 22, 22]]} for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
            for _ in range(20):
                writer.write(np.zeros((80, 80, 3), dtype=np.uint8))
            writer.release()
            args = parse_args([str(video), "--output-dir", str(Path(directory) / "results"), "--scan-interval", "0.5",
                               "--min-sharpness", "0", "--tag-threshold", "0.2", "--candidate-threshold", "0.2",
                               "--confirmations", "2", "--second-look"])
            checks = run(args, tracker=FakeTracker(), matcher=FakeMatcher())["checks"]
            first, second = checks[0], checks[1]
            self.assertTrue(first["positive"] and not first["second_look"])
            self.assertTrue(second["second_look"])
            self.assertEqual(second["processed_at_frame"], first["processed_at_frame"])  # immediately, same frame
            self.assertGreaterEqual(abs(second["frame_idx"] - first["frame_idx"]), 2)  # 0.2 s at 10 fps
            self.assertEqual(sum(c["second_look"] for c in checks), 1)  # two hits confirm: no more second looks


if __name__ == "__main__":
    unittest.main()
