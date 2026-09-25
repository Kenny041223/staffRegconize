"""Regression tests for false staff transfers, bounded evidence, and exports."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staff_identity import StaffPolicy, build_staff_decisions, export_decisions
from render_evidence_video import parse_args, render, visible_detections
from staff_scan import ScanCandidate, ScanScheduler, owned_tag_match
from tag_overlay import build_tag_display_map


def person(tid=1, box=(10, 10, 50, 70), segment=0):
    return {"track_id": tid, "segment": segment, "bbox": box}


def hit(frame, tid=1, segment=0, score=.95):
    return {"frame_idx": frame, "track_id": tid, "segment": segment,
            "raw_score": score, "tag_bbox": [20, 25, 26, 31], "person_bbox": [10, 10, 50, 70]}


def labelled(result):
    return {(f, d["track_id"]) for f, detections in result.frames.items()
            for d in detections if d["status"] == "confirmed_staff"}


class StaffIdentityTests(unittest.TestCase):
    def policy(self, **kwargs):
        return StaffPolicy(**dict({"min_score": .9, "min_hits": 2}, **kwargs))

    def strict(self, **kwargs):
        return self.policy(**dict({"mode": "continuity", "evidence_window_seconds": 5,
                                   "evidence_ttl_seconds": 8}, **kwargs))

    # Track mode (default)

    def test_track_mode_keeps_evidence_through_a_brief_overlap(self):
        # One hit on each side of a one-frame overlap still confirms the track;
        # only the overlap frame itself, a possible handoff, is left unlabelled.
        frames = {i: [person()] for i in range(21)}
        frames[10].append(person(2, (12, 12, 52, 72)))
        result = build_staff_decisions([hit(5), hit(15)], frames, 10, self.policy())
        self.assertEqual(labelled(result), {(i, 1) for i in range(21) if i != 10})

    def test_track_mode_label_stops_where_the_person_briefly_disappears(self):
        # Undetected for 0.5 s (> 0.2 s grace, < 1 s gap): the ID may have moved to
        # someone else, so the label is not carried past it without a new hit.
        frames = {i: [person()] for i in range(41) if not 15 <= i < 20}
        result = build_staff_decisions([hit(5), hit(8)], frames, 10, self.policy())
        self.assertEqual(labelled(result), {(i, 1) for i in range(15)})

    def test_track_mode_combines_hits_far_apart_on_one_track(self):
        frames = {i: [person()] for i in range(201)}
        result = build_staff_decisions([hit(0), hit(170)], frames, 10, self.policy())
        self.assertEqual(labelled(result), {(i, 1) for i in range(201)})

    def test_track_mode_label_extends_only_to_the_nearest_crossings(self):
        frames = {i: [person()] for i in range(31)}
        for crossing in (3, 25):
            frames[crossing].append(person(2, (12, 12, 52, 72)))
        result = build_staff_decisions([hit(10), hit(15)], frames, 10, self.policy())
        self.assertEqual(labelled(result), {(i, 1) for i in range(4, 25)})
        self.assertEqual(result.intervals[0]["confirmed_at_frame"], 15)

    # Continuity mode (strict)

    def test_crossing_between_two_hits_cannot_confirm_either_side(self):
        frames = {i: [person()] for i in range(12)}
        frames[5].append(person(2, (12, 12, 52, 72)))
        result = build_staff_decisions([hit(1), hit(9)], frames, 10, self.strict())
        self.assertEqual(labelled(result), set())
        self.assertEqual(result.frames[5][0]["status"], "uncertain")
        self.assertNotEqual(result.frames[4][0]["continuity_id"], result.frames[6][0]["continuity_id"])

    def test_confirmed_history_stops_at_crossing_and_requires_fresh_hits(self):
        frames = {i: [person()] for i in range(16)}
        frames[6].append(person(2, (12, 12, 52, 72)))
        result = build_staff_decisions([hit(0), hit(3), hit(8)], frames, 10, self.strict())
        self.assertEqual(labelled(result), {(i, 1) for i in range(6)})

    def test_strict_mode_needs_recent_hits(self):
        frames = {i: [person()] for i in range(90)}
        self.assertFalse(labelled(build_staff_decisions([hit(0), hit(80)], frames, 10, self.strict())))

    # Both modes

    def test_same_id_gap_and_large_jump_break_evidence(self):
        for policy in (self.policy(), self.strict()):
            for frames in ({0: [person()], 30: [person()]},
                           {0: [person()], 1: [person(box=(300, 10, 340, 70))], 3: [person()]}):
                checks = [hit(min(frames)), hit(max(frames))]
                result = build_staff_decisions(checks, frames, 10, policy)
                self.assertFalse(labelled(result))

    def test_new_tracker_id_does_not_inherit_staff(self):
        frames = {i: [person(1 if i < 6 else 2)] for i in range(12)}
        for policy in (self.policy(), self.strict()):
            result = build_staff_decisions([hit(0), hit(3)], frames, 10, policy)
            self.assertEqual(labelled(result), {(i, 1) for i in range(6)})

    def test_hits_must_be_independent(self):
        frames = {i: [person()] for i in range(90)}
        for policy in (self.policy(), self.strict()):
            for checks in ([hit(0), hit(0)], [hit(0), hit(1)]):
                self.assertFalse(labelled(build_staff_decisions(checks, frames, 10, policy)))

    def test_confirmation_expires_and_backfill_is_bounded(self):
        frames = {i: [person()] for i in range(80)}
        policy = self.strict(evidence_window_seconds=1, evidence_ttl_seconds=2, backfill_seconds=.5)
        result = build_staff_decisions([hit(20), hit(25)], frames, 10, policy)
        self.assertEqual(labelled(result), {(i, 1) for i in range(15, 46)})
        self.assertEqual(result.frames[15][0]["confirmed_at_frame"], 25)
        self.assertEqual(result.frames[46][0]["status"], "unknown")

    def test_prediction_cannot_supply_ownership_or_observed_presence(self):
        frames = {0: [dict(person(), predicted=True)], 3: [person()]}
        self.assertFalse(labelled(build_staff_decisions([hit(0), hit(3)], frames, 10, self.policy())))

    def test_badge_owner_ambiguity_applies_even_below_crossing_iou(self):
        frames = {i: [person(), person(2, (19, 20, 27, 35))] for i in range(4)}
        result = build_staff_decisions([hit(0), hit(3)], frames, 10, self.policy())
        self.assertFalse(labelled(result))
        args = parse_args(["unused", "--only-candidates", "--staff-min-hits", "2"])
        self.assertTrue(all(not dets for dets in visible_detections(result, [hit(0), hit(3)], args).values()))

    def test_review_mode_keeps_all_labels_unknown_or_uncertain(self):
        frames = {i: [person()] for i in range(4)}
        self.assertFalse(labelled(build_staff_decisions([hit(0), hit(3)], frames, 10)))

    def test_badge_overlay_stops_at_crossing_and_does_not_restart_after_it(self):
        frames = {i: [person()] for i in range(8)}
        frames[2].append(person(2, (12, 12, 52, 72)))
        display = build_tag_display_map([hit(0)], frames, 8, 10, display_seconds=.8)
        self.assertEqual(set(display), {0, 1})

    def test_scanner_rejects_a_badge_inside_a_second_person(self):
        candidate = ScanCandidate(0, (10, 10, 50, 70), (0, 0), np.zeros((80, 80, 3), np.uint8), 1,
                                  other_people=((19, 20, 27, 35),))
        self.assertEqual(owned_tag_match({"scores": [.99], "boxes": [[20, 25, 26, 31]]}, candidate), (0., None))

    def test_scheduler_drops_pending_crop_at_handoff_and_expires_confirmation(self):
        scheduler = ScanScheduler(2, 20, 10, evidence_window_frames=5, evidence_ttl_frames=8)
        track = scheduler.observe(0, 1)
        candidate = ScanCandidate(0, (10, 10, 50, 70), (0, 0), np.zeros((80, 80, 3), np.uint8), 1)
        scheduler.consider(track, candidate)
        scheduler.observe(1, 1, reset=True)
        self.assertEqual(scheduler.take_due(2), [])
        current = scheduler.observe(2, 1)
        current.record(candidate, .99, [20, 25, 26, 31], 2, .9, 1)
        self.assertIsNotNone(current.confirmed_at)
        self.assertIsNone(scheduler.observe(9, 1).confirmed_at)

    def test_rendered_decisions_and_csv_match_scanner_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = {i: [person()] for i in range(10)}
            checks = [dict(hit(0), processed_at_frame=1), dict(hit(3), processed_at_frame=4)]
            decisions = build_staff_decisions(checks, frames, 10, self.policy())
            export_decisions(decisions, 10, root / "observations.csv", root / "staff_frames.csv")
            video = root / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
            for _ in range(10):
                writer.write(np.zeros((80, 80, 3), np.uint8))
            writer.release()
            report = {"fps": 10, "processed_frames": 10, "checks": checks, "video": str(video),
                      "staff_decisions": decisions.metadata(), "confirmation_enabled": True}
            (root / "report.json").write_text(json.dumps(report))
            metadata = render(parse_args([str(root), str(video), str(root / "review.mp4")]))
            self.assertEqual(metadata["confirmed_staff_frame_count"], 10)
            self.assertEqual((root / "staff_frames.csv").read_text(), (root / "review.staff.csv").read_text())
            self.assertEqual(metadata["intervals"], decisions.intervals)
            cap = cv2.VideoCapture(str(root / "review.mp4"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            cap.release()


if __name__ == "__main__":
    unittest.main()
