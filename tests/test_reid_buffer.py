"""Test the actual BoxMOT lifetime via the wrapper, without loading weights."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reid_track import ReidPersonTracker


@unittest.skipUnless(importlib.util.find_spec("boxmot"), "Optional BoxMOT is not installed")
class ReidBufferTests(unittest.TestCase):
    def test_six_seconds_is_six_seconds_at_non_thirty_fps(self):
        from boxmot import BotSort
        for fps in (25., 29.97):
            created = []
            def factory(**kwargs):
                for key in ("reid_weights", "device", "half"):
                    kwargs.pop(key)
                tracker = BotSort(use_embeddings=False, **kwargs)
                created.append(tracker)
                return tracker
            wrapper = ReidPersonTracker.__new__(ReidPersonTracker)
            wrapper._tracker_cls = factory
            wrapper.model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=[])])
            wrapper.device, wrapper.conf, wrapper.imgsz = "cpu", .1, 1280
            wrapper.track_buffer_seconds, wrapper.reid_weights = 6., Path("unused.pt")
            cap = SimpleNamespace(isOpened=lambda: True, get=lambda _: fps, release=lambda: None)
            reads = iter([(True, np.zeros((80, 80, 3), np.uint8)), (False, None)])
            cap.read = lambda: next(reads)
            with patch("reid_track.cv2.VideoCapture", return_value=cap):
                self.assertEqual(len(list(wrapper.track("unused"))), 1)
            self.assertEqual(created[0].max_time_lost, round(6 * fps))


if __name__ == "__main__":
    unittest.main()
