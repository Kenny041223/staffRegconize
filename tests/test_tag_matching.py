"""Regression tests for comparable raw scores and square-padded crop geometry."""
import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tag_match import postprocess_raw, resolve_reference_box, rotate_reference, select_reference_embedding, badge_proposal_mask


class TagPostprocessTests(unittest.TestCase):
    def test_scores_remain_comparable_across_crops(self):
        logits = torch.logit(torch.tensor([[[0.05]], [[0.8]]]))
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]] * 2)
        results = postprocess_raw(logits, boxes, [(100, 100)] * 2)
        self.assertAlmostEqual(results[0]["scores"][0], 0.05, places=6)
        self.assertAlmostEqual(results[1]["scores"][0], 0.8, places=6)

    def test_square_padding_scale_and_padded_region_rejection(self):
        boxes = torch.tensor([[[0.5, 0.25, 0.2, 0.2], [0.5, 0.8, 0.2, 0.2]]])
        logits = torch.logit(torch.tensor([[[0.7], [0.9]]]))
        result = postprocess_raw(logits, boxes, [(50, 100)])[0]
        self.assertEqual(len(result["boxes"]), 1)
        torch.testing.assert_close(torch.tensor(result["boxes"][0]), torch.tensor([40., 15., 60., 35.]))

    def test_nms_keeps_highest_score_and_filters_invalid_boxes(self):
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2],
                               [0.5, 0.5, 0.0, 0.0], [float("nan"), 0.5, 0.1, 0.1]]])
        logits = torch.logit(torch.tensor([[[0.7], [0.9], [0.99], [0.99]]]))
        result = postprocess_raw(logits, boxes, [(100, 100)])[0]
        self.assertEqual(len(result["boxes"]), 1)
        self.assertAlmostEqual(result["scores"][0], 0.9, places=6)

    def test_threshold_and_empty_input_preserve_batch_slots(self):
        logits = torch.logit(torch.tensor([[[0.05]], [[0.8]]]))
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]] * 2)
        results = postprocess_raw(logits, boxes, [(100, 100)] * 2, threshold=0.1)
        self.assertEqual(results[0], {"scores": [], "boxes": []})
        self.assertEqual(len(results[1]["scores"]), 1)
        self.assertEqual(postprocess_raw(torch.empty(1, 0, 1), torch.empty(1, 0, 4), [(100, 100)]),
                         [{"scores": [], "boxes": []}])

    def test_background_is_removed_before_top_k_can_displace_badge(self):
        logits = torch.logit(torch.tensor([[[0.999], [0.75]]]))
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
        result = postprocess_raw(logits, boxes, [(100, 100)], pre_nms_top_k=1,
                                 proposal_mask=torch.tensor([[False, True]]))[0]
        self.assertAlmostEqual(result["scores"][0], 0.75, places=6)

    def test_evidence_indices_preserve_original_proposals_through_nms(self):
        logits = torch.logit(torch.tensor([[[.99], [.7], [.8], [.9]]]))
        boxes = torch.tensor([[[.5, .5, .2, .2], [.5, .5, .2, .2],
                               [.2, .2, .1, .1], [.5, .5, .2, .2]]])
        result = postprocess_raw(logits, boxes, [(100, 100)], return_indices=True,
                                 proposal_mask=torch.tensor([[False, True, True, True]]))[0]
        self.assertEqual(result["indices"], [3, 2])
        empty = postprocess_raw(logits, boxes, [(100, 100)], threshold=1, return_indices=True)[0]
        self.assertEqual(empty["indices"], [])


class ReferenceAndFilterTests(unittest.TestCase):
    def test_reference_selection_uses_marked_badge_and_square_padding_scale(self):
        # On a 100x50 reference, the full shirt has higher objectness but only
        # the small proposal covers the annotated badge sufficiently.
        boxes = torch.tensor([[[0.5, 0.25, 1.0, 0.5], [0.8, 0.25, 0.2, 0.2]]])
        classes = torch.tensor([[[1., 0.], [0., 1.]]])
        query, info = select_reference_embedding(classes, boxes, torch.logit(torch.tensor([[0.99, 0.1]])),
                                                  (100, 50), (70, 15, 90, 35))
        torch.testing.assert_close(query, classes[:, 1:2])
        self.assertAlmostEqual(info["iou"], 1, places=5)

    def test_missing_reference_proposal_does_not_silently_choose_clothing(self):
        with self.assertRaisesRegex(ValueError, "no proposal"):
            select_reference_embedding(torch.ones(1, 1, 2), torch.tensor([[[0.1, 0.1, 0.1, 0.1]]]),
                                       torch.zeros(1, 1), (100, 50), (70, 15, 90, 35))

    def test_reference_annotation_cannot_follow_an_unrelated_replacement_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jpg"
            path.write_bytes(b"original image")
            path.with_suffix(".tag.json").write_text(json.dumps({
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "box": [70, 15, 90, 35]}))
            self.assertEqual(resolve_reference_box(path, (100, 50))[0], (70, 15, 90, 35))
            path.write_bytes(b"replacement image")
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_reference_box(path, (100, 50))
            self.assertEqual(resolve_reference_box(path, (100, 50), (1, 1, 10, 10))[0], (1, 1, 10, 10))

    def test_badge_filter_rejects_background_edges_large_boxes_and_padding(self):
        boxes = torch.tensor([[[.5, .3, .15, .10], [.5, .3, .15, .10], [.5, .3, .15, .10],
                               [.02, .3, .10, .10], [.5, .3, .40, .4], [.5, .8, .15, .10],
                               [.5, .3, .40, .04]]])
        obj = torch.logit(torch.tensor([[.05, .05, .001, .05, .05, .05, .05]]))
        # Badge text is lower than a background label on proposal 1.
        semantics = torch.tensor([[[2., 1.], [1., 2.], [2., 1.], [2., 1.], [2., 1.], [2., 1.], [2., 1.]]])
        mask = badge_proposal_mask(boxes, [(60, 100)], obj, semantics, positive_count=1)
        self.assertEqual(mask.tolist(), [[True, False, False, False, False, False, False]])

    def test_ambiguous_badge_background_tie_is_rejected(self):
        boxes = torch.tensor([[[.5, .3, .15, .10], [.5, .3, .15, .10]]])
        obj = torch.logit(torch.tensor([[.05, .05]]))
        semantics = torch.tensor([[[1.2, 1.], [1.6, 1.]]])
        mask = badge_proposal_mask(boxes, [(100, 100)], obj, semantics, positive_count=1)
        self.assertEqual(mask.tolist(), [[False, True]])


class ReferenceRotationTests(unittest.TestCase):
    def test_badge_box_follows_the_badge_at_every_rotation(self):
        import numpy as np
        from PIL import Image
        pixels = np.zeros((50, 100, 3), dtype=np.uint8)
        pixels[15:35, 70:90] = 255  # the "badge": x 70-90, y 15-35 in a 100x50 image
        image, box = Image.fromarray(pixels), (70, 15, 90, 35)
        for degrees in (0, 90, 180, 270):
            rotated, (x1, y1, x2, y2) = rotate_reference(image, box, degrees)
            array = np.asarray(rotated)
            self.assertEqual(rotated.size, (100, 50) if degrees in (0, 180) else (50, 100))
            self.assertTrue((array[y1:y2, x1:x2] == 255).all(), degrees)
            self.assertEqual(int((array == 255).all(-1).sum()), (x2 - x1) * (y2 - y1), degrees)

    def test_turning_a_box_back_restores_it(self):
        from tag_match import rotate_box
        box, size = (70, 15, 90, 35), (100, 50)
        for degrees in (90, 180, 270):
            turned_size = size if degrees == 180 else size[::-1]
            turned = rotate_box(box, size, degrees)
            self.assertEqual(rotate_box(turned, turned_size, (360 - degrees) % 360), box)

    def test_only_quarter_turns_are_allowed(self):
        from PIL import Image
        with self.assertRaises(ValueError):
            rotate_reference(Image.new("RGB", (10, 10)), (1, 1, 5, 5), 45)


if __name__ == "__main__":
    unittest.main()
