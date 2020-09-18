import os
import unittest

import torch
import numpy as np

from embryovision.managedata import AnnotatedImage
from embryovision.tests.common import get_loadable_filenames
from embryovision.objectdetector.objectpredictor import (
    MaskRCNNPredictor,
    PronucleiPredictor,
    BlastomerePredictor,
    non_max_suppression_slow,
    load_maskrcnn)
from embryovision.tvmaskrcnn.objectsegmentation import DetectionResult


class TestMaskRCNNPredictor(unittest.TestCase):
    def test_iou_threshold(self):
        self.assertEqual(MaskRCNNPredictor.iou_threshold, 0.7)

    def test_load_network(self):
        self.assertEqual(MaskRCNNPredictor.load_network, load_maskrcnn)

    def test_default_blocksize_is_1(self):
        self.assertEqual(MaskRCNNPredictor.default_blocksize, 1)

    def test_input_shape_is_correct(self):
        self.assertEqual(MaskRCNNPredictor.input_shape, (500, 500))

    def test_read_image_returns_correct_shape(self):
        filenames = get_loadable_filenames()[:1]
        names_and_boxes = [(nm, (10, 10, 300, 300)) for nm in filenames]

        p = MockMaskRCNNPredictor()
        for name, box in names_and_boxes:
            loaded = p.read_image(name, box)
            self.assertEqual(loaded.size(), (3,) + p.input_shape)

    def test_read_image_crops(self):
        # we crop to a 1-px region and ensure that the std is 0:
        filenames = get_loadable_filenames()[:1]
        names_and_boxes = [(nm, (100, 100, 1, 1)) for nm in filenames]
        p = MockMaskRCNNPredictor(device='cpu')
        for name, box in names_and_boxes:
            loaded = p.read_image(name, box)
            n = loaded.detach().numpy()
            self.assertAlmostEqual(n.std(), 0, places=6)  # single precision

    def test_transform_detection_result_to_xy_polygon_correct_return_type(self):
        detection_result = make_simple_detection_result()
        p = MockMaskRCNNPredictor(device='cpu')
        box = (0, 0, 500, 500)
        out = p.transform_detection_result_to_xy_polygon(detection_result, box)
        self.assertIsInstance(out, dict)
        self.assertEqual(
            set(out.keys()),
            {'xy_polygon', 'confidence'})

    def test_transform_detection_result_shifts_box(self):
        detection_result = make_simple_detection_result()
        p = MockMaskRCNNPredictor(device='cpu')
        box = (7e3, 7e3, 100, 100)
        out = p.transform_detection_result_to_xy_polygon(detection_result, box)
        polygon = out['xy_polygon']

        for i in range(2):
            self.assertGreater(polygon[:, i].min(), box[i])

    def test_transform_to_final_output_returns_nested_lists(self):
        detections_per_image = [1, 2, 3]
        detection_results = [
            [make_simple_detection_result() for _ in range(n_detections)]
            for n_detections in detections_per_image]
        boxes = [(0, 0, 500, 500) for _ in detections_per_image]

        p = MockMaskRCNNPredictor(device='cpu')
        out = p._transform_to_final_output(detection_results, boxes)
        self.assertEqual(len(out), len(detections_per_image))
        for o, n in zip(out, detections_per_image):
            self.assertEqual(len(o), n)

    def test_predict_returns_annotated_images(self):
        p = MockMaskRCNNPredictor(device='cpu')
        filenames = get_loadable_filenames()[:1]
        names_and_boxes = [(nm, (100, 100, 200, 200)) for nm in filenames]
        out = p._predict(names_and_boxes)
        for entry in out:
            self.assertIsInstance(entry, AnnotatedImage)


class TestPronucleiPredictor(unittest.TestCase):
    def test_loadname(self):
        self.assertIn('pronucleidetector.pkl', PronucleiPredictor.loadname)
        self.assertTrue(os.path.exists(PronucleiPredictor.loadname))


class TestBlastomerePredictor(unittest.TestCase):
    def test_loadname(self):
        self.assertIn('blastomeredetector.pkl', BlastomerePredictor.loadname)
        self.assertTrue(os.path.exists(BlastomerePredictor.loadname))


class TestNonMaximumSuppression(unittest.TestCase):
    def test_when_no_candidates_found(self):
        boxes = []
        confidences = []
        iou_threshold = 0.75
        indices = non_max_suppression_slow(boxes, confidences, iou_threshold)
        self.assertEqual(indices, list())

    def test_when_non_overlapping_found(self):
        boxes = [
            [30, 30, 50, 50],
            [100, 100, 120, 120],
            ]
        confidences = [0.7, 0.8]
        iou_threshold = 0.75
        indices = non_max_suppression_slow(boxes, confidences, iou_threshold)
        self.assertEqual(sorted(indices), [0, 1])

    def test_when_overlapping(self):
        boxes = [
            [30, 30, 50, 50],
            [30, 30, 50, 50],
            [100, 100, 120, 120],
            ]
        confidences = [0.7, 0.8, 0.9]
        iou_threshold = 0.75
        indices = non_max_suppression_slow(boxes, confidences, iou_threshold)
        self.assertEqual(sorted(indices), [1, 2])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def make_simple_detection_result():
    mask = np.zeros([500, 500])
    box = (100, 100, 150, 150)
    mask[box[0]: box[0] + box[2], box[1]:box[1] + box[2]] = 0.98

    detection_result = DetectionResult(
        confidence=0.99,
        box=box,
        mask=mask)
    return detection_result


class MockMaskRCNNPredictor(MaskRCNNPredictor):
    @staticmethod
    def load_network(loadname):
        return MockMaskRCNNNetwork()


class MockMaskRCNNNetwork(torch.nn.Module):
    def __init__(self):
        # We just make a simple pytorch network that can
        # 1. take an image as input
        # 2. get sent to a device
        super(MockMaskRCNNNetwork, self).__init__()
        self.layer = torch.nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.layer(x)

    def predict(self, x):
        # The actual embryovision Mask-RCNN networks return a list of
        # DetectionResults.
        # So we just return a list of detection results, with
        # of a random length
        n_detected = np.random.randint(7)
        return [make_simple_detection_result() for _ in range(n_detected)]

    def predict_proba(self, x):
        return self.predict(x)



if __name__ == '__main__':
    unittest.main()

