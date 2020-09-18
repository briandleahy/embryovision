import os
import unittest

import torch

from embryovision.managedata import AnnotatedImage
from embryovision.tests.common import get_loadable_filenames
from embryovision.predictor import load_classifier
from embryovision.stage.stagepredictor import StagePredictor
from embryovision.tests.test_predictor import MockPredictor


class TestStagePredictor(unittest.TestCase):
    def test_loadname(self):
        self.assertIn('stageclassifier.pkl', StagePredictor.loadname)
        self.assertTrue(os.path.exists(StagePredictor.loadname))

    def test_load_network(self):
        self.assertEqual(StagePredictor.load_network, load_classifier)

    def test_input_shape_is_correct(self):
        self.assertEqual(StagePredictor.input_shape, (224, 224))

    def test_under_predict(self):
        p = MockStagePredictor(device='cpu')
        filenames = get_loadable_filenames()[:1]
        boxes = [
            (10, 10, p.input_shape[0], p.input_shape[1])
            for _ in filenames]
        filenames_and_boxes = [(f, b) for f, b in zip(filenames, boxes)]
        out = p._predict(filenames_and_boxes)
        for value in out:
            self.assertIsInstance(value, AnnotatedImage)

    def test_load_images_at_one_timepoint_returns_correct_shape(self):
        p = MockStagePredictor(device='cpu')
        f0_filename = get_loadable_filenames()[0]
        box = (0, 0, 500, 500)
        loaded = p._load_inputs_at_one_timepoint(f0_filename, box)
        self.assertEqual(loaded.shape, (3,) + p.input_shape)

    def test_read_images_for_stage_returns_correct_type(self):
        p = MockStagePredictor(device='cpu')
        filenames = get_loadable_filenames()[:1]
        box = (0, 0, 500, 500)
        names_and_boxes = [(nm, box) for nm in filenames]
        tensor = p._read_images_for_stage(names_and_boxes)
        self.assertIsInstance(tensor, torch.Tensor)

    def test_read_images_for_stage_returns_correct_size(self):
        p = MockStagePredictor(device='cpu')
        filenames = get_loadable_filenames()[:1]
        box = (0, 0, 500, 500)
        names_and_boxes = [(nm, box) for nm in filenames]
        tensor = p._read_images_for_stage(names_and_boxes)
        self.assertEqual(tensor.size(), (1, 3) + p.input_shape)


class MockStagePredictor(MockPredictor, StagePredictor):
    pass


if __name__ == '__main__':
    unittest.main()

