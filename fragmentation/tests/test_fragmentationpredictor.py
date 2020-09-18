import os
import io
import unittest

from embryovision.managedata import AnnotatedImage
from embryovision.tests.common import get_loadable_filenames
from embryovision.predictor import load_classifier
from embryovision.fragmentation.fragmentationpredictor import FragPredictor
from embryovision.tests.test_predictor import MockPredictor


class TestFragPredictor(unittest.TestCase):
    def test_loadname(self):
        self.assertIn('fragmentationclassifier.pkl', FragPredictor.loadname)
        self.assertTrue(os.path.exists(FragPredictor.loadname))

    def test_load_network(self):
        self.assertEqual(FragPredictor.load_network, load_classifier)

    def test_input_shape_is_correct(self):
        self.assertEqual(FragPredictor.input_shape, (299, 299))

    def test_under_predict(self):
        p = MockFragPredictor(device='cpu')
        filenames = get_loadable_filenames()[:1]  # we can only augment 001.jpg
        boxes = [
            (10, 10, p.input_shape[0], p.input_shape[1])
            for _ in filenames]
        filenames_and_boxes = [(f, b) for f, b in zip(filenames, boxes)]
        out = p._predict(filenames_and_boxes)
        for value in out:
            self.assertIsInstance(value, AnnotatedImage)

    def test_crop_images_in_ram(self):
        p = MockFragPredictor(device='cpu')
        filenames = get_loadable_filenames()[:2]
        boxes = [
            (10, 10, p.input_shape[0], p.input_shape[1])
            for _ in filenames]
        filenames_and_boxes = [(f, b) for f, b in zip(filenames, boxes)]
        cropped = p._crop_images_in_ram(filenames_and_boxes)

        for i, nm in enumerate(filenames):
            box = boxes[i]
            out = cropped[i]
            self.assertIsInstance(out, io.BytesIO)


class MockFragPredictor(MockPredictor, FragPredictor):
    pass


if __name__ == '__main__':
    unittest.main()

