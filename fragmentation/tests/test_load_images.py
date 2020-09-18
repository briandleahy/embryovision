import unittest

import numpy as np
import torch

from embryovision import util
from embryovision.fragmentation.load_images import (
    load_images_for_fragmentation_regressor)
from embryovision.tests.common import get_loadable_filenames


class TestLoadImages(unittest.TestCase):
    def test_shape_is_correct(self):
        filenames = get_loadable_filenames()
        loaded = load_images_for_fragmentation_regressor(filenames)
        correct_leading_dimensions = (len(filenames), 3)
        self.assertEqual(loaded.size()[:2], correct_leading_dimensions)

    def test_dimension_is_correct(self):
        filenames = get_loadable_filenames()[:2]
        loaded = load_images_for_fragmentation_regressor(filenames)
        correct_leading_dimensions = (len(filenames), 3)
        self.assertEqual(len(loaded.size()), 4)

    def test_datatype_is_torch_tensor(self):
        filenames = get_loadable_filenames()[:2]
        loaded = load_images_for_fragmentation_regressor(filenames)
        self.assertIsInstance(loaded, torch.Tensor)

    def test_numeric_dtype_is_float32(self):
        filenames = get_loadable_filenames()[:2]
        loaded = load_images_for_fragmentation_regressor(filenames)
        self.assertEqual(loaded.dtype, torch.float32)

    def test_normalization_is_correct(self):
        filename = get_loadable_filenames()[:1]
        loaded = load_images_for_fragmentation_regressor(filename)
        raw = util.read_image(filename[0])

        # The normalization should increase the standard deviation of
        # the image by ~1/ 0.229, which is at least 4:
        self.assertGreater(loaded.std(), 4 * raw.std())


if __name__ == '__main__':
    unittest.main()

