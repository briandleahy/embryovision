import unittest

import torch
import numpy as np
from nose.plugins.attrib import attr

from embryovision.tvmaskrcnn.objectsegmentation import ObjectSegmentationModel


@attr('slow')
class TestOutputObjectSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create model
        cls.base_model = ObjectSegmentationModel(input_size=(500, 500),
                                                 num_classes=2)
        cls.base_model.eval()
        # Generate input image
        nsamples = 2
        input_shape = (nsamples,) + cls.base_model.input_shape
        valid_input = torch.rand(input_shape).type(torch.float32)
        valid_input = [img for img in valid_input]
        # Predict output
        cls.pred_out = cls.base_model.predict(valid_input)

    def test_output_is_list(self):
        self.assertEqual(type(self.pred_out), type(list()))

    def test_output_confidence_is_float(self):
        for pred_inst in self.pred_out:
            self.assertEqual(pred_inst.confidence.dtype, np.float32)

    def test_output_mask_is_float(self):
        for pred_inst in self.pred_out:
            self.assertEqual(pred_inst.mask.dtype, np.float32)

    def test_output_mask_has_correct_shape(self):
        for pred_inst in self.pred_out:
            self.assertEqual(pred_inst.mask.shape,
                             (self.base_model.input_shape[1],
                              self.base_model.input_shape[2]))

    def test_output_box_is_float(self):
        for pred_inst in self.pred_out:
            self.assertEqual(pred_inst.box.dtype, np.float32)

    def test_output_box_has_correct_shape(self):
        for pred_inst in self.pred_out:
            self.assertEqual(pred_inst.box.shape, (4,))


if __name__ == '__main__':
    unittest.main()
