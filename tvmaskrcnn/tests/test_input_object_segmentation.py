import unittest
import torch
import numpy as np

from embryovision.tvmaskrcnn.objectsegmentation import ObjectSegmentationModel


class TestInputObjectSegmentation(unittest.TestCase):
    def test_invalid_input_dtype(self):
        segmentation_model = fast_load_model()
        nsamples = 2
        input_shape = (nsamples,) + segmentation_model.input_shape
        invalid_input = torch.rand(input_shape).type(torch.uint8)
        invalid_input = [img for img in invalid_input]
        self.assertRaisesRegex(
            ValueError,
            "I*float32*",
            segmentation_model.predict,
            invalid_input)

    def test_invalid_input_type_list(self):
        segmentation_model = fast_load_model()
        nsamples = 2
        input_shape = (nsamples,) + segmentation_model.input_shape
        invalid_input = torch.rand(input_shape).type(torch.float32)
        self.assertRaisesRegex(
            ValueError,
            "I*list*",
            segmentation_model.predict,
            invalid_input)

    def test_invalid_input_type_numpy(self):
        segmentation_model = fast_load_model()
        nsamples = 2
        input_shape = (nsamples,) + segmentation_model.input_shape
        invalid_input = np.random.rand(*input_shape).astype(np.float32)
        invalid_input = [img for img in invalid_input]
        self.assertRaisesRegex(
            ValueError,
            "I*torch*",
            segmentation_model.predict,
            invalid_input)

    def test_invalid_input_resolution(self):
        segmentation_model = fast_load_model()
        nsamples = 2
        input_shape = (nsamples, 3, 200, 200)
        invalid_input = torch.rand(input_shape).type(torch.float32)
        invalid_input = [img for img in invalid_input]
        self.assertRaisesRegex(
            ValueError,
            "I*shape*",
            segmentation_model.predict,
            invalid_input)

    def test_invalid_input_channel_num(self):
        segmentation_model = fast_load_model()
        nsamples = 2
        input_shape = (nsamples, 1, ) + \
                      segmentation_model.input_shape[1:]
        invalid_input = torch.rand(input_shape).type(torch.float32)
        invalid_input = [img for img in invalid_input]
        self.assertRaisesRegex(
            ValueError,
            "I*shape*",
            segmentation_model.predict,
            invalid_input)


base_model = ObjectSegmentationModel(input_size=(500, 500), num_classes=2)

def fast_load_model():
    base_model.eval()
    return base_model


if __name__ == '__main__':
    unittest.main()
