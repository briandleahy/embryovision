import unittest

import numpy as np
import torch

from embryovision.stage.stagedp import DynamicProgramming
from embryovision.stage.stage_to_number import stage_to_number


class TestDynamicProgramming(unittest.TestCase):

    def test_invalid_classwise_weights_dynamic_programming(self):
        num_classes = 13
        wrong_weights = list(np.random.rand(num_classes-1))
        self.assertRaisesRegex(
            ValueError,
            "C*the same number*",
            DynamicProgramming,
            num_classes,
            wrong_weights)

    def test_predict_on_invalid_input_shape(self):
        dp_model = DynamicProgramming()
        num_frames = 100
        wrong_input = np.random.rand(num_frames, dp_model.num_classes-1)
        wrong_input = wrong_input.astype(np.float32)
        self.assertRaisesRegex(
            ValueError,
            "I*shape*",
            dp_model.predict,
            wrong_input)

    def test_predict_on_invalid_input_dtype(self):
        dp_model = DynamicProgramming()
        num_frames = 100
        wrong_input = np.random.rand(num_frames, dp_model.num_classes)
        wrong_input = wrong_input.astype(np.float64)
        self.assertRaisesRegex(
            ValueError,
            "I*float32",
            dp_model.predict,
            wrong_input)

    def test_predict_has_correct_output_shape(self):
        dp_model = DynamicProgramming()
        num_frames = 100
        correct_input = np.random.rand(num_frames, dp_model.num_classes)
        correct_input = correct_input.astype(np.float32)
        dp_out = dp_model.predict(correct_input)
        correct_size = (num_frames, )
        self.assertEqual(dp_out.shape, correct_size)

    def test_predict_has_correct_output_dtype(self):
        dp_model = DynamicProgramming()
        num_frames = 100
        correct_input = np.random.rand(num_frames, dp_model.num_classes)
        correct_input = correct_input.astype(np.float32)
        dp_out = dp_model.predict(correct_input)
        correct_type = np.int64
        self.assertEqual(dp_out.dtype, correct_type)

    def test_predict_is_monotonically_nondecreasing(self):
        dp_model = DynamicProgramming()
        num_frames = 100
        correct_input = np.random.rand(num_frames, dp_model.num_classes)
        correct_input = correct_input.astype(np.float32)
        dp_out = dp_model.predict(correct_input)
        nonempty_out = dp_out[dp_out!=stage_to_number['empty']]
        for elem_id, dp_elem in enumerate(nonempty_out[:-1]):
            self.assertTrue(dp_elem <= nonempty_out[elem_id+1])


if __name__ == '__main__':
    unittest.main()

