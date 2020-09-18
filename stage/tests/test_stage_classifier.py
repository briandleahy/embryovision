import unittest

import torch

from embryovision.stage.stageclassifier import StageClassificationModel


class TestStageClassificationModel(unittest.TestCase):
    def test_forward_does_not_crash_when_in_eval_mode(self):
        resnext = fast_load_resnext()
        nsamples = 2
        input_shape = (nsamples, 3) + resnext.input_size
        input_x = torch.rand(input_shape)
        output = resnext._forward(input_x)
        correct_size = (nsamples, resnext.num_classes)
        self.assertEqual(tuple(output.size()), correct_size)

    def test_forward_does_not_crash_when_in_train_mode(self):
        resnext = fast_load_resnext()
        resnext.train()
        nsamples = 2
        input_shape = (nsamples, 3) + resnext.input_size
        input_x = torch.rand(input_shape)
        output = resnext._forward(input_x)
        correct_size = (nsamples, resnext.num_classes)
        self.assertEqual(tuple(output.size()), correct_size)

    def test_predict_runs_on_correct_format_data(self):
        resnext = fast_load_resnext()
        nsamples = 2
        input_shape = (nsamples, 3) + resnext.input_size
        input_x = torch.rand(input_shape)
        labels = resnext.predict(input_x)
        correct_size = (nsamples,)
        self.assertEqual(labels.shape, correct_size)

    def test_invalid_input_message_states_float32(self):
        resnext = fast_load_resnext()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnext.input_size)
        self.assertRaisesRegex(
            ValueError,
            "I*float32*",
            resnext.predict,
            wrong_input)

    def test_invalid_input_message_states_torch_tensor(self):
        resnext = fast_load_resnext()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnext.input_size)
        self.assertRaisesRegex(
            ValueError,
            "I*torch.Tensor*",
            resnext.predict,
            wrong_input)

    def test_invalid_input_message_states_correct_shape(self):
        resnext = fast_load_resnext()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnext.input_size)
        self.assertRaisesRegex(
            ValueError,
            r"I*\?, 3, {}, {}*".format(*resnext.input_size),
            resnext.predict,
            wrong_input)


model_toTest = StageClassificationModel(
    input_size=(224, 224), num_classes=13, pretrained=False)


def fast_load_resnext():
    model_toTest.eval()
    return model_toTest


if __name__ == '__main__':
    unittest.main()

