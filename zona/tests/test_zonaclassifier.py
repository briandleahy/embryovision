import unittest

import torch

from embryovision.zona.zonaclassifier import Resnet101TransferLearned


class TestResnet101TransferLearned(unittest.TestCase):
    def test_initializes_constructs_resnet(self):
        resnet = fast_load_resnet()
        self.assertTrue(hasattr(resnet, 'resnet'))

    def test_last_layer_returns_valid_layer(self):
        resnet = fast_load_resnet()
        last_layer = resnet.last_layer
        self.assertIsInstance(last_layer, torch.nn.Conv2d)

    def test_last_layer_has_correct_shape(self):
        resnet = fast_load_resnet()
        last_layer = resnet.last_layer
        self.assertEqual(last_layer.out_channels, resnet.num_classes)

    def test_freeze_earlier_layers_freezes_earlier_layers(self):
        resnet = fast_load_resnet()
        resnet.freeze_earlier_layers()
        last_layer_parameters = list(resnet.last_layer.parameters())
        earlier_parameters = [
            p for p in resnet.parameters()
            if all([p is not plast for plast in last_layer_parameters])]
        for parameter in earlier_parameters:
            self.assertFalse(parameter.requires_grad)

    def test_freeze_earlier_layers_does_not_freeze_last_layer(self):
        resnet = fast_load_resnet()
        resnet.freeze_earlier_layers()
        last_layer_parameters = list(resnet.last_layer.parameters())
        for parameter in last_layer_parameters:
            self.assertTrue(parameter.requires_grad)

    def test_unfreeze_earlier_layers(self):
        resnet = fast_load_resnet()
        resnet.freeze_earlier_layers()
        resnet.unfreeze_earlier_layers()
        for parameter in resnet.parameters():
            self.assertTrue(parameter.requires_grad)

    def test_forward_does_not_crash_when_in_eval_mode(self):
        resnet = fast_load_resnet()
        nsamples = 2
        input_shape = (nsamples, 3) + resnet.input_size
        input_x = torch.rand(input_shape)
        output = resnet._forward(input_x)
        correct_size = (nsamples, resnet.num_classes) + resnet.input_size
        self.assertEqual(tuple(output.size()), correct_size)

    def test_forward_does_not_crash_when_in_train_mode(self):
        resnet = fast_load_resnet()
        resnet.train()
        nsamples = 2
        input_shape = (nsamples, 3) + resnet.input_size
        input_x = torch.rand(input_shape)
        output = resnet._forward(input_x)
        correct_size = (nsamples, resnet.num_classes) + resnet.input_size
        self.assertEqual(tuple(output.size()), correct_size)

    def test_predict_runs_on_correct_format_data(self):
        resnet = fast_load_resnet()
        nsamples = 2
        input_shape = (nsamples, 3) + resnet.input_size
        input_x = torch.rand(input_shape)

        labels = resnet.predict(input_x)
        correct_size = (nsamples,) + resnet.input_size
        self.assertEqual(labels.shape, correct_size)

    def test_invalid_input_message_states_float32(self):
        resnet = fast_load_resnet()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnet.input_size)
        self.assertRaisesRegex(
            ValueError,
            "I*float32*",
            resnet.predict,
            wrong_input)

    def test_invalid_input_message_states_torch_tensor(self):
        resnet = fast_load_resnet()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnet.input_size)
        self.assertRaisesRegex(
            ValueError,
            "I*torch.Tensor*",
            resnet.predict,
            wrong_input)

    def test_invalid_input_message_states_correct_shape(self):
        resnet = fast_load_resnet()
        nsamples = 2
        wrong_input = torch.rand((nsamples, 2) + resnet.input_size)
        self.assertRaisesRegex(
            ValueError,
            r"I*\?, 3, {}, {}*".format(*resnet.input_size),
            resnet.predict,
            wrong_input)


RESNET = Resnet101TransferLearned(
    input_size=(125, 125), num_classes=4, pretrained=False)


def fast_load_resnet():
    # Returns a resnet in a consistent state
    RESNET.unfreeze_earlier_layers()
    RESNET.eval()
    return RESNET


if __name__ == '__main__':
    unittest.main()

