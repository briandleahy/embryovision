import sys

import torch
import torchvision

from embryovision.torchnn import Classifier


class Resnet101TransferLearned(Classifier):
    def __init__(self, input_size=(125, 125), num_classes=4, pretrained=True):
        """
        Explain why input_size is here to be helpful.
        """
        self.pretrained = pretrained
        super(Resnet101TransferLearned, self).__init__(
            input_size=input_size, num_classes=num_classes)

    def _initialize(self):
        r101 = torchvision.models.segmentation.fcn_resnet101(
            pretrained=self.pretrained)
        r101.classifier[4] = torch.nn.Conv2d(
            512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.resnet = r101

    def freeze_earlier_layers(self):
        last_layer_parameters = list(self.last_layer.parameters())
        for param in self.parameters():
            if all([param is not plast for plast in last_layer_parameters]):
                param.requires_grad = False

    def unfreeze_earlier_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def last_layer(self):
        return self.resnet.classifier[4]

    def _forward(self, x):
        result = self.resnet.forward(x)
        return torch.nn.functional.log_softmax(result['out'], dim=1)

    def is_input_valid(self, input_x):
        valid = (type(input_x) == torch.Tensor and
                 input_x.dtype == torch.float32 and
                 len(input_x.size()) == 4 and
                 input_x.size()[1:] == (3,) + tuple(self.input_size))
        return valid

    @property
    def input_error_msg(self):
        msg = ("Input must be a torch.Tensor of shape (?, 3, " +
               "{}, {}) and dtype float32.".format(*self.input_size))
        return msg

