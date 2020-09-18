import torch
import torch.nn as nn
import torchvision

from embryovision.torchnn import Classifier


class StageClassificationModel(Classifier):
    def __init__(self, input_size=(224, 224), num_classes=13,
                 pretrained=True):
        self.pretrained = pretrained
        super(StageClassificationModel, self).__init__(
            input_size=input_size, num_classes=num_classes)

    def _initialize(self):
        self.network = self._create_network()

    def _create_network(self):
        conv = torchvision.models.resnext101_32x8d(pretrained=self.pretrained)
        num_ftrs = conv.fc.in_features

        fc_dim1 = 200
        fc_dim2 = 100

        conv.fc = nn.Linear(num_ftrs, fc_dim1)

        relu1 = torch.nn.ReLU()
        bn1 = torch.nn.BatchNorm1d(fc_dim1, eps=1e-5, affine=True)
        fc1 = nn.Linear(fc_dim1, fc_dim2)
        relu2 = torch.nn.ReLU()
        bn2 = torch.nn.BatchNorm1d(fc_dim2, eps=1e-5, affine=True)
        fc2 = nn.Linear(fc_dim2, self.num_classes)
        drop1 = nn.Dropout(0.5)
        drop2 = nn.Dropout(0.3)

        fc_layer_1 = torch.nn.Sequential(
            relu1,
            bn1,
            drop1,
            fc1,
            )
        fc_layer_2 = torch.nn.Sequential(
            relu2,
            bn2,
            drop2,
            fc2,
            )
        network = torch.nn.Sequential(
            conv,
            fc_layer_1,
            fc_layer_2,
            )
        return network

    def _forward(self, x):
        return self.network.forward(x)

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
