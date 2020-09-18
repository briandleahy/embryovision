from collections import deque

import numpy as np
import torch
from torchvision import models


# FIXME some of the common code between Regression & Classifier should get
# factored out once Won-Dong has his code up

class Regressor(torch.nn.Module):
    input_error_msg = 'Invalid input'

    def __init__(self):
        super(Regressor, self).__init__()
        self._initialize()

    def predict(self, x):
        with torch.no_grad():
            zs = self.forward(x)
        zs_np = zs.cpu().detach().numpy()
        return zs_np[:, 0]

    def forward(self, input_x):
        """``sequence`` must be a torch.tensor of dtype float32"""
        if not self.is_input_valid(input_x):
            raise ValueError(self.input_error_msg)
        return self._forward(input_x)

    def is_input_valid(self, input_x):
        raise NotImplementedError("Implement in subclass")

    def grab_parameters_as_numpy(self):
        parameters_raw = [p for p in self.parameters()]
        params = deque()
        for p in parameters_raw:
            params.extend(p.detach().cpu().numpy().ravel())
        return np.array(params)

    def _initialize(self):
        """Contains code that actually sets up the network."""
        raise NotImplementedError("Implement in subclass")

    def _forward(self, input_x):
        """Does the actual calculation of the network."""
        raise NotImplementedError("Implement in subclass")


class InceptionV3Regressor(Regressor):
    image_shape = (3, 299, 299)
    input_error_msg = (
        "Input must be a torch.tensor of dtype float32 and shape " +
        "(?, {}, {}, {})".format(*image_shape))

    def __init__(self, pretrained=True):
        self.pretrained = pretrained
        super(InceptionV3Regressor, self).__init__()

    def is_input_valid(self, input_x):
        valid = (type(input_x) == torch.Tensor and
                 input_x.dtype == torch.float32 and
                 len(input_x.size()) == 4 and
                 input_x.size()[1:] == self.image_shape)
        return valid

    def _initialize(self):
        inception = models.inception_v3(pretrained=self.pretrained)
        inception.aux_logits = False
        inception.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features=2048, out_features=1, bias=True)
            )
        self.network = inception

    def _forward(self, input_x):
        return self.network(input_x)

