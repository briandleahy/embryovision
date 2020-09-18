from collections import deque

import numpy as np
import torch


class Classifier(torch.nn.Module):
    input_error_msg = 'Invalid input'

    def __init__(self, input_size, num_classes):
        """
        Parameters
        ----------
        input_size : int
            The number of elements in each vector.
        num_classes : int
            The number of classes to predict.
        """
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self._initialize()

    def predict_proba(self, x):
        with torch.no_grad():
            zs = self.forward(x)
            ps = torch.nn.functional.softmax(zs, dim=1)
        # And I like numpy:
        return ps.cpu().detach().numpy()

    def predict(self, x):
        with torch.no_grad():
            zs = self.forward(x)
        zs_np = zs.cpu().detach().numpy()
        return zs_np.argmax(axis=1)

    def forward(self, input_x):
        """``sequence`` must be a torch.tensor of dtype float32"""
        if not self.is_input_valid(input_x):
            raise ValueError(self.input_error_msg)
        zs = self._forward(input_x)
        return torch.nn.functional.log_softmax(zs, dim=1)

    def is_input_valid(self, input_x):
        valid = (type(input_x) == torch.Tensor and
                 input_x.dtype == torch.float32 and
                 len(input_x.size()) == 2 and
                 input_x.size()[1] == self.input_size)
        return valid

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


class LogisticClassifier(Classifier):
    """
    Classify a vector x_i.
    """

    input_error_msg = (
        'Input must be torch.Tensor of dtype float32, shape (?, input_size)')

    def _forward(self, x):
        """``x`` must be a torch.tensor of dtype float32"""
        return self._classification_layer(x)

    def _initialize(self):
        self._classification_layer = self._create_classification_layer()

    def _create_classification_layer(self):
        return torch.nn.Linear(self.input_size, self.num_classes)

