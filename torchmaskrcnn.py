import torch


class MaskRCNN(torch.nn.Module):

    def __init__(self, input_shape, input_type, input_dtype):
        super(MaskRCNN, self).__init__()
        self.input_shape = input_shape
        self.input_type = input_type
        self.input_dtype = input_dtype
        self.input_error_message = (
            "Input must be a list of torch.Tensor(s) of dtype {} and " + 
            "shape ({}, {}, {}).".format(input_dtype, *input_shape))
        self._initialize()

    def _predict(self, input_x):
        """Does the actual calculation of the network."""
        raise NotImplementedError("Implement in subclass")
        
    def predict(self, input_x):
        if not self._is_valid_input(input_x):
            raise ValueError(self.input_error_message)
        return self._predict(input_x)
    
    def _is_valid_input(self, input_x):
        valid = type(input_x) == type(list())
        for x in input_x:
            valid = valid and (
                type(x) == self.input_type and
                x.shape == self.input_shape and
                x.dtype == self.input_dtype
                )
        return valid

    def _initialize(self):
        """Contains code that actually sets up the network."""
        raise NotImplementedError("Implement in subclass")
