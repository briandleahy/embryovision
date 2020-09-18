import numpy as np

from embryovision.stage.stage_to_number import stage_to_number


class DynamicProgramming(object):
    def __init__(self, num_classes=13, classwise_weights=None):
        if classwise_weights == None:
            classwise_weights = [0.4, 1, 1, 1.2, 1, 1.2, 1.2, 1.2, 1.2, 1.1,
                                 1, 1, 0.01]
        self.num_classes = num_classes
        if not self._is_valid_weight(num_classes, classwise_weights):
            raise ValueError(self.weight_error_message)
        self.classwise_weights = np.reshape(classwise_weights, (1, -1))

    def _manhattan_dynamic_programming(self, pred_prob):
        num_frames = pred_prob.shape[0]
        # Compute cost matrix for dynamic programming
        dp_arr = np.zeros((num_frames+1, self.num_classes))
        # First column
        for i in range(0, num_frames):
            dp_arr[i+1, 0] = dp_arr[i, 0] + pred_prob[i, 0]
        # First row
        for j in range(0, self.num_classes-1):
            dp_arr[0, j+1] = dp_arr[0, j]
        # Other elements by applying the max operation
        for i in range(0, num_frames):
            for j in range(0, self.num_classes-1):
                dp_arr[i+1, j+1] = max(dp_arr[i+1, j],
                                       dp_arr[i, j+1] + pred_prob[i, j+1])
        # Suppress values to zeros if they are not on the optimal path
        # The last column
        j = self.num_classes-1
        for i in range(num_frames-1,-1,-1):
            if dp_arr[i, j] + pred_prob[i, j] != dp_arr[i+1, j]:
                dp_arr[i, j] = 0
        # The last row
        i = num_frames
        for j in range(self.num_classes-2,-1,-1):
            if dp_arr[i, j] != dp_arr[i, j+1]:
                dp_arr[i, j] = 0
        # Other elements
        for i in range(num_frames-1,-1,-1):
            for j in range(self.num_classes-2,-1,-1):
                if (dp_arr[i, j] != dp_arr[i, j+1]) and \
                   ((dp_arr[i, j] + pred_prob[i, j]) != dp_arr[i+1, j]):
                    dp_arr[i, j] = 0
        # DP result as numpy array
        outcome =  np.asarray([np.argmax(dp_arr[frame_id+1] > 0)
                               for frame_id in range(0, num_frames)])
        return outcome

    def predict(self, output_prob):
        if not self._is_valid_input(output_prob):
            raise ValueError(self.input_error_message)
        # Apply class-wise weights
        output_scaled = output_prob * self.classwise_weights
        # Dynamic programming
        output_label = self._manhattan_dynamic_programming(output_scaled)
        # Empty frames
        label_woDP = np.argmax(output_prob, axis =1)
        empty_frames = (label_woDP == stage_to_number['empty'])
        output_label[empty_frames] = stage_to_number['empty']
        return output_label

    def _is_valid_input(self, input_x):
        valid = (type(input_x) == np.ndarray and
                 input_x.dtype == np.float32 and
                 len(input_x.shape) == 2 and
                 input_x.shape[1] == self.num_classes)
        return valid

    def _is_valid_weight(self, num_classes, classwise_weights):
        valid = num_classes == len(classwise_weights)
        return valid

    @property
    def input_error_message(self):
        msg = ("Input must be a numpy array of shape (?, {})" \
               " and dtype float32.".format(self.num_classes))
        return msg

    @property
    def weight_error_message(self):
        msg = ("Class-wise weights should be a list with the same number" \
               " of entries to the stage labels, which is {}.".format(
                self.num_classes))
        return msg

