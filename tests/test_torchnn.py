import unittest

import numpy as np
import torch

from embryovision import torchnn


INPUT_SIZE = 13
NUM_CLASSES = 5
TOLS = {'atol': 1e-13, 'rtol': 1e-13}
FLOAT_TOLS = {'atol': 1e-6, 'rtol': 1e-6}
_no_cuda = not torch.cuda.is_available()
_no_cuda_message = '`cuda` is not available'


class TestClassifier(unittest.TestCase):
    # We test via calling the super() methonds on the initializable LC
    def test_initialize_raises_error(self):
        lc = make_logistic_classifier()
        self.assertRaises(
            NotImplementedError, super(lc.__class__, lc)._initialize)

    def test_forward_raises_error(self):
        lc = make_logistic_classifier()
        self.assertRaises(
            NotImplementedError, super(lc.__class__, lc)._forward, None)

    def test_grab_parameters_returns_correct_size(self):
        lc = torchnn.LogisticClassifier(INPUT_SIZE, NUM_CLASSES)
        lc_params = lc.grab_parameters_as_numpy()

        n_matrix_parameters = INPUT_SIZE * NUM_CLASSES
        n_offset_parameters = NUM_CLASSES
        n_total_params = n_matrix_parameters + n_offset_parameters

        self.assertEqual(lc_params.size, n_total_params)

    def test_grab_parameters_returns_correct_flat(self):
        lc = torchnn.LogisticClassifier(INPUT_SIZE, NUM_CLASSES)
        lc_params = lc.grab_parameters_as_numpy()
        self.assertEqual(lc_params.ndim, 1)

    @unittest.skipIf(_no_cuda, _no_cuda_message)
    def test_grab_parameters_on_cuda(self):
        lc = torchnn.LogisticClassifier(INPUT_SIZE, NUM_CLASSES)
        params_cpu = lc.grab_parameters_as_numpy()
        lc.to('cuda')
        params_cuda = lc.grab_parameters_as_numpy()
        self.assertTrue(np.allclose(params_cpu, params_cuda, **FLOAT_TOLS))

    @unittest.skipIf(_no_cuda, _no_cuda_message)
    def test_predict_proba_works_when_on_gpu(self):
        lc = make_logistic_classifier()
        data = make_random_input_of_size(40)
        probs_cpu = lc.predict_proba(data)

        data_gpu = data.to('cuda')
        lc.to('cuda')
        probs_gpu = lc.predict_proba(data_gpu)
        self.assertTrue(np.allclose(probs_cpu, probs_gpu, **FLOAT_TOLS))

    @unittest.skipIf(_no_cuda, _no_cuda_message)
    def test_predict_works_when_on_gpu(self):
        lc = make_logistic_classifier()
        data = make_random_input_of_size(40)
        labels_cpu = lc.predict(data)

        data_gpu = data.to('cuda')
        lc.to('cuda')
        labels_gpu = lc.predict(data_gpu)
        self.assertTrue(np.all(labels_cpu == labels_gpu))


class TestLogisticClassifier(unittest.TestCase):
    def test_init_does_not_crash(self):
        lc = make_logistic_classifier()
        self.assertTrue(lc is not None)

    def test_probability_prediction_gives_correct_shape(self):
        lc = make_logistic_classifier()
        each_ok = []
        for number_of_points in [9, 13, 3]:
            data = make_random_input_of_size(number_of_points)
            probs = lc.predict_proba(data)
            correct_shape = (number_of_points, NUM_CLASSES)
            each_ok.append(probs.shape == correct_shape)
        self.assertTrue(all(each_ok))

    def test_probabilities_sum_to_one(self):
        lc = make_logistic_classifier()
        each_ok = []
        for number_of_points in [9, 13, 3]:
            data = make_random_input_of_size(number_of_points)
            probs = lc.predict_proba(data)
            total_probability = probs.sum(axis=1)
            this_ok = np.allclose(total_probability, 1.0, **FLOAT_TOLS)
            each_ok.append(this_ok)
        self.assertTrue(all(each_ok))

    def test_probabilities_and_predictions_agree(self):
        lc = make_logistic_classifier()
        data = make_random_input_of_size(500)
        probs = lc.predict_proba(data)
        predicted_from_probs = probs.argmax(axis=1)
        predicted = lc.predict(data)
        self.assertTrue(np.all(predicted == predicted_from_probs))

    def test_forward_raises_error_if_input_is_not_torch(self):
        data_torch = make_random_input_of_size(12)
        data_numpy = data_torch.numpy()
        data_list = data_numpy.tolist()

        lc = make_logistic_classifier()
        for data in [data_numpy, data_list]:
            self.assertRaises(ValueError, lc.forward, data)

        # And ensure that it does not raise an error when passed a torch
        # tensor:
        _ = lc.forward(data_torch)
        self.assertTrue(True)

    def test_forward_raises_error_if_input_is_not_float32(self):
        data_torch32 = make_random_input_of_size(12)
        data_numpy = data_torch32.numpy()
        data_torch64 = torch.from_numpy(data_numpy.astype('float64'))

        lc = make_logistic_classifier()
        self.assertRaises(ValueError, lc.forward, data_torch64)

    def test_forward_raises_error_if_input_is_incorrect_size(self):
        data_torch = make_random_input_of_size(29)
        data_numpy = data_torch.numpy()
        wrong_size_numpy = data_numpy[:, :-1]
        wrong_size_torch = torch.from_numpy(wrong_size_numpy)

        lc = make_logistic_classifier()
        self.assertRaises(ValueError, lc.forward, wrong_size_torch)

        # And ensure that it does not raise an error when the right size:
        _ = lc.forward(data_torch)
        self.assertTrue(True)

    def test_forward_raises_error_if_input_is_incorrect_dimensions(self):
        data_torch = make_random_input_of_size(1)
        data_numpy = data_torch.numpy().squeeze()
        wrong_dim_torch = torch.from_numpy(data_numpy)

        lc = make_logistic_classifier()
        self.assertRaises(ValueError, lc.forward, wrong_dim_torch)


def make_logistic_classifier():
    lc = torchnn.LogisticClassifier(INPUT_SIZE, NUM_CLASSES)
    return lc


def make_random_input_of_size(number_of_points):
    data = np.random.randn(number_of_points, INPUT_SIZE).astype('float32')
    return torch.from_numpy(data)


if __name__ == '__main__':
    unittest.main()
