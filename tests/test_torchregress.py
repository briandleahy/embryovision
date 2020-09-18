import unittest

import numpy as np
import scipy
import torch

from embryovision import torchregress


INPUT_SIZE = (3, 299, 299)
NUM_CLASSES = 1
TOLS = {'atol': 1e-13, 'rtol': 1e-13}
FLOAT_TOLS = {'atol': 1e-6, 'rtol': 1e-6}
_no_cuda = not torch.cuda.is_available()
_no_cuda_message = '`cuda` is not available'
_slow_scipy_version = float(scipy.__version__[:3]) > 1.3
_slow_scipy_message = (
"""
Due to a performance regression in scipy's random variables,
instantiating an inception network is slow in scipy >= 1.4.0. As a
result, these tests are skipped if your scipy version is 1.4.0 or
greater. If you want to run these tests, you can either (i) downgrade
your scipy, or (ii) modify this file locally.
"""
)


class TestRegressor(unittest.TestCase):
    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_initialize_raises_error(self):
        inception = make_regressor()
        self.assertRaises(
            NotImplementedError,
            super(inception.__class__, inception)._initialize)

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_forward_raises_error(self):
        inception = make_regressor()
        self.assertRaises(
            NotImplementedError,
            super(inception.__class__, inception)._forward,
            None)

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_grab_parameters_returns_correct_flat(self):
        # FIXME this test is slow. Use it w/o an Inception
        inception = make_regressor()
        params = inception.grab_parameters_as_numpy()
        self.assertEqual(params.ndim, 1)

    @unittest.skipIf(_no_cuda or _slow_scipy_version, _no_cuda_message)
    def test_predict_works_when_on_gpu(self):
        inception = make_regressor()
        np.random.seed(1234)
        data = make_random_input_of_size(2)
        labels_cpu = inception.predict(data)

        data_gpu = data.to('cuda')
        inception.to('cuda')
        labels_gpu = inception.predict(data_gpu)
        self.assertTrue(np.allclose(labels_cpu, labels_gpu, **FLOAT_TOLS))


class TestInceptionV3Regressor(unittest.TestCase):
    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_prediction_gives_correct_shape(self):
        inception = make_regressor()
        number_of_points = 5
        data = make_random_input_of_size(number_of_points)
        preds = inception.predict(data)
        self.assertEqual(preds.shape, (number_of_points,))

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_forward_raises_error_if_input_is_not_torch(self):
        data_torch = make_random_input_of_size(1)
        data_numpy = data_torch.numpy()
        data_list = data_numpy.tolist()

        inception = make_regressor()
        for data in [data_numpy, data_list]:
            self.assertRaises(ValueError, inception.forward, data)

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_forward_raises_error_if_input_is_not_float32(self):
        data_torch32 = make_random_input_of_size(1)
        data_numpy = data_torch32.numpy()
        data_torch64 = torch.from_numpy(data_numpy.astype('float64'))

        inception = make_regressor()
        self.assertRaises(ValueError, inception.forward, data_torch64)

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_forward_raises_error_if_input_is_incorrect_size(self):
        data_torch = make_random_input_of_size(1)
        data_numpy = data_torch.numpy()
        wrong_size_numpy = data_numpy[:, :-1]
        wrong_size_torch = torch.from_numpy(wrong_size_numpy)

        inception = make_regressor()
        self.assertRaises(ValueError, inception.forward, wrong_size_torch)

    @unittest.skipIf(_slow_scipy_version, _slow_scipy_message)
    def test_forward_raises_error_if_input_is_incorrect_dimensions(self):
        data_torch = make_random_input_of_size(1)
        data_numpy = data_torch.numpy().squeeze()
        wrong_dim_torch = torch.from_numpy(data_numpy)

        inception = make_regressor()
        self.assertRaises(ValueError, inception.forward, wrong_dim_torch)



class RegressorMaker(object):
    # Singleton pattern

    def __init__(self):
        self._inception = None

    def make_regressor(self):
        if self._inception is None:
            self._make_regressor()
        # We want the regressor in a consistent state:
        self._inception.to('cpu')
        self._inception.eval()
        return self._inception

    def _make_regressor(self):
        self._inception = torchregress.InceptionV3Regressor()


_regressor_maker = RegressorMaker()

def make_regressor():
    return _regressor_maker.make_regressor()


def make_random_input_of_size(number_of_points):
    data = np.random.randn(number_of_points, *INPUT_SIZE).astype('float32')
    return torch.from_numpy(data)


if __name__ == '__main__':
    unittest.main()
