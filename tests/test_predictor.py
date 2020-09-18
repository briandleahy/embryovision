import unittest

import torch

from embryovision.managedata import AnnotatedImage
from embryovision.predictor import Predictor
from embryovision.tests.common import INFOS


class TestPredictor(unittest.TestCase):
    def test_init_stores_device(self):
        for device in ['cuda', 'cpu']:
            p = MockPredictor(device=device)
            self.assertEqual(p.device, device)

    def test_loads_network(self):
        p = MockPredictor('cpu')
        self.assertIsInstance(p.network, torch.nn.Module)

    def test_sends_network_to_device(self):
        for device in ['cuda', 'cpu']:
            p = MockPredictor(device=device)
            parameter = next(p.network.parameters())
            self.assertEqual(parameter.device.type, device)

    def test_sets_default_blocksize_to_10(self):
        p = MockPredictor('cpu')
        self.assertEqual(p.blocksize, 10)

    def test_stores_blocksize(self):
        for b in [1, 10]:
            p = MockPredictor(device='cpu', blocksize=b)
            self.assertEqual(p.blocksize, b)

    def test_pack_into_annotated_images(self):
        p = MockPredictor(device='cpu')
        annotations = ['test' for _ in INFOS]
        out = p.pack_into_annotated_images(INFOS, annotations)
        for info, annotation, both in zip(INFOS, annotations, out):
            self.assertIsInstance(both, AnnotatedImage)
            self.assertEqual(both.info, info)
            self.assertEqual(both.annotation, annotation)


class MockPredictor(Predictor):
    @staticmethod
    def load_network(loadname):
        return MockNetwork()


class MockNetwork(torch.nn.Module):
    def __init__(self):
        # We just make a simple pytorch network that can
        # 1. take an image as input
        # 2. get sent to a device
        super(MockNetwork, self).__init__()
        self.layer = torch.nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.layer(x)

    def predict(self, x):
        return self(x).detach().numpy()

    def predict_proba(self, x):
        return self.predict(x)


if __name__ == '__main__':
    unittest.main()
