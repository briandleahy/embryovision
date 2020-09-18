import os
import unittest

from embryovision.managedata import AnnotatedImage
from embryovision.tests.common import get_loadable_filenames
from embryovision.predictor import load_classifier
from embryovision.zona.zonapredictor import ZonaPredictor
from embryovision.tests.test_predictor import MockPredictor


class TestZonaPredictor(unittest.TestCase):
    def test_loadname(self):
        self.assertIn('zonaclassifier.pkl', ZonaPredictor.loadname)
        self.assertTrue(os.path.exists(ZonaPredictor.loadname))

    def test_load_network(self):
        self.assertEqual(ZonaPredictor.load_network, load_classifier)

    def test_under_predict(self):
        filenames = get_loadable_filenames()[:2]
        p = MockZonaPredictor(device='cpu')
        out = p._predict(filenames)
        for value in out:
            self.assertIsInstance(value, AnnotatedImage)


class MockZonaPredictor(MockPredictor, ZonaPredictor):
    pass


if __name__ == '__main__':
    unittest.main()

