import os
import unittest

import torch
import numpy as np
from PIL import Image

from embryovision import util
from embryovision.tests.common import get_loadable_filenames


class TestReadImage(unittest.TestCase):
    def test_read_image_returns_numpy(self):
        filename = get_loadable_filenames()[0]
        image = util.read_image(filename)
        self.assertIsInstance(image, np.ndarray)

    def test_read_image_returns_correct_shape(self):
        filename = get_loadable_filenames()[0]
        image = util.read_image(filename)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

    def test_read_image_returns_float_on_01(self):
        filename = get_loadable_filenames()[0]
        image = util.read_image(filename)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 1)


class TestReadImageForTorch(unittest.TestCase):
    def test_read_image_for_torch_returns_torch(self):
        filenames = get_loadable_filenames()
        as_torch = util.read_images_for_torch(filenames)
        self.assertIsInstance(as_torch, torch.Tensor)

    def test_read_image_for_torch_returns_correct_shape(self):
        # torch expects (n_images, channels, size
        filenames = get_loadable_filenames()
        as_torch = util.read_images_for_torch(filenames)
        n_channels = 3
        self.assertEqual(as_torch.size()[:2], (len(filenames), n_channels))


class TestLoadAndCropImage(unittest.TestCase):
    def test_returns_pil_image(self):
        filename = get_loadable_filenames()[0]
        box = (1, 1, 2, 2)
        image = util.load_and_crop_image(filename, box)
        self.assertIsInstance(image, Image.Image)

    def test_output_image_is_correct_shape(self):
        filename = get_loadable_filenames()[0]
        box = (1, 1, 100, 100)
        shape = (150, 140)
        image = util.load_and_crop_image(filename, box, output_shape=shape)
        self.assertEqual(image.size, shape)

    def test_crop_box_is_used_with_resize_nearest(self):
        # we crop to a 1 px image, and check that all image values
        # are the same value
        filename = get_loadable_filenames()[0]
        box = (1, 1, 2, 2)
        image = util.load_and_crop_image(filename, box)

        correct_px_value = np.array(Image.open(filename))[box[0], box[1]]
        self.assertTrue(np.all(np.array(image) == correct_px_value))


class TestLoadImageIntoRam(unittest.TestCase):
    def test_load_image_as_bytes_io(self):
        filename = get_loadable_filenames()[0]
        loaded_into_ram = util.load_image_into_ram(filename)
        image0 = util.read_image(filename)
        image1 = util.read_image(loaded_into_ram)
        self.assertTrue(np.all(image0 == image1))


class TestTransformingCollection(unittest.TestCase):
    def test_getitem_transforms(self):
        np.random.seed(400)
        data = np.random.randn(20)
        transform = lambda x: -2 * x
        loader = util.TransformingCollection(data, transform)

        index = 0
        self.assertEqual(transform(data[index]), loader[index])

    def test_len(self):
        data = np.random.randn(20)
        transform = lambda x: -2 * x
        loader = util.TransformingCollection(data, transform)
        self.assertEqual(len(loader), data.size)

    def test_on_images(self):
        filenames = get_loadable_filenames()
        images_ram = [util.load_image_into_ram(nm) for nm in filenames]
        loader = util.TransformingCollection(images_ram, util.read_image)

        index = 0
        image_filename = util.read_image(filenames[index])
        image_loader = loader[index]
        self.assertTrue(np.all(image_filename == image_loader))


class TestMisc(unittest.TestCase):
    def test_split_all(self):
        dummy_folder = '/some/long/directory/structure/'
        filename = 'D2017_05_05_S1477_I313_pdb/WELL06/F0/016.jpg'
        fullname = os.path.join(dummy_folder, filename)

        fullname_f0_split = util.split_all(fullname)
        correct_answer = (
            '/', 'some', 'long', 'directory', 'structure',
            'D2017_05_05_S1477_I313_pdb', 'WELL06', 'F0', '016.jpg')
        self.assertEqual(fullname_f0_split, correct_answer)

    def test_augment_focus(self):
        filename = get_loadable_filenames()[0]
        augmented = util.augment_focus(filename)
        for foundname, focus_correct in zip(augmented, ['F-15', 'F0', 'F15']):
            *head, focus_found, image_number = util.split_all(foundname)
            self.assertTrue(os.path.exists(foundname))
            self.assertEqual(focus_found, focus_correct)

    def test_augment_focus_raises_error_when_no_filename(self):
        unloadable_filename = '/some/wrong/directory/structure/001.jpg'
        assert not os.path.exists(unloadable_filename)
        self.assertRaises(
            FileNotFoundError,
            util.augment_focus,
            unloadable_filename,)


def make_loader():
    filenames = get_loadable_filenames()
    return util.ImageTransformingCollection(filenames)


if __name__ == '__main__':
    unittest.main()

