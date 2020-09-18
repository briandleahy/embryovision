import io
import unittest

import numpy as np
from nose.plugins.attrib import attr

from embryovision.pipeline import (
    Pipeline, compress_zona, decompress_zona, predict)
from embryovision.managedata import (
    AnnotatedDataset, AnnotatedImage, ImageInfo, FilenameParser as FP)
from embryovision.tests.common import get_loadable_filenames, INFOS


@attr('slow')
class TestPredictWithCompress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filenames = get_loadable_filenames()[:1]
        cls.result = predict(filenames, device='cpu', zona_compressed=True)

    def test_has_correct_keys(self):
        correct_keys = {
            'zona', 'boxes', 'frag', 'stage_raw', 'stage_smooth',
            'pronuclei', 'blastomeres'}
        self.assertEqual(set(self.result.keys()), correct_keys)

    def test_has_compressed_zona(self):
        for image in self.result['zona'].iterate_over_images():
            self.assertIsInstance(image, AnnotatedImage)
            self.assertIsInstance(image.annotation, io.BytesIO)


@attr('slow')
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = Pipeline('cpu')
        filenames = get_loadable_filenames()[:1]
        cls.out = p.predict_all(filenames)
        cls.keys = ['zona', 'boxes', 'frag', 'stage_raw', 'stage_smooth',
                    'pronuclei', 'blastomeres']
        # Then we want to make sure that the detector actually picked up
        # pronuclei & cells, otherwise these are crappy tests:
        for key in ['pronuclei', 'blastomeres']:
            assert len(cls.out[key].iterate_over_images()) > 0

    def test_predict_all_returns_datasets(self):
        for k in self.keys:
            self.assertIsInstance(self.out[k], AnnotatedDataset)

    def test_zona_is_fully_convolutional_labels(self):
        zona = self.out['zona']
        for i in zona.iterate_over_images():
            self.assertEqual(i.annotation.shape, (500, 500))
            self.assertEqual(i.annotation.dtype, 'uint8')

    def test_boxes_are_boxes(self):
        boxes = self.out['boxes']
        for i in boxes.iterate_over_images():
            self.assertEqual(len(i.annotation), 4)

    def test_frag_values_are_floats(self):
        frag = self.out['frag']
        for i in frag.iterate_over_images():
            self.assertEqual(float(i.annotation), i.annotation)

    def test_stage_raw_is_13_elements(self):
        stage_raw = self.out['stage_raw']
        for i in stage_raw.iterate_over_images():
            self.assertEqual(len(i.annotation), 13)

    def test_stage_smooth_is_smoothed(self):
        stage_smooth = self.out['stage_smooth']
        for i in stage_smooth.iterate_over_images():
            self.assertEqual(i.annotation, 1)

    def test_pronuclei_outputs_are_correct_type(self):
        pronuclei = self.out['pronuclei']
        for i in pronuclei.iterate_over_images():
            self.assertIsInstance(i.annotation, list)
            for detection in i.annotation:
                self.assertEqual(
                    set(detection.keys()),
                    {'confidence', 'xy_polygon'})

    def test_blastomeres_outputs_are_correct_type(self):
        blastomeres = self.out['blastomeres']
        for i in blastomeres.iterate_over_images():
            self.assertIsInstance(i.annotation, list)
            for detection in i.annotation:
                self.assertEqual(
                    set(detection.keys()),
                    {'confidence', 'xy_polygon'})

    def test_grab_1cell_names_and_boxes(self):
        stages = make_fake_smoothed_stages()
        box = (1, 1, 100, 100)
        names_and_boxes = [
            (FP.get_partial_filename_from_imageinfo(i.info), box)
            for i in stages.iterate_over_images()]
        p = Pipeline('cpu')

        onecell_names_boxes = p._grab_1cell_names_and_boxes(
            stages, names_and_boxes)

        images_in_onecell_stage = [
            i for i in stages.iterate_over_images() if i.annotation == 1]
        for name, _ in onecell_names_boxes:
            info = FP.get_imageinfo_from_filename(name)
            self.assertEqual(stages[info].annotation, 1)
        self.assertEqual(
            len(images_in_onecell_stage),
            len(onecell_names_boxes))

    def test_grab_cleavage_names_and_boxes(self):
        stages = make_fake_smoothed_stages()
        box = (1, 1, 100, 100)
        names_and_boxes = [
            (FP.get_partial_filename_from_imageinfo(i.info), box)
            for i in stages.iterate_over_images()]
        p = Pipeline('cpu')

        cleavage_names_boxes = p._grab_cleavage_names_and_boxes(
            stages, names_and_boxes)

        cleavage_stages = {1, 2, 3, 4, 5, 6, 7, 8}
        images_in_cleavage_stage = [
            i for i in stages.iterate_over_images()
            if i.annotation in cleavage_stages]
        for name, _ in cleavage_names_boxes:
            info = FP.get_imageinfo_from_filename(name)
            self.assertIn(stages[info].annotation, cleavage_stages)
        self.assertEqual(
            len(images_in_cleavage_stage),
            len(cleavage_names_boxes))


class TestZonaCompression(unittest.TestCase):
    def test_compress_zona_transforms_to_bytes_io(self):
        np.random.seed(1659)
        raw = make_zona_dataset(INFOS)
        compressed = compress_zona(raw)
        for entry in compressed.iterate_over_images():
            self.assertIsInstance(entry.annotation, io.BytesIO)

    def test_compress_decompress_zona_is_identity(self):
        np.random.seed(1659)
        dataset_raw = make_zona_dataset(INFOS)
        dataset_compressed = compress_zona(dataset_raw)
        dataset_decompressed = decompress_zona(dataset_compressed)

        for info in INFOS:
            raw = dataset_raw[info].annotation
            decompressed = dataset_decompressed[info].annotation
            self.assertTrue(np.all(raw == decompressed))


def make_zona_dataset(infos):
    images = [
        AnnotatedImage(i, np.random.randint(low=0, high=4, size=(500, 500)))
        for i in infos]
    return AnnotatedDataset.create_from_list(images)


def make_fake_smoothed_stages():
    base_info = INFOS[0]
    out = list()
    image_number = 0
    for stage in range(13):
        for _ in range(10):
            image_number += 1
            info = ImageInfo(*base_info[:3], image_number)
            out.append(AnnotatedImage(info, stage))
    return AnnotatedDataset.create_from_list(out)


if __name__ == '__main__':
    unittest.main()
