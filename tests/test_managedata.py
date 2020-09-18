import os
import sys
import shutil
import tempfile
import unittest
from collections import OrderedDict

from embryovision.managedata import (
    FilenameParser, ImageInfo, AnnotatedImage, AnnotatedImageCollection,
    AnnotatedFocusFolder, AnnotatedWellFolder, AnnotatedSlideFolder,
    AnnotatedDataset)
from embryovision.tests.common import (
    INFO_F0, INFO_F0_2, INFO_F0_3, INFO_FM, INFO_FM_2, INFO_FP, INFOS)


EMBRYO_FOLDER = os.path.join('some', 'complicated', 'directory', 'structure')


class TestFilenameParser(unittest.TestCase):
    def test_get_imageinfo_from_filename(self):
        true_info = INFO_F0
        parser = FilenameParser(EMBRYO_FOLDER)
        filename = parser.get_filename_from_imageinfo(true_info)
        parsed_info = parser.get_imageinfo_from_filename(filename)
        self.assertEqual(parsed_info, true_info)

    def test_get_partial_filename_from_imageinfo(self):
        partial_name = FilenameParser.get_partial_filename_from_imageinfo(
            INFO_F0)
        correct = os.path.join(
            INFO_F0.slide,
            'WELL' + str(INFO_F0.well).rjust(2, '0'),
            'F{}'.format(INFO_F0.focus),
            str(INFO_F0.image_number).rjust(3, '0') + '.jpg')
        self.assertEqual(correct, partial_name)

    def test_get_filename_from_imageinfo(self):
        parser = FilenameParser(EMBRYO_FOLDER)
        for info in INFOS:
            partial = parser.get_partial_filename_from_imageinfo(info)
            filename = parser.get_filename_from_imageinfo(info)
            truth = os.path.join(EMBRYO_FOLDER, partial)
            self.assertEqual(filename, truth)

    def test_get_partial_filename_from_imageinfo_when_values_are_float(self):
        parser = FilenameParser
        for info in INFOS:
            slide, well, focus, image_number = info
            floatinfo = ImageInfo(
                slide, float(well), float(focus), float(image_number))
            fromfloat = parser.get_partial_filename_from_imageinfo(floatinfo)
            fromint = parser.get_partial_filename_from_imageinfo(info)
            self.assertEqual(fromfloat, fromint)


class TestAnnotatedImage(unittest.TestCase):
    def test_stores_info(self):
        labeled_image = AnnotatedImage(INFO_F0, '1-cell')
        self.assertEqual(labeled_image.info, INFO_F0)

    def test_stores_annotation(self):
        annotation = '1-cell'
        labeled_image = AnnotatedImage(INFO_F0, annotation)
        self.assertEqual(labeled_image.annotation, annotation)

    # For these comparison tests, we use the fact that slide_P < slide_0
    # for the particular definitions of slide_p, slide_0 above
    def test_label_equals_when_true(self):
        labeled_image = AnnotatedImage(INFO_F0, 'first')
        labeled_same = AnnotatedImage(INFO_F0, 'different_annotation')
        self.assertEqual(labeled_image, labeled_same)

    def test_label_equals_when_false(self):
        labeled_image = AnnotatedImage(INFO_F0, '')
        labeled_different = AnnotatedImage(INFO_FM, '')
        self.assertNotEqual(labeled_image, labeled_different)

    def test_label_lt_when_equal(self):
        labeled_image = AnnotatedImage(INFO_F0, '')
        labeled_same = AnnotatedImage(INFO_F0, '')
        self.assertGreaterEqual(labeled_image, labeled_same)

    def test_label_lt_when_lt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertLess(less, greater)

    def test_label_lt_when_gt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertFalse(greater < less)

    def test_label_gt_when_equal(self):
        labeled_image = AnnotatedImage(INFO_F0, '')
        labeled_same = AnnotatedImage(INFO_F0, '')
        self.assertFalse(labeled_image > labeled_same)

    def test_label_gt_when_lt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertFalse(less > greater)

    def test_label_gt_when_gt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertGreater(greater, less)

    def test_label_le_when_equal(self):
        labeled_image = AnnotatedImage(INFO_F0, '')
        labeled_same = AnnotatedImage(INFO_F0, '')
        self.assertLessEqual(labeled_image, labeled_same)

    def test_label_le_when_lt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertLessEqual(less, greater)

    def test_label_le_when_gt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertFalse(greater <= less)

    def test_label_ge_when_equal(self):
        labeled_image = AnnotatedImage(INFO_F0, '')
        labeled_same = AnnotatedImage(INFO_F0, '')
        self.assertGreaterEqual(labeled_image, labeled_same)

    def test_label_ge_when_lt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertFalse(less >= greater)

    def test_label_ge_when_gt(self):
        greater = AnnotatedImage(INFO_F0, '')
        less = AnnotatedImage(INFO_FP, '')
        self.assertGreaterEqual(greater, less)

    def test_copy_returns_different_object(self):
        image = AnnotatedImage(INFO_F0, '')
        copied = image.copy()
        self.assertTrue(image is not copied)

    def test_copy_returns_equal_object(self):
        image = AnnotatedImage(INFO_F0, '')
        copied = image.copy()
        self.assertEqual(image, copied)

    def test_cast_as_list_for_int(self):
        one = AnnotatedImage(INFO_F0, 1)
        one_list = one._cast_as_list(one.annotation)
        self.assertEqual(one_list, [1])

    def test_cast_as_list_for_string(self):
        one = AnnotatedImage(INFO_F0, 'abc')
        one_list = one._cast_as_list(one.annotation)
        self.assertEqual(one_list, ['abc'])

    def test_cast_as_list_for_list(self):
        datum = [1, 2, 3]
        one = AnnotatedImage(INFO_F0, datum)
        one_list = one._cast_as_list(one.annotation)
        self.assertEqual(one_list, datum)

    def test_cast_as_list_for_tuple(self):
        datum = (1, 2, 3)
        datum_list = [d for d in datum]
        one = AnnotatedImage(INFO_F0, datum)
        one_list = one._cast_as_list(one.annotation)
        self.assertEqual(one_list, datum_list)

    def test_collate_annotations_with_when_images_with_scalar_data(self):
        truth = [1, 2]
        one = AnnotatedImage(INFO_F0, truth[0])
        two = AnnotatedImage(INFO_F0, truth[1])

        collated = one.collate_annotations_with(two)
        self.assertEqual(collated.annotation, truth)

    def test_collate_annotations_with_scalar_and_list(self):
        truth = [1, 2, 3]
        one = AnnotatedImage(INFO_F0, truth[0])
        two_and_three = AnnotatedImage(INFO_F0, truth[1:])
        collated = one.collate_annotations_with(two_and_three)
        self.assertEqual(collated.annotation, truth)

    def test_collate_annotations_with_list_and_scalar(self):
        truth = [1, 2, 3]
        one_and_two = AnnotatedImage(INFO_F0, truth[:2])
        three = AnnotatedImage(INFO_F0, truth[2])

        collated = one_and_two.collate_annotations_with(three)
        self.assertEqual(collated.annotation, truth)

    def test_collate_annotations_with_list_and_tuple(self):
        truth = [1, 2, 3, 4]
        one_and_two = AnnotatedImage(INFO_F0, truth[:2])
        three_and_four = AnnotatedImage(INFO_F0, tuple(truth[2:]))
        collated = one_and_two.collate_annotations_with(three_and_four)
        self.assertEqual(collated.annotation, truth)

    def test_collate_annotations_with_raises_error_different_images(self):
        one = AnnotatedImage(INFO_F0, 1)
        different = AnnotatedImage(INFO_FP, 2)
        self.assertRaises(ValueError, one.collate_annotations_with, different)

    def test_create_from_filename_and_data(self):
        filename = (
            '/media/brian/Data/Dalit-Embryo-Images/training-data-folders/' +
            'D2016_06_03_S1113_I313_pdb/WELL06/F0/168.jpg')
        data = [1, 2, 4]
        image = AnnotatedImage.create_from_filename_and_data(
            filename, data)

        true_info = FilenameParser.get_imageinfo_from_filename(filename)
        self.assertEqual(image.info, true_info)
        self.assertEqual(image.annotation, data)

    def test_repr(self):
        labeled_image = AnnotatedImage(INFO_F0, '1-cell')
        true_repr = 'AnnotatedImage({}, {})'.format(
            labeled_image.info, labeled_image.annotation)
        self.assertEqual(str(labeled_image), true_repr)

    def test_pack_for_json_when_annotation_is_simple(self):
        annotation = 3
        image = AnnotatedImage(INFO_F0, annotation)
        packed = image.pack_for_json()
        self.assertEqual(packed, annotation)

    def test_pack_for_json_when_annotation_has_pack_for_json_method(self):
        data = 3
        annotation = AnnotatedImage(INFO_F0, data)
        image = AnnotatedImage(INFO_F0, annotation)
        packed = image.pack_for_json()
        self.assertEqual(packed, data)


class TestAnnotatedImageCollection(unittest.TestCase):
    def test_create_from_dict_not_implemented(self):
        self.assertRaises(
            NotImplementedError,
            AnnotatedImageCollection.create_from_dict,
            {})

    def test_prettify_not_implemented(self):
        self.assertRaises(
            NotImplementedError,
            AnnotatedImageCollection._prettify,
            2)

    def test_de_prettify_not_implemented(self):
        self.assertRaises(
            NotImplementedError,
            AnnotatedImageCollection._de_prettify,
            "WELL02")

    def test_add_entry_from_raises_exception(self):
        image_collection = AnnotatedImageCollection()
        labeled_image = AnnotatedImage(INFO_F0, 'first')
        self.assertRaises(
            Exception,
            image_collection._add_entry_from,
            labeled_image)

    def test_lower_class_is_none(self):
        self.assertIs(AnnotatedImageCollection._lower_class, None)


class TestAnnotatedFocusFolder(unittest.TestCase):
    def test_init_does_not_crash(self):
        focus = AnnotatedFocusFolder()
        self.assertTrue(focus is not None)

    def test_len_is_zero_on_init(self):
        focus = AnnotatedFocusFolder()
        self.assertEqual(len(focus), 0)

    def test_update_folder_info_from(self):
        focus = AnnotatedFocusFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus._update_info_from(labeled_f0.info)
        is_ok = [getattr(focus, a) == getattr(labeled_f0.info, a)
                 for a in ['slide', 'well', 'focus']]
        self.assertTrue(all(is_ok))

    def test_update_with_when_empty(self):
        focus = AnnotatedFocusFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus.update_with(labeled_f0)

        self.assertEqual(len(focus), 1)

        info_ok = [getattr(focus, a) == getattr(labeled_f0.info, a)
                   for a in ['slide', 'well', 'focus']]
        self.assertTrue(all(info_ok))

    def test_does_not_update_with_if_different_focus(self):
        focus = AnnotatedFocusFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus.update_with(labeled_f0)
        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        self.assertRaises(ValueError, focus.update_with, labeled_fp)

    def test_does_update_with_if_same_focus(self):
        focus = AnnotatedFocusFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus.update_with(labeled_f0)
        labeled_f0_2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus.update_with(labeled_f0_2)
        self.assertEqual(len(focus), 2)

    def test_getitem(self):
        focus = AnnotatedFocusFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus.update_with(labeled_f0)

        gotten = focus[labeled_f0.info.image_number]
        self.assertIs(gotten, labeled_f0)

    def test_equals_when_equal(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus_v0 = AnnotatedFocusFolder()
        focus_v0.update_with(labeled_f0)

        focus_v1 = AnnotatedFocusFolder()
        focus_v1.update_with(labeled_f0)

        self.assertEqual(focus_v0, focus_v1)

    def test_equals_when_not_equal(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus_v0 = AnnotatedFocusFolder()
        focus_v0.update_with(labeled_f0)

        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        focus_v1 = AnnotatedFocusFolder()
        focus_v1.update_with(labeled_fp)

        self.assertFalse(focus_v0 == focus_v1)

    def test_nequals_when_equal(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus_v0 = AnnotatedFocusFolder()
        focus_v0.update_with(labeled_f0)

        focus_v1 = AnnotatedFocusFolder()
        focus_v1.update_with(labeled_f0)

        self.assertFalse(focus_v0 != focus_v1)

    def test_nequals_when_not_equal(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus_v0 = AnnotatedFocusFolder()
        focus_v0.update_with(labeled_f0)

        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        focus_v1 = AnnotatedFocusFolder()
        focus_v1.update_with(labeled_fp)

        self.assertNotEqual(focus_v0, focus_v1)

    def test_create_from_gives_correct_type(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus = AnnotatedFocusFolder.create_from(labeled_f0)
        self.assertEqual(type(focus), AnnotatedFocusFolder)

    def test_create_from_gives_correct_value(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        focus_v0 = AnnotatedFocusFolder()
        focus_v0.update_with(labeled_f0)

        focus_v1 = AnnotatedFocusFolder.create_from(labeled_f0)
        self.assertEqual(focus_v0, focus_v1)

    def test_remove_when_image_is_in_focusfolder(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus = AnnotatedFocusFolder.create_from(labeled1)
        focus.update_with(labeled2)

        focus.remove(labeled1)
        self.assertTrue(labeled1 not in focus)
        self.assertGreater(len(focus), 0)

    def test_remove_when_image_is_not_in_focusfolder(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus = AnnotatedFocusFolder.create_from(labeled1)
        self.assertRaises(ValueError, focus.remove, labeled2)

    def test_iterate_over_images(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus = AnnotatedFocusFolder.create_from(labeled1)
        focus.update_with(labeled2)

        images = [i for i in focus.iterate_over_images()]
        self.assertEqual(len(images), 2)
        self.assertIn(labeled1, images)
        self.assertIn(labeled2, images)

    def test_is_image_in_focus_when_in_different_folder(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        info1 = labeled1.info
        info2 = INFO_FP
        focus = AnnotatedFocusFolder.create_from(labeled1)
        self.assertFalse(focus.contains_image(info2))

    def test_is_image_in_focus_when_in_same_folder_but_not_present(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        info1 = labeled1.info
        info2 = INFO_F0_2
        focus = AnnotatedFocusFolder.create_from(labeled1)
        self.assertFalse(focus.contains_image(info2))

    def test_is_image_in_focus_when_does_contain(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus = AnnotatedFocusFolder.create_from(labeled1)
        focus.update_with(labeled2)

        self.assertTrue(focus.contains_image(labeled1.info))
        self.assertTrue(focus.contains_image(labeled2.info))

    def test_prettify(self):
        image_number = 1
        correct = '001.jpg'
        output = AnnotatedFocusFolder._prettify(image_number)
        self.assertEqual(output, correct)

    def test_deprettify(self):
        image_number = 1
        pretty = AnnotatedFocusFolder._prettify(image_number)
        back = AnnotatedFocusFolder._de_prettify(pretty)
        self.assertEqual(back, image_number)


class TestAnnotatedWellFolder(unittest.TestCase):
    def test_init_does_not_crash(self):
        well = AnnotatedWellFolder()
        self.assertTrue(well is not None)

    def test_len_is_zero_on_init(self):
        well = AnnotatedWellFolder()
        self.assertEqual(len(well), 0)

    def test_update_folder_info_from(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well._update_info_from(labeled_f0.info)
        is_ok = [getattr(well, a) == getattr(labeled_f0.info, a)
                 for a in well._check_attrs]
        self.assertTrue(all(is_ok))

    def test_update_with_when_empty(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(labeled_f0)

        self.assertEqual(len(well), 1)

        info_ok = [getattr(well, a) == getattr(labeled_f0.info, a)
                   for a in well._check_attrs]
        self.assertTrue(all(info_ok))
        type_ok = type(well[labeled_f0.info.focus]) == AnnotatedFocusFolder
        self.assertTrue(type_ok)

    def test_does_not_update_with_if_different_well(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(labeled_f0)
        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        self.assertRaises(ValueError, well.update_with, labeled_fp)

    def test_does_update_with_if_same_focus(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(labeled_f0)
        labeled_f0_2 = AnnotatedImage(INFO_F0_2, '1-cell')
        well.update_with(labeled_f0_2)
        # Then the length should be 1, since it should update the same
        # dictionary!
        self.assertEqual(len(well), 1)

    def test_does_update_with_if_same_well(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(labeled_f0)
        labeled_fm_2 = AnnotatedImage(INFO_FM_2, '1-cell')
        well.update_with(labeled_fm_2)
        self.assertEqual(len(well), 2)

    def test_getitem(self):
        well = AnnotatedWellFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(labeled_f0)

        gotten = well[labeled_f0.info.focus]
        self.assertIs(type(gotten), AnnotatedFocusFolder)

    def test_remove_recursively_when_image_is_in_wellfolder(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        well = AnnotatedWellFolder.create_from(labeled1)
        well.update_with(labeled2)

        well.remove(labeled1, recursive=True)
        # There should still be 1 focus:
        self.assertEqual(len(well), 1)
        # But the image should not be in the focus folder:
        focus = [i for i in well][0]
        self.assertTrue(labeled1 not in focus)

    def test_remove_recursively_when_image_is_not_in_wellfolder(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        well = AnnotatedWellFolder.create_from(labeled1)
        well.update_with(labeled2)

        labeled_notin = AnnotatedImage(INFO_FP, '1-cell')
        self.assertRaises(ValueError, well.remove, labeled_notin)

    def test_iterate_over_images(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        well = AnnotatedWellFolder.create_from(labeled1)
        well.update_with(labeled2)

        images = [i for i in well.iterate_over_images()]
        self.assertEqual(len(images), 2)
        self.assertIn(labeled1, images)
        self.assertIn(labeled2, images)

    def test_prettify(self):
        focus_value = 15
        correct = 'F15'
        output = AnnotatedWellFolder._prettify(focus_value)
        self.assertEqual(output, correct)

    def test_deprettify(self):
        focus_value = 15
        pretty = AnnotatedWellFolder._prettify(focus_value)
        back = AnnotatedWellFolder._de_prettify(pretty)
        self.assertEqual(back, focus_value)

    def test_lower_class_is_focus(self):
        self.assertIs(AnnotatedWellFolder._lower_class, AnnotatedFocusFolder)

    def test_add_entry_from_includes_in_dict(self):
        well = AnnotatedWellFolder()
        image1 = AnnotatedImage(INFO_F0, '1-cell')
        well.update_with(image1)
        image2 = AnnotatedImage(INFO_F0_2, '1-cell')
        well._add_entry_from(image2)
        self.assertTrue(well.contains_image(image2.info))


class TestAnnotatedSlideFolder(unittest.TestCase):
    def test_init_does_not_crash(self):
        slide = AnnotatedSlideFolder()
        self.assertTrue(slide is not None)

    def test_len_is_zero_on_init(self):
        slide = AnnotatedSlideFolder()
        self.assertEqual(len(slide), 0)

    def test_update_with_when_empty(self):
        slide = AnnotatedSlideFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        slide.update_with(labeled_f0)

        self.assertEqual(len(slide), 1)

        info_ok = [getattr(slide, a) == getattr(labeled_f0.info, a)
                   for a in slide._check_attrs]
        self.assertTrue(all(info_ok))
        type_ok = type(slide[labeled_f0.info.well]) == AnnotatedWellFolder
        self.assertTrue(type_ok)

    def test_does_not_update_with_if_different_slide(self):
        slide = AnnotatedSlideFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        slide.update_with(labeled_f0)
        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        self.assertRaises(ValueError, slide.update_with, labeled_fp)

    def test_does_update_with_if_same_focus(self):
        slide = AnnotatedSlideFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        slide.update_with(labeled_f0)
        labeled_f0_2 = AnnotatedImage(INFO_F0_2, '1-cell')
        slide.update_with(labeled_f0_2)
        # Then the length should be 1, since it should update the same
        # dictionary!
        self.assertEqual(len(slide), 1)

    def test_getitem(self):
        slide = AnnotatedSlideFolder()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        slide.update_with(labeled_f0)

        gotten = slide[labeled_f0.info.well]
        self.assertIs(type(gotten), AnnotatedWellFolder)

    def test_prettify(self):
        well_number = 2
        correct = 'WELL02'
        output = AnnotatedSlideFolder._prettify(well_number)
        self.assertEqual(output, correct)

    def test_deprettify(self):
        well_number = 2
        pretty = AnnotatedSlideFolder._prettify(well_number)
        back = AnnotatedSlideFolder._de_prettify(pretty)
        self.assertEqual(back, well_number)

    def test_lower_class_is_well(self):
        self.assertIs(AnnotatedSlideFolder._lower_class, AnnotatedWellFolder)

    def test_add_entry_from_includes_in_dict(self):
        slide = AnnotatedSlideFolder()
        image1 = AnnotatedImage(INFO_F0, '1-cell')
        slide.update_with(image1)
        image2 = AnnotatedImage(INFO_F0_2, '1-cell')
        slide._add_entry_from(image2)
        self.assertTrue(slide.contains_image(image2.info))


class TestAnnotatedDataset(unittest.TestCase):
    def test_init_does_not_crash(self):
        slide = AnnotatedDataset()
        self.assertTrue(slide is not None)

    def test_len_is_zero_on_init(self):
        slide = AnnotatedDataset()
        self.assertEqual(len(slide), 0)

    def test_update_with_when_empty(self):
        dataset = AnnotatedDataset()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset.update_with(labeled_f0)

        self.assertEqual(len(dataset), 1)
        type_ok = type(dataset[labeled_f0.info.slide]) == AnnotatedSlideFolder
        self.assertTrue(type_ok)

    def test_update_with_if_different_slide(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        dataset = AnnotatedDataset()
        dataset.update_with(labeled_f0)
        dataset.update_with(labeled_fp)

        self.assertEqual(len(dataset), 2)

    def test_does_update_with_if_same_focus(self):
        dataset = AnnotatedDataset()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset.update_with(labeled_f0)
        labeled_f0_2 = AnnotatedImage(INFO_F0_2, '1-cell')
        dataset.update_with(labeled_f0_2)
        # Then the length should be 1, since it should update the same
        # dictionary!
        self.assertEqual(len(dataset), 1)

    def test_getitem(self):
        dataset = AnnotatedDataset()
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset.update_with(labeled_f0)

        gotten = dataset[labeled_f0.info.slide]
        self.assertIs(type(gotten), AnnotatedSlideFolder)

    def test_create_from(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(labeled_f0)
        self.assertEqual(len(dataset), 1)

    def test_create_from_list(self):
        images = [
            AnnotatedImage(info, '{}-cell'.format(index))
            for index, info in enumerate(INFOS)]

        dataset = AnnotatedDataset.create_from_list(images)
        for image in dataset.iterate_over_images():
            self.assertIn(image, images)

    def test_copy_returns_separate_object(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        copied = dataset.copy()
        self.assertTrue(dataset is not copied)

    def test_copy_returns_equal_dataset(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        copied = dataset.copy()
        self.assertEqual(dataset, copied)

    def test_copy_is_deep(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        copied = dataset.copy()

        key = dataset.keys()[0]
        self.assertTrue(dataset[key] is not copied[key])

    def test_keys(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        keys = dataset.keys()
        truth = list(dataset._dict.keys())
        self.assertEqual(keys, truth)
        self.assertEqual(type(keys), list)

    def test_values(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        values = dataset.values()
        truth = list(dataset._dict.values())
        self.assertEqual(values, truth)
        self.assertEqual(type(values), list)

    def test_items(self):
        image = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(image)
        items = dataset.items()
        truth = dataset._dict.items()
        self.assertEqual(items, truth)

    def test_iter(self):
        image_f0 = AnnotatedImage(INFO_F0, '1-cell')
        image_fm = AnnotatedImage(INFO_FM, '1-cell')
        image_fp = AnnotatedImage(INFO_FP, '1-cell')
        dataset = AnnotatedDataset.create_from(image_f0)
        dataset.update_with(image_fm)
        dataset.update_with(image_fp)
        entries_manual = [v for v in dataset._dict.values()]
        entries_iter = [v for v in dataset]
        ok = [m == i for m, i in zip(entries_manual, entries_iter)]
        self.assertTrue(all(ok))

    def test_create_from_dict(self):
        truth = AnnotatedDataset()
        image = AnnotatedImage(INFO_F0, [1, 2, 3])
        truth.update_with(image)

        the_dict = truth.pack_for_json(pretty=True)
        new_dataset = AnnotatedDataset.create_from_dict(the_dict)
        self.assertEqual(new_dataset, truth)

    def test_iterate_over_images(self):
        labeled_images = [AnnotatedImage(info, 'annotation') for info in INFOS]
        dataset = AnnotatedDataset.create_from_list(labeled_images)

        iterated_images = [i for i in dataset.iterate_over_images()]
        each_in = [l in iterated_images for l in labeled_images]
        self.assertEqual(len(iterated_images), len(labeled_images))
        self.assertTrue(all(each_in))

    def test_is_image_in_dataset_when_in_same_folder_but_not_present(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        info1 = labeled1.info
        info2 = INFO_F0_2
        dataset = AnnotatedDataset.create_from(labeled1)
        self.assertFalse(dataset.contains_image(info2))

    def test_is_image_in_focus_when_does_contain(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        dataset = AnnotatedDataset.create_from(labeled1)
        dataset.update_with(labeled2)

        self.assertTrue(dataset.contains_image(labeled1.info))
        self.assertTrue(dataset.contains_image(labeled2.info))

    def test_sort(self):
        dataset = make_simple_dataset()
        initial_list = dataset.iterate_over_images()
        assert sorted(initial_list) != initial_list
        sorted_dataset = dataset.sort()
        sorted_list = sorted_dataset.iterate_over_images()
        self.assertEqual(sorted(sorted_list), sorted_list)

    def test_index_with_info(self):
        dataset = make_simple_dataset()
        entry = dataset.iterate_over_images()[0]

        indexed = dataset[entry.info]
        self.assertEqual(indexed.info, entry.info)
        self.assertEqual(indexed.annotation, entry.annotation)
        # self.assertTrue(indexed is entry)

    def test_prettify(self):
        slide_name = 'name'
        correct = slide_name  # should not alter names
        output = AnnotatedDataset._prettify(slide_name)
        self.assertEqual(output, correct)

    def test_deprettify(self):
        slide_name = 'name'
        pretty = AnnotatedDataset._prettify(slide_name)
        back = AnnotatedDataset._de_prettify(pretty)
        self.assertEqual(back, slide_name)

    def test_lower_class_is_slide(self):
        self.assertIs(AnnotatedDataset._lower_class, AnnotatedSlideFolder)

    def test_add_entry_from_includes_in_dict(self):
        dataset = AnnotatedDataset()
        image1 = AnnotatedImage(INFO_F0, '1-cell')
        dataset.update_with(image1)
        image2 = AnnotatedImage(INFO_F0_2, '1-cell')
        dataset._add_entry_from(image2)
        self.assertTrue(dataset.contains_image(image2.info))


class TestFileIO(unittest.TestCase):
    def setUp(self):
        self._test_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._test_directory)

    def test_save_as_pickle(self):
        dataset = make_simple_dataset()
        filename = make_safe_filename_in(self._test_directory)
        dataset.save_as_pickle(filename)

        loaded = AnnotatedDataset.load_from_pickle(filename)
        # They should be different objects that are equal:
        self.assertFalse(loaded is dataset)
        self.assertEqual(loaded, dataset)
        # And we need to clean the file:
        os.remove(filename)

    def test_save_as_json(self):
        dataset = make_simple_dataset()
        filename = make_safe_filename_in(self._test_directory)
        dataset.save_as_json(filename)

        loaded = AnnotatedDataset.load_from_json(filename)
        # They should be different objects that are equal:
        self.assertFalse(loaded is dataset)
        self.assertEqual(loaded, dataset)
        # And we need to clean the file:
        os.remove(filename)

    def test_save_as_json_when_file_already_exists(self):
        dataset = make_simple_dataset()
        filename = make_safe_filename_in(self._test_directory)
        dataset.save_as_json(filename)

        # Now if we save it again w/o overwite it should raise an error:
        self.assertRaises(FileExistsError, dataset.save_as_json, filename)
        # And if we use `overwrite` it should not:
        _ = dataset.save_as_json(filename, overwrite=True)
        # And we need to clean the file:
        os.remove(filename)

    def test_save_as_pickle_when_file_already_exists(self):
        dataset = make_simple_dataset()
        filename = make_safe_filename_in(self._test_directory)
        dataset.save_as_pickle(filename)

        # Now if we save it again w/o overwite it should raise an error:
        self.assertRaises(FileExistsError, dataset.save_as_pickle, filename)
        # And if we use `overwrite` it should not:
        _ = dataset.save_as_pickle(filename, overwrite=True)
        # And we need to clean the file:
        os.remove(filename)


class TestFolderComparison(unittest.TestCase):
    def test_focus_for_same_image(self):
        labeled = AnnotatedImage(INFO_F0, '1-cell')
        data = AnnotatedImage(labeled.info, [1, 2, 3])
        focus_labeled = AnnotatedFocusFolder.create_from(labeled)
        focus_data = AnnotatedFocusFolder.create_from(data)
        self.assertTrue(focus_labeled.is_same_folder_as(focus_data))

    def test_focus_for_same_focus(self):
        labeled1 = AnnotatedImage(INFO_F0, '1-cell')
        labeled2 = AnnotatedImage(INFO_F0_2, '1-cell')
        focus_1 = AnnotatedFocusFolder.create_from(labeled1)
        focus_2 = AnnotatedFocusFolder.create_from(labeled2)
        self.assertTrue(focus_1.is_same_folder_as(focus_2))

    def test_focus_for_different_focus(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        labeled_fp = AnnotatedImage(INFO_FP, '1-cell')
        focus_1 = AnnotatedFocusFolder.create_from(labeled_f0)
        focus_2 = AnnotatedFocusFolder.create_from(labeled_fp)
        self.assertFalse(focus_1.is_same_folder_as(focus_2))

    # Then I should test the wells etc but later.


class TestPackForJson(unittest.TestCase):
    def test_labeledimage_pack_for_json(self):
        annotation = '1-cell'
        labeled = AnnotatedImage(INFO_F0, annotation)
        packed = labeled.pack_for_json()
        # truth = {INFO_F0.image_number: '1-cell'}
        self.assertEqual(packed, annotation)

    def test_labeledimage_pack_for_json_pretty(self):
        labeled = AnnotatedImage(INFO_F0, '1-cell')
        packed_true = labeled.pack_for_json(pretty=True)
        packed_false = labeled.pack_for_json(pretty=False)
        # truth = {INFO_F0.image_number: '1-cell'}
        truth = '1-cell'
        self.assertEqual(packed_true, truth)
        self.assertEqual(packed_false, truth)

    def test_labeleddataset_pack_for_json_not_pretty(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(labeled_f0)
        for_json = dataset.pack_for_json(pretty=False)
        truth = OrderedDict(
            {labeled_f0.info.slide: OrderedDict(
                {labeled_f0.info.well: OrderedDict(
                    {labeled_f0.info.focus: OrderedDict(
                        {labeled_f0.info.image_number: labeled_f0.annotation})
                    })
                })
            })
        self.assertEqual(for_json, truth)

    def test_labeleddataset_pack_for_json_pretty(self):
        labeled_f0 = AnnotatedImage(INFO_F0, '1-cell')
        dataset = AnnotatedDataset.create_from(labeled_f0)
        for_json = dataset.pack_for_json(pretty=True)

        slide_key = labeled_f0.info.slide
        well_key = 'WELL' + str(labeled_f0.info.well).rjust(2, '0')
        focus_key = 'F' + str(labeled_f0.info.focus)
        image_key = str(labeled_f0.info.image_number).rjust(3, '0') + '.jpg'
        truth = OrderedDict(
            {slide_key: OrderedDict(
                {well_key: OrderedDict(
                    {focus_key: OrderedDict(
                        {image_key: labeled_f0.annotation})
                    })
                })
            })
        self.assertEqual(for_json, truth)


class TestCollateImageCollection(unittest.TestCase):
    def test_collate_focuses_on_same_image_returns_correct_type(self):
        label = '1-cell'
        datum = [1, 2, 3]
        image_labeled = AnnotatedImage(INFO_F0, label)
        image_datum = AnnotatedImage(INFO_F0, datum)

        focus_labeled = AnnotatedFocusFolder.create_from(image_labeled)
        focus_datum = AnnotatedFocusFolder.create_from(image_datum)

        collated = focus_labeled.collate_annotations_with(focus_datum)
        self.assertIsInstance(collated, AnnotatedFocusFolder)

    def test_collate_focuses_on_same_image_returns_correct_value(self):
        label = '1-cell'
        datum = [1, 2, 3]
        image_labeled = AnnotatedImage(INFO_F0, label)
        image_datum = AnnotatedImage(INFO_F0, datum)

        focus_labeled = AnnotatedFocusFolder.create_from(image_labeled)
        focus_datum = AnnotatedFocusFolder.create_from(image_datum)

        collated_focus = focus_labeled.collate_annotations_with(focus_datum)
        collated_value = collated_focus[collated_focus.keys()[0]].annotation
        truth = [label] + datum

        self.assertEqual(collated_value, truth)

    def test_collate_focuses_on_different_image_returns_correct_type(self):
        label = '1-cell'
        datum = [1, 2, 3]
        image_labeled = AnnotatedImage(INFO_F0, label)
        different_datum = AnnotatedImage(INFO_F0_2, datum)

        focus_labeled = AnnotatedFocusFolder.create_from(image_labeled)
        focus_datum = AnnotatedFocusFolder.create_from(different_datum)

        collated = focus_labeled.collate_annotations_with(focus_datum)
        self.assertIsInstance(collated, AnnotatedFocusFolder)

    def test_collate_focuses_on_different_image_is_length_0(self):
        label = '1-cell'
        datum = [1, 2, 3]
        image_labeled = AnnotatedImage(INFO_F0, label)
        different_datum = AnnotatedImage(INFO_F0_2, datum)

        focus_labeled = AnnotatedFocusFolder.create_from(image_labeled)
        focus_datum = AnnotatedFocusFolder.create_from(different_datum)

        collated = focus_labeled.collate_annotations_with(focus_datum)
        # Since these two are different images, neither is present in both,
        # so the length should be 0:
        self.assertEqual(len(collated), 0)

    def test_collate_focuses_raises_error_when_different_folders(self):
        label = '1-cell'
        datum = [1, 2, 3]
        image_labeled = AnnotatedImage(INFO_F0, label)
        different_datum = AnnotatedImage(INFO_FP, datum)

        focus_0 = AnnotatedFocusFolder.create_from(image_labeled)
        focus_plus = AnnotatedFocusFolder.create_from(different_datum)

        self.assertRaises(
            ValueError,
            focus_0.collate_annotations_with,
            focus_plus)

    def test_collate_focuses_on_overlapping_datasets_is_correct_length(self):
        # Collate images [0, 1] with [0, 1, 2]
        label = '1-cell'
        datum = [1, 2, 3]
        infos = [INFO_F0, INFO_F0_2, INFO_F0_3]
        labels = [AnnotatedImage(i, label) for i in infos]
        data = [AnnotatedImage(i, datum) for i in infos[:-1]]

        focus_labeled = AnnotatedFocusFolder.create_from(labels[0])
        for im in labels[1:]:
            focus_labeled.update_with(im)
        focus_datum = AnnotatedFocusFolder.create_from(data[0])
        for im in data[1:]:
            focus_datum.update_with(im)

        collated = focus_labeled.collate_annotations_with(focus_datum)
        self.assertEqual(len(collated), len(data))

    def test_collate_wells(self):
        label = AnnotatedImage(INFO_F0, '1-cell')
        data = AnnotatedImage(label.info, [0, 1, 2])

        well_labeled = AnnotatedWellFolder.create_from(label)
        well_data = AnnotatedWellFolder.create_from(data)

        collated = well_labeled.collate_annotations_with(well_data)
        well_labeled.update_with(data)
        self.assertEqual(well_labeled, collated)


def make_simple_dataset():
    dataset = AnnotatedDataset()
    for i, info in enumerate(INFOS):
        data = list(range(3*i, 3*i + 3, 3))
        image = AnnotatedImage(info, data)
        dataset.update_with(image)
    return dataset


def make_safe_filename_in(directory, extension='pkl'):
    savefilename = os.path.join(directory, 'tmp.{}'.format(extension))
    if os.path.exists(savefilename):
        raise FileExistsError('{} already exists!'.format(savefilename))
    return savefilename


if __name__ == '__main__':
    unittest.main()

