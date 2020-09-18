import sys
import unittest

import numpy as np

from embryovision.attentionbox import BoundingBoxCalculator, find_bounding_box


LABEL_SHAPE = (500, 500)


class TestConvenienceFunction(unittest.TestCase):
    def test_does_the_same_as_calculator(self):
        box_zona = (20, 20, 33, 33)
        box_well = (10, 10, 70, 70)
        box_cell = (25, 25, 15, 15)
        labels = create_embryo_like_labels(box_zona, box_well, box_cell)

        calc = BoundingBoxCalculator()
        out_from_calc = calc.find_bounding_box(labels)
        out_from_convenience = find_bounding_box(labels)

        self.assertEqual(out_from_calc, out_from_convenience)

    def test_honors_box_size(self):
        box_zona = (20, 20, 33, 33)
        box_well = (30, 30, 70, 70)
        box_cell = (25, 25, 15, 15)
        labels = create_embryo_like_labels(box_zona, box_well, box_cell)

        box_side = 200
        out_from_convenience = find_bounding_box(
            labels, box_side=box_side)
        self.assertEqual(out_from_convenience[2:], (box_side, box_side))

    def test_returns_centered_box_on_empty_labels(self):
        empty_labels = np.zeros(LABEL_SHAPE, dtype='int')
        empty_labels[:] = BoundingBoxCalculator.outside_well_label

        box = find_bounding_box(empty_labels)
        image_center = [i // 2 for i in LABEL_SHAPE]
        box_center = [box[0] + box[2] / 2, box[1] + box[3] / 2]
        self.assertEqual(image_center, box_center)

    def test_returns_valid_box_on_random_labels(self):
        np.random.seed(1033)
        random_labels = np.random.randint(
            low=0, high=4, size=LABEL_SHAPE, dtype='int')

        box = find_bounding_box(random_labels)
        for i in range(2):
            self.assertGreaterEqual(box[i], 0)
            self.assertLessEqual(box[i] + box[i + 2], LABEL_SHAPE[i])


class TestBoundingBoxCalculator(unittest.TestCase):
    def test_find_bounding_box_keeps_all_zona_and_within_zona(self):
        calc = BoundingBoxCalculator()

        box_zona = (20, 20, 33, 33)
        box_well = (10, 10, 70, 70)
        box_cell = (25, 25, 15, 15)
        labels = create_embryo_like_labels(box_zona, box_well, box_cell)

        out = calc.find_bounding_box(labels)
        cropped = labels[out[0]: out[0] + out[2], out[1]: out[1] + out[3]]
        for which_label in [calc.zona_label, calc.inside_zona_label]:
            n_pixels_raw = (labels == which_label).sum()
            n_pixels_cropped = (cropped == which_label).sum()
            self.assertEqual(n_pixels_raw.sum(), n_pixels_cropped.sum())

    def test_init_stores_box_side(self):
        side = 100
        calc = BoundingBoxCalculator(side)
        self.assertEqual(calc.box_side, side)

    def test_raises_value_error_when_shape_is_incorrect(self):
        calc = BoundingBoxCalculator()
        correct_shape = calc.image_shape
        wrong_shape = [i + 10 for i in correct_shape]
        image_wrong = np.zeros(wrong_shape)
        self.assertRaises(ValueError, calc.find_bounding_box, image_wrong)

    def test_default_box_side_is_328(self):
        calc = BoundingBoxCalculator()
        self.assertEqual(calc.box_side, 328)

    def test_init_sets_up_box_halfside(self):
        side = 100
        calc = BoundingBoxCalculator(side)
        self.assertEqual(calc._box_halfside, side / 2)

    def test_expand_box_to_correct_size_gives_correct_width_and_height(self):
        side = 31
        calc = BoundingBoxCalculator(side)

        np.random.seed(1130)
        box_initial = np.random.randint(low=0, high=LABEL_SHAPE[0], size=4)
        box_recentered = calc._expand_box_to_correct_size(box_initial)
        self.assertEqual(box_recentered[2:], (side, side))

    def test_find_bounding_box_returns_correct_shape(self):
        side = 300
        calc = BoundingBoxCalculator(side)

        box_zona = (20, 20, 15, 15)
        box_well = (30, 30, 70, 70)
        box_cell = (25, 25, 5, 5)
        labels = create_embryo_like_labels(box_zona, box_well, box_cell)

        out = calc._find_bounding_box(labels)
        self.assertEqual(out[2:], (side, side))

    def test_find_bounding_box_returns_indexable_of_correct_size(self):
        box_side = 300
        calc = BoundingBoxCalculator(box_side)

        box_well = (10, 10, 70, 70)
        box_zona = (20, 20, 33, 33)
        box_cell = (25, 25, 15, 15)
        labels = create_embryo_like_labels(box_zona, box_well, box_cell)

        out = calc.find_bounding_box(labels)
        cropped = labels[out[0]: out[0] + out[2], out[1]: out[1] + out[3]]
        self.assertEqual(cropped.shape, (box_side, box_side))

    def test_find_mask_center(self):
        center = (72, 36)
        height = 11
        width = 23
        bbox = (center[0] - height // 2, center[1] - width // 2, height, width)
        mask = create_mask(bbox)
        calc = BoundingBoxCalculator(40)

        predicted = calc._find_mask_center(mask)
        self.assertEqual(center, predicted)

    def test_mask_well_interior(self):
        calc = BoundingBoxCalculator(40)
        np.random.seed(1123)
        labels = np.random.randint(
            low=0, high=4, size=LABEL_SHAPE, dtype='int8')

        inside_well = (labels != calc.outside_well_label)
        predicted = calc._mask_well_interior(labels)
        self.assertTrue(np.all(inside_well == predicted))

    def test_mask_embryo(self):
        calc = BoundingBoxCalculator(40)
        np.random.seed(1123)
        labels = np.random.randint(
            low=0, high=4, size=LABEL_SHAPE, dtype='int8')

        embryo = ((labels == calc.zona_label) |
                  (labels == calc.inside_zona_label))
        predicted = calc._mask_embryo(labels)
        self.assertTrue(np.all(embryo == predicted))

    def test_calculate_minimal_bounding_box_for(self):
        bbox = (65, 22, 11, 41)
        mask = create_mask(bbox)
        calc = BoundingBoxCalculator(40)
        minimal_bbox = calc._calculate_minimal_bounding_box_for(mask)
        self.assertEqual(bbox, minimal_bbox)

    def test_shift_box1_to_enclose_box2_when_already_enclosed(self):
        center = (62, 62)
        side1 = 20
        side2 = 10
        box1 = tuple([c - side1 // 2 for c in center]) + (side1, side1)
        box2 = tuple([c - side2 // 2 for c in center]) + (side2, side2)

        box_shifted = BoundingBoxCalculator._shift_box1_to_enclose_box2(
            box1, box2)
        self.assertEqual(box_shifted, box1)

    def test_shift_box1_to_enclose_box2_when_lower_left_of_box1(self):
        box1 = (42, 42, 20, 20)
        box2 = (35, 35, 10, 10)

        box_shifted = BoundingBoxCalculator._shift_box1_to_enclose_box2(
            box1, box2)
        box_correct = box2[:2] + box1[2:]
        self.assertEqual(box_shifted, box_correct)

    def test_shift_box1_to_enclose_box2_when_upper_right_of_box1(self):
        box1 = (42, 42, 22, 20)
        box2 = (60, 60, 12, 10)

        box_shifted = BoundingBoxCalculator._shift_box1_to_enclose_box2(
            box1, box2)

        upper2 = box2[0] + box2[2]
        right2 = box2[1] + box2[3]
        height1, width1 = box1[2:]
        box_correct = (upper2 - height1, right2 - width1) + box1[2:]
        self.assertEqual(box_shifted, box_correct)

    def test_clip_box_to_be_within_boxside_when_negative(self):
        calc = BoundingBoxCalculator()
        box_outside = (-10, -10, 100, 100)
        box_clipped = calc._clip_box_to_be_within_image(box_outside)
        for i in range(2):
            self.assertGreaterEqual(box_clipped[i], 0)

    def test_clip_box_to_be_within_boxside_when_too_large(self):
        box_side = 300
        calc = BoundingBoxCalculator(box_side)
        max_size = calc.image_shape
        box_outside = (max_size[0] + 10, max_size[1] + 10, 100, 100)
        box_clipped = calc._clip_box_to_be_within_image(box_outside)
        for i in range(2):
            self.assertLessEqual(box_clipped[i] + box_side, max_size[i])

    def test_centered_empty_box(self):
        calc = BoundingBoxCalculator()
        centered_empty_box = calc._centered_empty_box
        correct_box = tuple([i // 2 for i in calc.image_shape]) + (0, 0)
        self.assertEqual(correct_box, centered_empty_box)


def create_mask(bbox):
    lower, left, height, width = bbox
    upper = lower + height
    right = left + width

    mask = np.zeros(LABEL_SHAPE, dtype='bool')
    mask[lower:upper, left:right] = True
    return mask


def create_embryo_like_labels(box_zona, box_well, box_cell):
    mask_zona = create_mask(box_zona)
    mask_well = create_mask(box_well)
    mask_cell = create_mask(box_cell)

    labels = np.full(
        LABEL_SHAPE,
        BoundingBoxCalculator.outside_well_label, dtype='int')
    labels[mask_well] = BoundingBoxCalculator.inside_well_label
    labels[mask_zona] = BoundingBoxCalculator.zona_label
    labels[mask_cell] = BoundingBoxCalculator.inside_zona_label

    return labels


if __name__ == '__main__':
    unittest.main()

