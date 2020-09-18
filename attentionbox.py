import sys
import time
import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from embryovision import managedata


def find_bounding_box(labels, box_side=328):
    calc = BoundingBoxCalculator(box_side=box_side)
    return calc.find_bounding_box(labels)


class BoundingBoxCalculator(object):
    image_shape = (500, 500)
    image_x, image_y = np.meshgrid(
        *[np.arange(i) for i in image_shape], indexing='ij')
    embryo_cutoff = 1000

    inside_well_label = 0
    zona_label = 1
    inside_zona_label = 2
    outside_well_label = 3

    def __init__(self, box_side=328):
        self.box_side = box_side
        self._box_halfside = self.box_side / 2

    def find_bounding_box(self, labels):
        if labels.shape != self.image_shape:
            msg = "`labels` must be of shape {}".format(self.image_shape)
            raise ValueError(msg)
        box = self._find_bounding_box(labels)
        return self._clip_box_to_be_within_image(box)

    def _find_bounding_box(self, labels):
        # 1. Get the bbox centered on the well center
        well = self._mask_well_interior(labels)
        box_well_minimal = (
            self._calculate_minimal_bounding_box_for(well)
            if well.any() else self._centered_empty_box)
        box_well = self._expand_box_to_correct_size(box_well_minimal)

        # 2. Get the minimal box for the embryo:
        embryo = self._mask_embryo(labels)
        box_embryo_minimal = (
            self._calculate_minimal_bounding_box_for(embryo)
            if embryo.sum() > self.embryo_cutoff else box_well)

        # 3. If any of the embryo box edges are outside the well,
        #    shift the well bbox to contain the embryo bbox
        return self._shift_box1_to_enclose_box2(box_well, box_embryo_minimal)

    def _clip_box_to_be_within_image(self, box):
        lower, left = [
            int(np.clip(pos, 0, max_pos - self.box_side))
            for pos, max_pos in zip(box[:2], self.image_shape)]
        return (lower, left, self.box_side, self.box_side)

    @classmethod
    def _calculate_minimal_bounding_box_for(cls, mask):
        x = cls.image_x[mask]
        y = cls.image_y[mask]
        return (x.min(), y.min(), x.ptp() + 1, y.ptp() + 1)

    def _expand_box_to_correct_size(self, box_centered):
        x0, y0, w0, h0 = box_centered
        x_center = x0 + 0.5 * w0
        y_center = y0 + 0.5 * h0
        x1 = x_center - self._box_halfside
        y1 = y_center - self._box_halfside
        return (x1, y1, self.box_side, self.box_side)

    @classmethod
    def _find_mask_center(cls, mask):
        # could try the median instead
        return cls.image_x[mask].mean(), cls.image_y[mask].mean()

    @classmethod
    def _mask_well_interior(cls, labels):
        return labels != cls.outside_well_label

    @classmethod
    def _mask_embryo(cls, labels):
        return (labels == cls.zona_label) | (labels == cls.inside_zona_label)

    @classmethod
    def _shift_box1_to_enclose_box2(cls, box1, box2):
        lower1, left1, height1, width1 = box1
        lower2, left2, height2, width2 = box2

        lower = (
            lower2 + height2 - height1 if lower2 + height2 > lower1 + height1
            else min(lower1, lower2))
        left = (
            left2 + width2 - width1 if left2 + width2 > left1 + width1
            else min(left1, left2))
        bbox = (lower, left, height1, width1)
        return bbox

    @property
    def _centered_empty_box(self):
        return tuple([i // 2 for i in self.image_shape]) + (0, 0)

