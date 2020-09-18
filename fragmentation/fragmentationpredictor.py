import io
import os

import numpy as np

from embryovision.util import load_and_crop_image, augment_focus
from embryovision.localfolders import embryovision_folder
from embryovision.predictor import Predictor, load_classifier
from embryovision.managedata import FilenameParser
from embryovision.fragmentation.load_images import (
    load_images_for_fragmentation_regressor)


class FragPredictor(Predictor):
    loadname = os.path.join(
        embryovision_folder, 'fragmentation', 'fragmentationclassifier.pkl')
    load_network = staticmethod(load_classifier)
    input_shape = (299, 299)

    def _predict(self, f0_filenames_and_boxes):
        filenames = [nm for nm, box in f0_filenames_and_boxes]
        boxes = [box for nm, box in f0_filenames_and_boxes]

        augmented_names = [augment_focus(nm) for nm in filenames]
        fm_filenames_and_boxes = [
            (names[0], box)
            for names, box in zip(augmented_names, boxes)]
        fp_filenames_and_boxes = [
            (names[2], box)
            for names, box in zip(augmented_names, boxes)]

        frag_f0 = self._predict_for_focus_level(f0_filenames_and_boxes)
        frag_fm = self._predict_for_focus_level(fm_filenames_and_boxes)
        frag_fp = self._predict_for_focus_level(fp_filenames_and_boxes)
        frag_values = np.mean([frag_f0, frag_fm, frag_fp], axis=0)

        infos = [
            FilenameParser.get_imageinfo_from_filename(nm)
            for nm in filenames]
        return self.pack_into_annotated_images(infos, frag_values)

    def _predict_for_focus_level(self, filenames_and_boxes):
        cropped_images = self._crop_images_in_ram(filenames_and_boxes)
        input_x = load_images_for_fragmentation_regressor(cropped_images)
        input_x = input_x.to(self.device)
        frag_values = self.network.predict(input_x)
        return frag_values

    def _crop_images_in_ram(self, filenames_and_boxes):
        cropped_images = []
        for filename, box in filenames_and_boxes:
            cropped = load_and_crop_image(
                filename, box, output_shape=self.input_shape)
            bytes_io = io.BytesIO()
            cropped.save(bytes_io, format='png')
            cropped_images.append(bytes_io)
        return cropped_images

