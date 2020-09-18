import os

import numpy as np
import torch

from embryovision.util import load_and_crop_image, augment_focus
from embryovision.localfolders import embryovision_folder
from embryovision.predictor import Predictor, load_classifier
from embryovision.managedata import FilenameParser


class StagePredictor(Predictor):
    loadname = os.path.join(
        embryovision_folder, 'stage', 'stageclassifier.pkl')
    input_shape = (224, 224)
    load_network = staticmethod(load_classifier)

    def _predict(self, filenames_and_boxes):
        input_x = self._read_images_for_stage(filenames_and_boxes)
        stage_probs = self.network.predict_proba(input_x)
        infos = [
            FilenameParser.get_imageinfo_from_filename(nm)
            for nm, box in filenames_and_boxes]
        return self.pack_into_annotated_images(infos, stage_probs)

    def _read_images_for_stage(self, filenames_and_boxes):
        images_numpy = np.array([
            self._load_inputs_at_one_timepoint(f0_filename, box)
            for f0_filename, box in filenames_and_boxes])
        images_torch = torch.from_numpy(images_numpy.astype('float32'))
        images_torch = images_torch.to(self.device)
        return images_torch

    def _load_inputs_at_one_timepoint(self, f0_filename, box):
        bottom, middle, top = augment_focus(f0_filename)
        images = [
            np.array(
                load_and_crop_image(
                    name,
                    box,
                    output_shape=self.input_shape)
                )
            for name in [middle, top, bottom]]
        images = np.array(images).astype('float32') / 255.0
        images = (images - 0.5) / 0.25
        return images

