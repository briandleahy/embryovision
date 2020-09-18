import os

import torch
import numpy as np
import scipy.ndimage as nd

from embryovision.util import (
    load_and_crop_image,
    augment_focus,
    sort_azimuthally)
from embryovision.localfolders import embryovision_folder
from embryovision.predictor import Predictor, load_maskrcnn
from embryovision.managedata import FilenameParser


class MaskRCNNPredictor(Predictor):
    load_network = staticmethod(load_maskrcnn)
    default_blocksize = 1
    input_shape = (500, 500)
    iou_threshold = 0.7

    def _predict(self, filenames_and_boxes):
        boxes = [box for filename, box in filenames_and_boxes]
        filenames = [filename for filename, box in filenames_and_boxes]

        outputs_raw = [
            self._predict_single_timepoint_raw(*name_and_box)
            for name_and_box in filenames_and_boxes]

        output = self._transform_to_final_output(outputs_raw, boxes)
        infos = [
            FilenameParser.get_imageinfo_from_filename(nm)
            for nm in filenames]
        return self.pack_into_annotated_images(infos, output)

    def _transform_to_final_output(self, raw_outputs, boxes):
        out = list()
        for raw_output, box in zip(raw_outputs, boxes):
            this_output = [
                self.transform_detection_result_to_xy_polygon(p, box)
                for p in raw_output]
            out.append(this_output)
        return out

    def read_image(self, filename, box):
        image_pil = load_and_crop_image(
            filename, box, output_shape=self.input_shape)
        image_np = np.array(image_pil).astype('float32')
        image_np /= 255.0
        image_np -= 0.5
        image_np /= 0.25

        image_torch = torch.from_numpy(image_np).repeat(3, 1, 1).to(self.device)
        return image_torch

    def transform_detection_result_to_xy_polygon(self, detection_result, box):
        confidence = detection_result.confidence
        xy_polygon = self._get_xy_polygon_from(detection_result.mask, box)
        return {'confidence': confidence, 'xy_polygon': xy_polygon}

    def _get_xy_polygon_from(self, mask, box):
        max_likelihod_prediction = (mask > 0.5)

        # largest connected component:
        labeled = nd.label(max_likelihod_prediction)[0]
        labels, counts = np.unique(labeled, return_counts=True)
        largest_cluster = counts[1:].argmax() + 1
        connected_mask = (labeled == largest_cluster)

        # xy of boundary in cropped coords
        smaller = nd.binary_erosion(connected_mask)
        y, x = np.nonzero(connected_mask & (~smaller))

        # xy of boundary in real coords:
        rescale = [b / s for b, s in zip(box[2:], mask.shape)]
        x = x * rescale[0]  # cast as float too
        y = y * rescale[1]

        # shift according to original crop-box corner:
        x += box[1]
        y += box[0]
        return np.transpose(sort_azimuthally(x, y))

    def _predict_single_timepoint_raw(self, filename, box):
        focuses = augment_focus(filename)
        focus_outputs = list()
        for focus in focuses:
            input_x = self.read_image(focus, box)
            focus_outputs.extend(self.network.predict([input_x]))
        bounding_boxes = [result.box for result in focus_outputs]
        confidence_scores = [result.confidence for result in focus_outputs]
        indices = non_max_suppression_slow(
            bounding_boxes, confidence_scores, self.iou_threshold)
        return [focus_outputs[i] for i in indices]


class PronucleiPredictor(MaskRCNNPredictor):
    loadname = os.path.join(
        embryovision_folder, 'objectdetector', 'pronucleidetector.pkl')


class BlastomerePredictor(MaskRCNNPredictor):
    loadname = os.path.join(
        embryovision_folder, 'objectdetector', 'blastomeredetector.pkl')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def non_max_suppression_slow(bounding_boxes, confidence_score, iou_threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return []
    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    pick = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        pick.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)

        intersection = w * h
        union = areas[index] + areas[order[:-1]] - intersection
        iou = intersection / union

        left = np.where(iou < iou_threshold)
        order = order[left]

    return pick

