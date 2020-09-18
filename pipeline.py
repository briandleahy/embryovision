import io

import numpy as np
from PIL import Image

from embryovision.managedata import (
    AnnotatedDataset, AnnotatedImage, FilenameParser as FP)
from embryovision.attentionbox import find_bounding_box
from embryovision.zona.zonapredictor import ZonaPredictor
from embryovision.fragmentation.fragmentationpredictor import FragPredictor
from embryovision.stage.stagepredictor import StagePredictor
from embryovision.stage.stagedp import DynamicProgramming
from embryovision.stage.stage_to_number import stage_to_number
from embryovision.objectdetector.objectpredictor import (
    PronucleiPredictor, BlastomerePredictor)


def predict(filenames, device='cuda', zona_compressed=True):
    pipeline = Pipeline(device=device)
    out = pipeline.predict_all(filenames)
    if zona_compressed:
        out['zona'] = compress_zona(out['zona'])
    return out


class Pipeline(object):
    def __init__(self, device='cuda'):
        self.device = device

    def predict_all(self, filenames):
        zona = self.predict_zona(filenames)
        boxes = self.calculate_boxes(zona)
        filenames_and_boxes = [
            (nm, b.annotation)
            for nm, b in zip(filenames, boxes)]
        frag = self.predict_frag(filenames_and_boxes)
        # FIXME check low frag
        stage_raw = self.predict_stage_raw(filenames_and_boxes)
        stage_smooth = self.predict_stage_smooth(stage_raw)

        names_boxes_1cell = self._grab_1cell_names_and_boxes(
            stage_smooth,
            filenames_and_boxes)
        pronuclei = self.predict_pronuclei(names_boxes_1cell)

        names_boxes_cleavage = self._grab_cleavage_names_and_boxes(
            stage_smooth,
            filenames_and_boxes)
        blastomeres = self.predict_blastomeres(names_boxes_cleavage)

        out = {
            'zona': zona,
            'boxes': AnnotatedDataset.create_from_list(boxes),
            'frag': frag,
            'stage_raw': stage_raw,
            'stage_smooth': stage_smooth,
            'pronuclei': pronuclei,
            'blastomeres': blastomeres,
            }
        return out

    def predict_zona(self, filenames):
        predictor = ZonaPredictor(device=self.device)
        annotated_images = predictor.predict(filenames)
        return AnnotatedDataset.create_from_list(annotated_images)

    def calculate_boxes(self, zona_dataset):
        boxes = list()
        for label in zona_dataset.iterate_over_images():
            info = label.info
            box = find_bounding_box(label.annotation)
            boxes.append(AnnotatedImage(info, box))
        return boxes

    def predict_frag(self, boxes):
        predictor = FragPredictor(device=self.device)
        annotated_images = predictor.predict(boxes)
        return AnnotatedDataset.create_from_list(annotated_images)

    def predict_stage_raw(self, boxes):
        predictor = StagePredictor(device=self.device)
        annotated_images = predictor.predict(boxes)
        return AnnotatedDataset.create_from_list(annotated_images)

    def predict_stage_smooth(self, stage_raw_dataset):
        predictor = DynamicProgramming()
        embryos = [well for slide in stage_raw_dataset for well in slide]

        out = list()
        for embryo in embryos:
            these_images = embryo.iterate_over_images()
            these_probs = np.array([i.annotation for i in these_images])
            these_smoothed = predictor.predict(these_probs.astype('float32'))
            for image, label in zip(these_images, these_smoothed):
                out.append(AnnotatedImage(image.info, label))
        return AnnotatedDataset.create_from_list(out)

    def predict_pronuclei(self, filenames_and_boxes):
        predictor = PronucleiPredictor(device=self.device)
        annotated_images = predictor.predict(filenames_and_boxes)
        return AnnotatedDataset.create_from_list(annotated_images)

    def predict_blastomeres(self, filenames_and_boxes):
        predictor = BlastomerePredictor(device=self.device)
        annotated_images = predictor.predict(filenames_and_boxes)
        return AnnotatedDataset.create_from_list(annotated_images)

    def _grab_cleavage_names_and_boxes(self, stage_smooth, names_and_boxes):
        valid_stages = {
            stage_to_number['{}-cell'.format(stage)]
            for stage in range(1, 9)}
        out = self._grab_names_and_boxes_for_stages(
            stage_smooth, valid_stages, names_and_boxes)
        return out

    def _grab_1cell_names_and_boxes(self, stage_smooth, names_and_boxes):
        valid_stages = {stage_to_number['1-cell']}
        out = self._grab_names_and_boxes_for_stages(
            stage_smooth, valid_stages, names_and_boxes)
        return out

    def _grab_names_and_boxes_for_stages(
            self, stage_smooth, valid_stages, names_and_boxes):
        infos = [
            FP.get_imageinfo_from_filename(nm) for nm, _ in names_and_boxes]
        out = [
            name_and_box
            for name_and_box, info in zip(names_and_boxes, infos)
            if stage_smooth[info].annotation in valid_stages]
        return out


def compress_zona(zona_dataset):
    compressed_images = list()
    for entry in zona_dataset.iterate_over_images():
        segmentation = entry.annotation.astype('uint8')
        as_pil = Image.fromarray(segmentation, mode="L")
        compressed = io.BytesIO()
        as_pil.save(compressed, format='png')
        compressed_images.append(
            AnnotatedImage(entry.info, compressed)
            )
    return AnnotatedDataset.create_from_list(compressed_images)


def decompress_zona(compressed_dataset):
    zona_images = list()
    for entry in compressed_dataset.iterate_over_images():
        compressed = entry.annotation
        as_pil = Image.open(compressed)
        segmentation = np.array(as_pil, dtype='uint8')
        zona_images.append(
            AnnotatedImage(entry.info, segmentation)
            )
    return AnnotatedDataset.create_from_list(zona_images)

