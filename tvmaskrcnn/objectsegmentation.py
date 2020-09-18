import torch
import torch.nn as nn
import torchvision
from collections import namedtuple

from embryovision.torchmaskrcnn import MaskRCNN


DetectionResult = namedtuple("DetectionResult", ("confidence", "box", "mask"))

class ObjectSegmentationModel(MaskRCNN):
    model_name = 'maskrcnn_resnet50_fpn'

    def __init__(self, input_size=(500, 500), num_classes=2):
        self.num_classes = num_classes
        super(ObjectSegmentationModel, self).__init__(
            input_shape=(3, input_size[0], input_size[1]), 
            input_type=torch.Tensor, 
            input_dtype=torch.float32
            )

    def _initialize(self):
        self.network = self._create_network()

    def _create_network(self):
        network = torchvision.models.detection.__dict__[self.model_name](
            num_classes=self.num_classes)
        return network

    def _predict(self, input_x):
        instances = self.network(input_x)
        out = list()
        for instance in instances:
            for inst_id, score in enumerate(instance['scores']):
                out.append(
                    DetectionResult(
                        confidence=score.detach().cpu().numpy(),
                        box=instance['boxes'][inst_id].detach().cpu().numpy(),
                        mask=instance['masks'][inst_id].detach().cpu().numpy()[0]))
        return out
