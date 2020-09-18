import pickle

from embryovision.managedata import AnnotatedImage
from embryovision.tvmaskrcnn.objectsegmentation import ObjectSegmentationModel


class Predictor(object):
    loadname = None  # implement in subclass
    load_network = None  # implement in subclass
    default_blocksize = 10

    def __init__(self, device='cuda', blocksize=None):
        self.device = device
        self.network = self.load_network(self.loadname)
        self.network.to(self.device)
        if blocksize is None:
            blocksize = self.default_blocksize
        self.blocksize = blocksize

    def predict(self, inputs):
        out = list()
        for start in range(0, len(inputs), self.blocksize):
            these_inputs = inputs[start: start + self.blocksize]
            these_predictions = self._predict(these_inputs)
            out.extend(these_predictions)
        return out

    def pack_into_annotated_images(self, image_infos, annotations):
        images = list()
        for image_info, annotation in zip(image_infos, annotations):
            image = AnnotatedImage(image_info, annotation)
            images.append(image)
        return images


def load_classifier(filename):
    with open(filename, 'rb') as f:
        classifier = pickle.load(f)
    classifier.to('cpu')
    classifier.eval()
    return classifier


def load_maskrcnn(filename):
    base_model = ObjectSegmentationModel(input_size=(500, 500), num_classes=2)
    with open(filename, 'rb') as f:
        loaded_state = pickle.load(f)
    base_model.load_state_dict(loaded_state)
    base_model.to('cpu')
    base_model.eval()
    return base_model

