import os

from embryovision.util import read_images_for_torch
from embryovision.localfolders import embryovision_folder
from embryovision.predictor import Predictor, load_classifier
from embryovision.managedata import FilenameParser


class ZonaPredictor(Predictor):
    loadname = os.path.join(embryovision_folder, 'zona', 'zonaclassifier.pkl')
    load_network = staticmethod(load_classifier)

    def _predict(self, filenames):
        images = read_images_for_torch(filenames).to(self.device)
        labels = self.network.predict(images).astype('uint8')
        infos = [FilenameParser.get_imageinfo_from_filename(nm)
                 for nm in filenames]
        return self.pack_into_annotated_images(infos, labels)

