import os
import glob

from embryovision.managedata import ImageInfo

_here = os.path.abspath(os.path.dirname(__file__))
image_directory = os.path.join(_here, 'images')


def get_loadable_filenames():
    globexpr = os.path.join(
        image_directory,
        "D2012_01_01_S0001_I313_pdb",
        "WELL01",
        "F0",
        "*.jpg")
    return sorted(glob.glob(globexpr))


INFO_F0 = ImageInfo(
    slide='D2017_05_05_S1477_I313_pdb', well=6, focus=0, image_number=16)
INFO_F0_2 = ImageInfo(
    slide='D2017_05_05_S1477_I313_pdb', well=6, focus=0, image_number=17)
INFO_F0_3 = ImageInfo(
    slide='D2017_05_05_S1477_I313_pdb', well=6, focus=0, image_number=18)
INFO_FM = ImageInfo(
    slide='D2017_07_04_S1911_I292_pdb', well=4, focus=-15, image_number=5)
INFO_FM_2 = ImageInfo(
    slide='D2017_05_05_S1477_I313_pdb', well=6, focus=-15, image_number=17)
INFO_FP = ImageInfo(
    slide='D2016_08_21_S1201_I313_pdb', well=12, focus=15, image_number=48)
INFOS = (INFO_F0, INFO_F0_2, INFO_FP, INFO_FM, INFO_FM_2)
