import os
import io

import torch
import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(filename):
    image_pil = Image.open(filename).convert("RGB")
    image_rgb = np.array(image_pil).astype('float')
    return image_rgb / 255.0


def read_images_for_torch(filenames):
    images_np = np.array(
        [read_image(nm) for nm in filenames]).astype('float32')
    images_np = np.moveaxis(images_np, 3, 1)
    images_torch = torch.from_numpy(images_np)
    return images_torch


def load_image_into_ram(filename):
    with open(filename, 'rb') as f:
        bytesio = io.BytesIO(f.read())
    return bytesio


def load_and_crop_image(filename, box, output_shape=(299, 299)):
    """
    Parameters
    ----------
    filenames : path-like object
    box : 4-element tuple
        (left, lower, width, height) of the box, in px.
    output_shape : 2-element tuple, optional
        The shape to resize the image to after cropping.

    Returns
    -------
    PIL.Image.Image
    """
    im = Image.open(filename)
    left, lower, width, height = box
    cropped = im.crop((lower, left, lower + height, left + width))
    return cropped.resize(output_shape, resample=Image.NEAREST)


def sort_azimuthally(x, y):
    """Given an x, y, sorts them in azimuthal order.

    Parameters
    ----------
    x, y : N-element np.ndarray's

    Returns
    -------
    xs, ys : N-element np.ndarray's sorted by angle
    """
    x0 = x.mean()
    y0 = y.mean()
    dx = x - x0
    dy = y - y0
    angle = np.arctan2(dy, dx)
    inds = np.argsort(angle % (2*np.pi))
    return x[inds].copy(), y[inds].copy()


def split_all(filename):
    splitted = os.path.split(filename)
    if len(splitted[0]) >= 1 and len(splitted[1]) > 1:
        answer = split_all(splitted[0]) + splitted[1:]
    else:
        answer = tuple([s for s in splitted if len(s) > 0])
    return answer


def augment_focus(filename):
    """
    Given a filename, returns the associated (F+, F0, F-) filenames.

    Parameters
    ----------
    filename : string
        Filename of the form path/slide/well/focus/image_numper.ext

    Returns
    ------
    (lower, central, upper)
        The filenames from above and below in focus, in the order
        F-, F0, F+
    """
    *head, slide, well, focus, image = split_all(filename)

    def _get_filename(focus_name):
        return os.path.join(*head, slide, well, focus_name, image)

    lower = _get_filename("F-15")
    central = _get_filename("F0")
    upper = _get_filename("F15")

    if not os.path.exists(upper):
        lower = _get_filename("F-13")
        upper = _get_filename("F13")

    out = (lower, central, upper)
    for name in out:
        if not os.path.exists(name):
            msg = "Unable to augment for: {}".format(filename)
            raise FileNotFoundError(msg)
    return out


class TransformingCollection(object):
    def __init__(self, items_raw, transform):
        """
        A utility for accessing and transforming items stored items in ram.

        Parameters
        ----------
        items_raw : list-like
        transform : function
            The transformation function to call on the raw data.
            Called every time an item is indexed, so if the
            transformation is random calling the same index twice will
            return two different values.

        Methods
        -------
        __getitem__ : Indexable like an ordinary numpy array
        __len__

        Examples
        --------
        Loading images as jpegs into ram:
        >>> filenames = ['000.jpg', '001.jpg', '002.jpg', '003.jpg']
        >>> images_ram = [load_image_into_ram(nm) for nm in filenames]
        >>> loader = TransformingCollection(images_ram, read_image)
        >>> from_loader = loader[0]
        >>> from_file = read_image(filenames[0])
        >>> assert np.all(from_loader == from_file)
        """
        self.items_raw = items_raw
        self.transform = transform

    def __getitem__(self, indexable):
        return self.transform(self.items_raw[indexable])

    def __len__(self):
        return len(self.items_raw)

