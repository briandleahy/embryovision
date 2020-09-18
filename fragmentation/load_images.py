import numpy as np
import torch
import torchvision.transforms.functional as transform
from embryovision.util import read_image


def load_images_for_fragmentation_regressor(image_filenames):
    """
    Parameters
    ----------
    image_filenames : iterable of filename-like objects
        The image filenames must point to pre-cropped images, in the
        shape of (299, 299) and centered on the zona using attentionbox.

    Returns
    -------
    torch.Tensor, shape (?, 3, 299, 299)
        Tensor of the shape, normalization, and data type required for
        the fragmentation classifier
    """
    inputs = []
    for filename in image_filenames:
        image_numpy = read_image(filename)
        image_tensor = transform.to_tensor(image_numpy.astype('float32'))
        image_norm = transform.normalize(
            image_tensor,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
        inputs.append(image_norm)
    return torch.stack(inputs,0)

