"""Tools for getting labels for images.

The AnnotatedDataset here provides easy functionality for something like:
{slide:
    well:
        focus:
            image: label
}
"""

import os
import json
import copy
import pickle
import itertools
from collections import namedtuple, OrderedDict

from embryovision.util import split_all


ImageInfo = namedtuple("ImageInfo", ['slide', 'well', 'focus', 'image_number'])


class AnnotatedImage(object):
    """A format for storing annotations on an image in a dataset.

    Methods
    -------
    copy() -> AnnotatedImage
    collate_annotations_with(AnnotatedImage) -> AnnotatedImage
    create_from_filename_and_data(filename, data) -> AnnotatedImage
    pack_for_json() -> ``self.annotation``

    In addition, ``AnnotatedImage`` supports comparisons <, >, ==, !=,
    <=, >= based on the image info, sorting first by slides, then by
    wells, then by focus. This allows for convenient, consistent sorting.
    """

    def __init__(self, info, annotation):
        """
        Parameters
        ----------
        info : ``ImageInfo``
            Information about the image.
        annotation:  object
            Any information about the image to store.


        Notes
        -----
        For collating with other AnnotatedImages, it is best if
        ``annotation`` can be cast as a list. Alternatively, if
        ``annotation`` is a user-defined object, giving it a
        `pack_for_json(bool pretty)` method will allow for clean
        collating.

        See Also
        --------
        AnnotatedImage.create_from_filename_and_data(filename, data)
        """
        self.info = info
        self.annotation = annotation

    def copy(self):
        new_info = ImageInfo(
            slide=self.info.slide, well=self.info.well, focus=self.info.focus,
            image_number=self.info.image_number)
        new_annotation = copy.deepcopy(self.annotation)
        return self.__class__(new_info, new_annotation)

    def collate_annotations_with(self, other):
        """
        Collates with another image `out-of-place`, leaving the original
        result un-modified.

        Parameters
        ----------
        other : AnnotatedImage
            The .info of both images must be the same.

        Returns
        -------
        collated : AnnotatedImage
            AnnotatedImage, with the same ``info`` as both and with
            ``annotation = list(self.annotation) + list(other.annotation)``,
            loosely.
        """
        if self != other:
            raise ValueError("Cannot collate different images.")
        self_annotation = self._cast_as_list(self.annotation)
        other_annotation = other._cast_as_list(other.annotation)
        new_annotation = self_annotation + other_annotation

        collated = self.__class__(self.info, new_annotation)
        return collated

    @classmethod
    def create_from_filename_and_data(cls, filename, annotation):
        """
        Parameters
        ----------
        filename : string
            Filename of the form ``head/slide/well/focus/image.jpg``,
            where ``head`` is any valid folder path.
        annotation : object
            Any information about the image to store.

        Returns
        -------
        AnnotatedImage
        """
        info = FilenameParser.get_imageinfo_from_filename(filename)
        return cls(info, annotation)

    def pack_for_json(self, pretty=False):
        if hasattr(self.annotation, 'pack_for_json'):
            packed = self.annotation.pack_for_json(pretty=pretty)
        else:
            packed = self.annotation
        return packed

    def _cast_as_list(cls, datum):
        if type(datum) in {list, set, tuple}:
            datum_list = list(datum)
        else:
            datum_list = [datum]
        return datum_list

    @classmethod
    def _get_label_from(cls, filename):
        directory, path = os.path.split(filename)
        root, label = os.path.split(directory)
        return label

    def __eq__(self, other):
        return self.info == other.info

    def __lt__(self, other):
        # self.__gt__ is done via self.__lt__
        if self.info.slide != other.info.slide:
            lt = self.info.slide < other.info.slide
        elif self.info.well != other.info.well:
            lt = self.info.well < other.info.well
        elif self.info.focus != other.info.focus:
            lt = self.info.focus < other.info.focus
        elif self.info.image_number != other.info.image_number:
            lt = self.info.image_number < other.info.image_number
        else:  # everything is the same!
            lt = False
        return lt

    def __le__(self, other):
        return not (self > other)

    def __repr__(self):
        result = '{}({}, {})'.format(
            self.__class__.__name__, self.info, self.annotation)
        return result


class FilenameParser(object):
    """Methods for parsing filenames into ImageInfo and vice versa

    This assumes that your directory structure is of the form
        embryo_folder/slide/wellname/focus/image_number.jpg
    with:
        embryo_folder: an arbitrary directory structure
        slide: The name of the slide; can be arbitrary
        wellname: Of the form WELLXX, where XX is a 2-digit number, pre-
            padded with zeros if less than 10.
        focus: Of the form FX, where X is a number.
        image_number: Of the form XXX.jpg, where XXX is a 3-digit
            number, pre-padded with zeros if needed

    For instance, some valid filenames are:
        /home/brian/Data/slidename-1/WELL01/F0/001.jpg
        /home/brian/Data/slidename-2/WELL03/F-15/010.jpg
        /home/brian/Data/slidename-3/WELL12/F45/123.jpg
    The FilenameParser works regardless of whether your platform is
    windows, linux, or mac.

    The FilenameParser works by initializing with a root directory that
    contains the images (`embryo_folder`; this would be
    `'/home/brian/Data/'` in the examples above). However, the
    `embryo_folder` is only needed if the full filename is desired; the
    other class methods work without setting the embryo_folder.

    Methods
    -------
    get_filename_from_imageinfo: ImageInfo -> full filename
        Requires setting the "embryo_folder" portion of the path in the
        init. Returns the full filename, e.g.
            /home/brian/Data/slidename-1/WELL01/F0/001.jpg
    get_partial_filename_from_imageinfo: ImageInfo -> partial filename
        Classmethod; does not require setting `embryo_folder` Returns a
        partial filename, e.g.
            slidename-1/WELL01/F0/001.jpg
    get_imageinfo_from_filename: filename -> ImageInfo
        Can take a full filename or a partial filename.
    """

    def __init__(self, embryo_folder=None):
        self.embryo_folder = embryo_folder

    def get_filename_from_imageinfo(self, image_info):
        """Transform an ImageInfo object into a valid filename"""
        partial_name = self.get_partial_filename_from_imageinfo(image_info)
        fullname = os.path.join(
            self.embryo_folder,
            partial_name,
            )
        return fullname

    @classmethod
    def get_partial_filename_from_imageinfo(cls, image_info):
        """Transform an ImageInfo object into a valid filename"""
        partialname = os.path.join(
            image_info.slide,
            'WELL' + str(int(image_info.well)).rjust(2, '0'),
            'F' + str(int(image_info.focus)),
            str(int(image_info.image_number)).rjust(3, '0') + '.jpg',
            )
        return partialname

    @classmethod
    def get_imageinfo_from_filename(cls, filename):
        *head, slide, wellname, focusname, imagename = split_all(filename)
        well = int(wellname.split("WELL")[1])
        focus = int(focusname.split("F")[1])
        image = int(os.path.splitext(imagename)[0])
        return ImageInfo(slide, well, focus, image)


_image_collection_docstring = """
Collections for storing images.

This essentially operates as a hierarchical dictionaries, with lots of
helper functions There are 4 instantiable classes of
AnnotatedImageCollection, at different levels of the heirarchy in the
structure of the embryo data:

    AnnotatedDataset
    AnnotatedSlideFolder
    AnnotatedWellFolder
    AnnotatedFocusFolder

Each level is composed of entries in the level below, with the exception
of the AnnotatedFocusFolder, which is composed of AnnotatedImages. These
sub-entries are build automatically. Typical use cases use the
AnnotatedDataset.

An AnnotatedImageCollection is indexable, with either:
    * a key corresponding to the next level down (e.g. a slide name or a
      well number), in which case it returns the next level down.
    * an ImageInfo, in which case it returns the relevant AnnotatedImage


Methods
-------
create_from: AnnotatedImage -> AnnotatedImageCollection
create_from_list: list of `AnnotatedImage`s -> AnnotatedImageCollection
update_with: AnnotatedImage
    Add an AnnotatedImage to the dataset.
collate_annotations_with: AnnotatedImageCollection
    Given another dataset with the same set of images, returns a copy
    with the annotations for each image combined, according to
    AnnotatedImage.collate_annotations_with.
copy:
    Copy the dataset
sort:
    Return a sorted copy of the dataset
keys:
    Return a list of the keys in the dataset.
values:
    Return a list of the values in the dataset, i.e. the entries which
    form the next level down.
items:
    Return a list of the (key, value) pairs in the dataset.
save_as_json: filename
    Save the dataset as a json file
save_as_pickle: filename
    Save the dataset as a pickle file
load_from_json: filename
    Load the dataset from a json file. Class method.
load_from_pickle: filename
    Load the dataset from a pickle file. Class method.
remove: valid entry
    Remove an entry from the dataset (recursively, if desired).
iterate_over_images:
    Return a list of every image in the dataset.
contains_image: ImageInfo
    Check if an image is in the dataset
"""


class AnnotatedImageCollection(object):
    """Abstract Base Class"""
    _check_attrs = []
    _key_attr = ''
    _invalid_image_msg = ''
    _null_image = ImageInfo(None, None, None, None)
    _lower_class = None

    def __init__(self):
        self._dict = OrderedDict()
        self._update_info_from(self._null_image)

    @classmethod
    def create_from(cls, annotated_image):
        collection = cls()
        collection.update_with(annotated_image)
        return collection

    @classmethod
    def create_from_list(cls, annotated_images):
        collection = cls()
        for image in annotated_images:
            collection.update_with(image)
        return collection

    def update_with(self, annotated_image):
        if len(self) == 0:
            # FIXME
            # If the annotated image had an attribute slide, well, focus
            # then _update_info_from would work for either an image
            # or an AnnotatedImageCollection
            self._update_info_from(annotated_image.info)
            # FIXME likewise with
            #   _get_key_from in _update_with
            # although I don't know about the others...
            self._update_with(annotated_image)
        elif self.is_same_folder_as(annotated_image.info):
            self._update_with(annotated_image)
        else:
            raise ValueError(self._invalid_image_msg)

    def collate_annotations_with(self, other):
        """
        Explain:
            what gets collated
            in-place vs out-of-place
            that it only includes keys which have been collated.
            raises an error when you try to collate two different folders.

        Returns
        -------
        collated : same class as `self`
            A copy of self, other with the annotations collated.
            The collation operation just gets called recursively, which
            ends up calling `AnnotatedImage.collate_annotations_with`,
            which finally just adds the data as a list.
        """
        if not self.is_same_folder_as(other):
            raise ValueError("Cannot collate from different folders.")
        collated = self.copy()
        for key in self.keys():
            if key in other.keys():
                collated[key] = collated[key].collate_annotations_with(
                    other[key])
            else:
                _ = collated._dict.pop(key)
        return collated

    def copy(self):
        # 1. Create an empty collection for the same folder:
        new = self.__class__()
        for attr in self._check_attrs:
            setattr(new, attr, getattr(self, attr))

        # 2. Update the elements:
        for key in self._dict.keys():
            new._dict.update({key: self._dict[key].copy()})
        return new

    def sort(self):
        """Returns a sorted copy of `self`; this is not in-place."""
        copy = self.__class__()
        for annotated_image in sorted(self.iterate_over_images()):
            copy.update_with(annotated_image)
        return copy

    def keys(self):
        return list(self._dict.keys())

    def values(self):
        return list(self._dict.values())

    def items(self):
        return self._dict.items()

    def is_same_folder_as(self, other):
        """Checks if it is the same folder as either another ImageCollection
        or a ImageInfo object
        """
        metadata_same = [
            getattr(self, a) == getattr(other, a)
            for a in self._check_attrs]
        return all(metadata_same)

    def pack_for_json(self, pretty=True):
        ordered_dict = OrderedDict()
        for key, value in self._dict.items():
            pretty_key = self._prettify(key) if pretty else key
            entry = {pretty_key: value.pack_for_json(pretty=pretty)}
            ordered_dict.update(entry)
        return ordered_dict

    def save_as_json(self, filename, overwrite=False):
        if os.path.exists(filename) and not overwrite:
            msg = '{} already exists. Pass `overwrite` kwarg.'.format(filename)
            raise FileExistsError(msg)
        ordered_dict = self.pack_for_json()
        with open(filename, 'w') as f:
            json.dump(ordered_dict, f, indent=4)
        return None

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            ordered_dict = json.load(f, object_pairs_hook=OrderedDict)
        return cls.create_from_dict(ordered_dict)

    def save_as_pickle(self, filename, overwrite=False):
        if os.path.exists(filename) and not overwrite:
            msg = '{} already exists. Pass `overwrite` kwarg.'.format(filename)
            raise FileExistsError(msg)
        ordered_dict = self.pack_for_json()
        with open(filename, 'wb') as f:
            pickle.dump(ordered_dict, f)
        return None

    @classmethod
    def load_from_pickle(cls, filename):
        with open(filename, 'rb') as f:
            ordered_dict = pickle.load(f)
        return cls.create_from_dict(ordered_dict)

    def remove(self, entry, recursive=False):
        for key, value in self._dict.items():
            if entry is value:
                self._dict.pop(key)
                break
            elif recursive:
                try:
                    value.remove(entry, recursive=True)
                    break
                except ValueError:
                    pass
        else:
            raise ValueError("entry not in {}".format(self))

    def iterate_over_images(self):
        return list(itertools.chain(*[c.iterate_over_images() for c in self]))

    def contains_image(self, image_info):
        is_same_folder = [
            getattr(image_info, a) == getattr(self, a)
            for a in image_info._fields
            if a in self._check_attrs]
        key_value = getattr(image_info, self._key_attr)
        if not all(is_same_folder):
            contains_image = False
        elif key_value in self._dict:
            contains_image = self._dict[key_value].contains_image(image_info)
        else:
            contains_image = False
        return contains_image

    def _check_if_image_is_valid(self, image_info):
        info_ok = [getattr(self, a) == getattr(image_info, a)
                   for a in self._check_attrs]
        return all(info_ok)

    def _update_info_from(self, image_info):
        for attr in self._check_attrs:
            setattr(self, attr, getattr(image_info, attr))

    def _update_with(self, annotated_image):
        key = self._get_key_from(annotated_image)
        if key not in self._dict:
            self._add_entry_from(annotated_image)
        else:
            self._dict[key].update_with(annotated_image)

    def _get_key_from(self, annotated_image):
        return getattr(annotated_image.info, self._key_attr)

    def _add_entry_from(self, annotated_image):
        new_collection = self._lower_class.create_from(
            annotated_image)
        key = self._get_key_from(annotated_image)
        self._dict.update({key: new_collection})

    def _getitem_from_imageinfo(self, key):
        truekey = getattr(key, self._key_attr)
        item = self._dict[truekey]
        if hasattr(item, '_getitem_from_imageinfo'):
            item = item._getitem_from_imageinfo(key)
        return item

    @classmethod
    def _create_from_dict(cls, ordered_dict, *parent_attrs):
        image_collection = cls()
        for attr, value in zip(cls._check_attrs, parent_attrs):
            setattr(image_collection, attr, value)

        for name, entry in ordered_dict.items():
            key = cls._de_prettify(name) if type(name) is str else name
            sub_collection = cls._lower_class._create_from_dict(
                entry, *parent_attrs, key)
            image_collection._dict.update({key: sub_collection})
        return image_collection

    @classmethod
    def create_from_dict(cls, ordered_dict):
        raise NotImplementedError

    @classmethod
    def _prettify(cls, key):
        raise NotImplementedError("Implement in subclass")

    @classmethod
    def _de_prettify(cls, key):
        """Inverse of _prettify"""
        raise NotImplementedError("Implement in subclass")

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        if isinstance(key, ImageInfo):
            item = self._getitem_from_imageinfo(key)
        else:
            item = self._dict[key]
        return item

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __iter__(self):
        return iter(self._dict.values())

    def __eq__(self, other):  # FIXME what was the design choice here?
        return self._dict == other._dict


class AnnotatedFocusFolder(AnnotatedImageCollection):
    _check_attrs = ['slide', 'well', 'focus']
    _key_attr = 'image_number'
    _invalid_image_msg = "Image is from a different focus"
    __doc__ = _image_collection_docstring

    def _update_with(self, annotated_image):
        key = self._get_key_from(annotated_image)
        entry = {key: annotated_image}
        self._dict.update(entry)

    def iterate_over_images(self):
        return list(self._dict.values())

    def contains_image(self, image_info):
        is_same_folder = [
            getattr(image_info, a) == getattr(self, a)
            for a in image_info._fields
            if a in self._check_attrs]
        key_value = getattr(image_info, self._key_attr)
        if not all(is_same_folder):
            contains_image = False
        else:
            contains_image = key_value in self._dict
        return contains_image

    @classmethod
    def _prettify(cls, image_number):
        return str(image_number).rjust(3, '0') + '.jpg'

    @classmethod
    def _de_prettify(cls, image_name):
        image_number = int(image_name.split('.')[0])
        return image_number

    @classmethod
    def _create_from_dict(cls, ordered_dict, *parent_attrs):
        focus_collection = cls()
        slide, well, focus = parent_attrs

        for image_name, image_annotation in ordered_dict.items():
            image_number = (
                cls._de_prettify(image_name)
                if type(image_name) is str else image_name)
            info = ImageInfo(slide, well, focus, image_number)
            annotated = AnnotatedImage(info, image_annotation)
            focus_collection.update_with(annotated)
        return focus_collection


class AnnotatedWellFolder(AnnotatedImageCollection):
    _check_attrs = ['slide', 'well']
    _key_attr = 'focus'
    _invalid_image_msg = "Image is from a different well"
    _lower_class = AnnotatedFocusFolder
    __doc__ = _image_collection_docstring

    @classmethod
    def _prettify(cls, focus_value):
        return 'F' + str(focus_value)

    @classmethod
    def _de_prettify(cls, focus_name):
        return int(focus_name[1:])


class AnnotatedSlideFolder(AnnotatedImageCollection):
    _check_attrs = ['slide']
    _key_attr = 'well'
    _invalid_image_msg = "Image is from a different slide"
    _lower_class = AnnotatedWellFolder
    __doc__ = _image_collection_docstring

    @classmethod
    def _prettify(cls, well_number):
        return "WELL" + str(well_number).rjust(2, '0')

    @classmethod
    def _de_prettify(cls, well_name):
        return int(well_name.split("WELL")[1])


class AnnotatedDataset(AnnotatedImageCollection):
    _check_attrs = []
    _key_attr = 'slide'
    _invalid_image_msg = 'This should not raise'
    _lower_class = AnnotatedSlideFolder
    __doc__ = _image_collection_docstring

    @classmethod
    def create_from_dict(cls, ordered_dict):
        return cls._create_from_dict(ordered_dict)

    @classmethod
    def _prettify(cls, slide_name):
        return slide_name

    @classmethod
    def _de_prettify(cls, slide_name):
        return slide_name

