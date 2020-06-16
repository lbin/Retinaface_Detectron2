import random
import sys

import numpy as np
from detectron2.data.transforms import ResizeTransform, TransformGen
from fvcore.transforms.transform import CropTransform, NoOpTransform
from PIL import Image

__all__ = ["WiderFace_ResizeShortestEdge", "WiderFace_NoOpTransform", "WiderFace_RandomCrop"]


class WiderFace_ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="choice", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        if min(h, w) >= self.short_edge_length[0]:
            return NoOpTransform()

        scale = self.short_edge_length[0] * 1.0 / min(h, w)
        newh = h * scale
        neww = w * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class WiderFace_NoOpTransform(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(self):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()

    def get_transform(self, img):

        return NoOpTransform()


class WiderFace_RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size

        # crop_size = np.asarray([0.1, 0.9], dtype=np.float32)
        # ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
        # return int(h * ch + 0.5), int(w * cw + 0.5)

        PRE_SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(h, w)
        w = int(scale * short_side)
        h = w
        return h, w
