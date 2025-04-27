from albumentations.augmentations.blur.transforms import (
    Defocus as RandomDefocusFromAlbumentations,
    MotionBlur as RandomMotionBlurFromAlbumentations,
)
from imgaug.augmenters.imgcorruptlike import(
    DefocusBlur as DefocusBlurFromImgaugInner,
    MotionBlur as MotionBlurFromImgaugInner,
)

from .utils import (
    AlbumentationAugmentation,
    ImgaugAugmentation,
)


# Augmentations from Albumentations

class DefocusBlurFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomDefocusFromAlbumentations, **kwargs)


class MotionBlurFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomMotionBlurFromAlbumentations, **kwargs)


# Augmentations from Imgaug

class DefocusBlurFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(DefocusBlurFromImgaugInner, **kwargs)


class MotionBlurFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(MotionBlurFromImgaugInner, **kwargs)
