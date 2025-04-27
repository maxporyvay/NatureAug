import numpy as np
from random import random

from albumentations.augmentations.transforms import (
    RandomFog as RandomFogFromAlbumentations,
    RandomRain as RandomRainFromAlbumentations,
    RandomShadow as RandomShadowFromAlbumentations,
    RandomSnow as RandomSnowFromAlbumentations,
    RandomSunFlare as RandomSunFlareFromAlbumentations,
)
from imgaug.augmenters.imgcorruptlike import(
    Frost as FrostFromImgaugInner,
)
from imgaug.augmenters.weather import(
    Clouds as CloudsFromImgaugInner,
    Fog as FogFromImgaugInner,
    Rain as RainFromImgaugInner,
    Snowflakes as SnowflakesFromImgaugInner,
)

from .utils import (
    AlbumentationAugmentation,
    ImgaugAugmentation,
    plasma_fractal,
)


# Augmentations from Albumentations

class FogFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomFogFromAlbumentations, **kwargs)


class RainFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomRainFromAlbumentations, **kwargs)


class ShadowFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomShadowFromAlbumentations, **kwargs)


class SnowFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomSnowFromAlbumentations, **kwargs)


class SunFlareFromAlbumentations(AlbumentationAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RandomSunFlareFromAlbumentations, **kwargs)


# Augmentations from Imgaug

class CloudsFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(CloudsFromImgaugInner, **kwargs)


class FogFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(FogFromImgaugInner, **kwargs)


class FrostFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(FrostFromImgaugInner, **kwargs)


class RainFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(RainFromImgaugInner, **kwargs)


class SnowflakesFromImgaug(ImgaugAugmentation):
    def __init__(self, **kwargs):
        super().__init__(SnowflakesFromImgaugInner, **kwargs)


# Augmentations from Common Corruptions

class FogFromCommonCorruptions:
    def __init__(self, p=0.5, severity=1):
        self.p = p
        self.severity = severity

    def __call__(self, sample):
        if random() < self.p:
            c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][self.severity - 1]
            sample = np.array(sample, dtype=np.float32) / 255.
            max_val = sample.max()
            sample += c[0] * plasma_fractal(wibbledecay=c[1])[:227, :227][..., np.newaxis]
            return np.clip(sample * max_val / (max_val + c[0]), 0, 1) * 255
        return sample
