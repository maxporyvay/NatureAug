from torchvision.transforms.v2 import Compose

from .camera_artifacts import (
    DefocusBlurFromAlbumentations,
    DefocusBlurFromImgaug,
    MotionBlurFromAlbumentations,
    MotionBlurFromImgaug,
)
from .weather_artifacts import (
    CloudsFromImgaug,
    FogFromAlbumentations,
    FogFromCommonCorruptions,
    FogFromImgaug,
    FrostFromImgaug,
    RainFromAlbumentations,
    RainFromImgaug,
    ShadowFromAlbumentations,
    SnowFromAlbumentations,
    SnowflakesFromImgaug,
    SunFlareFromAlbumentations,
)

AUGMENTATIONS = {
    'clouds_from_imgaug': CloudsFromImgaug,
    'defocus_blur_from_albumentations': DefocusBlurFromAlbumentations,
    'defocus_blur_from_imgaug': DefocusBlurFromImgaug,
    'fog_from_albumentations': FogFromAlbumentations,
    'fog_from_common_corruptions': FogFromCommonCorruptions,
    'fog_from_imgaug': FogFromImgaug,
    'frost_from_imgaug': FrostFromImgaug,
    'motion_blur_from_albumentations': MotionBlurFromAlbumentations,
    'motion_blur_from_imgaug': MotionBlurFromImgaug,
    'rain_from_albumentations': RainFromAlbumentations,
    'rain_from_imgaug': RainFromImgaug,
    'shadow_from_albumentations': ShadowFromAlbumentations,
    'snow_from_albumentations': SnowFromAlbumentations,
    'snowflakes_from_imgaug': SnowflakesFromImgaug,
    'sunflare_from_albumentations': SunFlareFromAlbumentations,
}


def load_augmentation_pipeline(config):
    aug_list = config['augmentations']
    aug_classes_list = []
    for aug in aug_list:
        aug_params = {}
        if 'params' in aug:
            aug_params = aug['params']
        aug_classes_list.append(AUGMENTATIONS[aug['name']](**aug_params))
    return Compose(aug_classes_list)
