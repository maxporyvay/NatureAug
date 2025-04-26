from torchvision.transforms.v2 import Compose

from .weather_artifacts import Fog

AUGMENTATIONS = {
    'fog': Fog,
}


def load_augmentation_pipeline(config):
    aug_list = config['augmentations']
    aug_classes_list = []
    for aug in aug_list:
        aug_classes_list.append(AUGMENTATIONS[aug['name']]())
    return Compose(aug_classes_list)
