from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToPILImage, ToTensor

from natureaug.augmentations import load_augmentation_pipeline
from natureaug.data import load_dataset


def get_train_test_loaders(config):
    augmentation_pipeline = load_augmentation_pipeline(config)
    normalize = Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    if augmentation_pipeline is not None:
        train_transform = Compose([
            ToPILImage(),
            Resize((224,224)),
            augmentation_pipeline,
            ToTensor(),
            normalize,
        ])
    else:
        train_transform = Compose([
            ToPILImage(),
            Resize((224,224)),
            ToTensor(),
            normalize,
        ])
    test_transform = Compose([
        ToPILImage(),
        Resize((224,224)),
        ToTensor(),
        normalize,
    ])

    dataset = load_dataset(config)
    dataset_class, dataset_num_classes = dataset['class'], dataset['num_classes']
    train_dataset = dataset_class(train=True, transform=train_transform)
    test_dataset = dataset_class(train=False, transform=test_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, dataset_num_classes
