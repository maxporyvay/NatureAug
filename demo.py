import json
import sys
import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToPILImage, ToTensor

from natureaug.augmentations import load_augmentation_pipeline
from natureaug.classifiers import load_classifier
from natureaug.data import load_dataset


if __name__ == '__main__':
    assert len(sys.argv) == 3
    assert sys.argv[1] == '--config-path'
    config_path = sys.argv[2]
    with open(config_path) as config_file:
        config = json.load(config_file)

    augmentation_pipeline = load_augmentation_pipeline(config)
    normalize = Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    if augmentation_pipeline is not None:
        train_transform = Compose([
            ToPILImage(),
            Resize((227,227)),
            augmentation_pipeline,
            ToTensor(),
            normalize,
        ])
    else:
        train_transform = Compose([
            ToPILImage(),
            Resize((227,227)),
            ToTensor(),
            normalize,
        ])
    test_transform = Compose([
        ToPILImage(),
        Resize((227,227)),
        ToTensor(),
        normalize,
    ])

    dataset_class = load_dataset(config)
    train_dataset = dataset_class(train=True, transform=train_transform)
    test_dataset = dataset_class(train=False, transform=test_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    classifier_class = load_classifier(config)

    num_classes = 62
    num_epochs = 1
    learning_rate = 0.005

    model = classifier_class(num_classes)

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    total_step = len(train_loader)

    # Device configuration
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('CUDA is available:', cuda_available)
    if cuda_available:
        print('CUDA device count:', torch.cuda.device_count())
        current_device = torch.cuda.current_device()
        print('CUDA current device index:', current_device)
        print('CUDA current device name:', torch.cuda.get_device_name(current_device))
    model.to(device)

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
               .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print('Accuracy of the network on the test images: {}%'.format(100 * correct / total))
