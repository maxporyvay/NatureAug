import time

import torch

from natureaug.classifiers import load_classifier
from .preprocess import get_train_test_loaders
from .test import test
from .train import train


def get_device():
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
    return device


def full_experiment_pipeline(config):
    for subexp_config in config['subexps']:
        print(f'Running subexperiment "{subexp_config["name"]}"')
        subexperiment_pipeline(subexp_config)


def subexperiment_pipeline(subexp_config):
    start_time = time.time()

    train_loader, test_loader, dataset_num_classes = get_train_test_loaders(subexp_config)

    model = load_classifier(subexp_config)(num_classes=dataset_num_classes)

    device = get_device()
    model.to(device)

    model.train()
    train(model=model, train_dataloader=train_loader, device=device, config=subexp_config)

    model.eval()
    test(model=model, test_dataloader=test_loader, device=device, num_classes=dataset_num_classes)

    print(f"Subexperiment time: {time.time() - start_time}")
