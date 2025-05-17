import time

import torch

from natureaug.classifiers import load_classifier
from .preprocess import get_train_test_loaders
from .report import report_exp_results
from .test import test
from .train import train

REPORT_FILE_NAME = 'report.csv'


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
    exp_names_list = []
    metrics_list = []
    times_list = []

    for subexp_config in config['subexps']:
        exp_name = subexp_config["name"]
        print(f'Running subexperiment "{exp_name}"')
        metrics, time_elapsed = subexperiment_pipeline(subexp_config)
        exp_names_list.append(exp_name)
        metrics_list.append(metrics)
        times_list.append(time_elapsed)

    report_exp_results(REPORT_FILE_NAME, exp_names_list, metrics_list, times_list)
    print()


def subexperiment_pipeline(subexp_config):
    start_time = time.time()

    train_loader, test_loader, dataset_num_classes = get_train_test_loaders(subexp_config)

    model = load_classifier(subexp_config)(num_classes=dataset_num_classes)

    device = get_device()
    model.to(device)

    model.train()
    train(model=model, train_dataloader=train_loader, device=device, config=subexp_config)

    model.eval()
    metrics = test(model=model, test_dataloader=test_loader, device=device, num_classes=dataset_num_classes)

    time_elapsed = time.time() - start_time
    print(f"Subexperiment time: {time_elapsed}")

    return metrics, time_elapsed
