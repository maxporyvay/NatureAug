from dataclasses import dataclass

import torch
import tqdm
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
)


@dataclass
class ResultMetrics:
    accuracy_micro: float
    accuracy_macro: float
    f1_macro: float


def test(model, test_dataloader, device, num_classes):
    with torch.no_grad():
        y_pred = []
        y_true = []
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            del images, labels, outputs
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)

        accuracy_micro = multiclass_accuracy(y_pred, y_true, average="micro").item()
        accuracy_macro = multiclass_accuracy(y_pred, y_true, average="macro", num_classes=num_classes).item()
        f1_macro = multiclass_f1_score(y_pred, y_true, average="macro", num_classes=num_classes).item()
        print(f'Accuracy (micro) on test images: {accuracy_micro}')
        print(f'Accuracy (macro) on test images: {accuracy_macro}')
        print(f'F1-score (macro) on test images: {f1_macro}')

        del y_pred, y_true

    return ResultMetrics(
        accuracy_micro=accuracy_micro,
        accuracy_macro=accuracy_macro,
        f1_macro=f1_macro,
    )
