import torch
import tqdm
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
)


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
        print(f'Accuracy (micro) on test images: {multiclass_accuracy(y_pred, y_true, average="micro")}')
        print(f'Accuracy (macro) on test images: {multiclass_accuracy(y_pred, y_true, average="macro", num_classes=num_classes)}')
        print(f'F1-score (macro) on test images: {multiclass_f1_score(y_pred, y_true, average="macro", num_classes=num_classes)}')
        del y_pred, y_true
