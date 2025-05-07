import torch
import tqdm


def test(model, test_dataloader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print('Accuracy of the network on the test images: {}%'.format(100 * correct / total))
