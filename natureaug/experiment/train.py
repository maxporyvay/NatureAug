from functools import partial

import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD


OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
}

LOSS_FUNCTIONS = {
    'cross_entropy_loss': CrossEntropyLoss,
}

setting_optimizer_params = lambda optimizer, params: partial(optimizer, **params)


def load_optimizer(config):
    optimizer_params = {}
    if 'params' in config:
        optimizer_params = config['params']
    return setting_optimizer_params(OPTIMIZERS[config['optimizer']['name']], optimizer_params)


def load_loss_function(config):
    loss_function_params = {}
    if 'params' in config:
        loss_function_params = config['params']
    return LOSS_FUNCTIONS[config['loss_function']['name']](**loss_function_params)


def train(model, train_dataloader, device, config):
    training_config = config['training']
    criterion = load_loss_function(training_config)
    optimizer = load_optimizer(training_config)(model.parameters())

    total_step = len(train_dataloader)
    num_epochs = training_config['num_epochs']

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (images, labels) in tqdm.tqdm(enumerate(train_dataloader)):
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
