from .alexnet import AlexNet

CLASSIFIERS = {
    'alexnet': AlexNet,
}


def load_classifier(config):
    return CLASSIFIERS[config['classifier']['name']]
