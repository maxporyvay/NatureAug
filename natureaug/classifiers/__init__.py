from functools import partial

from torchvision.models import alexnet, resnet18, vgg11, inception_v3

setting_no_weights = lambda classifier: partial(classifier, weights=None)
disabling_inception_aux_logits = lambda classifier: partial(classifier, aux_logits=False)

CLASSIFIERS = {
    'alexnet': setting_no_weights(alexnet),
    'resnet18': setting_no_weights(resnet18),
    'vgg11': setting_no_weights(vgg11),
    'inception_v3': setting_no_weights(disabling_inception_aux_logits(inception_v3)),
}


def load_classifier(config):
    return CLASSIFIERS[config['classifier']['name']]
