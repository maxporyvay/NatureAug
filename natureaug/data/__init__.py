from .belgium_tsc_dataset import BelgiumTSCDataset
from .gtsrb_dataset import GTSRBDataset

DATASETS = {
    'belgium_tsc': {
        'class': BelgiumTSCDataset,
        'num_classes': 62,
    },
    'gtsrb': {
        'class': GTSRBDataset,
        'num_classes': 43,
    },
}


def load_dataset(config):
    return DATASETS[config['dataset']['name']]
