from .belgium_tsc_dataset import BelgiumTSCDataset
from .gtsrb_dataset import GTSRBDataset

DATASETS = {
    'belgium_tsc': BelgiumTSCDataset,
    'gtsrb': GTSRBDataset,
}


def load_dataset(config):
    return DATASETS[config['dataset']['name']]
