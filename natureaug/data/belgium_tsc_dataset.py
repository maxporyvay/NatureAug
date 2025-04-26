from skimage import io
from torch.utils.data import Dataset

from .utils import load_label_named_dir_data


class BelgiumTSCDataset(Dataset):
    def __init__(self, train=True, transform=None):
        if train:
            subfolder_path = 'Training'
        else:
            subfolder_path = 'Testing'
        root_dir = f'datasets/belgium_tsc/{subfolder_path}'
        self.image_paths, self.labels = self.load_data(root_dir)
        assert len(self.image_paths) == len(self.labels)
        self.transform = transform

    def load_data(self, data_directory):
        return load_label_named_dir_data(data_directory, ['.ppm'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample, label = io.imread(self.image_paths[idx]), self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
