import csv
import os

from skimage import io
from torch.utils.data import Dataset

from .utils import load_label_named_dir_data


class GTSRBDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.img_extensions = ['.ppm']
        root_dir_prefix = 'datasets/gtsrb'
        if train:
            self.image_paths, self.labels = self.load_train_data(
                f'{root_dir_prefix}/GTSRB/Final_Training/Images',
            )
        else:
            self.image_paths, self.labels = self.load_test_data(
                f'{root_dir_prefix}/GTSRB/Final_Test/Images',
                f'{root_dir_prefix}/GT-final_test.csv',
            )
        assert len(self.image_paths) == len(self.labels)
        self.transform = transform

    def load_train_data(self, data_directory):
        return load_label_named_dir_data(data_directory, self.img_extensions)

    def load_test_data(self, data_directory, gt_csv_path):
        file_names = [file for file in os.listdir(data_directory)
                      if any([file.endswith(ext) for ext in self.img_extensions])]
        gt_img_to_label_dct = {}
        with open(gt_csv_path, mode='r') as gt_csv_file:
            gt_csv_reader = csv.DictReader(gt_csv_file, delimiter=';')
            for row in gt_csv_reader:
                gt_img_to_label_dct[row['Filename']] = row['ClassId']
        file_paths = []
        labels = []
        for file_name in file_names:
            assert file_name in gt_img_to_label_dct
            file_paths.append(os.path.join(data_directory, file_name))
            labels.append(int(gt_img_to_label_dct[file_name]))
        return file_paths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample, label = io.imread(self.image_paths[idx]), self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
