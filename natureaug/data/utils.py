import os


def load_label_named_dir_data(data_dir, img_extensions):
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    image_paths = []
    labels = []
    for d in directories:
        label_directory = os.path.join(data_dir, d)
        file_paths = [os.path.join(label_directory, file) for file in os.listdir(label_directory)
                      if any([file.endswith(ext) for ext in img_extensions])]
        for file_path in file_paths:
            image_paths.append(file_path)
            labels.append(int(d))
    return image_paths, labels
