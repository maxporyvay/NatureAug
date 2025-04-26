import json

from natureaug.data import load_dataset


if __name__ == '__main__':
    with open('config_example.json') as config_file:
        config = json.load(config_file)
    dataset_class = load_dataset(config)
    train_dataset = dataset_class(train=True)
    test_dataset = dataset_class(train=False)
    print(len(train_dataset))
    print(len(test_dataset))
