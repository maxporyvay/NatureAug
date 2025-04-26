from natureaug.dataset import BelgiumTSCDataset, GTSRBDataset


if __name__ == '__main__':
    belgium_tsc_train_dataset = BelgiumTSCDataset(train=True)
    belgium_tsc_test_dataset = BelgiumTSCDataset(train=False)
    gtsrb_train_dataset = GTSRBDataset(train=True)
    gtsrb_test_dataset = GTSRBDataset(train=False)
    print(len(belgium_tsc_train_dataset))
    print(len(belgium_tsc_test_dataset))
    print(len(gtsrb_train_dataset))
    print(len(gtsrb_test_dataset))