#!/bin/sh
DATASET_DIR=gtsrb
rm -rf $DATASET_DIR
mkdir $DATASET_DIR && cd $DATASET_DIR

TRAIN_ZIP=gtsrb_train.zip
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip -O $TRAIN_ZIP
unzip $TRAIN_ZIP
rm -f $TRAIN_ZIP

TEST_ZIP=gtsrb_test.zip
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip -O $TEST_ZIP
unzip $TEST_ZIP
rm -f $TEST_ZIP
