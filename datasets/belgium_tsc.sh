#!/bin/sh
DATASET_DIR=belgium_tsc
rm -rf $DATASET_DIR
mkdir $DATASET_DIR && cd $DATASET_DIR

TRAIN_ZIP=belgium_tsc_train.zip
wget https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip -O $TRAIN_ZIP
unzip $TRAIN_ZIP
rm -f $TRAIN_ZIP

TEST_ZIP=belgium_tsc_test.zip
wget https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip -O $TEST_ZIP
unzip $TEST_ZIP
rm -f $TEST_ZIP
