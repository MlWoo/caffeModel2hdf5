#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e
TOOLS=../build/tools

NETWORK=../models/bvlc_googlenet
CAFFEMODEL=/root
OUTPUT=./
GLOG_logtostderr=1 $TOOLS/dump_network_hdf5 \
    $NETWORK/train_val.prototxt  \
    $CAFFEMODEL/caffe_init.caffemodel \
    $OUTPUT/train_val.hdf5
