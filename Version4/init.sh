#!/bin/bash

mkdir -p ./output
mkdir -p ./external/mxnet
mkdir -p ./model/pretrained_model

cd lib/bbox
python2 setup_linux.py build_ext --inplace
cd ../nms
python2 setup_linux.py build_ext --inplace
cd ../..
