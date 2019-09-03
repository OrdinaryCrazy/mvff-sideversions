# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'mvff_rfcn'))

import train_end2end
import test

if __name__ == "__main__":
    train_end2end.main()
    test.main()

# (mvff-env) jingtun@winnie:~/mvff-sideversions2$ sudo nohup python2 -u experiments/mvff_rfcn/mvff_rfcn_end2end_train_ test.py --cfg experiments/mvff_rfcn/cfgs/resnet_v1_101_motion_vector_imagenet_vid_rfcn_end2end_ohem.yaml >version3_e poch3.out 2>&1 &