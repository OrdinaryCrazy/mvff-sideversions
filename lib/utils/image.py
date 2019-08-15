# --------------------------------------------------------
# Rivulet
# Licensed under The MIT License [see LICENSE for details]
# Modified by Boyuan Feng
# --------------------------------------------------------
# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
import pickle

# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

mv_not_found_count = 0

# Used for test only
def get_image_mv(roidb, config):
    '''
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    '''
    global mv_not_found_count
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    processed_mvs = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        # TODO. This position should read the motion vector.
        # Original: path_to_mv_pattern: ['self.data_path', 'Data', 'DET', 'train', 'ILSVRC2015_VID_train_0000', 'ILSVRC2015_train_00000000', '000010.JPEG']
        path_to_mv_pattern = roi_rec['image'].split('/')
        # Expect: path_to_mv_pattern: ['self.data_path', 'MV', 'DET', 'train', 'ILSVRC2015_VID_train_0000', 'ILSVRC2015_train_00000000', '000010.JPEG']
        path_to_mv_pattern[6] = 'MV'
        path_to_mv = '/'.join(path_to_mv_pattern)
        path_to_mv = path_to_mv[:-5] + '.pkl'

        # print('img_path: ', roi_rec['image'], ', mv_path: ', path_to_mv)


        if not os.path.exists(path_to_mv):
            mv_not_found_count += 1
            # mv = np.ones(2*600*1000).reshape((600,1000,2))
            mv = np.zeros(2*600*1000).reshape((600,1000,2))
            print('mv_not_found_count: ', mv_not_found_count, ', path_to_mv: ', path_to_mv)
        else:
            mv = pickle.load(open(path_to_mv, 'rb'))
            # For debugging, mv is set to zero now. This indicates that the cur feature should be same as the key feature.
            # mv = np.zeros(2*600*1000).reshape((600,1000,2))
        #assert os.path.exists(path_to_mv), '%s does not exist'.format(path_to_mv)
        # mv = pickle.load(open(path_to_mv, 'rb'))


        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            mv = mv[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        mv_tensor = mv
        processed_ims.append(im_tensor)
        processed_mvs.append(mv_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)



        return processed_ims, processed_roidb, processed_mvs


def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal. 0 for non-key frame, 1 for key frame.
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb


train_mv_not_found_count = 0


def get_pair_image_mv(roidb, config):
    """
    Preprocess image and return processed gt_roidb
    :param roidb: a list of gt_roidb
    :return: list of img as in mxnet format
    """
    global train_mv_not_found_count
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    processed_mvs = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal. 0 for non-key frame, 1 for key frame.
        # roi_rec['image'] = self.data_path/Data/DET/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/000010.JPEG
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            # Case I: VID train data.

            # TODO. Not sure whether we should use +5 here.
            ref_id = (roi_rec['frame_seg_id'] // 12) * 12
            # print('ref_id: ', ref_id)
            # ref_id = min(roi_rec['frame_seg_id'] + 5, roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1

            '''
            if not os.path.exists(ref_image):
                print("ref_image does not exist: ", ref_image)
                ref_im = im.copy()
                eq_flag = 1
            else:
                ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                eq_flag = 0
            '''

            # TODO. This position should read the motion vector.
            # Original: path_to_mv_pattern: ['self.data_path', 'Data', 'DET', 'train', 'ILSVRC2015_VID_train_0000', 'ILSVRC2015_train_00000000', '000010.JPEG']
            path_to_mv_pattern = roi_rec['image'].split('/')
            # Expect: path_to_mv_pattern: ['self.data_path', 'MV', 'DET', 'train', 'ILSVRC2015_VID_train_0000', 'ILSVRC2015_train_00000000', '000010.JPEG']
            path_to_mv_pattern[6] = 'MV'
            path_to_mv = '/'.join(path_to_mv_pattern)
            path_to_mv = path_to_mv[:-5] + '.pkl'

            if not os.path.exists(path_to_mv):
                train_mv_not_found_count += 1
                mv = np.zeros(2*600*1000).reshape((600,1000,2))
                print('train_mv_not_found_count: ', train_mv_not_found_count, ', path_to_mv: ', path_to_mv)
            else:
                mv = pickle.load(open(path_to_mv, 'rb'))


            # assert os.path.exists(path_to_mv), '%s does not exist'.format(path_to_mv)
            # mv = pickle.load(open(path_to_mv, 'rb'))
        else:
            # Case II: DET train data.
            ref_im = im.copy()
            eq_flag = 1
            # Should have a definition of mv. No mv for DET train data.
            mv = np.zeros(2*600*1000).reshape((600,1000,2))


        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]
            mv = mv[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        #mv, mv_scale = resize(mv, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        #mv = cv2.resize(mv, (36, 63), interpolation = cv2.INTER_AREA)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        #mv_tensor = transform(mv, [0,0,0])
        mv_tensor = mv
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_mvs.append(mv_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb, processed_mvs



def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = im.astype(np.float32)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    depth = im.shape[2]
    im_tensor = np.zeros((1, depth, im.shape[0], im.shape[1]))
    for i in range(depth):
        im_tensor[0, i, :, :] = im[:, :, depth-1 - i] - pixel_means[depth-1 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
