# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

from multiprocessing.pool import ThreadPool as Pool
import cPickle
import os
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from utils.PrefetchingIter import PrefetchingIter


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict, scale) in enumerate(zip(scores_all, boxes_all, data_dict_all, scales)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


im_shape = None

def im_detect(predictor, data_batch, data_names, scales, cfg, count):
    global im_shape
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        # im_shape = data_dict['data'].shape
        im_shape_tmp = data_dict['data'].shape
        if len(im_shape_tmp) == 4:
            im_shape = im_shape_tmp



        # import pickle

        # pickle.dump(im_shape, open('im_shape'+str(count)+'.pkl', 'wb'))

        # pickle.dump(rois, open('rois'+str(count)+'.pkl','wb'))

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        # pickle.dump(scores, open('scores'+str(count)+'.pkl','wb'))

        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        # pickle.dump(bbox_deltas, open('bbox_deltas'+str(count)+'.pkl','wb'))


        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        # pickle.dump(pred_boxes, open('bbox_pred'+str(count)+'.pkl','wb'))

        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
        # pickle.dump(pred_boxes, open('clip_boxes'+str(count)+'.pkl','wb'))

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)

    if output_all[0].has_key('feat_conv_3x3_relu_output'):
        feat = output_all[0]['feat_conv_3x3_relu_output']
        feat_after_motion = None
    else:
        feat = None
        feat_after_motion = output_all[0]['warping_feat_output']

    #print(output_all[0])

    # For debugging
    # rpn_cls_score_output = output_all[0]['rpn_cls_score_output']
    # rpn_bbox_pred_output = output_all[0]['rpn_bbox_pred_output']
    # rpn_cls_prob_output = output_all[0]['rpn_cls_prob_output']
    # rois_output = output_all[0]['rois_output']
    # rfcn_cls_output = output_all[0]['rfcn_cls_output']
    # rfcn_bbox_output = output_all[0]['rfcn_bbox_output']
    # cls_score_output = output_all[0]['ave_cls_scors_rois_output']
    bbox_pred1_output = output_all[0]['ave_bbox_pred_rois_output']
    bbox_pred2_output = output_all[0]['bbox_pred_reshape2_output']
    bbox_pred_output = output_all[0]['bbox_pred_reshape_output']

    return scores_all, pred_boxes_all, data_dict_all, feat, bbox_pred1_output, bbox_pred2_output, bbox_pred_output

def im_batch_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        im_infos = data_dict['im_info'].asnumpy()
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        rois = output['rois_output'].asnumpy()
        for im_idx in xrange(im_infos.shape[0]):
            bb_idxs = np.where(rois[:,0] == im_idx)[0]
            im_shape = im_infos[im_idx, :2].astype(np.int)

            # post processing
            pred_boxes = bbox_pred(rois[bb_idxs, 1:], bbox_deltas[bb_idxs, :])
            pred_boxes = clip_boxes(pred_boxes, im_shape)

            # we used scaled image & roi to train, so it is necessary to transform them back
            pred_boxes = pred_boxes / scale[im_idx]

            scores_all.append(scores[bb_idxs, :])
            pred_boxes_all.append(pred_boxes)

    return scores_all, pred_boxes_all, data_dict_all



feat = None
count = 0


def pred_eval(gpu_id, key_predictor, cur_predictor, test_data, imdb, cfg, vis=False, thresh=1e-4, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    global feat
    global count

    det_file = os.path.join(imdb.result_path, imdb.name + '_'+ str(gpu_id) + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes, frame_ids = cPickle.load(fid)
        return all_boxes, frame_ids

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]
    num_images = test_data.size
    roidb_frame_ids = [x['frame_id'] for x in test_data.roidb]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    frame_ids = np.zeros(num_images, dtype=np.int)

    roidb_idx = -1
    roidb_offset = -1
    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    # test_data is an instance of the CLASS TestLoader
    for im_info, key_frame_flag, data_batch in test_data:
        count += 1

        # print('im_info: ', im_info, ', key_frame_flag: ', key_frame_flag, ', data_batch: ', data_batch)
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        if key_frame_flag != 2:
            # print('pred_eval, key_predictor')
            tmp = data_names[:]
            # print('pred_eval, data_names: ', tmp)
            #tmp.remove('motion_vector')
            # print('pred_eval, data_names: ', tmp)
            # print('pred_eval, data_batch: ', data_batch)

            scores_all, boxes_all, data_dict_all, feat, bbox_pred1_output, bbox_pred2_output, bbox_pred_output = im_detect(key_predictor, data_batch, tmp, scales, cfg, count)
            # scores_all, boxes_all, data_dict_all, feat, feat_after_motion = im_detect(key_predictor, data_batch, tmp, scales, cfg)
            # print('key frame, feature after motion: ', feat_after_motion)

            import pickle
            #pickle.dump(feat, open('feat_'+str(count)+'.pkl', 'wb'))
            #pickle.dump(feat_after_motion, open('feat_after_motion'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rpn_cls_score_output, open('rpn_cls_score_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(rpn_bbox_pred_output, open('rpn_bbox_pred_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rpn_cls_prob_output, open('rpn_cls_prob_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rois_output, open('rois_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rfcn_cls_output, open('rfcn_cls_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rfcn_bbox_output, open('rfcn_bbox_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred1_output, open('bbox_pred1_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred2_output, open('bbox_pred2_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred_output, open('bbox_pred_output'+str(count)+'.pkl', 'wb'))



        else:
            # print('pred_eval, cur_predictor')
            data_batch.data[0][-1] = feat
            data_batch.provide_data[0][-1] = ('feat_key', feat.shape)



            #print('pred_eval(), non-key branch, data_batch.key_frame_flag: ', data_batch.key_frame_flag)
            # print('pred_eval(), non-key branch, data_batch.provide_data[0]: ', data_batch.provide_data[0])
            # print('pred_eval(), non-key branch, data_batch.data[0]: ', data_batch.data[0])

            import pickle
            #pickle.dump(data_batch.data[0][1], open('mv_'+str(count)+'.pkl', 'wb'))
            #pickle.dump(data_batch.data[0][2], open('feat_'+str(count)+'.pkl', 'wb'))

            scores_all, boxes_all, data_dict_all, _, bbox_pred1_output, bbox_pred2_output, bbox_pred_output = im_detect(cur_predictor, data_batch, data_names, scales, cfg, count)



            # scores_all, boxes_all, data_dict_all, _, feat_after_motion = im_detect(cur_predictor, data_batch, data_names, scales, cfg)
            # print('non-key frame, feature after motion: ', feat_after_motion)

            # pickle.dump(feat_after_motion, open('feat_after_motion'+str(count)+'.pkl', 'wb'))
            # pickle.dump(rpn_cls_score_output, open('rpn_cls_score_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(rpn_cls_prob_output, open('rpn_cls_prob_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rois_output, open('rois_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rfcn_cls_output, open('rfcn_cls_output'+str(count)+'.pkl', 'wb'))
            #pickle.dump(rfcn_bbox_output, open('rfcn_bbox_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred1_output, open('bbox_pred1_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred2_output, open('bbox_pred2_output'+str(count)+'.pkl', 'wb'))
            # pickle.dump(bbox_pred_output, open('bbox_pred_output'+str(count)+'.pkl', 'wb'))


        if key_frame_flag == 0:
            roidb_idx += 1
            roidb_offset = 0
        else:
            roidb_offset += 1

        frame_ids[idx] = roidb_frame_ids[roidb_idx] + roidb_offset

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_dets[keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))

    with open(det_file, 'wb') as f:
        cPickle.dump((all_boxes, frame_ids), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return all_boxes, frame_ids

def pred_eval_multiprocess(gpu_num, key_predictors, cur_predictors, test_datas, imdb, cfg, vis=False, thresh=1e-4, logger=None, ignore_cache=True):
    if gpu_num == 1:
        res = [pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis, thresh, logger, ignore_cache),]
    else:
        pool = Pool(processes=gpu_num)
        multiple_results = [pool.apply_async(pred_eval,args=(i, key_predictors[i], cur_predictors[i], test_datas[i], imdb, cfg, vis, thresh, logger, ignore_cache)) for i in range(gpu_num)]
        pool.close()
        pool.join()
        res = [res.get() for res in multiple_results]
    info_str = imdb.evaluate_detections_multiprocess(res)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-4):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

