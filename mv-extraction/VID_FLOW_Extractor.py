import os
import numpy as np
import cv2
from glob import glob
import pickle

_IMAGE_SIZE = 256
VAL_OR_TRAIN = 0 # 0 for train, 1 for validation

def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow

def compute_TVL1(prev, curr, bound=15):
    """
        Compute the TV-L1 optical flow.
    """
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    # flow = (flow + bound) * (255.0 / (2*bound))
    # flow = np.round(flow).astype(int)
    # flow[flow >= 255] = 255
    # flow[flow <= 0] = 0

    return flow
    
def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)), flow[:, :, 1])

def extract_flow(video_path,flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return

def get_frame_segment_id_collection():
    if VAL_OR_TRAIN == 0:
        handler = open('VID_train_15frames_1.txt', 'r')
    else:
        handler = open('VID_val_videos_eval.txt', 'r')

    lines = handler.readlines()
    splits = []
    for line in lines:
        splits.append(line[:-1].split(' '))

    frame_segment_id_collection = {}
    for split in splits:
        if VAL_OR_TRAIN == 0:
            path = split[0]
        else:
            path = split[0][:-7]
        
        if path not in frame_segment_id_collection:
            frame_segment_id_collection[path] = []

        if VAL_OR_TRAIN == 0:
            frame_segment_id_collection[path].append(split[2])
        else:
            frame_segment_id_collection[path].append(int(split[0][-6:]))

    return frame_segment_id_collection

def mv_extraction_per_video_collection(path_to_video_directory, target_directory, collection):
    for idx in collection :
        idx_int = int(idx)
        group_idx = idx_int // 12
        frame_idx = idx_int % 12
        # try:
        print("path_to_video:!!!" + path_to_video_directory + '  ind: ' + str(idx_int))
        
        frames = glob(os.path.join(path_to_video_directory, '*.JPEG'))
        frames.sort()
            
        prev = cv2.imread(frames[group_idx])
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        # print(prev.shape)
        curr = cv2.imread(frames[frame_idx])
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # print(curr.shape)
        flow = compute_TVL1(prev, curr)
        flow = flow.astype('int8')
        flow_path = '%06d' % idx_int
        flow_path = target_directory + '/' + flow_path + '.pkl'
        print(flow_path)
        pickle.dump(flow, open(flow_path, 'wb'), protocol=2)
        # except:
        #     print("Error\n\n\n\n\n")
        #     exit(0)
        # loaded_mv = load_mv(mv_path)
        # assert (loaded_mv == mv).all()

def mv_extraction_train_part(frame_segment_id_collection):
    count = 0
    for path in frame_segment_id_collection:
        path_to_video_directory = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/Data/VID/' + path
        collection = frame_segment_id_collection[path]

        # Define path_to_target_directory
        # target_directory = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/MV/VID/' + path
        # target_directory = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/Res/VID/' + path
        target_directory = '/home/ssd1T_2/boyuan/ImageNetVID/ILSVRC2015/flow/VID/' + path

        # target_directory may not exist. If so, we create one.
        if not os.path.isdir(target_directory):
            os.system("mkdir -p " + target_directory)

        # Conduct the functionality.
        try:
            mv_extraction_per_video_collection(path_to_video_directory, target_directory, collection)
        except:
            print(collection)
            print("Error: path_to_video_directory: " + path_to_video_directory + ", target_directory: " + target_directory + ", collection: " + collection + "\n\n\n")

        # Report Progress
        count += 1
        print("Progress: " + str(count) + "/" + str(len(frame_segment_id_collection)) + '\n\n')

error_recorder = []

def wrapper_collection():
    frame_segment_id_collection = get_frame_segment_id_collection()
    mv_extraction_train_part(frame_segment_id_collection)
    print(error_recorder)

wrapper_collection()

# if __name__ =='__main__':

#     video_paths="/home/xueqian/bishe/extrat_feature/output"
#     flow_paths="/home/xueqian/bishe/extrat_feature/flow"
#     video_lengths = 109 

#     extract_flow(video_paths, flow_paths)
