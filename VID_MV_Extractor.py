import os
import coviar
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import pickle

def img2video(path_to_directory, target_directory):
    os.system('ffmpeg -framerate 30 -pattern_type glob -i ' + path_to_directory + '/\'*.JPEG\' -c:v libx264 -pix_fmt yuv420p ' + target_directory + '/out.mp4')
    os.system('ffmpeg -i ' + target_directory + '/out.mp4 -c:v mpeg4 -f rawvideo ' + target_directory + '/output.mp4')
    os.system('rm ' + target_directory + '/out.mp4')

def load_mv(path_to_mv):
    mv = pickle.load(open(path_to_mv, 'rb'))
    return mv

def get_frame_segment_id_collection():
    handler = open('VID_train_15frames_1.txt', 'r')
    # handler = open('VID_val_videos_eval.txt', 'r')

    lines = handler.readlines()

    splits = []
    for line in lines:
        splits.append(line[:-1].split(' '))

    frame_segment_id_collection = {}
    for split in splits:
        path = split[0]
        # path = split[0][:-7]
        if path not in frame_segment_id_collection:
            frame_segment_id_collection[path] = []
        frame_segment_id_collection[path].append(split[2])
        # frame_segment_id_collection[path].append(int(split[0][-6:]))

    return frame_segment_id_collection

error_recorder = []

def video2mv_collection(path_to_video, target_directory, collection):
    # num_group = coviar.get_num_gops(path_to_video)
    for idx in collection:
        idx_int = int(idx)
        group_idx = idx_int // 12
        frame_idx = idx_int % 12 + 5
        try:
            print("path_to_video:!!!" + path_to_video + '  ind: ' + str(idx_int))
            mv = coviar.load(path_to_video, group_idx, frame_idx, 2, True)
            mv = mv.astype('int8')
            mv_path = '%06d' % idx_int
            mv_path = target_directory + '/' + mv_path + '.pkl'
            pickle.dump(mv, open(mv_path, 'wb'), protocol=2)
        except:
            print("Error\n\n\n\n\n")
            error_recorder.append(mv_path)
            return

        loaded_mv = load_mv(mv_path)
        assert (loaded_mv == mv).all()

def mv_extraction_per_video_collection(path_to_video_directory, target_directory, collection):
    path_to_video = target_directory + '/output.mp4'

    img2video(path_to_video_directory, target_directory)
    video2mv_collection(path_to_video, target_directory, collection)

    os.system('rm '+path_to_video)

def mv_extraction_train_part(frame_segment_id_collection):
    count = 0
    for path in frame_segment_id_collection:
        path_to_video_directory = '/home/ssd1T_2/boyuan/ImageNetVID/ILSVRC2015/Data/VID/' + path
        collection = frame_segment_id_collection[path]

        # Define path_to_target_directory
        # target_directory = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/MV/VID/' + path
        target_directory = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/Res/VID/' + path
        # target_directory = '/home/jingtun/Res/VID/' + path

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

def wrapper_collection():
    frame_segment_id_collection = get_frame_segment_id_collection()
    mv_extraction_train_part(frame_segment_id_collection)
    print(error_recorder)

wrapper_collection()