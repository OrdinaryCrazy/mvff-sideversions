import numpy as np
import copy
import os
import pickle

DEBUG = 0

def avg_mv(mv_full, xmin, ymin, xmax, ymax):
    # mv_full is a tensor of H x W x 2
    xmax = int(xmax)
    ymax = int(ymax)
    xmin = int(xmin)
    ymin = int(ymin)

    selected_area_x = mv_full[ymin:ymax, xmin:xmax, 0]
    selected_area_y = mv_full[ymin:ymax, xmin:xmax, 1]
    avg_x = np.mean(selected_area_x)
    avg_y = np.mean(selected_area_y)

    return avg_x, avg_y

def bbox_motion(xmin, ymin, xmax, ymax, mv_x, mv_y):
    xmin = xmin + mv_x
    xmax = xmax + mv_x
    ymin = ymin + mv_y
    ymax = ymax + mv_y

    return xmin, ymin, xmax, ymax

def idx_file_mapping():
    root = '/home/ssd1T_1/boyuan/ImageNetVID/ILSVRC2015/MV/VID/'
    handler = open('../VID_val_videos_eval.txt', 'r')
    lines = handler.readlines()
    
    mapping = {}
    for line in lines:
        # LINE = 'val/ILSVRC2015_val_00000000/000000 1\n'
        line_split = line[:-1].split(' ')
        mapping[line_split[1]] = root + line_split[0]

    return mapping

# Pred file
# line_split = [1, 2, 0.0002, 694.72, 41.69, 1262.16, 556.21]
def propagate_single_key_line(line_split, mv_full, new_frame_id):
    # Assume current line belongs to a key frame

    xmin = int(float(line_split[3]))
    ymin = int(float(line_split[4]))
    xmax = int(float(line_split[5]))
    ymax = int(float(line_split[6]))

    avg_x, avg_y = avg_mv(mv_full, xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = bbox_motion(xmin, ymin, xmax, ymax, avg_x, avg_y)

    new_line = line_split[:]
    new_line[0] = new_frame_id
    new_line[3] = str(xmin)
    new_line[4] = str(ymin)
    new_line[5] = str(xmax)
    new_line[6] = str(ymax)

    new_line = ' '.join(new_line) + '\n'
    return new_line

mv_store = {}
val_mv_not_found_count = 0

def load_mv(path_to_mv):
    path_to_mv += '.pkl'
    global val_mv_not_found_count
    if not os.path.exists(path_to_mv):
        val_mv_not_found_count += 1
        mv = np.zeros(2*2000*4000).reshape((2000,4000,2))
        print('train_mv_not_found_count: ', val_mv_not_found_count, ', path_to_mv: ', path_to_mv)
    else:
        mv = pickle.load(open(path_to_mv, 'rb'))
    return mv

def propagate_key_lines(key_frame_predictions, mapping, num_image_in_group):
    if DEBUG:
        print('DEBUG: Enter propagate_key_lines')

    global mv_store
    key_frame_global_id = int(key_frame_predictions[0][0])

    group_collection = [] # Containing propagated lines of non-key frames
    for group_idx in range(1,num_image_in_group+1):
        if DEBUG:
            print('DEBUG: propagate_key_lines, group_idx: ' + str(group_idx))
        # Load mv
        if key_frame_global_id + group_idx >= 176127:
            break
        file_path = mapping[str(key_frame_global_id+group_idx)]
        if file_path in mv_store:
            mv_full = mv_store[file_path]
        else:
            mv_store.clear()
            # TODO
            mv_full = load_mv(file_path)
            mv_store[file_path] = mv_full


        for key_line_idx in range(len(key_frame_predictions)):
            new_frame_global_id = str(key_frame_global_id + group_idx)
            line_split = key_frame_predictions[key_line_idx]
            processed_new_line = propagate_single_key_line(line_split, mv_full, new_frame_global_id)
            group_collection.append(processed_new_line)

    return group_collection

def get_num_image_in_group(lines, startingIdx, mapping):
    if DEBUG:
        print('DEBUG: Enter get_num_image_in_group()')

    key_frame_line = lines[startingIdx]
    line_split = key_frame_line[:-1].split(' ')
    key_frame_global_id = int(line_split[0])
    key_frame_path = mapping[str(key_frame_global_id)]
    key_frame_local_id = int(key_frame_path.split('/')[-1])

    num_image_in_group = 0

    for idx in range(10000):
        if startingIdx + idx < 19614314:
            group_line = lines[startingIdx + idx]
            group_line_split = group_line[:-1].split(' ')
            group_line_global_id = int(group_line_split[0])        
            group_line_path = mapping[str(group_line_global_id)]
            group_line_local_id = int(group_line_path.split('/')[-1])
            if group_line_local_id < key_frame_local_id:
                return num_image_in_group
            else:
                num_image_in_group = group_line_local_id - key_frame_local_id
                if num_image_in_group == 11:
                    return 11
        else:
            return num_image_in_group
    
def process():
    mapping = idx_file_mapping()
    output_collection = []
    handler = open('det_VID_val_videos_all.txt', 'r')
    lines = handler.readlines()

    frame_local_id = None

    key_frame_predictions = []

    last_line_local_id = -1

    # line_split = [1, 2, 0.0002, 694.72, 41.69, 1262.16, 556.21]
    for line_idx in range(len(lines)):
        if line_idx % 100000 == 0:
            print('Progress: '+str(line_idx) + '/' + str(len(lines)))
        line = lines[line_idx]
        line_split = line[:-1].split(' ')
        file_path = mapping[line_split[0]]
        frame_local_id = int(file_path.split('/')[-1])
        if frame_local_id % 12 == 0:
            # Key frame
            # Collect key frame predictions.
            confidence = float(line_split[2])
            if confidence < 0.005:
                continue
            key_frame_predictions.append(line_split)
            output_collection.append(line)
        else:
            # Non-key frame.
            if last_line_local_id % 12 == 0:
                # The first non-key after key frame.
                # First count the num_image_in_group
                num_image_in_group = get_num_image_in_group(lines, line_idx, mapping)
                # key_frame_predictions should contain all predictions from the key frame. 
                #                Do propagation for the following 11 non-key frames
                group_collection = propagate_key_lines(key_frame_predictions, mapping, num_image_in_group)
                output_collection += group_collection
                # Clear up 
                key_frame_predictions[:] = []

        last_line_local_id = frame_local_id

    out_file = open('propagated_results', 'w')
    out_file.writelines(output_collection)


process()
