import os 
import copy
from tqdm import tqdm
import numpy as np
from collections import deque

from shared_utils import find_unqiue_videos, write_json_file, find_unique_objects
from vis_detections import load_detector_output, vis_detection_video
from detection.run_tf_detector_batch import write_results_to_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def limit_xy_min(coord):
    if coord < 0:
        coord = 0
    return coord


def limit_wh_rel(coord):
    if coord > 1:
        coord = 1
    return coord


def bbbox_buffer(bbox_shape, buffer_scale):
    """
    bbox_shape = list, containing x_min, y_min, w_rel, h_rel. 
        These values are from 0 to 1
        x_min and y_min denotes the position of the top left corner
        w_rel denotes the length of the width proportional to the width of the image
        h_rel denotes the length of the height proportional to the height of the image
    
    buffer_scale = float, size of the buffer. 
        E.g., 1.1 denotes that the buffer will be 10% larger than the bounding box
    """

    x_min, y_min, w_rel, h_rel = bbox_shape
    w_rel_new = w_rel * buffer_scale
    x_min_new = x_min - (w_rel_new - w_rel)
    h_rel_new = h_rel * buffer_scale
    y_min_new = y_min - (h_rel_new - h_rel)

    x_min_buf = limit_xy_min(x_min_new)
    y_min_buf = limit_xy_min(y_min_new)
    w_rel_buf = limit_wh_rel(w_rel_new)
    h_rel_buf = limit_wh_rel(h_rel_new)

    buf_shape = [x_min_buf, y_min_buf, w_rel_buf, h_rel_buf]
    return buf_shape


def bbox_iou(bbox_1, bbox_2):
    """
    Function to calculate Intersection over Union (IoU), where 
        IoU = Area of overlap / Area of union

    Arguments:
    bbox_1; bbox_2 = list, containing x_min, y_min, w_rel, h_rel. 
        These values are from 0 to 1
        x_min and y_min denotes the position of the top left corner
        w_rel denotes the length of the width proportional to the width of the image
        h_rel denotes the length of the height proportional to the height of the image
    """
    # determine the coordinates of bounding boxes 1 and 2
    bb1_x_min, bb1_y_min, bb1_w_rel, bb1_h_rel = bbox_1
    bb1_y_max = bb1_y_min + bb1_h_rel
    bb1_x_max = bb1_x_min + bb1_w_rel
    bb2_x_min, bb2_y_min, bb2_w_rel, bb2_h_rel = bbox_2
    bb2_y_max = bb2_y_min + bb2_h_rel
    bb2_x_max = bb2_x_min + bb2_w_rel

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1_x_min, bb2_x_min)
    y_top = max(bb1_y_min, bb2_y_min)
    x_right = min(bb1_x_max, bb2_x_max)
    y_bottom = min(bb1_y_max, bb2_y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0 #no intersection

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1_x_max - bb1_x_min) * (bb1_y_max - bb1_y_min)
    bb2_area = (bb2_x_max - bb2_x_min) * (bb2_y_max - bb2_y_min)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert 0.0 <= iou <= 1.0
    
    return iou


def rm_bad_detections(images, conf_threshold, conf_threshold_buf):
    # provide some buffer and accept detections with a conf threshold 
    # conf_threshold_buf% less for rolling prediction averaging
    conf_threshold_limit = conf_threshold_buf * conf_threshold 

    for image in images: 
        detections = image['detections']

        updated_detections = []
        updated_max_conf = []
        for detection in detections:
            conf_score = detection['conf']
            if conf_score >= conf_threshold_limit:
                updated_detections.append(detection)
                updated_max_conf.append(conf_score)

        image['detections'] = updated_detections
        if updated_max_conf: 
            image['max_detection_conf'] = max(updated_max_conf)
        else: #updated_max_conf list is empty
            image['max_detection_conf'] = 0
    
    return images


def add_video_layer(images):
    """
    Converts the list of images into a list of videos by adding an 
    additional "layer" above the images "layer". 
    Each item in the video list is a dictionary containing:
        "video" = 'name of the video'
        "images" = 'details of each frame of that video'
    """
    video_paths = find_unqiue_videos(images)

    videos = []
    for video_path in video_paths:
        frames = []
        for image in images:
            frame_path = os.path.dirname(image['file'])
            if frame_path == video_path:
                frames.append(image)
        video = {
            'video': video_path,
            'images': frames}
        videos.append(video)

    return videos


def add_object_number(videos):
    
    updated_videos = copy.copy(videos)

    for video in updated_videos:
        
        frames = video['images']
        objects = []
        object_num = 1
        for frame in frames:
            # print(frame['file'])
            detections = frame['detections']
            
            if detections:
                for det_idx, detection in enumerate(detections):
                    
                    # print('Detection number {}'.format(det_idx))
                    #first detection of the video is always a new unique object
                    if not objects: 
                        
                        detection["object_number"] = "object_" + str(object_num)

                        objects.append(detection)
                        
                        # print("Creating new object {}".format(detection["object_number"]))                         
                    
                    else:
                            
                        # Check if the detection is an alr recorded object
                        det_categorised = False
                        for i in range(len(objects)): 
                            object_number = objects[i]["object_number"]
                            bbox_object = objects[i]['bbox']
                            bbox_detection = detection['bbox']
                            
                            iou = bbox_iou(bbox_object, bbox_detection)
                            
                            # Bounding boxes overlap significantly, 
                            # and so it is the same object
                            if iou >= iou_threshold:
                                detection["object_number"] = object_number

                                objects[i] = detection
                                
                                # print("Same object, which is {}".format(detection["object_number"]))
                                det_categorised = True

                                break #break the objects for loop
                        
                        if not det_categorised:
                            # Since the detection is NOT an alr recorded object, 
                            # add in a new object
                            object_num = object_num + 1
                            detection["object_number"] = "object_" + str(object_num)

                            objects.append(detection)
                            
                            # print("Creating new object {}".format(detection["object_number"]))                           
                                
    return updated_videos


def rolling_pred_avg(objects):

    updated_objects = copy.copy(objects)
    
    for video in updated_objects: 
        
        frames = video['images']

        unique_objects = find_unique_objects(frames)
        Q_objs = []
        for unique_object in unique_objects:
            Q_obj = {
                'object_number': unique_object,
                'object_Q': deque(maxlen = rolling_avg_size)
            }
            Q_objs.append(Q_obj)

        for frame in frames: 
            
            for Q_obj in Q_objs:
                Q_obj['object_Q'].append([0,0,0])
            
            detections = frame['detections']

            if detections: 
                for detection in detections: 
                    det_cat = int(detection['category'])
                    det_conf = detection['conf']
                    det_obj_num = detection['object_number']

                    Q = [Q_obj['object_Q'] for Q_obj in Q_objs if Q_obj['object_number'] == det_obj_num][0]
                    Q[-1][det_cat-1] = det_conf
                
                for detection in detections: 
                    det_obj_num = detection['object_number']
                    Q = [Q_obj['object_Q'] for Q_obj in Q_objs if Q_obj['object_number'] == det_obj_num][0]

                    # find the rolling prediction average...
                    np_mean = np.array(Q).mean(axis = 0)

                    # update the conf and cat of the detection 
                    max_conf = max(np_mean)
                    max_index = np.where(np_mean == max_conf)[0].tolist()[0] + 1

                    detection['conf'] = max_conf
                    detection['category'] = str(max_index)                
    
    return updated_objects


def main():

    frames_json = os.path.join(output_dir, "CT_models_test_frames_det.json")
    images_full, detector_label_map = load_detector_output(frames_json)

    conf_threshold_buf = 0.7
    images = rm_bad_detections(images_full, conf_threshold, conf_threshold_buf)
    output_file = os.path.splitext(frames_json)[0] + '_conf_' + str(conf_threshold) + '.json'
    write_results_to_file(images, output_file)

    videos = add_video_layer(images)
    videos_path = os.path.join(output_dir, "CT_models_test_videos.json")
    write_json_file(videos, videos_path)

    objects = add_object_number(videos)
    objects_path = os.path.join(output_dir, "CT_models_test_objects.json")
    write_json_file(objects, objects_path)

    roll_avg = rolling_pred_avg(objects)
    roll_avg_path = os.path.join(output_dir, "CT_models_test_videos_rolling_avg.json")
    write_json_file(roll_avg, roll_avg_path)

    for video in tqdm(roll_avg): 
        video_name = video['video']
        video_Fs = 30
        images_set = video['images']

        vis_detection_video(images_set, detector_label_map, frames_dir, 
            conf_threshold, output_dir, video_name, video_Fs)


if __name__ == '__main__':
    ## Arguments
    output_dir = "results/CT_models_test_2"
    frames_dir = "results/CT_models_test_2/video_frames"
    conf_threshold = 0.8
    iou_threshold = 0.5
    rolling_avg_size = 32

    main()