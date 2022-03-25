import os 
import copy
import argparse
import numpy as np
from collections import deque

# Functions imported from this project
import config
from shared_utils import find_unique_videos, write_json_file, find_unique_objects
from shared_utils import VideoOptions, default_path_from_none
from shared_utils import write_frame_results, write_roll_avg_video_results
from vis_detections import load_detector_output, vis_detection_videos
from run_det_video import write_frame_results

# Imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
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
    NOT USED.
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


def rm_bad_detections(options, images):
    # provide some buffer and accept detections with a conf threshold 
    # conf_threshold_buf% less for rolling prediction averaging
    conf_threshold_limit = options.conf_threshold_buf * options.rendering_confidence_threshold 

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
    video_paths = find_unique_videos(images)

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


def add_object_number(videos, iou_threshold):
    
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
                        
                        detection["object_number"] = "object_" + str(object_num).zfill(2)

                        objects.append(detection)
                        
                        # print("Creating new object {}".format(detection["object_number"]))                         
                    
                    else:
                            
                        # Check if the detection is an alr recorded object
                        det_categorised = False
                        for object in objects: 
                            object_number = object["object_number"]
                            bbox_object = object['bbox']
                            bbox_detection = detection['bbox']
                            
                            iou = bbox_iou(bbox_object, bbox_detection)
                            
                            # Bounding boxes overlap significantly, 
                            # and so it is the same object
                            if iou >= iou_threshold:
                                detection["object_number"] = object_number

                                object = detection
                                
                                # print("Same object, which is {}".format(detection["object_number"]))
                                det_categorised = True

                                break #break the objects for loop
                        
                        if not det_categorised:
                            # Since the detection is NOT an alr recorded object, 
                            # add in a new object
                            object_num = object_num + 1
                            detection["object_number"] = "object_" + str(object_num).zfill(2)

                            objects.append(detection)
                            
                            # print("Creating new object {}".format(detection["object_number"]))                           
                                
    return updated_videos


def rolling_pred_avg(objects, rolling_avg_size):

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


def remove_video_layer(roll_avg):
    images_only = []
    for video in roll_avg:
        images_only = images_only + video['images']

    return images_only


def rolling_avg(options, images, Fs):
    
    images = rm_bad_detections(options, images)
    output_images = copy.deepcopy(images)

    videos = add_video_layer(images)
    output_videos = copy.deepcopy(videos)

    objects = add_object_number(videos, options.iou_threshold)
    output_objects = copy.deepcopy(objects)

    roll_avg = rolling_pred_avg(objects, options.rolling_avg_size)
    output_roll_avg = copy.deepcopy(roll_avg)

    roll_avg_images = remove_video_layer(roll_avg)

    ## Write the output files
    options.roll_avg_frames_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.roll_avg_frames_json, '_roll_avg_frames.json'
    )

    write_frame_results(roll_avg_images, Fs, options.roll_avg_frames_json, options.frame_folder)
    write_roll_avg_video_results(options)

    return roll_avg_images, output_images, output_videos, output_objects, output_roll_avg


def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    images_full, detector_label_map, Fs = load_detector_output(options.full_det_frames_json)

    roll_avg, output_images, output_videos, output_objects, output_roll_avg = rolling_avg(options, images_full, Fs)

    write_roll_avg_video_results(options)

    vis_detection_videos(options)

    ## Save outputs for checking
    output_file = os.path.splitext(
        options.full_det_frames_json)[0] + '_conf_' + str(options.rendering_confidence_threshold) + '.json'
    write_results_to_file(output_images, output_file)

    videos_path = os.path.join(options.output_dir, "CT_models_test_videos.json")
    write_json_file(output_videos, videos_path)

    objects_path = os.path.join(options.output_dir, "CT_models_test_objects.json")
    write_json_file(output_objects, objects_path)

    roll_avg_path = os.path.join(options.output_dir, "CT_models_test_videos_rolling_avg.json")
    write_json_file(output_roll_avg, roll_avg_path)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to implement rolling prediction averaging on results from MegaDetector.')
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--frame_folder', type = str,
                        default = config.FRAME_FOLDER,
                        help = 'Path to folder containing the video frames processed by det_videos.py'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--rolling_avg_size', type=float,
                        default = config.ROLLING_AVG_SIZE, 
                        help = "Number of frames in which rolling prediction averaging will be calculated from. A larger number will remove more inconsistencies, but will cause delays in detecting an object."
    )
    parser.add_argument('--iou_threshold', type=float,
                        default = config.IOU_THRESHOLD, 
                        help = "Threshold for Intersection over Union (IoU) to determine if bounding boxes are detecting the same object."
    )
    parser.add_argument('--conf_threshold_buf', type=float,
                        default = config.CONF_THRESHOLD_BUF, 
                        help = "Buffer for the rendering confidence threshold to allow for 'poorer' detections to be included in rolling prediction averaging."
    )
    parser.add_argument('--nth_highest_confidence', type=float,
                        default = config.NTH_HIGHEST_CONFIDENCE, 
                        help="nth-highest-confidence frame to choose a confidence value for each video"
    )
    return parser


if __name__ == '__main__':

    main()