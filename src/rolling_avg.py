import os 
import copy
import time
import argparse
import numpy as np
import humanfriendly
from tqdm import tqdm
import multiprocessing as mp
from collections import deque
import ijson.backends.yajl2_c as ijson

# Functions imported from this project
import config
from shared_utils import find_unique_objects, VideoOptions, default_path_from_none
from shared_utils import write_frame_results, write_roll_avg_video_results
from vis_detections import vis_detection_videos

# Imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def limit_xy_min(coord):
    if coord < 0:
        coord = 0
    return coord


def limit_wh_rel(coord):
    if coord > 1:
        coord = 1
    return coord


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
    
    ## Provide some buffer and accept detections with a conf threshold that is
    ## conf_threshold_buf% less for rolling prediction averaging
    ## DEPRECIATED WITH LOAD_DETECTOR_ROLL_AVG()
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


def id_object(frames, iou_threshold):
    '''
    Function to identify individual objects based on the intersection over union (IOU) between 
    the bounding boxes of the current and previous frame. 

    Args:
    frames: list of frames from one video
    '''
    objects = []
    object_num = 1
    for frame in frames: 
        
        detections = frame['detections']
        
        if detections:
            for detection in detections:
                
                #first detection of the video is always a new unique object
                if not objects: 
                    
                    detection['object_number'] = "object_" + str(object_num).zfill(2)
                    objects.append(detection)
                                           
                else:
                        
                    # Check if the detection is an alr recorded object
                    det_categorised = False
                    for object in objects: 
                        object_number = object['object_number']
                        bbox_object = object['bbox']
                        bbox_detection = detection['bbox']
                        
                        iou = bbox_iou(bbox_object, bbox_detection)
                        
                        # Same object if bounding boxes overlap significantly
                        if iou >= iou_threshold:
                            
                            detection['object_number'] = object_number
                            object = detection
                            
                            det_categorised = True

                            break #break the objects for loop
                    
                    # Add in new object since the detection is NOT an alr recorded object
                    if not det_categorised:
                        
                        object_num = object_num + 1
                        detection['object_number'] = "object_" + str(object_num).zfill(2)
                        objects.append(detection)

    return frames


def rpa_calc(frames, rolling_avg_size):

    unique_objects = find_unique_objects(frames)

    ## Create deque to store conf for each unique object
    Q_objs = []
    for unique_object in unique_objects:
        
        initial_deque = [[0,0,0]] * 3 #start deque with 3 "empty frames" to remove any errornous first frames

        Q_obj = {
            'object_number': unique_object,
            'object_Q': deque(initial_deque, maxlen = rolling_avg_size)
        }
        Q_objs.append(Q_obj)

    ## Run rpa for frames
    for frame in frames: 
        
        # Pre-allocate a confidence of zero for all classes for all objects.
        # This means that for objects without a detection in this frame, 
        # the confidence of the classes will be set to 0. 
        for Q_obj in Q_objs:
            Q_obj['object_Q'].append([0,0,0])
        
        detections = frame['detections']

        if detections: 
            
            # Fill in the confidence of each detection into their respective 
            # object deque
            for detection in detections: 
                det_cat = int(detection['category'])
                det_conf = detection['conf']
                det_obj_num = detection['object_number']

                Q = [Q_obj['object_Q'] for Q_obj in Q_objs if Q_obj['object_number'] == det_obj_num][0]
                Q[-1][det_cat-1] = det_conf

            # Calculate the mean confidence for each object and replace
            # the confidence value in the detection
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

    return frames


def rpa_video(options, images, video_path):
    
    images_copy = copy.deepcopy(images)

    frames = [image for image in images_copy if os.path.dirname(image['file']) == video_path]
    
    frames = id_object(frames, options.iou_threshold)

    frames = rpa_calc(frames, options.rolling_avg_size)

    return frames


def load_detector_roll_avg(options):

    start = time.time()
    print("\nLoading animal detections from full_det_frames.json for RPA now...")
    
    images = []
    detector_label_map = {}
    video_paths = []
    Fs = []
    with open(options.full_det_frames_json, 'rb') as f:
        
        ## Load detections with a conf >= conf_threshold_limit
        ## Provide some buffer and accept detections with a conf threshold that is
        ## conf_threshold_buf% less for rolling prediction averaging
        conf_threshold_limit = options.conf_threshold_buf * options.rendering_confidence_threshold 
        raw_images = ijson.items(f, 'images.item', use_float = True)
        for raw_image in raw_images:

            updated_detections = []
            updated_max_conf_list = []
            for detection in raw_image['detections']:
                conf_score = detection['conf']
                if conf_score >= conf_threshold_limit:
                    updated_detections.append(detection)
                    updated_max_conf_list.append(conf_score)

            if updated_max_conf_list: 
                updated_max_conf = max(updated_max_conf_list)
            else: #updated_max_conf list is empty
                updated_max_conf = 0

            image = {
                'file': raw_image['file'],
                'max_detection_conf': updated_max_conf,
                'detections': updated_detections
            }
            images.append(image)
    
        f.seek(0) #Reads the file twice. Not optimal, but can't think of a way around this. 
        parse_events = ijson.parse(f, use_float = True)
        for prefix, event, value in parse_events:

            ## Load detector_label_map. Only works with 3 categories. 
            if prefix == 'detection_categories.1':
                detector_label_map['1'] = value
            elif prefix == 'detection_categories.2':
                detector_label_map['2'] = value
            elif prefix == 'detection_categories.3':
                detector_label_map['3'] = value
            ## Load unique video names/path for each video
            elif prefix == 'videos.video_names.item':
                video_paths.append(value)
            ## Load frame rates for each video
            elif prefix == 'videos.frame_rates.item':
                Fs.append(value)

    end = time.time()
    elapsed = end - start
    print('Detections loaded in {}'.format(humanfriendly.format_timespan(elapsed)))

    return images, detector_label_map, video_paths, Fs


def rolling_avg(options, mute = False):
    
    ## Load images from full_det_frames_json
    ## Remove detections with conf <= conf_threshold_limit to save RAM memory
    images, detector_label_map, video_paths, Fs = load_detector_roll_avg(options)
    
    ## Conduct RPA
    print("Conducting rolling prediction averaging now...")

    roll_avg = []
    def callback_func(result):
        roll_avg.extend(result)
        pbar.update(1)

    pool = mp.Pool(mp.cpu_count())
    pbar = tqdm(total = len(video_paths))

    for video_path in video_paths: 
        pool.apply_async(
            rpa_video, args = (options, images, video_path), callback = callback_func
        )

    pool.close()
    pool.join()
    pbar.close()

    ## Write the output files
    options.roll_avg_frames_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.roll_avg_frames_json, '_roll_avg_frames.json'
    )

    write_frame_results(
        roll_avg, Fs, 
        options.roll_avg_frames_json, mute = mute)
    write_roll_avg_video_results(options, mute = mute)

    return roll_avg


def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    rolling_avg(options)

    # vis_detection_videos(options)


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
    parser.add_argument('--full_det_frames_json', type=str,
                        default = config.FULL_DET_FRAMES_JSON, 
                        help = 'Path to json file containing the frame-level results of all MegaDetector detections.'
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
    parser.add_argument('--check_accuracy', type=bool,
                        default = config.CHECK_ACCURACY, 
                        help = 'Whether accuracy of MegaDetector should be checked with manual ID. Folder names must contain species and quantity.'
    )
    return parser


if __name__ == '__main__':

    main()
