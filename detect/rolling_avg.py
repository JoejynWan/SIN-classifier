import os 
import copy
import time
import json
import argparse
import numpy as np
import humanfriendly
from tqdm import tqdm
import multiprocessing as mp
from collections import deque
import ijson.backends.yajl2_c as ijson

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions, find_unique_objects 
from general.shared_utils import process_video_obj_results, json_to_csv
from detect.detect_utils import default_path_from_none, write_frame_results
from detect.vis_detections import vis_detection_videos

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
    Function to identify individual objects based on the intersection over union
    (IOU) between the bounding boxes of the current and previous frame. 

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


def rpa_calc(frames, rolling_avg_size, num_classes):

    unique_objects = find_unique_objects(frames)

    ## Create deque to store conf for each unique object
    Q_objs = []
    for unique_object in unique_objects:
        
        empty_frame = np.zeros(num_classes).tolist()
        initial_deque = [empty_frame] * 3 #start deque with 3 "empty frames" to remove any errornous first frames

        Q_obj = {
            'object_number': unique_object,
            'object_Q': deque(initial_deque, maxlen = rolling_avg_size)
        }
        Q_objs.append(Q_obj)

    ## Run rpa for frames
    for frame in frames: 
        
        detections = frame['detections']
        if detections: 
            
            # Fill in the confidence of each detection into their respective 
            # object deque
            for detection in detections: 
                det_cat = int(detection['category'])
                det_conf = detection['conf']
                det_obj_num = detection['object_number']

                filled_frame = empty_frame.copy()
                filled_frame[det_cat-1] = det_conf

                for Q_obj in Q_objs:
                    if Q_obj['object_number'] == det_obj_num:

                        # find the rolling prediction average...
                        temp_Q = Q_obj['object_Q'].copy()
                        temp_Q.append(filled_frame)
                        np_mean = np.array(temp_Q).mean(axis = 0)

                        # update the confidence for each class in Q_obj
                        Q_obj['object_Q'].append(np_mean.tolist())

                # update the conf and cat of the detection in frames
                max_conf = max(np_mean)
                max_index = np.where(np_mean == max_conf)[0].tolist()[0] + 1

                detection['conf'] = round(max_conf, 3)
                detection['category'] = str(max_index)    

    return frames


def rpa_calc_classifications(frames, rolling_avg_size, num_classes):

    unique_objects = find_unique_objects(frames)
    
    ## Create deque to store conf for each unique object
    Q_objs = []
    for unique_object in unique_objects:

        empty_frame = np.zeros(num_classes).tolist()
        initial_deque = [empty_frame] * 3 #start deque with 3 "empty frames" to remove any errornous first frames

        Q_obj = {
            'object_number': unique_object,
            'object_Q': deque(initial_deque, maxlen = rolling_avg_size)
        }
        Q_objs.append(Q_obj)
    
    ## Run rpa for frames
    for frame in frames: 
        
        detections = frame['detections']
        if detections: 
            
            for detection in detections:
                
                if 'classifications' in detection:

                    det_obj_num = detection['object_number']
                    classifications = detection['classifications']
                    
                    # Get the confidence of each species class
                    filled_frame = empty_frame.copy()
                    for classification in classifications:
                        class_cat = int(classification[0])
                        class_conf = classification[1]

                        filled_frame[class_cat] = class_conf

                    for Q_obj in Q_objs:
                        if Q_obj['object_number'] == det_obj_num:
                            
                            # find the rolling prediction average...
                            temp_Q = Q_obj['object_Q'].copy()
                            temp_Q.append(filled_frame)
                            np_mean = np.array(temp_Q).mean(axis = 0)

                            # update the confidence for each class in Q_obj
                            Q_obj['object_Q'].append(np_mean.tolist())

                    # update the confidence for each class in frames
                    for classification in classifications:
                        class_idx = int(classification[0])
                        updated_conf = round(np_mean[class_idx], 3)
                        classification[1] = updated_conf                            

            # Calculate the mean confidence for each object and replace
            # the confidence value in the detection
            # for detection in detections: 

            #     if 'classifications' in detection:
                
            #         det_obj_num = detection['object_number']
            #         classifications = detection['classifications']
            #         Q = [Q_obj['object_Q'] for Q_obj in Q_objs if Q_obj['object_number'] == det_obj_num][0]

            #         # find the rolling prediction average...
            #         np_mean = np.array(Q).mean(axis = 0)

            #         # update the confidence for each class in Q_obj
            #         for Q_obj in Q_objs:
            #             if Q_obj['object_number'] == det_obj_num:
            #                 Q_obj['object_Q'][-1] = np_mean.tolist()

            #         # update the confidence for each class in frames
            #         for classification in classifications:
            #             class_idx = int(classification[0])
            #             updated_conf = round(np_mean[class_idx], 3)
            #             classification[1] = updated_conf

    return frames


def rpa_video(options, images, video_path, 
              num_classes, species_classifications = False):
    
    images_copy = copy.deepcopy(images)

    frames = [image for image in images_copy if os.path.dirname(image['file']) == video_path]
    
    if species_classifications:
        frames = rpa_calc_classifications(frames, options.rolling_avg_size, num_classes)

    else:

        frames = id_object(frames, options.iou_threshold)
        frames = rpa_calc(frames, options.rolling_avg_size, num_classes)

    return frames


def load_detector_roll_avg(frames_json, options, species_classifications = False):

    start = time.time()
    print("\nLoading animal detections for rolling prediction averaging...")
    
    images_updated = []
    with open(frames_json, 'rb') as f:
        
        ## Load detections with a conf >= conf_threshold_limit
        ## Provide some buffer and accept detections with a conf threshold that 
        ## is conf_threshold_buf% less for rolling prediction averaging
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

            image_updated = {
                'file': raw_image['file'],
                'max_detection_conf': round(updated_max_conf, 3),
                'detections': updated_detections
            }
            images_updated.append(image_updated)

        f.seek(0)
        detection_categories = ijson.items(f, 'detection_categories', 
                                           use_float = True)
        detector_label_map = list(detection_categories)[0]

        f.seek(0)
        videos = ijson.items(f, 'videos', use_float = True)
        videos_dict = list(videos)[0]

        f.seek(0)
        info = ijson.items(f, 'info', use_float = True)
        info_dict = list(info)[0]
        
        if species_classifications:
            f.seek(0)
            classification_categories = ijson.items(f, 
                                                    'classification_categories', 
                                                    use_float = True)
            classification_label_map = list(classification_categories)[0]
        else:
            classification_label_map = None

    end = time.time()
    elapsed = end - start
    print('Detections loaded in {}'.format(
        humanfriendly.format_timespan(elapsed)
        ))

    return images_updated, detector_label_map, videos_dict, info_dict, classification_label_map


def write_roll_avg_video_results(frames_json, videos_json, csv_json, options, 
                                 mute = False):
    
    output_data = process_video_obj_results(
        frames_json,
        options.nth_highest_confidence,
        options.rendering_confidence_threshold
    )
    
    # Write the output files
    with open(videos_json,'w') as f:
        json.dump(output_data, f, indent = 1)
    
    video_pd = json_to_csv(options, videos_json)

    video_pd.to_csv(csv_json, index = False)

    if not mute:
        print('Output file saved at {}'.format(videos_json))
        print('Output file saved at {}'.format(csv_json))


def get_output_paths(options, species_classifications):

    if species_classifications:
        options.classification_roll_avg_frames_json = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.classification_roll_avg_frames_json, 
            '_classification_roll_avg_frames.json'
        )
        options.classification_roll_avg_video_json = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.classification_roll_avg_video_json, 
            '_classification_roll_avg_videos.json'
        )
        options.classification_roll_avg_video_csv = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.classification_roll_avg_video_csv, 
            '_classification_roll_avg_videos.csv'
        )

        frames_json = options.classification_roll_avg_frames_json
        video_json = options.classification_roll_avg_video_json
        video_csv = options.classification_roll_avg_video_csv

    else:
        options.roll_avg_frames_json = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.roll_avg_frames_json, '_roll_avg_frames.json'
        )
        options.roll_avg_video_json = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.roll_avg_video_json, '_roll_avg_videos.json'
        )
        options.roll_avg_video_csv = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.roll_avg_video_csv, '_roll_avg_videos.csv'
        )

        frames_json = options.roll_avg_frames_json
        video_json = options.roll_avg_video_json
        video_csv = options.roll_avg_video_csv

    return frames_json, video_json, video_csv


def rolling_avg(frames_json, options, species_classifications = False, 
                mute = False):
    
    ## Load images from full_det_frames_json
    ## Remove detections with conf <= conf_threshold_limit to save RAM memory
    images, detector_label_map, video_info, info, classification_label_map = load_detector_roll_avg(
        frames_json, options, species_classifications)
    
    ## Get video_paths and Fs
    video_paths = video_info['video_names']
    Fs = video_info['frame_rates']

    ## Get total number of classes
    if species_classifications:
        num_classes = len(classification_label_map)
    else:
        num_classes = len(detector_label_map)

    print("Conducting rolling prediction averaging...")
    roll_avg = []
    def callback_func(result):
        roll_avg.extend(result)
        pbar.update(1)

    pool = mp.Pool(options.n_cores)
    pbar = tqdm(total = len(video_paths))
    
    for video_path in video_paths: 
        pool.apply_async(
            rpa_video, args = (options, images, video_path, num_classes, 
                               species_classifications), 
            callback = callback_func
        )

    pool.close()
    pool.join()
    pbar.close()

    ## Write the output files
    frames_json, video_json, video_csv = get_output_paths(options, species_classifications)

    write_frame_results(options, roll_avg, Fs, frames_json, info = info, 
                        classification_map = classification_label_map, 
                        mute = mute)
    
    write_roll_avg_video_results(frames_json, video_json, video_csv, options, 
                                 mute = mute)

    return roll_avg


def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    frames_json = options.classification_frames_json
    options.output_files_dir = os.path.join(options.output_dir, 
                                            "SINClassifier_outputs")
    rolling_avg(frames_json, options, species_classifications = True)

    # vis_detection_videos(options)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to implement rolling prediction averaging on '
                    'results from MegaDetector.')
    parser.add_argument(
        '--output_dir', type=str,
        default = config.OUTPUT_DIR, 
        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument(
        '--input_dir', type=str, 
        default = config.INPUT_DIR, 
        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument(
        '--full_det_frames_json', type=str,
        default = config.FULL_DET_FRAMES_JSON, 
        help = 'Path to json file containing the frame-level results of all '
               'MegaDetector detections.'
    )
    parser.add_argument(
        '--frame_folder', type = str,
        default = config.FRAME_FOLDER,
        help = 'Path to folder containing the video frames processed by '
               'det_videos.py'
    )
    parser.add_argument(
        '--rendering_confidence_threshold', type=float,
        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
        help = 'Do not render boxes with confidence below this threshold'
    )
    parser.add_argument(
        '--rolling_avg_size', type=float,
        default = config.ROLLING_AVG_SIZE, 
        help = 'Number of frames in which rolling prediction averaging will be '
               'calculated from. A larger number will remove more '
               'inconsistencies, but will cause delays in detecting an object'
    )
    parser.add_argument(
        '--iou_threshold', type=float,
        default = config.IOU_THRESHOLD, 
        help = 'Threshold for Intersection over Union (IoU) to determine if '
               'bounding boxes are detecting the same object'
    )
    parser.add_argument(
        '--conf_threshold_buf', type=float,
        default = config.CONF_THRESHOLD_BUF, 
        help = 'Buffer for the rendering confidence threshold to allow for '
               '"poorer" detections to be included in rolling prediction '
               'averaging'
    )
    parser.add_argument(
        '--nth_highest_confidence', type=float,
        default = config.NTH_HIGHEST_CONFIDENCE, 
        help = 'nth-highest-confidence frame to choose a confidence value for '
               'each video'
    )
    parser.add_argument(
        '--check_accuracy', type=bool,
        default = config.CHECK_ACCURACY, 
        help = 'Whether accuracy of MegaDetector should be checked with manual '
               'ID. Folder names must contain species and quantity'
    )
    parser.add_argument(
        '--n_cores', type=int,
        default = config.N_CORES, 
        help = 'number of cores to use for detection and cropping (CPU only)'
    )
    parser.add_argument(
        '--classification_frames_json', type=str,
        default = config.CLASSIFICATION_FRAMES_JSON, 
        help = 'Path to json file containing the frame-level detection results '
               'after merging with species classification results'
    )
    parser.add_argument(
        '--classification_roll_avg_frames_json', type=str,
        default = config.CLASSIFICATION_ROLL_AVG_FRAMES_JSON, 
        help = 'Path to json file containing the frame-level detection results '
               'after merging with species classification results and '
               'conducting rolling prediction averaging'
    )
    parser.add_argument(
        '-c', '--classifier_categories', type=str,
        default = config.CLASSIFIER_CATEGORIES, 
        help = 'path to JSON file for classifier categories. If not given, '
               'classes are numbered "0", "1", "2", ...'
    )
    return parser


if __name__ == '__main__':

    main()
