import os 
from tqdm import tqdm
import numpy as np
from collections import deque

from shared_utils import find_unqiue_videos, write_json_file
from vis_detections import load_detector_output
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


def add_object_layer(videos):

    for vid_idx, video in tqdm(enumerate(videos)):
        
        frames = video['images']
        objects = []
        for frame_idx, frame in enumerate(frames):
            
            detections = frame['detections']
            if detections:
                for det_idx, detection in enumerate(detections):
                    
                    object_num = 1
                    det_categorised = False
                    while not det_categorised:
                        
                        #first detection of the video is always a new unique object
                        if not objects: 
                            object_dict = {
                                "object_number": "object_" + str(object_num),
                                "details": detection
                            }
                            
                            # save detection as a new object
                            objects.append(object_dict)

                            # update the detection with object number
                            videos[vid_idx]['images'][frame_idx]['detections'][det_idx] = object_dict

                            det_categorised = True
                        
                        else:
                            for object in objects: 
                                object_number = object["object_number"]
                                bbox_object = object["details"]['bbox']
                                bbox_detection = detection['bbox']
                                
                                iou = bbox_iou(bbox_object, bbox_detection)
                                
                                # Bounding boxes overlap significantly, 
                                # and so it is the same object
                                if iou >= iou_threshold:
                                    
                                    object_dict = {
                                        "object_number": object_number,
                                        "details": detection
                                    }

                                    # update object as the detection
                                    object = object_dict

                                    # update the detection with object number
                                    videos[vid_idx]['images'][frame_idx]['detections'][det_idx] = object_dict
                                
                                    det_categorised = True

                                else:
                                    
                                    object_num = object_num + 1
                                    object_dict = {
                                        "object_number": "object_" + str(object_num),
                                        "details": detection
                                    }

                                    # add new object for that video
                                    objects.append(object_dict)

                                    # update the detection with object number
                                    videos[vid_idx]['images'][frame_idx]['detections'][det_idx] = object_dict
                                    
                                    det_categorised = True

                                    break
                                
    return videos


def rolling_pred_avg(videos):

    updated_videos = []
    
    for video in videos: 
        
        frames = video['images']
        updated_frames = video['images']
        Q = deque(maxlen = rolling_avg_size) #reset conf for each video
        for frame_idx, frame in enumerate(frames):
            
            detections = frame['detections']

            # if there are no bounding boxes...
            if not detections:
                
                # update the conf deque with 0 for all classes ...
                conf_all = [0,0,0]
                Q.append(conf_all)
            
            # if there are bounding boxes...
            else: 
                # for each bounding box...
                for det_idx, detection in enumerate(detections): 
                    det_conf = float(detection['conf'])
                    det_cat = int(detection['category'])
                    
                    # update the conf deque with the confidence of the detected class... 
                    conf_all = [0,0,0]
                    conf_all[det_cat-1] = det_conf
                    Q.append(conf_all)

                    # find the rolling prediction average...
                    np_mean = np.array(Q).mean(axis = 0)

                    # update the conf of the detection 
                    updated_frames[frame_idx]['detections'][det_idx]['conf'] = np_mean[det_cat-1] #TODO fix for specific class
        
        updated_video = {
            'video': video['video'], 
            'images': updated_frames}
        updated_videos.append(updated_video)
    
    return(updated_videos)


def main():

    frames_json = os.path.join(output_dir, "CT_models_test_frames_det.json")
    images_full, detector_label_map = load_detector_output(frames_json)

    conf_threshold_buf = 0.7
    images = rm_bad_detections(images_full, conf_threshold, conf_threshold_buf)

    videos = add_video_layer(images)

    objects = add_object_layer(videos)
    
    # roll_avg = rolling_pred_avg(videos)
    
    ## Export out the updated files
    output_file = os.path.splitext(frames_json)[0] + '_conf_' + str(conf_threshold) + '.json'
    write_results_to_file(images, output_file)

    videos_path = os.path.join(output_dir, "CT_models_test_videos.json")
    write_json_file(videos, videos_path)

    objects_path = os.path.join(output_dir, "CT_models_test_objects.json")
    write_json_file(objects, objects_path)
    
    # roll_avg_path = os.path.join(output_dir, "CT_models_test_videos_rolling_avg.json")
    # write_json_file(roll_avg, roll_avg_path)


if __name__ == '__main__':
    ## Arguments
    output_dir = "results/CT_models_test_2"
    conf_threshold = 0.8
    iou_threshold = 0.5
    rolling_avg_size = 64

    main()