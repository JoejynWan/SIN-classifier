import os 
from collections import deque

from shared_utils import find_unqiue_videos
from vis_detections import load_detector_output
from detection.run_tf_detector_batch import write_results_to_file


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


def images_list_to_videos_list(images):

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
            'frames': frames}
        videos.append(video)

    return videos


def rolling_pred_avg(images):
    ##TODO identify unique videos
    objects = []
    for image in images:
        detections = image['detections']
        Q = deque(maxlen = rolling_avg_size)
        object_Q = {
            'animal_Q' : Q,
            'person_Q' : Q,
            'vehicle_Q' : Q}

        if detections:
            for detection in detections:
                if not objects: #detected_objects is still empty (first detection)
                    objects.append({
                        "detection": detection, 
                        "conf_Q": object_Q})
                else:
                    for object in objects:
                        bbox_object = object["detection"]['bbox']
                        bbox_detection = detection['bbox']

                        iou = bbox_iou(bbox_object, bbox_detection)
                        if iou >= iou_threshold: 
                            #append conf for each class respectively
                            detection_cat = detection['category']
                            detection_conf = detection['conf']
                            animal_Q = object['conf_Q']['animal_Q']
                            person_Q = object['conf_Q']['person_Q']
                            vehicle_Q = object['conf_Q']['vehicle_Q']

                            if detection_cat == '1':
                                animal_Q.append(detection_conf)
                                person_Q.append(0)
                                vehicle_Q.append(0)
                            elif detection_cat == '2':
                                animal_Q.append(0)
                                person_Q.append(detection_conf)
                                vehicle_Q.append(0)
                            elif detection_cat == '3':
                                animal_Q.append(0)
                                person_Q.append(0)
                                vehicle_Q.append(detection_conf)

                            
                            
                            
                            pass #TODO
                        else: #different object
                            objects.append([detection, object_Q])
        else:
            pass
            #TODO add 0 to the object_Q

def main():
    images_full, detector_label_map = load_detector_output(frames_json)

    conf_threshold_buf = 0.7
    images = rm_bad_detections(images_full, conf_threshold, conf_threshold_buf)

    videos = images_list_to_videos_list(images)

    print(videos[0])

    output_file = os.path.splitext(frames_json)[0] + '_conf_' + str(conf_threshold) + '.json'
    write_results_to_file(images, output_file)

if __name__ == '__main__':
    ## Arguments
    frames_json = "results/CT_models_test/CT_models_test_frames_det.json"
    conf_threshold = 0.8
    iou_threshold = 0.5
    rolling_avg_size = 64

    main()