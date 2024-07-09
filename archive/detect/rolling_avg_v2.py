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
                print(detection)
                #first detection of the video is always a new unique object
                if not objects: 
                    
                    object_name = "object_" + str(object_num).zfill(2)
                    frame['object_number'] = frame['object_number'].append(object_name)
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