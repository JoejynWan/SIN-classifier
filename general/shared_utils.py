import os
import json

class VideoOptions:

    input_dir = ''
    output_dir = None

    model_file = ''
    species_database_file = None

    recursive = True 
    n_cores = 1
    
    full_det_frames_json = None
    full_det_video_json = None
    
    resume_from_checkpoint = None
    checkpoint_path = None
    checkpoint_frequency = -1 #default of -1 to not run any checkpointing

    frame_sample = None
    debug_max_frames = -1
    reuse_results_if_available = False
    json_confidence_threshold = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold

    render_output_video = False
    delete_output_frames = True
    frame_folder = None
    rendering_confidence_threshold = 0.2

    rolling_avg_size = 32
    iou_threshold = 0.5
    conf_threshold_buf = 0.7
    nth_highest_confidence = 1

    roll_avg_frames_json = None
    roll_avg_video_json = None
    roll_avg_video_csv = None
    
    check_accuracy = False
    manual_id_csv = None
    manual_vs_md_csv = None
    species_list_csv = None

    rolling_avg_size_range = None
    iou_threshold_range = None
    conf_threshold_buf_range = None
    rendering_confidence_threshold_range = None
    roll_avg_acc_csv = None

    run_species_classifier = False
    cropped_images_dir = None
    species_model = None
    classification_csv = None
    classification_frames_json = None
    classification_video_json = None
    classifier_categories = None
    image_size = 224
    batch_size = 1
    device = None
    num_workers = 8


def unique(x):
    x = list(sorted(set(x)))
    return x


def find_unique_videos(images, output_dir = None, from_frame_names = False):
    """
    Function to extract the unique videos in the frames.json file. 
    
    Arguments: 
    images = list containing dictionaries of image frames.
        Each dictionary should contain:
            'file' = 'path/to/frame/image'
    output_dir = str, optional. Path to save directory. 
        If provided, folders with the names of the unique videos will be created
        in the save directory provided

    """

    frames_paths = []
    video_save_paths = []
    for entry in images:

        if from_frame_names:
            frame_name = entry
        else:
            frame_name = entry['file']

        frames_path = os.path.dirname(frame_name)      
        frames_paths.append(frames_path)
        
        if output_dir: 
            video_path = os.path.dirname(frames_path)
            video_save_paths.append(os.path.join(output_dir, (video_path)))

    if output_dir: 
        [os.makedirs(make_path, exist_ok=True) for make_path in unique(video_save_paths)]
    
    unique_videos = unique(frames_paths)
    
    return unique_videos


def find_unique_objects(images):
    
    obj_nums = []
    for image in images:
        for detection in image['detections']:
            obj_num = detection['object_number']
            obj_nums.append(obj_num)

    unq_obj_nums = unique(obj_nums)

    return unq_obj_nums


def process_video_obj_results(frames_json, 
                              nth_highest_confidence, 
                              rendering_confidence_threshold):
    """
    Given an API output file produced at the *frame* level, corresponding to a 
    directory created with video_folder_to_frames, map those frame-level results 
    back to the video level for use in Timelapse.

    Function adapted from detection.video_utils.frame_results_to_video_results, 
    except frame rate is added, and unique objects are considered. 
    """
    
    # Load frame-level results
    with open(frames_json,'r') as f:
        input_data = json.load(f)

    images = input_data['images']
    Fs = input_data['videos']['frame_rates']

    # Find one object detection for each video
    unique_videos = find_unique_videos(images)

    output_images = []
    for unique_video, fs in zip(unique_videos, Fs):
        frames = [im for im in images if unique_video in im['file']]

        unique_objs = find_unique_objects(frames)

        all_detections_this_video = []
        for frame in frames:
            all_detections_this_video.extend(frame['detections'])
        
        # Extract only the detection with the highest confidence for each 
        # object per video
        top_obj_detections = []
        for unique_obj in unique_objs:
            obj_detections = [det for det in all_detections_this_video if \
                              unique_obj in det['object_number']]
    
            # Find the nth-highest-confidence frame to choose a confidence value
            if len(obj_detections) > nth_highest_confidence:
                
                obj_detections_by_conf = sorted(obj_detections, 
                                                key = lambda i: i['conf'], 
                                                reverse = True)
                top_obj_detection = obj_detections_by_conf[nth_highest_confidence-1]

                if top_obj_detection['conf'] >= rendering_confidence_threshold:
                    top_obj_detections.append(top_obj_detection)

        # Output dict for this video
        im_out = {}
        im_out['file'] = unique_video
        im_out['frame_rate'] = int(float(fs))
        im_out['detections'] = top_obj_detections
        im_out['max_detection_conf'] = 0
        if len(top_obj_detections) > 0:
            confidences = [d['conf'] for d in top_obj_detections]
            im_out['max_detection_conf'] = max(confidences)
        
        output_images.append(im_out)

    output_data = input_data
    output_data['images'] = output_images

    return(output_data, output_images)
