import os
import json
import copy
import json
import shutil
from tqdm import tqdm
from typing import Container
from datetime import datetime
from collections import defaultdict

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_tf_detector import TFDetector


def delete_temp_dir(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        pass


def unique(x):
    x = list(sorted(set(x)))
    return x


VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg')

def is_video_file(s: str, video_extensions: Container[str] = VIDEO_EXTENSIONS
                  ) -> bool:
    """
    Checks a file's extension against a hard-coded set of video file
    extensions.
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def find_unique_videos(images, output_dir = None):
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
        frames_path = os.path.dirname(entry['file'])      
        frames_paths.append(frames_path)
        
        if output_dir: 
            video_save_paths.append(os.path.join(output_dir, os.path.dirname(frames_path)))

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


def write_json_file(output_object, output_path):
    
    with open(output_path, 'w') as f:
        json.dump(output_object, f, indent=1)
    print('Output file saved at {}'.format(output_path))


class VideoOptions:

    model_file = ''
    input_dir = ''
    recursive = True 
    
    output_dir = None
    
    full_det_frames_json = None
    full_det_video_json = None
    roll_avg_frames_json = None
    roll_avg_video_json = None

    render_output_video = False
    delete_output_frames = True
    frame_folder = None
    
    rendering_confidence_threshold = 0.8
    frame_sample = None
    
    rolling_avg_size = 32
    iou_threshold = 0.5
    conf_threshold_buf = 0.7
    nth_highest_confidence = 1

    n_cores = 1

    json_confidence_threshold = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold
    debug_max_frames = -1
    reuse_results_if_available = False


def make_output_path(output_dir, input_dir, file_suffix):
    if output_dir is None:
        output_file_name = input_dir + file_suffix
       
    else:
        input_folder_name = os.path.basename(input_dir)
        output_file_name = os.path.join(output_dir, input_folder_name + file_suffix)

    return output_file_name


def write_frame_results(results, Fs, output_file, relative_path_base=None):
    """
    Writes a list of detection results to a JSON output file. 
    Function is adapted from detection.run_tf_detector_batch.write_results_to_file, 
    except video info is added. 

    Args
    - results: list of dict, each dict represents detections on one image
    - Fs: list of int, where each int represents the frame rate for each unique video.
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    unique_videos = find_unique_videos(results)
    if len(unique_videos) != len(Fs):
        raise IndexError("The number of frame rates provided do not match the number of unique videos.")

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'videos':{
            'num_videos': len(unique_videos),
            'video_names': unique_videos,
            'frame_rates': Fs
        },
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


def write_video_results(input_file,output_file, nth_highest_confidence = 1):
    """
    Given an API output file produced at the *frame* level, corresponding to a directory 
    created with video_folder_to_frames, map those frame-level results back to the 
    video level for use in Timelapse.

    Function adapted from detection.video_utils.frame_results_to_video_results, except
    frame rate is added.
    """
        
    # Load results
    with open(input_file,'r') as f:
        input_data = json.load(f)

    images = input_data['images']
    detection_categories = input_data['detection_categories']
    Fs = input_data['videos']['frame_rates']
    
    ## Break into videos
    
    video_to_frames = defaultdict(list) 
    
    for im in tqdm(images):
        
        fn = im['file']
        video_name = os.path.dirname(fn)
        assert is_video_file(video_name)
        video_to_frames[video_name].append(im)
    
    ## For each video...
    print('Converting {} frame-level results to {} video-level results by keeping only one detection per category.'.format(
        len(images), len(video_to_frames)))

    output_images = []
    for video_name, fs in tqdm(zip(video_to_frames, Fs)):
        
        frames = video_to_frames[video_name]
        
        all_detections_this_video = []
        
        # frame = frames[0]
        for frame in frames:
            all_detections_this_video.extend(frame['detections'])
            
        # At most one detection for each category for the whole video
        canonical_detections = []
            
        # category_id = list(detection_categories.keys())[0]
        for category_id in detection_categories:
            
            category_detections = [det for det in all_detections_this_video if det['category'] == category_id]
            
            # Find the nth-highest-confidence video to choose a confidence value
            if len(category_detections) > nth_highest_confidence:
                
                category_detections_by_confidence = sorted(category_detections, key = lambda i: i['conf'],reverse=True)
                canonical_detection = category_detections_by_confidence[nth_highest_confidence]
                canonical_detections.append(canonical_detection)
                                      
        # Prepare the output representation for this video
        im_out = {}
        im_out['file'] = video_name
        im_out['frame_rate'] = int(fs)
        im_out['detections'] = canonical_detections
        im_out['max_detection_conf'] = 0
        if len(canonical_detections) > 0:
            confidences = [d['conf'] for d in canonical_detections]
            im_out['max_detection_conf'] = max(confidences)
        
        output_images.append(im_out)
        
    # ...for each video
    
    output_data = input_data
    output_data['images'] = output_images
    s = json.dumps(output_data,indent=1)
    
    # Write the output file
    with open(output_file,'w') as f:
        f.write(s)


def find_unique_objects(images):

    all_objs = []
    for image in images:
        
        detections = image['detections']

        for detection in detections:
            obj = detection['object_number']
            all_objs.append(obj)

    unique_objs = unique(all_objs)

    return(unique_objs)


def write_roll_avg_video_results(options, ):
    
    # Load frame-level results
    with open(options.roll_avg_frames_json,'r') as f:
        input_data = json.load(f)

    images = input_data['images']
    Fs = input_data['videos']['frame_rates']

    # Find one object detection for each video
    unique_videos = find_unique_videos(images)

    print('Converting {} frame-level results to {} video-level results by keeping only one detection per object.'.format(
        len(images), len(unique_videos)))

    output_images = []
    for unique_video, fs in tqdm(zip(unique_videos, Fs)):
        frames = [im for im in images if unique_video in im['file']]

        unique_objs = find_unique_objects(frames)

        all_detections_this_video = []
        for frame in frames:
            all_detections_this_video.extend(frame['detections'])
        
        # Extract only the detection with the highest confidence for each object per video
        top_obj_detections = []
        for unique_obj in unique_objs:
            obj_detections = [det for det in all_detections_this_video if unique_obj in det['object_number']]
    
            # Find the nth-highest-confidence frame to choose a confidence value
            if len(obj_detections) > options.nth_highest_confidence:
                
                obj_detections_by_conf = sorted(obj_detections, key = lambda i: i['conf'], reverse=True)
                top_obj_detection = obj_detections_by_conf[options.nth_highest_confidence]

                if top_obj_detection['conf'] >= options.rendering_confidence_threshold:
                    top_obj_detections.append(top_obj_detection)

        # Output dict for this video
        im_out = {}
        im_out['file'] = unique_video
        im_out['frame_rate'] = int(fs)
        im_out['detections'] = top_obj_detections
        im_out['max_detection_conf'] = 0
        if len(top_obj_detections) > 0:
            confidences = [d['conf'] for d in top_obj_detections]
            im_out['max_detection_conf'] = max(confidences)
        
        output_images.append(im_out)

    output_data = input_data
    output_data['images'] = output_images
    s = json.dumps(output_data,indent=1)
    
    # Write the output file
    with open(options.roll_avg_video_json,'w') as f:
        f.write(s)
