import os
import json
import shutil
from typing import Container


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
