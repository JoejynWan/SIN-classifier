import os
import sys
import json
import copy
import shutil
import pandas as pd
from tqdm import tqdm
from typing import Container
from datetime import datetime
from collections import defaultdict

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP


def check_output_dir(options):
    if os.path.exists(options.output_dir) and os.listdir(options.output_dir):
        while True:
            rewrite_input = input('\nThe output directory specified is not empty. Do you want to continue and rewrite the files in the directory? (y/n)')
            if rewrite_input.lower() not in ('y', 'n'):
                print('Input invalid. Please enter y or n.')
            else:
                break
        
        if rewrite_input.lower() == 'n':
            sys.exit('Stopping script. Please input the correct output directory. ')


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


def write_json_file(output_object, output_path):
    
    with open(output_path, 'w') as f:
        json.dump(output_object, f, indent=1)
    print('Output file saved at {}'.format(output_path))


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
    checkpoint_json = None
    checkpoint_frequency = -1 #default of -1 to not run any checkpointing

    frame_sample = None
    debug_max_frames = -1
    reuse_results_if_available = False
    json_confidence_threshold = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold

    render_output_video = False
    delete_output_frames = True
    frame_folder = None
    rendering_confidence_threshold = 0.8

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
    roll_avg_acc_csv = None


def load_detector_output(detector_output_path):
    with open(detector_output_path) as f:
        detector_output = json.load(f)
    assert 'images' in detector_output, (
        'Detector output file should be a json with an "images" field.')
    images = detector_output['images']

    if 'detection_categories' in detector_output:
        print('detection_categories provided')
        detector_label_map = detector_output['detection_categories']
    else:
        detector_label_map = DEFAULT_DETECTOR_LABEL_MAP

    Fs = detector_output['videos']['frame_rates']

    return images, detector_label_map, Fs


def default_path_from_none(output_dir, input_dir, file_path, file_suffix):
    """
    Creates the default path for the output files if not defined by the user.
    Default path is based on the output_dir, input_dir, and file_suffix. 
    """
    if file_path is None: 
    
        if output_dir is None:
            output_file_name = input_dir + file_suffix
        
        else:
            input_folder_name = os.path.basename(input_dir)
            output_file_name = os.path.join(output_dir, input_folder_name + file_suffix)

    else:
        output_file_name = file_path

    return output_file_name


def process_frame_results(results, Fs, relative_path_base=None):
    """
    Preps the results from MegaDetector for saving into the frames.json file.
    Function is adapted from detection.run_tf_detector_batch.write_results_to_file, 
    except video info is added. 
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

    frame_results = {
        'images': results,
        'detection_categories': DEFAULT_DETECTOR_LABEL_MAP,
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

    return frame_results


def process_video_results(frame_results, nth_highest_confidence):
    """
    Given an API output file produced at the *frame* level, corresponding to a directory 
    created with video_folder_to_frames, map those frame-level results back to the 
    video level for use in Timelapse.

    Function adapted from detection.video_utils.frame_results_to_video_results, except
    frame rate is added.
    """

    images = frame_results['images']
    detection_categories = frame_results['detection_categories']
    Fs = frame_results['videos']['frame_rates']
    
    ## Break into videos
    video_to_frames = defaultdict(list) 
    
    for im in images:
        
        fn = im['file']
        video_name = os.path.dirname(fn)
        assert is_video_file(video_name)
        video_to_frames[video_name].append(im)
    
    ## For each video...
    output_images = []
    for video_name, fs in zip(video_to_frames, Fs):
        
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
    
    video_results = frame_results
    video_results['images'] = output_images

    return video_results


def write_results(options, results, Fs,  
    relative_path_base=None, 
    nth_highest_confidence = 1, mute = False):
    """
    Writes a list of detection results to a JSON output file. 

    Args
    - results: list of dict, each dict represents detections on one image
    - Fs: list of int, where each int represents the frame rate for each unique video.
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    
    frame_results = process_frame_results(results, Fs, relative_path_base=relative_path_base)
    
    ## write frame.json file
    options.full_det_frames_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.full_det_frames_json, '_full_det_frames.json'
    )
    with open(options.full_det_frames_json, 'w') as f:
        json.dump(frame_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(options.full_det_frames_json))

    ## write video.json file
    video_results = process_video_results(frame_results, nth_highest_confidence)
    
    options.full_det_video_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.full_det_video_json, '_full_det_video.json'
    )
    with open(options.full_det_video_json, 'w') as f:
        json.dump(video_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(options.full_det_video_json))


def write_frame_results(results, Fs, frame_output_file, 
    relative_path_base=None, mute = False):
    """
    Writes a list of detection results to a JSON output file. 

    Args
    - results: list of dict, each dict represents detections on one image
    - Fs: list of int, where each int represents the frame rate for each unique video.
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    frame_results = process_frame_results(results, Fs, relative_path_base=relative_path_base)
    
    ## write frame.json file
    with open(frame_output_file, 'w') as f:
        json.dump(frame_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(frame_output_file))


def write_video_results(output_file, 
    frames_json_inputdata = None,  
    frames_json_inputfile = None, 
    nth_highest_confidence = 1, mute = False):

    # Load results
    if frames_json_inputfile is not None:
        with open(frames_json_inputfile,'r') as f:
            frames_json_inputdata = json.load(f)
    
    video_results = process_video_results(frames_json_inputdata, nth_highest_confidence)
    
    # Write the output file
    with open(output_file,'w') as f:
        json.dump(video_results, f, indent=1)
    
    if not mute:
        print('Output file saved at {}'.format(output_file))


def find_unique_objects(images):

    all_objs = []
    for image in images:
        
        detections = image['detections']

        for detection in detections:
            obj = detection['object_number']
            all_objs.append(obj)

    unique_objs = unique(all_objs)

    return(unique_objs)


def json_to_csv(options, images):

    video_pd = pd.DataFrame()
    for image in images:
        video = image['file'].replace('\\','/')
        frame_rate = image['frame_rate']
        detections = image['detections']

        if options.check_accuracy:
            station_sampledate, _, vid_name = video.split("/")
            uniquefile = os.path.join(station_sampledate, vid_name).replace('\\','/')
        else:
            uniquefile = 'NA'

        if not detections: #no detections, so false trigger

            obj_row = {
                'FullVideoPath': video,
                'UniqueFileName': uniquefile, 
                'FrameRate': frame_rate,
                'Category': 0,
                'DetectedObj': 'NA',
                'MaxConf': 'NA',
                'BboxXmin': 'NA',
                'BboxYmin': 'NA',
                'BboxWrel': 'NA',
                'BboxHrel': 'NA'
            }
            obj_row_pd = pd.DataFrame(obj_row, index = [0])

            video_pd = video_pd.append(obj_row_pd)

        else: 
            for detection in detections:

                obj_row = {
                    'FullVideoPath': video,
                    'UniqueFileName': uniquefile, 
                    'FrameRate': frame_rate,
                    'Category': detection['category'],
                    'DetectedObj': detection['object_number'],
                    'MaxConf': detection['conf'],
                    'BboxXmin': detection['bbox'][0],
                    'BboxYmin': detection['bbox'][1],
                    'BboxWrel': detection['bbox'][2],
                    'BboxHrel': detection['bbox'][3]
                }
                obj_row_pd = pd.DataFrame(obj_row, index = [0])

                video_pd = video_pd.append(obj_row_pd)
    
    options.roll_avg_video_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.roll_avg_video_csv, '_roll_avg_videos.csv'
    )
    video_pd.to_csv(options.roll_avg_video_csv, index = False)


def write_roll_avg_video_results(options, mute = False):
    
    # Load frame-level results
    with open(options.roll_avg_frames_json,'r') as f:
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
    options.roll_avg_video_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.roll_avg_video_json, '_roll_avg_videos.json'
    )

    with open(options.roll_avg_video_json,'w') as f:
        f.write(s)
    
    if not mute:
        print('Output file saved at {}'.format(options.roll_avg_video_json))

    json_to_csv(options, output_images)
    
    if not mute:
        print('Output file saved at {}'.format(options.roll_avg_video_csv))

def export_fn(options, video_summ):

    print("Copying false negative videos to output directory now...")

    video_summ_copy = video_summ.copy()
    export_pd = video_summ_copy[video_summ_copy['AccClass'] == 'FN']
    export_pd = export_pd.reset_index()

    root = os.path.abspath(os.curdir)

    for idx, row in tqdm(export_pd.iterrows(), total=export_pd.shape[0]):
        
        input_vid = os.path.join(root, options.input_dir, row['FullVideoPath'])
        output_vid_dir = os.path.join(root, options.output_dir, 'false_nagative_videos', os.path.dirname(row['UniqueFileName']))
        os.makedirs(output_vid_dir, exist_ok = True)

        _ = shutil.copy2(input_vid, output_vid_dir)

