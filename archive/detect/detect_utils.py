import os
import sys
import json
import copy
import shutil
from tqdm import tqdm
from typing import Container
from datetime import datetime
from collections import defaultdict

# Functions imported from this project
from general.shared_utils import find_unique_videos, default_path_from_none

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP, \
    get_detector_version_from_filename,\
    get_detector_metadata_from_version_string


def check_output_dir(options):
    assert os.path.isdir(options.output_dir),'{} is not a folder'.format(options.output_dir)
    
    if os.path.exists(options.output_dir) and os.listdir(options.output_dir):
        while True:
            rewrite_input = input(
                '\nThe output directory specified is not empty. Do you want ' \
                    'to continue and rewrite the files in the directory? (y/n)')
            if rewrite_input.lower() not in ('y', 'n'):
                print('Input invalid. Please enter y or n.')
            else:
                break
        
        if rewrite_input.lower() == 'n':
            sys.exit('Error: Please input the correct output directory.')

    os.makedirs(options.output_files_dir, exist_ok=True)


def delete_temp_dir(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        pass


VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg')

def is_video_file(s: str, video_extensions: Container[str] = VIDEO_EXTENSIONS
                  ) -> bool:
    """
    Checks a file's extension against a hard-coded set of video file
    extensions.
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def write_json_file(output_object, output_path):
    
    with open(output_path, 'w') as f:
        json.dump(output_object, f, indent=1)
    print('Output file saved at {}'.format(output_path))


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


def process_frame_results(results, Fs, relative_path_base=None,
                          detector_file=None, info=None, 
                          classification_map=None):
    """
    Preps the results from MegaDetector for saving into the frames.json file.
    Function is adapted from detection.run_tf_detector_batch.write_results_to_file, 
    except video info is added. 
    """
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], 
                                                 start = relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    ## Build the videos structure
    unique_videos = find_unique_videos(results)
    if len(unique_videos) != len(Fs):
        raise IndexError(
            'The number of frame rates provided do not match the number of '
            'unique videos ({} videos and {} frame rates).'.format(
                len(unique_videos), len(Fs)))

    ## Build the info structure
    if info is None:

        info = { 
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.2' 
        }
            
        if detector_file is not None:
            detector_filename = os.path.basename(detector_file)
            detector_version = get_detector_version_from_filename(detector_filename)
            detector_metadata = get_detector_metadata_from_version_string(detector_version)
            info['detector'] = detector_filename  
            info['detector_metadata'] = detector_metadata
        else:
            info['detector'] = 'unknown'
            info['detector_metadata'] = get_detector_metadata_from_version_string('unknown')

    else: #user provided the info structure

        if detector_file is not None:
            
            print('Warning (write_results_to_file): info struct and ' + \
                  'detector file supplied, ignoring detector file')

    if classification_map:
        frame_results = {
            'images': results,
            'detection_categories': DEFAULT_DETECTOR_LABEL_MAP,
            'videos':{
                'num_videos': len(unique_videos),
                'video_names': unique_videos,
                'frame_rates': Fs
            },
            'info': info,
            'classification_categories': classification_map
        }

    else:
        frame_results = {
            'images': results,
            'detection_categories': DEFAULT_DETECTOR_LABEL_MAP,
            'videos':{
                'num_videos': len(unique_videos),
                'video_names': unique_videos,
                'frame_rates': Fs
            },
            'info': info
        }

    return frame_results


def process_video_results(frame_results, nth_highest_confidence):
    """
    Given an API output file produced at the *frame* level, corresponding to a 
    directory created with video_folder_to_frames, map those frame-level results 
    back to the video level for use in Timelapse.

    Function adapted from detection.video_utils.frame_results_to_video_results, 
    except frame rate is added.
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
            
            category_detections = [det for det in all_detections_this_video if \
                                   det['category'] == category_id]
            
            # Find the nth-highest-confidence video to choose a confidence value
            if len(category_detections) > nth_highest_confidence:
                
                category_detections_by_conf = sorted(category_detections, 
                                                    key = lambda i: i['conf'],
                                                    reverse=True)
                canonical_detection = category_detections_by_conf[nth_highest_confidence-1]
                canonical_detections.append(canonical_detection)
                                      
        # Prepare the output representation for this video
        im_out = {}
        im_out['file'] = video_name
        im_out['frame_rate'] = int(float(fs))
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
    - Fs: list of int, where each int represents the frame rate for each unique 
      video.
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative 
      paths
    """
    
    frame_results = process_frame_results(
        results, Fs, 
        relative_path_base = relative_path_base,
        detector_file = options.model_file)
    
    ## write frame.json file
    options.full_det_frames_json = default_path_from_none(
        options.output_files_dir, options.input_dir, 
        options.full_det_frames_json, '_full_det_frames.json'
    )
    with open(options.full_det_frames_json, 'w') as f:
        json.dump(frame_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(options.full_det_frames_json))

    ## write video.json file
    video_results = process_video_results(frame_results, nth_highest_confidence)
    
    options.full_det_video_json = default_path_from_none(
        options.output_files_dir, options.input_dir, 
        options.full_det_video_json, '_full_det_video.json'
    )
    with open(options.full_det_video_json, 'w') as f:
        json.dump(video_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(options.full_det_video_json))


def write_frame_results(options, results, Fs, frame_output_file, 
                        relative_path_base=None, info = None, 
                        classification_map = None, mute = False):
    """
    Writes a list of detection results to a JSON output file. 

    Args
    - results: list of dict, each dict represents detections on one image
    - Fs: list of int, where each int represents the frame rate for each unique 
      video
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative 
      paths
    """
    frame_results = process_frame_results(
        results, Fs, 
        relative_path_base = relative_path_base,
        detector_file = options.model_file,
        info = info, 
        classification_map = classification_map)
    
    ## write frame.json file
    with open(frame_output_file, 'w') as f:
        json.dump(frame_results, f, indent=1)

    if not mute:
        print('Output file saved at {}'.format(frame_output_file))


# def write_video_results(output_file, 
#     frames_json_inputdata = None,  
#     frames_json_inputfile = None, 
#     nth_highest_confidence = 1, mute = False):

#     # Load results
#     if frames_json_inputfile is not None:
#         with open(frames_json_inputfile,'r') as f:
#             frames_json_inputdata = json.load(f)
    
#     video_results = process_video_results(
#         frames_json_inputdata, nth_highest_confidence)
    
#     # Write the output file
#     with open(output_file,'w') as f:
#         json.dump(video_results, f, indent=1)
    
#     if not mute:
#         print('Output file saved at {}'.format(output_file))


def export_fn(options, video_summ):

    print("Copying false negative videos to output directory now...")

    video_summ_copy = video_summ.copy()
    export_pd = video_summ_copy[video_summ_copy['AccClass'] == 'FN']
    export_pd = export_pd.reset_index()

    root = os.path.abspath(os.curdir)

    for idx, row in tqdm(export_pd.iterrows(), total=export_pd.shape[0]):
        
        input_vid = os.path.join(root, options.input_dir, row['FullVideoPath'])
        output_vid_dir = os.path.join(
            root, options.output_dir, 'false_negative_videos', 
            os.path.dirname(row['UniqueFileName'])
        )
        os.makedirs(output_vid_dir, exist_ok = True)

        _ = shutil.copy2(input_vid, output_vid_dir)


def summarise_cat(full_df):
    """
    Summarises dataset into two columns: 'UniqueFileName' and 'Category', 
    where each row is a unique UniqueFileName. If a UniqueFileName has multiple 
    Categories (more than one detection or animal), they are summarised 
    using the following rules:
    1) Where there are multiples of the same category, only one is kept
    2) Where there is an animal (1) detection, and human (2) and/or vehicle (3) 
       detection, the video is considered to be an animal category (1)
    3) Where there is a human (2) and vehicle (3), the video is considered to 
       be a human category (2)
    """

    video_cat = full_df[['UniqueFileName', 'Category', 'SpeciesClass']]
    video_cat_sort = video_cat.sort_values(by = ['UniqueFileName', 'Category'])
    summ_cat = video_cat_sort.drop_duplicates(
        subset = ['UniqueFileName'], keep = 'first', ignore_index = True
    )

    return summ_cat

