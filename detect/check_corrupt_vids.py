from multiprocessing.sharedctypes import Value
import os
from pickle import FALSE
import cv2
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions

## Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from detection.video_utils import find_videos


def check_corrupt(
    options, video_file, 
    move_or_copy = "move", copy_non_corrupt = False,
    Fs_threshold = 20, vid_duration_threshold = 5):

    input_fn_absolute = os.path.join(options.input_dir,video_file)
    assert os.path.isfile(input_fn_absolute), 'File {} not found'.format(input_fn_absolute)

    vidcap = cv2.VideoCapture(input_fn_absolute)
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/Fs

    if Fs < Fs_threshold or duration < vid_duration_threshold:

        ## Move/copy the corrupt videos into corrupt folder
        video_dir_relative = os.path.dirname(video_file)
        if Fs < Fs_threshold:
            video_dir_abs = os.path.join(options.output_dir, video_dir_relative, 'corrupt_vids', 'low_fps')
        elif duration < vid_duration_threshold: 
            video_dir_abs = os.path.join(options.output_dir, video_dir_relative, 'corrupt_vids', 'short_duration')
        os.makedirs(video_dir_abs, exist_ok=True)

        if move_or_copy == "move":
            _ = shutil.move(input_fn_absolute, video_dir_abs)
        elif move_or_copy == "copy":
            _ = shutil.copy2(input_fn_absolute, video_dir_abs)
        else:
            raise ValueError("Either choose to move or copy corrupt files.")

        ## Record the names and frame rates of the corrupt videos
        corrupt_fs_row = {
            'video_path': input_fn_absolute,
            'Fs': Fs
        }
        corrupt_fs_row_pd = pd.DataFrame(corrupt_fs_row, index = [0])

        return corrupt_fs_row_pd
    
    elif copy_non_corrupt == True:
        ## Copy the non-corrupt videos into the for_processing folder
        video_dir_relative = os.path.dirname(video_file)
        video_dir_abs = os.path.join(options.output_dir, video_dir_relative, 'for_processing')
        os.makedirs(video_dir_abs, exist_ok=True)

        _  = shutil.copy2(input_fn_absolute, video_dir_abs)
    
    
def check_corrupt_dir(
    options, 
    move_or_copy = "move", copy_non_corrupt = False, 
    Fs_threshold = 20, vid_duration_threshold = 5):

    print("Checking for corrupt videos now...")

    ## Find videos in folder
    input_files_full_paths = find_videos(options.input_dir, recursive=True)
    input_files_relative_paths = [os.path.relpath(s,options.input_dir) for s in input_files_full_paths]
    input_files_relative_paths = [s.replace('\\','/') for s in input_files_relative_paths]
    
    corrupt_fs = pd.DataFrame()
    for input_file_relative in tqdm(input_files_relative_paths):
    
        corrupt_fs_row_pd = check_corrupt(options, input_file_relative, 
            move_or_copy = move_or_copy, copy_non_corrupt = copy_non_corrupt, 
            Fs_threshold = Fs_threshold, vid_duration_threshold = vid_duration_threshold)
        corrupt_fs = pd.concat([corrupt_fs, corrupt_fs_row_pd])

    if not corrupt_fs.empty:
        print(
            "Warning: There are {} videos either with an extremely low frame rate (< 20fps), "
            "or they have a short duration (< 5 seconds)."
            "They have been moved/copied to the corrupt_vids folder.".format(len(corrupt_fs)))


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Check for corrupt files')
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    return parser


if __name__ == '__main__':

    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    check_corrupt_dir(options, move_or_copy = "copy", copy_non_corrupt = True)
