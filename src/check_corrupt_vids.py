import os
import cv2
import shutil
import argparse
import warnings
import pandas as pd
from tqdm import tqdm

# Functions imported from this project
import config
from shared_utils import VideoOptions

## Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from detection.video_utils import find_videos


def check_corrupt(options, video_file):
    input_fn_absolute = os.path.join(options.input_dir,video_file)
    assert os.path.isfile(input_fn_absolute), 'File {} not found'.format(input_fn_absolute)

    vidcap = cv2.VideoCapture(input_fn_absolute)
    Fs = vidcap.get(cv2.CAP_PROP_FPS)

    if Fs < 20:

        ## Move the corrupt videos out of input_dir
        video_dir_relative = os.path.dirname(video_file)
        video_dir_abs = os.path.join(options.output_dir, 'corrupt_vids', video_dir_relative)
        os.makedirs(video_dir_abs, exist_ok=True)
        _ = shutil.move(input_fn_absolute, video_dir_abs)

        ## Record the names and frame rates of the corrupt videos
        corrupt_fs_row = {
            'video_path': input_fn_absolute,
            'Fs': Fs
        }
        corrupt_fs_row_pd = pd.DataFrame(corrupt_fs_row, index = [0])

        return corrupt_fs_row_pd
    
    
def check_corrupt_dir(options):

    print("Checking for corrupt videos now...")

    ## Find videos in folder
    input_files_full_paths = find_videos(options.input_dir, recursive=True)
    input_files_relative_paths = [os.path.relpath(s,options.input_dir) for s in input_files_full_paths]
    input_files_relative_paths = [s.replace('\\','/') for s in input_files_relative_paths]
    
    corrupt_fs = pd.DataFrame()
    for input_file_relative in tqdm(input_files_relative_paths):
    
        corrupt_fs_row_pd = check_corrupt(options, input_file_relative)
        corrupt_fs = corrupt_fs.append(corrupt_fs_row_pd)

    if not corrupt_fs.empty:
        warnings.warn(
            "There are {} videos with an extremely low frame rate (< 20fps)."
            "They have been moved to the corrupt_vids folder.".format(len(corrupt_fs)),
            UserWarning, stacklevel = 2)


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

    check_corrupt_dir(options)
