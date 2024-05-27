
import os
import cv2
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from typing import Container,Iterable,List

VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg')

def is_video_file(s: str, video_extensions: Container[str] = VIDEO_EXTENSIONS
                  ) -> bool:
    """
    Checks a file's extension against a hard-coded set of video file
    extensions.
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def find_video_strings(strings: Iterable[str]) -> List[str]:
    """
    Given a list of strings that are potentially video file names, looks for
    strings that actually look like video file names (based on extension).
    """
    return [s for s in strings if is_video_file(s.lower())]


def find_videos(dirname: str, recursive: bool = False) -> List[str]:
    """
    Finds all files in a directory that look like video file names. Returns
    absolute paths.
    """
    if recursive:
        strings = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirname, '*.*'))
    return find_video_strings(strings)

def check_corrupt_file(
    video_file, input_dir, output_dir, 
    move_or_copy = "move", copy_non_corrupt = False,
    Fs_threshold = 20, vid_duration_threshold = 5):

    input_fn_absolute = os.path.join(input_dir,video_file)
    assert os.path.isfile(input_fn_absolute), 'File {} not found'.format(input_fn_absolute)

    vidcap = cv2.VideoCapture(input_fn_absolute)
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/Fs

    if Fs < Fs_threshold or duration < vid_duration_threshold:

        ## Move/copy the corrupt videos into corrupt folder
        video_dir_relative = os.path.dirname(video_file)
        if Fs < Fs_threshold:
            video_dir_abs = os.path.join(output_dir, video_dir_relative, 'low_fps')
        elif duration < vid_duration_threshold: 
            video_dir_abs = os.path.join(output_dir, video_dir_relative, 'short_duration')
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
        video_dir_abs = os.path.join(output_dir, video_dir_relative, 'for_processing')
        os.makedirs(video_dir_abs, exist_ok=True)

        _  = shutil.copy2(input_fn_absolute, video_dir_abs)
    
    
def check_corrupt_dir(
    input_dir, 
    output_dir, 
    move_or_copy = "move", copy_non_corrupt = False, 
    Fs_threshold = 20, vid_duration_threshold = 5):

    print("Checking for corrupt videos now...")

    ## Find videos in folder
    input_files_full_paths = find_videos(input_dir, recursive=True)
    input_files_relative_paths = [os.path.relpath(s, input_dir) for s in input_files_full_paths]
    input_files_relative_paths = [s.replace('\\','/') for s in input_files_relative_paths]
    
    corrupt_fs = pd.DataFrame()
    for input_file_relative in tqdm(input_files_relative_paths):
    
        corrupt_fs_row_pd = check_corrupt_file(
            input_file_relative, 
            input_dir, output_dir, 
            move_or_copy = move_or_copy, copy_non_corrupt = copy_non_corrupt, 
            Fs_threshold = Fs_threshold, vid_duration_threshold = vid_duration_threshold
        )
        corrupt_fs = pd.concat([corrupt_fs, corrupt_fs_row_pd])

    if not corrupt_fs.empty:
        print(
            "Warning: There are {} videos either with an extremely low frame rate (< 20fps), "
            "or they have a short duration (< 5 seconds)."
            "They have been moved/copied to the low_fps or short_duration folder.".format(len(corrupt_fs)))
