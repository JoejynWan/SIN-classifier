import os
import shutil


def delete_temp_dir(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        pass


def unique(x):
    x = list(set(x))
    return x


def find_unqiue_videos(images, output_dir = None):
    """
    Function to extract the unique videos in the frames.json file. 
    
    Arguments: 
    images = list containing dictionaries of image frames.
        Each dictionary should contain:
            'file' = 'path/to/frame/image'


    """

    frames_paths = []
    video_save_paths = []
    for entry in images:
        frames_path = os.path.dirname(entry['file'])      
        frames_paths.append(frames_path)
        
        if output_dir: 
            video_save_paths.append(os.path.join(output_dir, os.path.dirname(frames_path)))

    unqiue_videos = unique(frames_paths)

    return unqiue_videos, video_save_paths