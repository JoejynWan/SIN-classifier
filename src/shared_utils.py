import os
import json
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
    
    unqiue_videos = unique(frames_paths)
    
    return unqiue_videos


def write_json_file(output_object, output_path):
    
    with open(output_path, 'w') as f:
        json.dump(output_object, f, indent=1)
    print('Output file saved at {}'.format(output_path))