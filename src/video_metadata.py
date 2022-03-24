import os
import ffmpeg

from detection.video_utils import find_videos

input_dir = 'data/Round_7_June_2019/'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
input_dir_full_paths = find_videos(input_dir, recursive=True)
vids_relative_paths = [os.path.relpath(s,input_dir) for s in input_dir_full_paths]
vids_relative_paths = [s.replace('\\','/') for s in vids_relative_paths]


vid_path = os.path.join(input_dir, vids_relative_paths[0])
assert os.path.isfile(vid_path)
print(vid_path)
video = ffmpeg.probe(vid_path)

print(video['streams'])