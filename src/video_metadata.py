import os
import ffmpeg
from datetime import datetime

from detection.video_utils import find_videos

input_dir = 'data/Round_14_January_2020'

input_dir_full_paths = find_videos(input_dir, recursive=True)
vids_relative_paths = [os.path.relpath(s,input_dir) for s in input_dir_full_paths]
vids_relative_paths = [s.replace('\\','/') for s in vids_relative_paths]

vid_rel_path = vids_relative_paths[0]

station_sampledate, species_abundance, vid_name = vid_rel_path.split("/")
station, sampledate =  station_sampledate.split("_")
species_details = species_abundance.split(" ")
file = os.path.join(station_sampledate, vid_name).replace('\\','/')


print(vids_relative_paths[0])

vid_full_path = os.path.join(input_dir, vid_rel_path)
creation_datetime = datetime.fromtimestamp(os.path.getctime(vid_full_path))

true_list = {
    'Station': station,
    'SamplingDate': sampledate,
    'Date': str(creation_datetime.date()),
    'Time': str(creation_datetime.time()),
    'DateTime': str(creation_datetime),
    'File': file
}

print(true_list)


# video = ffmpeg.probe(vid_full_path)
# print(video)