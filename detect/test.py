import time
import json
import argparse
import humanfriendly

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

from detect_utils import process_video_obj_results


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module for testing.')
    parser.add_argument(
        '--rendering_confidence_threshold', type=float,
        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
        help = 'do not render boxes with confidence below this threshold'
    )
    parser.add_argument(
        '--nth_highest_confidence', type=float,
        default = config.NTH_HIGHEST_CONFIDENCE, 
        help = 'nth-highest-confidence frame to choose a confidence value for '
               'each video'
    )
    return parser


start = time.time()
input = 'results\\test\\test_roll_avg_frames_classified.json'
output = 'results\\test\\test_roll_avg_video_classified.json'

parser = get_arg_parser()
args = parser.parse_args()
options = VideoOptions()
args_to_object(args, options)

output_data, output_images = process_video_obj_results(input, 1, 0.2)

with open(output,'w') as f:
    json.dump(output_data, f, indent = 1)

end = time.time()
elapsed = end - start
print('File completed in {}'.format(humanfriendly.format_timespan(elapsed)))
