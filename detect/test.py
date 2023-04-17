import time
import json
import argparse
import humanfriendly

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions
from detect_utils import sort_videos

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module for testing.')
    parser.add_argument(
        '--output_dir', type=str,
        default = config.OUTPUT_DIR, 
        help = 'Path to folder where results will be saved.'
    )
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
    parser.add_argument(
        '--check_accuracy', type=bool,
        default = config.CHECK_ACCURACY, 
        help = 'Whether accuracy of MegaDetector should be checked with manual '
               'ID. Folder names must contain species and quantity.'
    )
    parser.add_argument(
        '-c', '--classifier_categories', type=str,
        default = config.CLASSIFIER_CATEGORIES, 
        help = 'path to JSON file for classifier categories. If not given, '
               'classes are numbered "0", "1", "2", ...'
    )
    return parser


start = time.time()
input = 'results\\test\\test_roll_avg_videos.csv'

parser = get_arg_parser()
args = parser.parse_args()
options = VideoOptions()
args_to_object(args, options)

sort_videos(options, input)

end = time.time()
elapsed = end - start
print('File completed in {}'.format(humanfriendly.format_timespan(elapsed)))
