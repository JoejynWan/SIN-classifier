import os
import argparse

from shared_utils import make_output_path, VideoOptions
from rolling_avg import rolling_avg
from vis_detections import load_detector_output

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    # Set output file name and create output folder if not present
    if options.roll_avg_frames_json is None: 
        options.roll_avg_frames_json = make_output_path(
            options.output_dir, options.input_dir, '_roll_avg_frames.json'
        )
    os.makedirs(options.output_dir, exist_ok=True)

    images_full, detector_label_map, Fs = load_detector_output(options.full_det_frames_json)

    roll_avg, _, _, _, _ = rolling_avg(options, images_full)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to optimise rolling prediction averaging results.')
    parser.add_argument('--output_dir', type=str,
                        default = default_output_dir, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--input_dir', type=str, 
                        default = default_input_dir, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--full_det_frames_json', type=str,
                        default = default_full_det_frames_json, 
                        help = '.json file depicting the detections for each frame of the video'
    )
    parser.add_argument('--roll_avg_frames_json', type=str,
                        default = None, 
                        help = 'Path of json file with rolling-averaged detections for each frame.'
    )
    parser.add_argument('--frame_folder', type = str,
                        default = default_frame_folder,
                        help = 'Path to folder containing the video frames processed by det_videos.py'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = 0.8, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--rolling_avg_size', type=float,
                        default = 32, 
                        help = "Number of frames in which rolling prediction averaging will be calculated from. A larger number will remove more inconsistencies, but will cause delays in detecting an object."
    )
    parser.add_argument('--iou_threshold', type=float,
                        default = 0.5, 
                        help = "Threshold for Intersection over Union (IoU) to determine if bounding boxes are detecting the same object."
    )
    parser.add_argument('--conf_threshold_buf', type=float,
                        default = 0.7, 
                        help = "Buffer for the rendering confidence threshold to allow for 'poorer' detections to be included in rolling prediction averaging."
    )
    return parser
    

if __name__ == '__main__':
    default_input_dir = 'data/example_test_set_frames'
    default_full_det_frames_json = 'data/example_test_set_frames/example_test_set_full_det_frames.json'
    default_frame_folder = 'data/example_test_set_frames/frames'
    default_output_dir = 'results/example_test_set_frames'

    main()