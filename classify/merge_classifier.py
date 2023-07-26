import os
import json
import argparse

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions, default_path_from_none
from general.shared_utils import process_video_obj_results, json_to_csv

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from classification.merge_classification_detection_output import main as \
    merge_classifier_md


def write_classified_video_results(options):
    
    output_data = process_video_obj_results(
        options.classification_frames_json,
        options.nth_highest_confidence,
        options.rendering_confidence_threshold
    )
    
    # Write json output file
    if options.classification_video_json is None:
        path = options.classification_frames_json.replace('frames', 'video')
        options.classification_video_json = path
    
    with open(options.classification_video_json,'w') as f:
        json.dump(output_data, f, indent = 1)

    print('Output file saved at {}'.format(options.classification_video_json))

    # Write csv output file
    video_pd = json_to_csv(options, options.classification_video_json)

    if options.classification_video_csv is None: 
        path = options.classification_video_json.replace('.json', '.csv')
        options.classification_video_csv = path

    video_pd.to_csv(options.classification_video_csv, index = False)

    print('Output file saved at {}'.format(options.classification_video_csv))


def merge_classifier(options):

    if options.classification_frames_json is None: 

        options.classification_frames_json = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.classification_frames_json, '_classified_frames.json'
        ) 

    sp_classifier_name = os.path.basename(options.species_model)

    merge_classifier_md(
        classification_csv_path = options.classification_csv,
        label_names_json_path = options.classifier_categories, 
        detection_json_path = options.roll_avg_frames_json,
        output_json_path = options.classification_frames_json, 
        classifier_name = sp_classifier_name, 
        threshold = 0,
        datasets = None,
        queried_images_json_path = None,
        detector_output_cache_base_dir = None,
        detector_version = None,
        samples_per_label = None,
        seed = None,
        label_pos = None,
        relative_conf = None,
        typical_confidence_threshold = None)
    
    write_classified_video_results(options)
        

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to crop out detections.')
    parser.add_argument(
        '--output_dir', type=str,
        default = config.OUTPUT_DIR, 
        help = 'Path to folder where results will be saved.'
    )
    parser.add_argument(
        '--roll_avg_frames_json', type=str,
        default = config.ROLL_AVG_FRAMES_JSON, 
        help = 'Path to the roll_avg_frames_json file.'
    )
    parser.add_argument(
        '--classification_csv', type=str,
        default = config.CLASSIFICATION_CSV, 
        help = 'Path to csv file containing the species classification results '
               'for cropped bounding boxes'
    )
    parser.add_argument(
        '-c', '--classifier_categories',
        default = config.CLASSIFIER_CATEGORIES, 
        help = 'path to JSON file for classifier categories. If not given, '
               'classes are numbered "0", "1", "2", ...'
    )
    parser.add_argument(
        '--classification_frames_json', type=str,
        default = config.CLASSIFICATION_FRAMES_JSON, 
        help = 'Path to json file containing the frame-level detection results '
               'after merging with species classification results'
    )
    parser.add_argument(
        '--classification_video_json', type=str,
        default = config.CLASSIFICATION_VIDEO_JSON, 
        help = 'Path to json file containing the video-level detection results '
               'after merging with species classification results'
    )
    parser.add_argument(
        '--classification_video_csv', type=str,
        default = config.CLASSIFICATION_VIDEO_CSV, 
        help = 'Path to csv file containing the video-level detection results '
               'after merging with species classification results'
    )
    parser.add_argument(
        '--species_model', type=str, 
        default = config.SPECIES_MODEL, 
        help = 'Path to .pt file containing the species classifier model.'
    )
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    merge_classifier(options)
