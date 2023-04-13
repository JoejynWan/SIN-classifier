import os
import argparse

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from classification.merge_classification_detection_output import main as merge_classifier_md


def merge_classifier(options):

    if options.classification_json is None: 
        filename = os.path.basename(options.roll_avg_frames_json)
        filename = os.path.splitext(filename)[0] + "_classified.json"

        options.classification_json = os.path.join(options.output_dir, filename)

    sp_classifier_name = os.path.basename(options.species_model)

    merge_classifier_md(
        classification_csv_path = options.classification_csv,
        label_names_json_path = options.classifier_categories, 
        detection_json_path = options.roll_avg_frames_json,
        output_json_path = options.classification_json, 
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
        '--classification_json', type=str,
        default = config.CLASSIFICATION_JSON, 
        help = 'Path to json file containing the detection results after '
               'merging with species classification results'
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
