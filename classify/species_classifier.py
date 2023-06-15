import os
import argparse

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from classification.run_classifier import main as sp_class_md


def sp_classifier(options):
    if options.classification_csv is None: 
        options.classification_csv = os.path.join(
            options.output_files_dir, "species_classifier_crops.csv.gz")

    sp_class_md(
        model_path = options.species_model,
        cropped_images_dir = options.cropped_images_dir,
        output_csv_path = options.classification_csv,
        detections_json_path = options.roll_avg_frames_json,
        classifier_categories_json_path = options.classifier_categories,
        img_size = options.image_size,
        batch_size = options.batch_size,
        num_workers = options.num_workers,
        device_id = options.device)
        

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to crop out detections.')
    parser.add_argument(
        '--output_dir', type=str,
        default = config.OUTPUT_DIR, 
        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument(
        '--species_model', type=str, 
        default = config.SPECIES_MODEL, 
        help = 'Path to .pt file containing the species classifier model.'
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
        '--cropped_images_dir', type=str,
        default = config.CROPPED_IMAGES_DIR, 
        help = 'Path to folder to save the cropped animal images, defaults to '
               'a folder in the system temporary folder'
    )
    parser.add_argument(
        '-c', '--classifier_categories', type=str,
        default = config.CLASSIFIER_CATEGORIES, 
        help = 'path to JSON file for classifier categories. If not given, '
               'classes are numbered "0", "1", "2", ...'
    )
    parser.add_argument(
        '--image_size', type=int, 
        default = config.IMAGE_SIZE,
        help = 'size of input image to model, usually 224px, but may be larger '
               'especially for EfficientNet models.'
    )
    parser.add_argument(
        '--batch_size', type=int, 
        default = config.BATCH_SIZE,
        help = 'batch size for evaluating model.'
    )
    parser.add_argument(
        '--device', type=int, 
        default = config.DEVICE,
        help = 'preferred CUDA device.'
    )
    parser.add_argument(
        '--num_workers', type=int, 
        default = config.NUM_WORKERS,
        help = 'Number of workers for data loading.'
    )
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    sp_classifier(options)
