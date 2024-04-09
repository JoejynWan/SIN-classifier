import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions
from detect.detect_utils import summarise_cat

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


def sort_videos(options, video_csv):

    print("Sorting the videos now...")

    vid_results = pd.read_csv(video_csv)

    if not vid_results['UniqueFileName'].any():
        vid_results['UniqueFileName'] = vid_results['FullVideoPath']
    vid_results_summ = summarise_cat(vid_results)

    root = os.path.abspath(os.curdir)
    vid_results_summ = vid_results_summ.reset_index()

    for idx, row in tqdm(vid_results_summ.iterrows(), total=vid_results_summ.shape[0]):

        station_dir = os.path.dirname(row['UniqueFileName'])
        input_vid = os.path.join(root,options.output_dir,row['UniqueFileName'])
        
        if row['Category'] == 0:

            false_neg_dir = os.path.join(root, options.output_dir, station_dir, 
                                         'False trigger')
            os.makedirs(false_neg_dir, exist_ok = True)
            _ = shutil.move(input_vid, false_neg_dir)

        elif row['Category'] == 1:
            
            if pd.isna(row['SpeciesClass']):
                animal_dir = os.path.join(root, options.output_dir, station_dir, 
                                          'Animal captures')
            else:
                spp_folder_name = row['SpeciesClass'].replace("_", " ")
                animal_dir = os.path.join(root, options.output_dir, station_dir, 
                                          spp_folder_name)
                
            os.makedirs(animal_dir, exist_ok = True)
            _ = shutil.move(input_vid, animal_dir)

        elif row['Category'] in (2,3):

            human_dir = os.path.join(root, options.output_dir, station_dir, 
                                     'Non targeted')
            os.makedirs(human_dir, exist_ok = True)
            _ = shutil.move(input_vid, human_dir)

        else:
            print('Warning: There are videos that are not sorted.')
            others_dir = os.path.join(root, options.output_dir, station_dir, 
                                     'Not sorted')
            os.makedirs(others_dir, exist_ok = True)
            _ = shutil.move(input_vid, others_dir)

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using '
                    'a trained MegaDetector model, where MegaDetector runs on '
                    'each frame in a video (or every Nth frame).')
    parser.add_argument(
        '--model_file', type=str, 
        default = config.MODEL_FILE, 
        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument(
        '--input_dir', type=str, 
        default = config.INPUT_DIR, 
        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument(
        '--recursive', type=bool, 
        default = config.RECURSIVE, 
        help = 'recurse into [input_dir]; only meaningful if a folder is '
               'specified as input'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default = config.OUTPUT_DIR, 
        help = 'Path to folder where results will be saved.'
    )
    parser.add_argument(
        '--render_output_video', type=bool,
        default = config.RENDER_OUTPUT_VIDEO, 
        help = 'Enable video output rendering.'
    )
    parser.add_argument(
        '--frame_folder', type=str, 
        default = config.FRAME_FOLDER, 
        help = 'Folder to use for intermediate frame storage, defaults to a '
               'folder in the system temporary folder'
    )
    parser.add_argument(
        '--run_species_classifier', type=bool,
        default = config.RUN_SPECIES_CLASSIFIER, 
        help = 'Enable/disable species classification (default false)'
    )
    parser.add_argument(
        '--cropped_images_dir', type=str,
        default = config.CROPPED_IMAGES_DIR, 
        help = 'Path to folder to save the cropped animal images, defaults to '
               'a folder in the system temporary folder'
    )
    parser.add_argument(
        '--species_model', type=str, 
        default = config.SPECIES_MODEL, 
        help = 'Path to .pt file containing the species classifier model.'
    )
    parser.add_argument(
        '--classification_csv', type=str,
        default = config.CLASSIFICATION_CSV, 
        help = 'Path to csv file containing the species classification results '
               'for cropped bounding boxes'
    )
    parser.add_argument(
        '-c', '--classifier_categories', type=str,
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
        '--classification_roll_avg_frames_json', type=str,
        default = config.CLASSIFICATION_ROLL_AVG_FRAMES_JSON, 
        help = 'Path to json file containing the frame-level detection results '
               'after merging with species classification results and '
               'conducting rolling prediction averaging'
    )
    parser.add_argument(
        '--classification_roll_avg_video_json', type=str,
        default = config.CLASSIFICATION_ROLL_AVG_VIDEO_JSON, 
        help = 'Path to json file containing the video-level detection results '
               'after merging with species classification results and '
               'conducting rolling prediction averaging'
    )
    parser.add_argument(
        '--classification_roll_avg_video_csv', type=str,
        default = config.CLASSIFICATION_ROLL_AVG_VIDEO_CSV, 
        help = 'Path to csv file containing the video-level detection results '
               'after merging with species classification results and '
               'conducting rolling prediction averaging'
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
    parser.add_argument(
        '--delete_output_frames', type=bool,
        default = config.DELETE_OUTPUT_FRAMES, 
        help = 'enable/disable temporary file deletion (default True)'
    )
    parser.add_argument(
        '--rendering_confidence_threshold', type=float,
        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
        help = 'do not render boxes with confidence below this threshold'
    )
    parser.add_argument(
        '--n_cores', type=int,
        default = config.N_CORES, 
        help = 'number of cores to use for detection and cropping (CPU only)'
    )
    parser.add_argument(
        '--frame_sample', type=int,
        default = config.FRAME_SAMPLE, 
        help='process every Nth frame (defaults to every frame)'
    )
    parser.add_argument(
        '--debug_max_frames', type=int,
        default = config.DEBUG_MAX_FRAMES, 
        help = 'trim to N frames for debugging (impacts model execution, not '
               'frame rendering)'
    )
    parser.add_argument(
        '--json_confidence_threshold', type=float,
        default = config.JSON_CONFIDENCE_THRESHOLD, 
        help = 'exclude detections in the .json file with confidence below '
                'this threshold'
    )
    parser.add_argument(
        '--check_accuracy', type=bool,
        default = config.CHECK_ACCURACY, 
        help = 'Whether accuracy of MegaDetector should be checked with manual '
               'ID. Folder names must contain species and quantity.'
    )
    parser.add_argument(
        '--species_database_file', type=str,
        default = config.SPECIES_DATABASE_FILE, 
        help = 'Path to the species_database.csv which lists species details'
    )
    parser.add_argument(
        '--manual_id_csv', type=str,
        default = config.MANUAL_ID_CSV, 
        help = 'Path to csv file containing the results of manual '
               'identification detections.'
    )
    parser.add_argument(
        '--roll_avg_video_csv', type=str,
        default = config.ROLL_AVG_VIDEO_CSV, 
        help = 'Path to the roll_avg_video csv file.'
    )
    parser.add_argument(
        '--roll_avg_frames_json', type=str,
        default = config.ROLL_AVG_FRAMES_JSON, 
        help = 'Path to the roll_avg_frames_json file.'
    )
    parser.add_argument(
        '--rolling_avg_size', type=float,
        default = config.ROLLING_AVG_SIZE, 
        help = 'Number of frames in which rolling prediction averaging will be '
               'calculated from. A larger number will remove more '
               'inconsistencies, but will cause delays in detecting an object.'
    )
    parser.add_argument(
        '--iou_threshold', type=float,
        default = config.IOU_THRESHOLD, 
        help = 'Threshold for Intersection over Union (IoU) to determine if '
               'bounding boxes are detecting the same object.'
    )
    parser.add_argument(
        '--conf_threshold_buf', type=float,
        default = config.CONF_THRESHOLD_BUF, 
        help = 'Buffer for the rendering confidence threshold to allow for '
               '"poorer" detections to be included in rolling prediction '
               'averaging.'
    )
    parser.add_argument(
        '--nth_highest_confidence', type=float,
        default = config.NTH_HIGHEST_CONFIDENCE, 
        help = 'nth-highest-confidence frame to choose a confidence value for '
               'each video'
    )
    parser.add_argument(
        '--resume_from_checkpoint', type=str,
        default = config.RESUME_FROM_CHECKPOINT, 
        help = 'path to JSON checkpoint file for which MD detection will be '
               'resumed from.'
    )
    parser.add_argument(
        '--checkpoint_path', type=str,
        default = config.CHECKPOINT_PATH, 
        help = 'path to JSON checkpoint file for which checkpoints will be '
               'written to.'
    )
    parser.add_argument(
        '--checkpoint_frequency', type=int,
        default = config.CHECKPOINT_FREQUENCY, 
        help = 'write results to JSON checkpoint file every N images.'
    )
    return parser


if __name__ == '__main__':
    
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    detector_csv = options.classification_roll_avg_video_csv
    sort_videos(options, detector_csv)
