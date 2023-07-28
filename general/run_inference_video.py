import os
import time
import argparse
import humanfriendly

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions, default_path_from_none
from detect.detect_utils import delete_temp_dir, check_output_dir
from detect.detect_utils import export_fn, sort_videos
from detect.run_det_video import video_dir_to_frames, det_frames
from detect.vis_detections import vis_detection_videos
from detect.manual_ID import manual_ID_results
from detect.optimise_roll_avg import true_vs_pred
from detect.rolling_avg import rolling_avg
from classify.crop_det import crop_detections, load_crop_log
from classify.species_classifier import sp_classifier
from classify.merge_classifier import merge_classifier

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def runtime_txt(
    options, script_start_time, checkpoint1_time, checkpoint2_time, 
    checkpoint3_time, checkpoint4_time, checkpoint5_time):

    ## Elapsed time for MegaDetector
    megadetector_elapsed = checkpoint2_time - checkpoint1_time

    ## Elapsed time for visualisation
    if options.render_output_video:
        vis_elapsed = checkpoint5_time - checkpoint4_time
    else:
        vis_elapsed = 0

    ## Elapsed time for species classification
    classification_elapsed = checkpoint4_time - checkpoint3_time    

    ## Elapsed time for manual check
    if options.check_accuracy:
        manual_id_elapsed = checkpoint1_time - script_start_time
        true_vs_pred_elapsed = checkpoint3_time - checkpoint2_time
        check_acc_elapsed = manual_id_elapsed + true_vs_pred_elapsed
    else:
        check_acc_elapsed = 0
    
    ## Total elapsed time
    script_elapsed = time.time() - script_start_time

    lines = [
        'Runtime for MegaDetector = {}'.format(
            humanfriendly.format_timespan(megadetector_elapsed)),
        'Runtime for Species Classification = {}'.format(
            humanfriendly.format_timespan(classification_elapsed)),
        'Runtime for visualisation of bounding boxes = {}'.format(
            humanfriendly.format_timespan(vis_elapsed)),
        'Runtime for manual ID vs MegaDetector comparison = {}'.format(
            humanfriendly.format_timespan(check_acc_elapsed)),
        'Total script runtime = {}'.format(
            humanfriendly.format_timespan(script_elapsed))
    ]

    runtime_txt_file = os.path.join(options.output_files_dir, 
                                    'script_runtime.txt')
    with open(runtime_txt_file, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    return script_elapsed


def main(): 
    script_start_time = time.time()

    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)
    
    ## Check arguments
    assert os.path.isdir(options.input_dir),\
        '{} is not a folder'.format(options.input_dir)

    options.output_files_dir = os.path.join(options.output_dir, 
                                            "SINClassifier_outputs")
    check_output_dir(options)

    ## Getting the results from manual identifications
    # Run this first to ensure that all species are in species_database.csv
    if options.check_accuracy:
        manual_ID_results(options)

    checkpoint1_time = time.time()

    ## Detecting subjects in each video frame using MegaDetector
    ## Includes rolling prediction averaging across frames
    if not options.resume_from_checkpoint:
        video_dir_to_frames(options)
    det_frames(options)
    
    checkpoint2_time = time.time()

    ## Comparing results of manual identification with MegaDetector detections
    if options.check_accuracy:
        
        acc_pd, video_summ = true_vs_pred(options)

        options.roll_avg_acc_csv = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.roll_avg_acc_csv, '_roll_avg_acc.csv'
        )
        acc_pd.to_csv(options.roll_avg_acc_csv, index = False)

        options.manual_vs_md_csv = default_path_from_none(
            options.output_files_dir, options.input_dir, 
            options.manual_vs_md_csv, '_manual_vs_md.csv'
        )
        video_summ.to_csv(options.manual_vs_md_csv, index = False)

        export_fn(options, video_summ)

    checkpoint3_time = time.time()
    
    ## Running species classifications 
    if options.run_species_classifier: 
        # Cropping out bounding box detections of animals
        crop_detections(options)

        crop_log = load_crop_log(options)
        num_crops = crop_log['num_new_crops']
        if num_crops == 0: 
            # Skip classifier if there are no animal crops
            detector_json = options.roll_avg_frames_json
            detector_csv = options.roll_avg_video_csv

        else:
            # Classifying cropped bounding boxes to species 
            sp_classifier(options)
            merge_classifier(options)
            rolling_avg(options.classification_frames_json, options, 
                        species_classifications = True, mute = False)

            detector_json = options.classification_roll_avg_frames_json
            detector_csv = options.classification_roll_avg_video_csv

    else:
        detector_json = options.roll_avg_frames_json
        detector_csv = options.roll_avg_video_csv

    checkpoint4_time = time.time()

    ## Annotating and exporting to video
    if options.render_output_video:
        vis_detection_videos(options, detector_json, parallel = True)

    # If we are not checking accuracy, means we are using to sort unknown videos
    # Thus filter out the false triggers and human videos
    if not options.check_accuracy: 
        sort_videos(options, detector_csv)

    ## Delete the frames stored in the temp folder (if delete_output_frames = T)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)
        delete_temp_dir(options.cropped_images_dir)
    else:
        print('Frames of videos not deleted and saved in {}'.format(
            options.frame_folder))

    checkpoint5_time = time.time()

    script_elapsed = runtime_txt(
        options, script_start_time, 
        checkpoint1_time, checkpoint2_time, checkpoint3_time, checkpoint4_time, 
        checkpoint5_time)
    print('Completed! Script successfully excecuted in {}'.format(
        humanfriendly.format_timespan(script_elapsed)))


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

    main()