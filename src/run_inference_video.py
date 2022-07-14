import os
import time
import argparse
import humanfriendly

# Functions imported from this project
import config
from shared_utils import delete_temp_dir, VideoOptions, check_output_dir
from shared_utils import default_path_from_none, export_fn, sort_empty_human
from run_det_video import video_dir_to_frames, det_frames
from vis_detections import vis_detection_videos
from manual_ID import manual_ID_results
from optimise_roll_avg import true_vs_pred

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def runtime_txt(options, script_start_time, checkpoint1_time, checkpoint2_time, checkpoint3_time, checkpoint4_time):

    if options.check_accuracy:
        manual_id_elapsed = checkpoint1_time - script_start_time
        true_vs_pred_elapsed = checkpoint3_time - checkpoint2_time
        check_acc_elapsed = manual_id_elapsed + true_vs_pred_elapsed
    else:
        check_acc_elapsed = 0

    megadetector_elapsed = checkpoint2_time - checkpoint1_time

    if options.render_output_video:
        vis_elapsed = checkpoint4_time - checkpoint3_time
    else:
        vis_elapsed = 0

    script_elapsed = time.time() - script_start_time

    lines = [
        'Runtime for MegaDetector = {}'.format(humanfriendly.format_timespan(megadetector_elapsed)),
        'Runtime for visualisation of bounding boxes = {}'.format(humanfriendly.format_timespan(vis_elapsed)),
        'Runtime for manual identification vs MegaDetector detections comparison = {}'.format(humanfriendly.format_timespan(check_acc_elapsed)),
        'Total script runtime = {}'.format(humanfriendly.format_timespan(script_elapsed))
    ]

    runtime_txt_file = os.path.join(options.output_dir, 'script_runtime.txt')
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
    assert os.path.isdir(options.input_dir),'{} is not a folder'.format(options.input_dir)

    check_output_dir(options)

    ## Getting the results from manual identifications
    # Run this first to ensure that all species are in species_database.csv
    if options.check_accuracy:
        manual_ID_results(options)

    checkpoint1_time = time.time()

    ## Detecting subjects in each video frame using MegaDetector
    if not options.resume_from_checkpoint:
        video_dir_to_frames(options)
    det_frames(options)
    
    checkpoint2_time = time.time()

    ## Comparing results of manual identification with MegaDetector detections
    if options.check_accuracy:
        
        acc_pd, video_summ = true_vs_pred(options)

        options.roll_avg_acc_csv = default_path_from_none(
            options.output_dir, options.input_dir, 
            options.roll_avg_acc_csv, '_roll_avg_acc.csv'
        )
        acc_pd.to_csv(options.roll_avg_acc_csv, index = False)

        options.manual_vs_md_csv = default_path_from_none(
            options.output_dir, options.input_dir, 
            options.manual_vs_md_csv, '_manual_vs_md.csv'
        )
        video_summ.to_csv(options.manual_vs_md_csv, index = False)

        export_fn(options, video_summ)

    checkpoint3_time = time.time()

    ## Annotating and exporting to video
    if options.render_output_video:
        vis_detection_videos(options, parallel = True)

    # If we are not checking accuracy, means we are using to sort unknown videos
    # Thus filter out the false triggers and human videos
    if not options.check_accuracy: 
        sort_empty_human(options)

    ## Delete the frames stored in the temp folder (if delete_output_frames == TRUE)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)
    else:
        print('Frames of videos not deleted and saved in {}'.format(options.frame_folder))

    checkpoint4_time = time.time()

    script_elapsed = runtime_txt(
        options, script_start_time, 
        checkpoint1_time, checkpoint2_time, checkpoint3_time, checkpoint4_time)
    print('Completed! Script successfully excecuted in {}'.format(humanfriendly.format_timespan(script_elapsed)))


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = config.MODEL_FILE, 
                        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--recursive', type=bool, 
                        default = config.RECURSIVE, 
                        help = 'recurse into [input_dir]; only meaningful if a folder is specified as input'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--render_output_video', type=bool,
                        default = config.RENDER_OUTPUT_VIDEO, 
                        help = 'Enable video output rendering.'
    )
    parser.add_argument('--frame_folder', type=str, 
                        default = config.FRAME_FOLDER, 
                        help = 'Folder to use for intermediate frame storage, defaults to a folder in the system temporary folder'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default = config.DELETE_OUTPUT_FRAMES, 
                        help = 'enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--n_cores', type=int,
                        default = config.N_CORES, 
                        help = 'number of cores to use for detection (CPU only)'
    )
    parser.add_argument('--frame_sample', type=int,
                        default = config.FRAME_SAMPLE, 
                        help = 'procss every Nth frame (defaults to every frame)'
    )
    parser.add_argument('--debug_max_frames', type=int,
                        default = config.DEBUG_MAX_FRAMES, 
                        help = 'trim to N frames for debugging (impacts model execution, not frame rendering)'
    )
    parser.add_argument('--check_accuracy', type=bool,
                        default = config.CHECK_ACCURACY, 
                        help = 'Whether accuracy of MegaDetector should be checked with manual ID. Folder names must contain species and quantity.'
    )
    parser.add_argument('--species_database_file', type=str,
                        default = config.SPECIES_DATABASE_FILE, 
                        help = 'Path to the species_database.csv which describes details of species.'
    )
    parser.add_argument('--manual_id_csv', type=str,
                        default = config.MANUAL_ID_CSV, 
                        help = 'Path to csv file containing the results of manual identification detections.'
    )
    parser.add_argument('--roll_avg_video_csv', type=str,
                        default = config.ROLL_AVG_VIDEO_CSV, 
                        help = 'Path to the roll_avg_video csv file.'
    )
    parser.add_argument('--roll_avg_frames_json', type=str,
                        default = config.ROLL_AVG_FRAMES_JSON, 
                        help = 'Path to the roll_avg_frames_json file.'
    )
    parser.add_argument('--rolling_avg_size', type=float,
                        default = config.ROLLING_AVG_SIZE, 
                        help = "Number of frames in which rolling prediction averaging will be calculated from. A larger number will remove more inconsistencies, but will cause delays in detecting an object."
    )
    parser.add_argument('--iou_threshold', type=float,
                        default = config.IOU_THRESHOLD, 
                        help = "Threshold for Intersection over Union (IoU) to determine if bounding boxes are detecting the same object."
    )
    parser.add_argument('--conf_threshold_buf', type=float,
                        default = config.CONF_THRESHOLD_BUF, 
                        help = "Buffer for the rendering confidence threshold to allow for 'poorer' detections to be included in rolling prediction averaging."
    )
    parser.add_argument('--nth_highest_confidence', type=float,
                        default = config.NTH_HIGHEST_CONFIDENCE, 
                        help="nth-highest-confidence frame to choose a confidence value for each video"
    )
    parser.add_argument('--resume_from_checkpoint', type=str,
                        default = config.RESUME_FROM_CHECKPOINT, 
                        help="path to JSON checkpoint file for which MD detection will be resumed from."
    )
    parser.add_argument('--checkpoint_path', type=str,
                        default = config.CHECKPOINT_PATH, 
                        help="path to JSON checkpoint file for which checkpoints will be written to."
    )
    parser.add_argument('--checkpoint_frequency', type=int,
                        default = config.CHECKPOINT_FREQUENCY, 
                        help="write results to JSON checkpoint file every N images."
    )
    return parser


if __name__ == '__main__':

    main()
