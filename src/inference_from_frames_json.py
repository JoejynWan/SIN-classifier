import json
import argparse

# Functions imported from this project
import config
from shared_utils import delete_temp_dir, VideoOptions, default_path_from_none
from shared_utils import write_video_results, export_fn
from vis_detections import vis_detection_videos
from optimise_roll_avg import true_vs_pred
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


def main():

    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)
    
    ## Load data
    with open(options.full_det_frames_json,'r') as f:
        input_data = json.load(f)

    results = input_data['images']
    Fs = input_data['videos']['frame_rates']

    ## Converting frames.json to videos.json
    options.full_det_video_json = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.full_det_video_json, '_full_det_video.json'
    )
    write_video_results(options.full_det_video_json, frames_json_inputdata = input_data)

    ## Rolling prediction average 
    rolling_avg(options, results, Fs, relative_path_base = None)

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

    ## Annotating and exporting to video
    if options.render_output_video:
        vis_detection_videos(options)

    ## Delete the frames stored in the temp folder (if delete_output_frames == TRUE)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)
    else:
        print('Frames of videos not deleted and saved in {}'.format(options.frame_folder))


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Script to continue inference from full_det_frames.json if a MemoryError occurs.')
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
    parser.add_argument('--full_det_frames_json', type=str,
                        default = config.FULL_DET_FRAMES_JSON, 
                        help = 'Path to json file containing the frame-level results of all MegaDetector detections.'
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
    return parser


if __name__ == '__main__':

    main()