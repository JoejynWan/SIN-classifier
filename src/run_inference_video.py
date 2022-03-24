import os
import sys
import time
import argparse
import humanfriendly

# Functions imported from this project
from shared_utils import delete_temp_dir, VideoOptions, make_output_path
from run_det_video import video_dir_to_frames, det_frames
from vis_detections import vis_detection_videos
from video_metadata import manual_ID_csv

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def check_output_dir(options):
    if os.path.exists(options.output_dir) and os.listdir(options.output_dir):
        while True:
            rewrite_input = input('\nThe output directory specified is not empty. Do you want to continue and rewrite the files in the directory? (y/n)')
            if rewrite_input.lower() not in ('y', 'n'):
                print('Input invalid. Please enter y or n.')
            else:
                break
        
        if rewrite_input.lower() == 'n':
            sys.exit('Stopping script. Please input the correct output directory. ')


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

    ## TODO
    # Check the memory of the output_dir and temp_dir to ensure that there is 
    # sufficient space to save the frames and videos 

    ## Getting the results from manual detection
    if options.check_accuracy:
        manualID_file = make_output_path(options.output_dir, options.input_dir, "_manual_ID.csv")
        manual_ID_csv(options.input_dir, options.species_database_file, manualID_file)

    ## Detecting subjects in each video frame using MegaDetector
    image_file_names, Fs = video_dir_to_frames(options)
    det_frames(options, image_file_names, Fs)

    ## Annotating and exporting to video
    if options.render_output_video:
        vis_detection_videos(options)

    ## Delete the frames stored in the temp folder (if delete_output_frames == TRUE)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)
    else:
        print('Frames saved in {}'.format(options.frame_folder))

    script_elapsed = time.time() - script_start_time
    print('Completed! Script successfully excecuted in {}'.format(humanfriendly.format_timespan(script_elapsed)))


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_dir', type=str, 
                        default = default_input_dir, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--recursive', type=bool, 
                        default = True, help = 'recurse into [input_dir]; only meaningful if a folder is specified as input'
    )
    parser.add_argument('--output_dir', type=str,
                        default = default_output_dir, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--full_det_frames_json', type=str,
                        default = None, 
                        help = 'Path of json file with all detections for each frame.'
    )
    parser.add_argument('--full_det_video_json', type=str,
                        default = None, 
                        help = 'Path of json file with consolidated detections (consolidated across categories) for each video.'
    )
    parser.add_argument('--roll_avg_frames_json', type=str,
                        default = None, 
                        help = 'Path of json file with rolling-averaged detections for each frame.'
    )
    parser.add_argument('--roll_avg_video_json', type=str,
                        default = None, 
                        help = 'Path of json file with consolidated rolling-averaged detections (consolidated across categories) for each video.'
    )
    parser.add_argument('--render_output_video', type=bool,
                        default = True, 
                        help = 'Enable video output rendering.'
    )
    parser.add_argument('--frame_folder', type=str, 
                        default = default_frame_folder, 
                        help = 'Folder to use for intermediate frame storage, defaults to a folder in the system temporary folder'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default = True, 
                        help = 'enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = 0.8, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--n_cores', type=int,
                        default = 15, help = 'number of cores to use for detection (CPU only)'
    )
    parser.add_argument('--frame_sample', type=int,
                        default = None, help = 'procss every Nth frame (defaults to every frame)'
    )
    parser.add_argument('--debug_max_frames', type=int,
                        default = -1, help = 'trim to N frames for debugging (impacts model execution, not frame rendering)'
    )
    parser.add_argument('--check_accuracy', type=bool,
                        default = True, help = 'Whether accuracy of MegaDetector should be checked with manual ID. Folder names must contain species and quantity.'
    )
    parser.add_argument('--species_database_file', type=str,
                        default = default_species_database_file, help = 'Path to the species_database.csv which describes details of species.'
    )
    return parser


if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "../MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_dir = "data/20211119_test"
    default_output_dir = 'results/20211119_test'
    default_species_database_file = 'data/species_database.csv'
    default_frame_folder = None

    main()
