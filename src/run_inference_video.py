import os
import time
import argparse
import humanfriendly

# Functions imported from this project
import config
from shared_utils import delete_temp_dir, VideoOptions, check_output_dir
from run_det_video import video_dir_to_frames, det_frames
from vis_detections import vis_detection_videos
from manual_ID import manual_ID_results

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


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
    # Run this first to ensure that all species are in species_database.csv
    if options.check_accuracy:
        manual_ID_results(options)

        manual_det_endtime = time.time()

    ## Detecting subjects in each video frame using MegaDetector
    image_file_names, Fs = video_dir_to_frames(options)
    det_frames(options, image_file_names, Fs)
    
    megadetector_endtime = time.time()
    
    ## Annotating and exporting to video
    if options.render_output_video:
        vis_detection_videos(options)

        vis_endtime = time.time()

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
    return parser


if __name__ == '__main__':

    main()
