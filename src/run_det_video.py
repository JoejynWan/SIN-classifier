import os
import argparse
import tempfile
import itertools
from uuid import uuid1
from pickle import FALSE, TRUE

# Functions imported from this project
from shared_utils import delete_temp_dir, VideoOptions, make_output_path
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_tf_detector_batch import load_and_run_detector_batch
from detection.run_tf_detector_batch import write_results_to_file
from detection.video_utils import video_folder_to_frames
from detection.video_utils import frame_results_to_video_results
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


def make_default_json(options):
    if options.full_det_frames_json is None:
        options.full_det_frames_json = make_output_path(
            options.output_dir, options.input_video_file, '_full_det_frames.json')

    if options.full_det_video_json is None:
        options.full_det_video_json = make_output_path(
            options.output_dir, options.input_video_file, '_full_det_videos.json')

    if options.roll_avg_frames_json is None:
        options.roll_avg_frames_json = make_output_path(
            options.output_dir, options.input_video_file, '_roll_avg_frames.json')

    if options.roll_avg_video_json is None:
        options.roll_avg_video_json = make_output_path(
            options.output_dir, options.input_video_file, '_roll_avg_videos.json')
    
    return options


def video_dir_to_frames(options):
    """
    Split every video into frames, save frames in temp dir, 
    and return the full path of each frame. 
    """

    if options.frame_folder is not None:
        frame_output_folder = options.frame_folder
    else:
        tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
        frame_output_folder = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
        options.frame_folder = frame_output_folder
    os.makedirs(frame_output_folder, exist_ok=True)
    
    print("Saving videos as frames in {}...".format(frame_output_folder))
    frame_filenames, Fs = video_folder_to_frames(
        input_folder = options.input_video_file, output_folder_base = frame_output_folder, 
        recursive = options.recursive, overwrite = True,
        n_threads = options.n_cores, every_n_frames = options.frame_sample)
    
    image_file_names = list(itertools.chain.from_iterable(frame_filenames))

    return image_file_names, Fs


def det_frames(options, image_file_names):
    ## Create the paths to save the .json file describing detections for each frame and video    
    options = make_default_json(options)
    os.makedirs(options.output_dir, exist_ok=True)

    ## Run detections on each frame
    print("Running MegaDetector on each video frame...")
    results = load_and_run_detector_batch(
        options.model_file, image_file_names,
        confidence_threshold=options.json_confidence_threshold,
        n_cores=options.n_cores)

    ## Save and export results of full detection
    write_results_to_file(results, options.full_det_frames_json, options.frame_folder)
    frame_results_to_video_results(options.full_det_frames_json, options.full_det_video_json)

    ## Rolling prediction average 
    roll_avg, _, _, _, _ = rolling_avg(options, results)

    ## Save and export results of rolling average
    write_results_to_file(roll_avg, options.roll_avg_frames_json, options.frame_folder)
    frame_results_to_video_results(options.roll_avg_frames_json, options.roll_avg_video_json)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_video_file', type=str, 
                        default = default_input_video_file, 
                        help = 'video file (or folder) to process'
    )
    parser.add_argument('--recursive', type=bool, 
                        default=True, 
                        help='recurse into [input_video_file]; only meaningful if a folder is specified as input'
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
                        default = default_render_output_video, 
                        help = 'enable video output rendering (not rendered by default)'
    )
    parser.add_argument('--frame_folder', type=str, 
                        default = default_frame_folder,
                        help = 'folder to use for intermediate frame storage, defaults to a folder in the system temporary folder'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default=False, help='enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--n_cores', type=int,
                        default = default_n_cores, help='number of cores to use for detection (CPU only)'
    )
    parser.add_argument('--frame_sample', type=int,
                        default=None, help='procss every Nth frame (defaults to every frame)'
    )
    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='trim to N frames for debugging (impacts model execution, not frame rendering)'
    )
    return parser


def main(): 
    # Split videos into frames
    assert os.path.isdir(options.input_video_file),'{} is not a folder'.format(options.input_video_file)  

    image_file_names, Fs = video_dir_to_frames(options)

    # Run MegaDetector on each frame
    det_frames(options, image_file_names)

    ## Delete the frames stored in the temp folder (if delete_output_frames == TRUE)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)


if __name__ == '__main__':
    ## Defining arguments within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "../MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_video_file = "C:/temp_for_SSD_speed/CT_models_test"
    default_frame_folder = "C:/temp_for_SSD_speed/CT_models_test_frames"
    default_n_cores = '15'
    default_render_output_video = FALSE

    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    main()
