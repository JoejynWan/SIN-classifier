import os
import json
import argparse
import tempfile
import itertools
import datetime
from math import ceil
from uuid import uuid1
from pickle import FALSE, TRUE
import tensorflow.compat.v1 as tf

# Functions imported from this project
import config
from shared_utils import delete_temp_dir, VideoOptions, write_results
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_detector_batch import load_and_run_detector_batch
from detection.video_utils import video_folder_to_frames
from ct_utils import args_to_object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #set to ignore INFO messages


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
            tempdir, os.path.basename(options.input_dir) + '_frames_' + str(uuid1()))
        options.frame_folder = frame_output_folder
    os.makedirs(frame_output_folder, exist_ok=True)
    
    print("Saving videos as frames in {}...".format(frame_output_folder))
    frame_filenames, Fs = video_folder_to_frames(
        input_folder = options.input_dir, output_folder_base = frame_output_folder, 
        recursive = options.recursive, overwrite = True,
        n_threads = options.n_cores, every_n_frames = options.frame_sample)
    
    image_file_names = list(itertools.chain.from_iterable(frame_filenames))

    return image_file_names, Fs


def checkpointing_test(options):
    # Load the checkpoint if available
    # Relative file names are only output at the end; all file paths in the checkpoint are
    # still full paths.
    if options.resume_from_checkpoint:
        assert os.path.exists(options.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(options.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Test that we can write to the output_file's dir if checkpointing requested
    if options.checkpoint_frequency != -1:
        
        if options.checkpoint_path is not None:
            checkpoint_path = options.checkpoint_path
        else:
            checkpoint_path = os.path.join(options.output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        
        # Confirm that we can write to the checkpoint path, rather than failing after 10000 images
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
        
    else:
        
        checkpoint_path = None

    return results, checkpoint_path


def det_frames(options, image_file_names, Fs, run_chunks = False):
    
    os.makedirs(options.output_dir, exist_ok=True)

    num_images = len(image_file_names)
    print("Running MegaDetector for {} video frames.".format(num_images))

    if run_chunks: 
        def chunk(list, chunk_size):
            for i in range(0, len(list), chunk_size):
                yield list[i:i+chunk_size]

        results = []
        chunk_num = 1
        chunk_size = 10000
        for chunk_image_file_names in chunk(image_file_names, chunk_size):
            
            num_chunks = ceil(num_images/chunk_size)
            print("Running for chunk number {} of {}.".format(chunk_num, num_chunks))
            
            chunk_results = load_and_run_detector_batch(
                options.model_file, chunk_image_file_names,
                confidence_threshold=options.json_confidence_threshold,
                n_cores=options.n_cores, quiet = True)

            tf.reset_default_graph()

            results.extend(chunk_results)
            chunk_num = chunk_num +1
    else:
        results, checkpoint_path = checkpointing_test(options)

        results = load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.json_confidence_threshold, n_cores=options.n_cores, 
            checkpoint_path = checkpoint_path, checkpoint_frequency = options.checkpoint_frequency, 
            quiet = True)

    ## Save and export results of full detection
    write_results(options, results, Fs, relative_path_base = options.frame_folder)

    ## Rolling prediction average 
    rolling_avg(options, results, Fs, relative_path_base = options.frame_folder)


def main(): 
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    # Split videos into frames
    assert os.path.isdir(options.input_dir),'{} is not a folder'.format(options.input_dir)  

    image_file_names, Fs = video_dir_to_frames(options)

    # Run MegaDetector on each frame
    det_frames(options, image_file_names, Fs)

    ## Delete the frames stored in the temp folder (if delete_output_frames == TRUE)
    if options.delete_output_frames:
        delete_temp_dir(options.frame_folder)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = config.MODEL_FILE, 
                        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'video file (or folder) to process'
    )
    parser.add_argument('--recursive', type=bool, 
                        default = config.RECURSIVE, 
                        help='recurse into [input_dir]; only meaningful if a folder is specified as input'
    )
    parser.add_argument('--render_output_video', type=bool,
                        default = config.RENDER_OUTPUT_VIDEO, 
                        help = 'enable video output rendering (not rendered by default)'
    )
    parser.add_argument('--frame_folder', type=str, 
                        default = config.FRAME_FOLDER,
                        help = 'folder to use for intermediate frame storage, defaults to a folder in the system temporary folder'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default = config.DELETE_OUTPUT_FRAMES, 
                        help = 'enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--nth_highest_confidence', type=float,
                        default = config.NTH_HIGHEST_CONFIDENCE, 
                        help = "nth-highest-confidence frame to choose a confidence value for each video"
    )
    parser.add_argument('--n_cores', type=int,
                        default = config.N_CORES, 
                        help = 'number of cores to use for detection (CPU only)'
    )
    parser.add_argument('--frame_sample', type=int,
                        default = config.FRAME_SAMPLE, 
                        help='procss every Nth frame (defaults to every frame)'
    )
    parser.add_argument('--debug_max_frames', type=int,
                        default = config.DEBUG_MAX_FRAMES, 
                        help='trim to N frames for debugging (impacts model execution, not frame rendering)'
    )
    return parser


if __name__ == '__main__':
    ## Defining arguments within this script
    # Comment out if passing arguments from terminal directly

    main()
