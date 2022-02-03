import os
import argparse
import tempfile
import itertools
from uuid import uuid1

# Imported from Microsoft/CameraTraps github repository
from detection.process_video import ProcessVideoOptions
from detection.video_utils import video_folder_to_frames, frame_results_to_video_results
from detection import run_tf_detector_batch
from ct_utils import args_to_object


def video_dir_to_frames(options):
    ## Split every video into frames, save frames in temp dir, and return the full path of each frame
    if options.frame_folder is not None:
        frame_output_folder = options.frame_folder
    else:
        tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
        frame_output_folder = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
    os.makedirs(frame_output_folder, exist_ok=True)

    frame_filenames, Fs = video_folder_to_frames(input_folder=options.input_video_file, output_folder_base=frame_output_folder, 
                            recursive=options.recursive, overwrite=True,
                            n_threads=options.n_cores,every_n_frames=options.frame_sample)
    
    image_file_names = list(itertools.chain.from_iterable(frame_filenames))

    return image_file_names, frame_output_folder


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help='Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_video_file', type=str, 
                        default = default_input_video_file, 
                        help='video file (or folder) to process'
    )
    parser.add_argument('--recursive', type=bool, default=True, 
                        help='recurse into [input_video_file]; only meaningful if a folder is specified as input'
    )
    parser.add_argument('--frame_folder', type=str, default=None,
                        help='folder to use for intermediate frame storage, defaults to a folder in the system temporary folder')
                        
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')

    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file (or folder), defaults to [video file].mp4 for files, or [video file]_annotated] for folders')

    parser.add_argument('--render_output_video', action='store_true',
                        help='enable video output rendering (not rendered by default)')

    parser.add_argument('--delete_output_frames', type=bool,
                        default=True, help='enable/disable temporary file deletion (default True)')

    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold")

    parser.add_argument('--json_confidence_threshold', type=float,
                        default=0.0, help="don't include boxes in the .json file with confidence below this threshold")

    parser.add_argument('--n_cores', type=int,
                        default = default_n_cores, help='number of cores to use for detection (CPU only)')

    parser.add_argument('--frame_sample', type=int,
                        default=None, help='procss every Nth frame (defaults to every frame)')

    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='trim to N frames for debugging (impacts model execution, not frame rendering)')
    return parser


def main(): 
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args, options)

    ## Processing videos with MegaDetector
    assert os.path.isdir(options.input_video_file),'{} is not a folder'.format(options.input_video_file)

    # Save videos as frames in a temp folder 
    print("Saving videos as frames... ...")
    image_file_names, frame_output_folder = video_dir_to_frames(options)
    print("Success. Video frames saved in {}".format(frame_output_folder))

    # Run MegaDetector on each frame
    if options.output_json_file is None:
        frames_json = options.input_video_file + '.frames.json'
        video_json = options.input_video_file + '.json'
    else:
        if '.json' in options.output_json_file:
            frames_json = options.output_json_file.replace('.json','.frames.json')
            video_json = options.output_json_file
        else:
            video_json = options.output_json_file
            frames_json = video_json + '_frames'
            
    if options.reuse_results_if_available and \
        os.path.isfile(frames_json):
            print('Loading results from {}'.format(frames_json))
            results = None
    else:
        results = run_tf_detector_batch.load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.json_confidence_threshold,
            n_cores=options.n_cores)
    
        run_tf_detector_batch.write_results_to_file(
            results, frames_json,
            relative_path_base=frame_output_folder)

    frame_results_to_video_results(frames_json,video_json)


if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_video_file = "C:/temp_for_SSD_speed/CT_models_test"
    default_n_cores = '15'

    main()
