import os
import tempfile
import argparse
from uuid import uuid1

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object
from classification.crop_detections import main as crop_detections_ms


def crop_detections():

    ## Create temp directory for cropped images if not provided
    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    if options.cropped_images_dir is not None:
        cropped_images_dir = options.cropped_images_dir
    else:
        cropped_images_dir = os.path.join(
            tempdir, os.path.basename(options.input_dir) + '_croppedimgs_' + str(uuid1()))
    os.makedirs(cropped_images_dir, exist_ok=True)

    ## Run crop_detections
    crop_detections_ms(detections_json_path = options.full_det_frames_json, 
                    cropped_images_dir = cropped_images_dir, 
                    images_dir = options.frame_folder,
                    container_url = None, 
                    detector_version = 'md_v5a', #TODO find from model
                    save_full_images = True,
                    square_crops = True,
                    check_crops_valid = True, 
                    confidence_threshold = options.rendering_confidence_threshold,
                    threads = options.n_cores,
                    logdir = cropped_images_dir)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to crop out detections.')
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--full_det_frames_json', type=str,
                        default = config.FULL_DET_FRAMES_JSON, 
                        help = 'Path to json file containing the frame-level results of all MegaDetector detections.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--frame_folder', type=str, 
                        default = config.FRAME_FOLDER, 
                        help = 'Folder to use for intermediate frame storage, defaults to a folder in the system temporary folder'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
                        help = "don't render boxes with confidence below this threshold"
    )
    parser.add_argument('--n_cores', type=int,
                        default = config.N_CORES, 
                        help = 'number of cores to use for detection and cropping (CPU only)'
    )
    parser.add_argument('--cropped_images_dir', type=str,
                        default = config.CROPPED_IMAGES_DIR, 
                        help = 'Path to folder to save the cropped animal images, defaults to a folder in the system temporary folder'
    )
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    crop_detections()
