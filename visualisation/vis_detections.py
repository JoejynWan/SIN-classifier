import os
import cv2
import argparse
import tempfile
from pickle import TRUE

# Imported from Microsoft/CameraTraps github repository
from detection.process_video import ProcessVideoOptions
from visualization.visualize_detector_output import visualize_detector_output
from ct_utils import args_to_object


def frames_to_video(images, Fs, output_file_name, codec_spec = 'DIVX'):
    """
    Given a list of image files and a sample rate, concatenate those images into
    a video and write to [output_file_name].
    """
    
    if len(images) == 0:
        print('No images/ image files are supplied to convert to video')
        return

    # Determine width and height from the first image
    frame = cv2.imread(images[0])
    height, width, channels = frame.shape 

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, Fs, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def vis_detections(tempdir, input_video_file, detector_output_path, images_dir, output_video_file):
    ## Rendering detections to images 
    rendering_output_dir = os.path.join(tempdir, os.path.basename(input_video_file) + '_detections')
    
    detected_frame_files = visualize_detector_output(
        detector_output_path = detector_output_path,
        out_dir = rendering_output_dir,
        images_dir = images_dir,
        confidence = 0.8)
    
    frames_to_video(detected_frame_files, 30, output_video_file)

    delete_temp_dir(rendering_output_dir)

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
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json'
    )
    parser.add_argument('--render_output_video', type=bool,
                        default = default_render_output_video, 
                        help='enable video output rendering (not rendered by default)'
    )
    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file (or folder), defaults to [video file].mp4 for files, or [video file]_annotated] for folders'
    )
    parser.add_argument('--frame_folder', type=str, default=None,
                        help='folder to use for intermediate frame storage, defaults to a folder in the system temporary folder')

    parser.add_argument('--delete_output_frames', type=bool,
                        default=True, help='enable/disable temporary file deletion (default True)'
    )
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

    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    images_dir = 'C:/Users/joejyn/AppData/Local/Temp/process_camera_trap_video/test_frames_ec9a556e-84e3-11ec-a182-f4463780b21c'
    detector_output_path = 'C:/temp_for_SSD_speed/test.frames.json'
    output_video_file = 'C:/temp_for_SSD_speed/test_anno.avi'
    input_video_file = "C:/temp_for_SSD_speed/test"

    vis_detections(options, input_video_file, detector_output_path, images_dir, output_video_file, tempdir)

if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_video_file = "C:/temp_for_SSD_speed/test"
    default_n_cores = '15'
    default_render_output_video = TRUE

    main()