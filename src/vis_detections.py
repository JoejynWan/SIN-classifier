import os
import cv2
import argparse
import tempfile
from pickle import TRUE

# Functions imported from this project
from shared_utils import delete_temp_dir

# Imported from Microsoft/CameraTraps github repository
from visualization.visualize_detector_output import visualize_detector_output


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


def vis_detection_video(tempdir, input_video_file, detector_output_path, images_dir, output_video_file, confidence):
    ## Rendering detections to images 
    rendering_output_dir = os.path.join(tempdir, os.path.basename(input_video_file) + '_detections')
    
    detected_frame_files = visualize_detector_output(
        detector_output_path = detector_output_path,
        out_dir = rendering_output_dir,
        images_dir = images_dir,
        confidence = confidence)
    
    frames_to_video(detected_frame_files, 30, output_video_file)

    delete_temp_dir(rendering_output_dir)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help='Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_videos', type=str, 
                        default = default_input_videos, 
                        help='Path to folder containing videos to process'
    )
    parser.add_argument('--input_video_frames', type = str,
                        default = default_input_video_frames,
                        help = 'Path to folder containing the video frames processed by det_videos.py'
    )
    parser.add_argument('--input_frames_anno_file', type=str,
                        default = default_input_frames_anno_file, 
                        help = '.json file depicting the detections for each frame of the video'
    )
    parser.add_argument('--output_video_file', type=str,
                        default = default_output_video_file, 
                        help='.json output file, defaults to [video file].json'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default=True, help='enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold"
    )
    return parser


def main():
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')

    vis_detection_video(tempdir, args.input_videos, args.input_frames_anno_file, 
                        args.input_video_frames, args.output_video_file, 
                        args.rendering_confidence_threshold)


if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_videos = "C:/temp_for_SSD_speed/CT_models_test"
    default_input_video_frames = 'C:/temp_for_SSD_speed/CT_models_test_frames'
    default_input_frames_anno_file = 'C:/temp_for_SSD_speed/CT_models_test_frames/CT_models_test.frames.json'
    default_output_video_file = 'C:/temp_for_SSD_speed/CT_models_test_frames/CT_models_test_anno.avi'

    main()