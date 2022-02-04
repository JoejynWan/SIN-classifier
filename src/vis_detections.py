import os
import cv2
import json
import argparse
import tempfile
from pickle import TRUE

# Functions imported from this project
from shared_utils import delete_temp_dir, unique

# Imported from Microsoft/CameraTraps github repository
from visualization.visualize_detector_output import visualize_detector_output

def create_save_paths(input_frames_anno_file, output_dir):
    with open(input_frames_anno_file) as f:
        detector_output = json.load(f)
    assert 'images' in detector_output, (
        'Detector output file should be a json with an "images" field.')
    images = detector_output['images']

    frames_paths = []
    output_video_names = []
    output_save_paths = []
    for entry in images:
        frames_path = os.path.dirname(entry['file'])      
        frames_paths.append(frames_path)
        output_frames_path = os.path.join(output_dir, frames_path)
        output_video_names.append(os.path.basename(output_frames_path))
        output_save_paths.append(os.path.dirname(output_frames_path))

    [os.makedirs(make_path, exist_ok=True) for make_path in unique(output_save_paths)]

    return frames_paths, output_video_names, output_save_paths

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


def vis_detection_video(tempdir, detector_output_path, images_dir, output_video_file, confidence):
    ## Rendering detections to images 
    rendering_output_dir = os.path.join(tempdir, 'detection_frames')
    
    detected_frame_files = visualize_detector_output(
        detector_output_path = detector_output_path,
        out_dir = rendering_output_dir,
        images_dir = images_dir,
        confidence = confidence)
    
    frames_to_video(detected_frame_files, 30, output_video_file)

    delete_temp_dir(rendering_output_dir)


def vis_detection_videos(tempdir, input_frames_anno_file, input_frames_base_dir, output_dir, confidence):
    
    frames_paths, output_video_names, output_save_paths = create_save_paths(input_frames_anno_file, output_dir)

    for frame_path, save_path, video_name in zip(frames_paths, output_save_paths, output_video_names): 
        
        input_frames_dir = os.path.join(input_frames_base_dir, frame_path)
        output_video_file = os.path.join(save_path, video_name)

        vis_detection_video(tempdir, input_frames_anno_file, 
                            input_frames_dir, output_video_file, confidence)


def main():
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')

    vis_detection_videos(tempdir, args.input_frames_anno_file, 
                        args.input_frames_base_dir, args.output_dir, 
                        args.rendering_confidence_threshold)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help='Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--input_frames_base_dir', type = str,
                        default = default_input_frames_base_dir,
                        help = 'Path to folder containing the video frames processed by det_videos.py'
    )
    parser.add_argument('--input_frames_anno_file', type=str,
                        default = default_input_frames_anno_file, 
                        help = '.json file depicting the detections for each frame of the video'
    )
    parser.add_argument('--output_dir', type=str,
                        default = default_output_dir, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--delete_output_frames', type=bool,
                        default=True, help='enable/disable temporary file deletion (default True)'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold"
    )
    return parser


if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = "MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_input_frames_base_dir = 'C:/temp_for_SSD_speed/CT_models_test_frames'
    default_input_frames_anno_file = 'C:/temp_for_SSD_speed/CT_models_test_frames/CT_models_test.frames.json'
    default_output_dir = 'D:/CNN_Animal_ID/CamphoraClassifier/SIN-classifier/results/CT_models_test'

    main()