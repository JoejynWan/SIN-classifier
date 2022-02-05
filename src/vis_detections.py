import os
import cv2
import json
from tqdm import tqdm
import argparse
import tempfile
from pickle import TRUE
from typing import Any
from multiprocessing.pool import ThreadPool

# Functions imported from this project
from shared_utils import delete_temp_dir, unique

# Imported from Microsoft/CameraTraps github repository
from visualization import visualization_utils as vis_utils
from data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int


#%% Constants

# Convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}


#%% Functions

def load_detector_output(detector_output_path):
    with open(detector_output_path) as f:
        detector_output = json.load(f)
    assert 'images' in detector_output, (
        'Detector output file should be a json with an "images" field.')
    images = detector_output['images']

    detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
    if 'detection_categories' in detector_output:
        print('detection_categories provided')
        detector_label_map = detector_output['detection_categories']

    return images, detector_label_map

def find_unqiue_videos(images, output_dir):

    frames_paths = []
    video_save_paths = []
    for entry in images:
        frames_path = os.path.dirname(entry['file'])      
        frames_paths.append(frames_path)
        video_save_paths.append(os.path.join(output_dir, os.path.dirname(frames_path)))

    unqiue_videos = unique(frames_paths)

    [os.makedirs(make_path, exist_ok=True) for make_path in unique(video_save_paths)]

    return unqiue_videos

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


def vis_detector_output(images, detector_label_map, images_dir, out_dir, confidence, output_image_width = 700):
    """
    Args: 
    images = list, detection dictionary for each of the image frames, loaded from the detector_output .json file
    detector_label_map = str, detector categories loaded from the detector_output .json file
    images_dir = str, path to the base folder containing the frames
    out_dir = str, temporary directory where annotated frame images will be saved
    confidence = float, confidence threshold above which annotations will be rendered
    """

    ## Arguments error checking
    assert confidence > 0 and confidence < 1, (
        f'Confidence threshold {confidence} is invalid, must be in (0, 1).')
    
    os.makedirs(out_dir, exist_ok=True)
    
    annotated_img_paths = []
    for entry in images:
        image_id = entry['file']
        
        if 'failure' in entry:
            print(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        image_obj = os.path.join(images_dir, image_id)
        if not os.path.exists(image_obj):
            print(f'Image {image_id} not found in images_dir; skipped.')
            continue

        # resize is for displaying them more quickly
        image = vis_utils.resize_image(
            vis_utils.open_image(image_obj), output_image_width)

        vis_utils.render_detection_bounding_boxes(
            entry['detections'], image, label_map=detector_label_map,
            confidence_threshold=confidence)

        for char in ['/', '\\', ':']:
            image_id = image_id.replace(char, '~')
        annotated_img_path = os.path.join(out_dir, f'anno_{image_id}')
        annotated_img_paths.append(annotated_img_path)
        image.save(annotated_img_path)

    return annotated_img_paths


def vis_detection_videos(tempdir, input_frames_anno_file, input_frames_base_dir, Fs_per_video, output_dir, confidence):
    """
    Args:
    tempdir = str, path to temporary folder where annotated frames will be saved
    input_frames_anno_file = str, path to .json file describing the detections for each frame
    input_frames_base_dir = str, path to base folder containing the video frames
    Fs_per_video = list, frame rate for each unique video
    output_dir = str, path to the base folder where the annotated videos will be saved
    confidence = float, confidence threshold above which annotations will be rendered
    """
    images, detector_label_map = load_detector_output(input_frames_anno_file)

    unqiue_videos = find_unqiue_videos(images, output_dir)
    rendering_output_dir = os.path.join(tempdir, 'detection_frames')

    print('Rendering detections above a confidence threshold of {} for {} videos...'.format(
        confidence, len(unqiue_videos)))
    
    for unqiue_video, Fs in tqdm(zip(unqiue_videos, Fs_per_video), total = len(unqiue_videos)):
        images_set = [s for s in images if unqiue_video in s['file']]

        detected_frame_files = vis_detector_output(
            images_set, detector_label_map, input_frames_base_dir, 
            rendering_output_dir, confidence)
        
        output_video_file = os.path.join(output_dir, unqiue_video)
        frames_to_video(detected_frame_files, Fs, output_video_file)

        delete_temp_dir(rendering_output_dir)


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