import os
import cv2
import json
from tqdm import tqdm
import argparse
import tempfile

# Functions imported from this project
from shared_utils import delete_temp_dir, find_unqiue_videos, VideoOptions

# Imported from Microsoft/CameraTraps github repository
from visualization import visualization_utils as vis_utils
from data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int
from ct_utils import args_to_object

#%% Constants
# Set to ignore INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

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


def frames_to_video(images, Fs, output_file_name):
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
    if output_file_name.lower().endswith('.avi'):
        codec_spec = 'DIVX'
    elif output_file_name.lower().endswith('.mp4'):
        codec_spec = 'mp4v'
    else: 
        video_ext = output_file_name.split('.')[-1]
        print('No codec_spec specified for video type .{}, defaulting to DIVX'.format(video_ext))
        codec_spec = 'DIVX'
        
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, Fs, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def vis_detector_output(options, images, detector_label_map, out_dir, output_image_width = 700):
    """
    Args: 
    images = list, detection dictionary for each of the image frames, loaded from the detector_output .json file
    detector_label_map = str, detector categories loaded from the detector_output .json file
    images_dir = str, path to the base folder containing the frames
    out_dir = str, temporary directory where annotated frame images will be saved
    """

    ## Arguments error checking
    assert options.rendering_confidence_threshold > 0 and options.rendering_confidence_threshold < 1, (
        f'Confidence threshold {options.rendering_confidence_threshold} is invalid, must be in (0, 1).')
    
    os.makedirs(out_dir, exist_ok=True)
    
    annotated_img_paths = []
    for entry in images:
        image_id = entry['file']
        
        if 'failure' in entry:
            print(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        image_obj = os.path.join(options.frame_folder, image_id)
        if not os.path.exists(image_obj):
            print(f'Image {image_id} not found in images_dir; skipped.')
            continue

        # resize is for displaying them more quickly
        image = vis_utils.resize_image(
            vis_utils.open_image(image_obj), output_image_width)

        vis_utils.render_detection_bounding_boxes(
            entry['detections'], image, label_map=detector_label_map,
            confidence_threshold = options.rendering_confidence_threshold)

        for char in ['/', '\\', ':']:
            image_id = image_id.replace(char, '~')
        annotated_img_path = os.path.join(out_dir, f'anno_{image_id}')
        annotated_img_paths.append(annotated_img_path)
        image.save(annotated_img_path)

    return annotated_img_paths

def vis_detection_video(options, images_set, detector_label_map, video_name, video_Fs):
    
    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    rendering_output_dir = os.path.join(tempdir, 'detection_frames')

    detected_frame_files = vis_detector_output(options, 
        images_set, detector_label_map, rendering_output_dir)
    
    output_video_file = os.path.join(options.output_dir, video_name)
    os.makedirs(os.path.split(output_video_file)[0], exist_ok=True)
    frames_to_video(detected_frame_files, video_Fs, output_video_file)
    
    delete_temp_dir(rendering_output_dir)


def vis_detection_videos(options, Fs_per_video):
    """
    Args:
    input_frames_anno_file = str, path to .json file describing the detections for each frame
    input_frames_base_dir = str, path to base folder containing the video frames
    Fs_per_video = list, frame rate for each unique video
    output_dir = str, path to the base folder where the annotated videos will be saved
    confidence = float, confidence threshold above which annotations will be rendered
    """
    images, detector_label_map = load_detector_output(options.roll_avg_frames_json)

    unqiue_videos = find_unqiue_videos(images, options.output_dir)

    print('Rendering detections above a confidence threshold of {} for {} videos...'.format(
        options.rendering_confidence_threshold, len(unqiue_videos)))
    
    for unqiue_video, Fs in tqdm(zip(unqiue_videos, Fs_per_video), total = len(unqiue_videos)):
        images_set = [s for s in images if unqiue_video in s['file']]

        vis_detection_video(options, images_set, detector_label_map, unqiue_video, Fs)


def main():
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    vis_detection_videos(options) #TODO fix faulty missing Fs argument


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model, where MegaDetector runs on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = default_model_file, 
                        help='Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = default_output_dir, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--frame_folder', type = str,
                        default = default_input_frames_base_dir,
                        help = 'Path to folder containing the video frames processed by det_videos.py'
    )
    parser.add_argument('--roll_avg_frames_json', type=str,
                        default = default_input_frames_anno_file, 
                        help = '.json file depicting the detections for each frame of the video'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold"
    )
    return parser


if __name__ == '__main__':
    ## Defining parameters within this script
    # Comment out if passing arguments from terminal directly
    default_model_file = 'MegaDetectorModel_v4.1/md_v4.1.0.pb'
    default_input_frames_base_dir = 'C:/temp_for_SSD_speed/CT_models_test_frames'
    default_input_frames_anno_file = 'C:/temp_for_SSD_speed/CT_models_test_frames/CT_models_test.frames.json'
    default_output_dir = 'D:/CNN_Animal_ID/CamphoraClassifier/SIN-classifier/results/CT_models_test'

    main()