import os
import cv2
import argparse
import tempfile
from tqdm import tqdm
import multiprocessing as mp

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions
from detect_utils import delete_temp_dir, find_unique_videos
from detect_utils import load_detector_output

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

def frames_to_video(images, Fs, output_file_name):
    """
    Given a list of image files and a sample rate, concatenate those images 
    into a video and write to [output_file_name].
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
        print('No codec_spec specified for video type .{},'\
        ' defaulting to DIVX'.format(video_ext))
        codec_spec = 'DIVX'
        
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(
        filename = output_file_name, 
        fourcc = fourcc, 
        fps = int(float(Fs)), 
        frameSize = (int(width), int(height)))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def vis_detector_output(options, images, detector_label_map, out_dir, 
                        output_image_width = 700):
    """
    Args: 
    images = list, detection dictionary for each of the image frames, loaded 
    from the detector_output .json file
    detector_label_map = str, detector categories loaded from the 
    detector_output .json file
    images_dir = str, path to the base folder containing the frames
    out_dir = str, temp directory where annotated frame images will be saved
    """

    ## Arguments error checking
    assert options.rendering_confidence_threshold > 0 and \
        options.rendering_confidence_threshold < 1, (
        f'Confidence threshold {options.rendering_confidence_threshold}'\
        ' is invalid, must be in (0, 1).')
    
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


def vis_detection_video(options, images, detector_label_map, fs_video):
    
    video_name, video_Fs = fs_video

    images_set = [s for s in images if video_name in s['file']]
    
    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    
    video_id = video_name.replace('\\', '~')
    rendering_output_dir = os.path.join(tempdir, f'detection_frames_{video_id}')
    
    detected_frame_files = vis_detector_output(options, 
        images_set, detector_label_map, rendering_output_dir, 
        output_image_width = -1) #no resizing to have full resolution
    
    output_video_file = os.path.join(options.output_dir, video_name)
    os.makedirs(os.path.dirname(output_video_file), exist_ok = True)
    frames_to_video(detected_frame_files, video_Fs, output_video_file)
    
    delete_temp_dir(rendering_output_dir)

    return video_name


def vis_detection_videos(options, detector_output, parallel = True):
    """
    Args:
    input_frames_anno_file = str, path to .json file describing the detections 
    for each frame
    input_frames_base_dir = str, path to base folder containing the video frames
    Fs_per_video = list, frame rate for each unique video
    output_dir = str, path to the base folder where the annotated videos will 
    be saved
    confidence = float, confidence threshold above which annotations will be 
    rendered
    """
    ## Check arguments
    assert os.path.isdir(options.frame_folder),\
        '{} is not a folder path for the video frames.'.format(options.frame_folder)

    images, detector_label_map, Fs = load_detector_output(detector_output)

    unique_videos = find_unique_videos(images)

    print('\nRendering detections above a confidence threshold of '\
        '{} for {} videos...'.format(
        options.rendering_confidence_threshold, len(unique_videos)))

    fs_videos = list(zip(unique_videos, Fs))

    if parallel: 
        result_list = []
        def callback_func(result):
            result_list.append(result)
            pbar.update()

        pool = mp.Pool(mp.cpu_count())
        pbar = tqdm(total = len(fs_videos))

        for fs_video in fs_videos:
            pool.apply_async(
                vis_detection_video, 
                args = (options, images, detector_label_map, fs_video),
                callback = callback_func
            )

        pool.close()
        pool.join()

    else:

        for fs_video in tqdm(fs_videos):
            vis_detection_video(options, images, detector_label_map, fs_video)
 

def main():
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    vis_detection_videos(options) 


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using '
                    'a trained MegaDetector model, where MegaDetector runs '
                    'on each frame in a video (or every Nth frame).')
    parser.add_argument('--model_file', type=str, 
                        default = config.MODEL_FILE, 
                        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--frame_folder', type = str,
                        default = config.FRAME_FOLDER,
                        help = 'Path to folder containing the video frames '
                                'processed by det_videos.py'
    )
    parser.add_argument('--roll_avg_frames_json', type=str,
                        default = config.ROLL_AVG_FRAMES_JSON, 
                        help = '.json file depicting the detections for each '
                                'frame of the video'
    )
    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD, 
                        help = 'do not render boxes with confidence below this '
                                'threshold'
    )
    return parser


if __name__ == '__main__':

    main()
