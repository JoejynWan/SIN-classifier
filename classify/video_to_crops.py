import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from munch import Munch
from pathlib import Path
import supervision as sv
from typing import Callable
from PytorchWildlife.models import detection as pw_detection
from sc_utils.check_corrupt import check_corrupt_dir, find_videos


def callback(frame: np.ndarray, frame_id: str = None, target_dir: str = None) -> np.ndarray:
    """
    Callback function to process each video frame
    """
    ## Run MegaDetector with tracking and smoothering
    result = detection_model.single_image_detection(img_path=frame, img_id=frame_id)
    result["detections"] = tracker.update_with_detections(result["detections"])
    result["detections"] = smoother.update_with_detections(result["detections"])
    
    ## Save out cropped images
    os.makedirs(target_dir, exist_ok=True)
    with sv.ImageSink(target_dir_path=target_dir) as sink:
        for i, (xyxy, _, _, cat, _, _) in enumerate(result["detections"]):

            cropped_img = sv.crop_image(image=frame, xyxy=xyxy)
            crop_id = frame_id + "_crop" + str(i).zfill(2) + "_cat" + str(cat).zfill(2) + ".jpg"

            sink.save_image(image=cropped_img, image_name=crop_id)
    
    return result


def video_to_crops(    
    source_video_file: str,
    target_dir: str,
    callback: Callable[[np.ndarray, int], np.ndarray]
    ):
    """
    Process a video frame-by-frame, applying a callback function to each frame and saving the 
    results to a new video. This version allows codec selection.
    
    Args:
        source_video_file (str): 
            Path to the source video file.
        target_dir (str): 
            Path to the directory where the processed video will be saved.
        callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and its index as input and returns the processed frame.
    """
    
    ## Run the callback
    results = []
    for index, frame in enumerate(
        sv.get_video_frames_generator(source_path=source_video_file)
    ):
        frame_id = Path(source_video_file).stem + "_frame" + str(index).zfill(3)
        crops_dir = os.path.join(target_dir, Path(source_video_file).stem)
        result = callback(frame, frame_id = frame_id, target_dir = crops_dir)
        results.append(result)


if __name__ == '__main__':

    ## Set the general arguments
    CONFIG_PATH = './classify/config_classify.yaml'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load and set configurations from the YAML file
    with open(CONFIG_PATH) as f:
        config = Munch(yaml.load(f, Loader=yaml.FullLoader))

    ## Check for corrupt videos before running MD
    corrupted = check_corrupt_dir(config.SOURCE_DIR, config.TARGET_DIR, 
                                  vid_duration_threshold = 0, Fs_threshold = 10)

    ## Load the detection model
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, weights="models/MDV6b-yolov9c.pt", 
                                                  pretrained=True)
    
    ## Run detection, saving out of animal crops
    video_files = find_videos(config.SOURCE_DIR, recursive=True)   
    for video_file in tqdm(video_files):     

        ## Initiate supervision objects
        source_video_info = sv.VideoInfo.from_video_path(video_path=video_file)
        tracker = sv.ByteTrack(frame_rate = source_video_info.fps)
        smoother = sv.DetectionsSmoother()

        ## Process a single video
        video_to_crops(source_video_file = video_file, target_dir = config.TARGET_DIR, 
                       callback = callback)
