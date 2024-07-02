import os
import cv2 
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import supervision as sv
from typing import Callable
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans


def callback(frame: np.ndarray, frame_id: str = None) -> np.ndarray:
    """
    Callback function to process each video frame
    """
    ## Run MegaDetector with tracking and smoothering
    result = detection_model.single_image_detection(img_path=frame, img_id=frame_id)
    result["detections"] = tracker.update_with_detections(result["detections"])
    result["detections"] = smoother.update_with_detections(result["detections"])
    
    ## Labels from MegaDetector
    labels = []
    class_names = []
    confs = []
    if result["detections"].xyxy.size != 0:
        for class_id, conf in zip(result["detections"].class_id, result["detections"].confidence):
            class_name = detection_model.CLASS_NAMES[class_id]
            labels.append("{} {:.2f}".format(class_name, conf))
            class_names.append(class_name)
            confs.append(conf)
    
    # ## Labels from Classifier
    # labels = []
    # for xyxy in results_det["detections"].xyxy:
    #     cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
    #     results_clf = classification_model.single_image_classification(trans_clf(Image.fromarray(cropped_image)))
    #     labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))
    
    annotated_frame = box_annotator.annotate(scene=frame, detections=result["detections"], 
                                             labels=labels)
    
    return annotated_frame, class_names, confs


def process_video(    
    source_path: str,
    target_dir: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    stride: int = 1,
    codec: str = "mp4v"
    ):
    """
    Process a video frame-by-frame, applying a callback function to each frame and saving the 
    results to a new video. This version allows codec selection.
    
    Args:
        source_path (str): 
            Path to the source video file.
        target_path (str): 
            Path to save the processed video.
        callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and its index as input and returns the processed frame.
        codec (str, optional): 
            Codec used to encode the processed video. Default is "avc1".
    """
    
    ## Run the callback
    result_frames = []
    class_names_all = []
    confs_all = []
    for index, frame in enumerate(
        sv.get_video_frames_generator(source_path=source_path, stride=stride)
    ):
        frame_id = Path(source_path).stem + "_frame" + str(index).zfill(3)
        result_frame, class_names, confs = callback(frame, frame_id = frame_id)
        result_frames.append(result_frame)
        class_names_all.append(class_names)
        confs_all.append(confs)

    ## Get the target video path to write and sort the videos
    if not [cm for cm in class_names_all if cm != []]:
        class_dir = ['False trigger']
    else:
        class_dir = class_names_all[confs_all.index(max(confs_all))]

    vid_path_parts=Path(source_path).parts
    last_input_dir=Path(SOURCE_VIDEO_DIR).parts[-1]
    relative_dir=Path(*vid_path_parts[vid_path_parts.index(last_input_dir)+1:-1])
    full_output_dir = os.path.join(target_dir, relative_dir, class_dir[0])
    os.makedirs(full_output_dir, exist_ok=True)

    video_name=Path(source_path).parts[-1]
    target_video_path=os.path.join(full_output_dir,video_name)

    with sv.VideoSink(target_path=target_video_path, video_info=source_video_info, codec=codec) as sink:
        for result_frame in result_frames:
            sink.write_frame(frame=cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))


## Arguments
SOURCE_VIDEO_DIR = 'C:\\TempDataForSpeed\\20240618 maintenance'
TARGET_VIDEO_DIR = 'results\\20240618 maintenance'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FPS = 100
CODEC = "mp4v"

## Prep models and settings
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, weights="../MD_weights/MDV6b-yolov9c.pt", pretrained=True)
# trans_det = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
#                                                stride=detection_model.STRIDE)

# classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, pretrained=True)
# trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

## Run detection, classification, visualisation, and sorting of videos
video_files = glob.glob(os.path.join(SOURCE_VIDEO_DIR, '**/*.AVI'), recursive=True)

for video_file in tqdm(video_files):
    
    ## Determine the stride from target_fps
    source_video_info = sv.VideoInfo.from_video_path(video_path=video_file)
    
    if source_video_info.fps > TARGET_FPS:
        stride = int(source_video_info.fps / TARGET_FPS)
        source_video_info.fps = TARGET_FPS
    else:
        stride = 1

    ## Initiate supervision functions
    tracker = sv.ByteTrack(track_thresh = 0.25, frame_rate = source_video_info.fps)
    smoother = sv.DetectionsSmoother()
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=.5)

    ## Process a single video
    process_video(source_path = video_file, target_dir = TARGET_VIDEO_DIR, 
                  callback = callback, stride = stride, codec = CODEC)
