import os
import cv2 
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm
from munch import Munch
from pathlib import Path
import supervision as sv
from itertools import chain
from typing import Callable
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from sc_utils.smoother import ClassificationSmoother


def callback(frame: np.ndarray, frame_id: str = None) -> np.ndarray:
    """
    Callback function to process each video frame
    """
    ## Run MegaDetector with tracking and smoothering
    results_det = detection_model.single_image_detection(img_path=frame, img_id=frame_id)
    results_det["detections"] = tracker_det.update_with_detections(results_det["detections"])
    results_det["detections"] = smoother_det.update_with_detections(results_det["detections"])
    
    ## Labels from MegaDetector
    labels = []
    class_names = []
    confs = []
    if results_det["detections"].xyxy.size != 0:
        for class_id, conf in zip(results_det["detections"].class_id, 
                                  results_det["detections"].confidence):
            class_name = detection_model.CLASS_NAMES[class_id]
            labels.append("{} {:.2f}".format(class_name, conf))
            class_names.append(class_name)
            confs.append(conf)
    
    ## Labels from Classifier
    for class_id, xyxy, tracker_id, class_names_tup in zip(results_det["detections"].class_id, 
                                                           results_det["detections"].xyxy, 
                                                           results_det["detections"].tracker_id, 
                                                           enumerate(class_names)):
        if class_id == 0:
            cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
            results_clf = classification_model.single_image_classification(img=cropped_image, 
                                                                           img_id=frame_id)
            
            results_clf["detections"] = sv.Detections(
                xyxy = np.array([xyxy]),
                confidence = np.array([results_clf["confidence"]]), 
                class_id = np.array([results_clf["class_id"]]),
                tracker_id = np.array([tracker_id]), 
                data = {'all_confs': np.array([[conf[1] for conf in results_clf["all_confidences"]]]), 
                        'all_class_id': np.array([[conf[0] for conf in results_clf["all_confidences"]]])}
            )
            
            results_clf["detections"] = smoother_cls.update_with_detections(results_clf["detections"])
 
            conf = max(results_clf["detections"].confidence)
            conf_idx = np.where(results_clf["detections"].confidence == conf)[0]
            conf_id = results_clf["detections"].class_id[conf_idx].item()
            class_name = classification_model.CLASS_NAMES[conf_id]

            labels[class_names_tup[0]] = "{} {:.2f}".format(class_name, conf)
            class_names[class_names_tup[0]] = class_name
            confs[class_names_tup[0]] = conf
    
    annotated_frame = bbox_annotator.annotate(scene=frame, detections=results_det["detections"])
    annotated_frame = label_annotator.annotate(annotated_frame, 
                                               detections=results_det["detections"],
                                               labels=labels)
    
    return annotated_frame, class_names, confs


def process_video(    
    source_video_file: str,
    source_video_dir: str, 
    target_dir: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    codec: str = "mp4v"
    ):
    """
    Process a video frame-by-frame, applying a callback function to each frame and saving the 
    results to a new video. This version allows codec selection.
    
    Args:
        source_video_file (str): 
            Path to the source video file.
        source_video_dir (str): 
            Path to the directory containing the source video file.
        target_dir (str): 
            Path to the directory where the processed video will be saved.
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
        sv.get_video_frames_generator(source_path=source_video_file)
    ):
        frame_id = Path(source_video_file).stem + "_frame" + str(index).zfill(3)
        result_frame, class_names, confs = callback(frame, frame_id = frame_id)
        
        result_frames.append(result_frame)
        class_names_all.append(class_names)
        confs_all.append(confs)

    ## Get the target video path to write and sort the videos
    if not [cm for cm in class_names_all if cm != []]:
        class_dir = 'False trigger'
    else:
        confs_all_chain = list(chain.from_iterable(confs_all))
        class_names_all_chain = np.array(list(chain.from_iterable(class_names_all)))
        max_conf_idx = list(np.where(np.array(confs_all_chain) == max(confs_all_chain))[0])
        max_conf_class_names = class_names_all_chain[max_conf_idx].tolist()
        class_dir = max(max_conf_class_names, key = max_conf_class_names.count)

    vid_path_parts=Path(source_video_file).parts
    last_input_dir=Path(source_video_dir).parts[-1]
    relative_dir=Path(*vid_path_parts[vid_path_parts.index(last_input_dir)+1:-1])
    full_output_dir = os.path.join(target_dir, relative_dir, class_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    video_name=Path(source_video_file).parts[-1]
    target_video_path=os.path.join(full_output_dir,video_name)
    
    ## Save out the processed video
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_video_file)
    with sv.VideoSink(target_path=target_video_path, video_info=source_video_info, codec=codec) as sink:
        for result_frame in result_frames:
            sink.write_frame(frame=cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    ## Set the general arguments
    CONFIG_PATH = './general/config.yaml'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load and set configurations from the YAML file
    with open(CONFIG_PATH) as f:
        config = Munch(yaml.load(f, Loader=yaml.FullLoader))

    ## Load the detection and classification models
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, weights="models/MDV6b-yolov9c.pt", 
                                                  pretrained=True)

    classification_model = pw_classification.SINClassifier(device=DEVICE, 
                                                           weights=config.CLS_WEIGHTS_PATH)
    
    ## Run detection, classification, visualisation, and sorting of videos
    video_files = glob.glob(os.path.join(config.SOURCE_DIR, '**/*.AVI'), recursive=True)
    for video_file in tqdm(video_files):     

        ## Initiate supervision objects
        source_video_info = sv.VideoInfo.from_video_path(video_path=video_file)
        tracker_det = sv.ByteTrack(frame_rate=source_video_info.fps)
        smoother_det = sv.DetectionsSmoother()
        smoother_cls = ClassificationSmoother()
        bbox_annotator = sv.BoundingBoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=.5)

        ## Process a single video
        process_video(source_video_file = video_file, 
                      source_video_dir = config.SOURCE_DIR, 
                      target_dir = config.TARGET_DIR, 
                      callback = callback, codec = config.CODEC)
