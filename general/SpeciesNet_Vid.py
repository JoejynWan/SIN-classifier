import os
import cv2 
import yaml
import torch
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from munch import Munch
from pathlib import Path
import supervision as sv
from typing import Callable
from collections import Counter
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from sc_utils.check_corrupt import check_corrupt_dir, find_videos


def detection_callback(frame: np.ndarray, frame_id: str = None) -> np.ndarray:
    """
    Callback function to process each video frame with MegaDetector. Tracking and smoothering is 
    used to average the bounding boxes (xyxy) and confidences across frames. 
    """

    results_det = detection_model.single_image_detection(img=frame, img_path=frame_id, verbose=False)
    results_det["detections"] = tracker_det.update_with_detections(results_det["detections"])
    results_det["detections"] = smoother_det.update_with_detections(results_det["detections"])

    return results_det


def classification_callback(frame_path: str, results_det = None, img_size = None) -> np.ndarray:
    """
    Callback function to process each video frame with SpeciesNet classifier. 
    """

    ## Convert xyxy to normalised bounding box to be compatiable with speciesnet
    normalized_coords = []
    for xyxy in results_det['detections'].xyxy:
        x1, y1, x2, y2 = xyxy
        normalized_xyxy = [x1 / img_size[1], y1 / img_size[0], x2 / img_size[1], y2 / img_size[0]]
        normalized_coords.append(normalized_xyxy)

    spp_results = []
    for i in range(len(normalized_coords)):
        r = {'img_id': frame_path,
             'normalized_coords': [normalized_coords[i]]}
        spp_result = classification_model.single_image_classification(frame_path, det_results=r)
        spp_results.extend(spp_result)

    ## Save out as sv.Detections class for compatibility with supervision
    clf_detections = sv.Detections(
        xyxy = results_det['detections'].xyxy,
        confidence = np.array([spp_result['confidence'] for spp_result in spp_results]), 
        class_id = np.array([spp_result['class_id'] for spp_result in spp_results]), 
        tracker_id = results_det['detections'].tracker_id, 
        data = {'prediction': [spp_result['prediction'] for spp_result in spp_results]}
    )
    clf_detections = smoother_cls.update_with_detections(clf_detections)

    ## Get labels
    clf_conf_thres = 0.8
    clf_labels = []
    for i in range(len(clf_detections)):
        # Set class_id to unknown if conf is below threshold
        if clf_detections.confidence[i] < clf_conf_thres:
            clf_detections.class_id[i] = list(classification_model.id_to_label.keys())[-1]
            clf_detections.data['prediction'][i] = list(classification_model.id_to_label.items())[-1]
    
        label = classification_model.id_to_label[clf_detections.class_id[i]].split(';')[-1]
        clf_labels.append("{} {:.2f}".format(label, clf_detections.confidence[i]))

    ## Match format with singe_image_detection
    results_clf = {
        'img_id': results_det['img_id'],
        'detections': clf_detections,
        'labels': clf_labels, 
        'normalized_coords': normalized_coords
    }

    return results_clf


def get_video_class(results):
    """
    Get the overall class name of the video across all frames. 
    Overall class name is based on the most common occurance across all frames. 
    """
    preds_all = []
    for result in results: 
        if result["detections"].xyxy.size != 0:
            preds_all.extend(result["detections"].data["prediction"])
    
    if not preds_all:
        video_class = 'blank'
    else:
        video_class, count = Counter(preds_all).most_common(1)[0]

    return video_class


def vis_video(results, video_class, source_video_file, source_video_dir, target_dir, codec):
    """
    Visualise videos with the annotated bounding boxes from MegaDetector (or species classifier, 
    if available). 
    """
    ## Get annotated frames with labels 
    annotated_frames = []
    for result, frame in zip(results, sv.get_video_frames_generator(source_path=source_video_file)):

        annotated_frame = bbox_annotator.annotate(scene=frame, detections=result["detections"])
        annotated_frame = label_annotator.annotate(annotated_frame, 
                                                   detections=result["detections"],
                                                   labels=result['labels'])
        annotated_frames.append(annotated_frame)

    ## Get the full output video path
    vid_path_parts=Path(source_video_file).parts
    last_input_dir=Path(source_video_dir).parts[-1]
    relative_dir=Path(*vid_path_parts[vid_path_parts.index(last_input_dir)+1:-1])
    full_output_dir = os.path.join(target_dir, relative_dir, video_class)
    os.makedirs(full_output_dir, exist_ok=True)

    video_name=Path(source_video_file).parts[-1]
    target_video_path=os.path.join(full_output_dir,video_name)
    
    ## Save out the processed video
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_video_file)
    with sv.VideoSink(target_path=target_video_path, video_info=source_video_info, codec=codec) as sink:
        for result_frame in annotated_frames:
            sink.write_frame(frame=result_frame)


def process_video(    
    source_video_file: str,
    source_video_dir: str, 
    target_dir: str,
    detection_callback: Callable[[np.ndarray, int], np.ndarray],
    classification_callback: Callable[[np.ndarray, int], np.ndarray],
    vis_media: bool,
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
        detection_callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and its index as input and returns the results
            from MegaDetector
        classification_callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and the results from MegaDetector and returns the 
            results from the species classifier. 
        vis_media (bool):
            Boolean that controls if videos with bounding boxes should be saved out. 
        codec (str, optional): 
            Codec used to encode the processed video. Default is "avc1".
    """
    
    results_clfs = []
    for index, frame in enumerate(
        sv.get_video_frames_generator(source_path=source_video_file)
    ):
        ## Run MegaDetector
        frame_id = Path(source_video_file).stem + "_frame" + str(index).zfill(3)
        results_det = detection_callback(frame, frame_id = frame_id)
        
        ## Run SpeciesNet Classifier
        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_height, frame_width)

        with tempfile.TemporaryDirectory(prefix = "frame_folder") as tmpdir: 
            frame_path = os.path.join(tmpdir, frame_id + '.jpg')
            cv2.imwrite(frame_path, frame)    
        
            results_clf = classification_callback(frame_path, results_det, frame_size)
        
        results_clfs.append(results_clf)
    
    video_class = get_video_class(results_clfs)
    if vis_media: 
        vis_video(results_clfs, video_class, source_video_file, source_video_dir, target_dir, codec)
    
    return video_class


if __name__ == '__main__':

    ## Set the general arguments
    CONFIG_PATH = './general/config.yaml'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load and set configurations from the YAML file
    with open(CONFIG_PATH) as f:
        config = Munch(yaml.load(f, Loader=yaml.FullLoader))
 
    ## Check for target_dir and corrupt videos
    assert os.path.exists(config.TARGET_DIR), "TARGET_DIR does not exist."
    check_corrupt_dir(config.SOURCE_DIR, config.TARGET_DIR, vid_duration_threshold=0)

    ## Load the detection and classification models
    # detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, 
    #                                               version=config.DET_VERSION)
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, weights=config.DET_WEIGHTS_PATH, 
                                                  version=config.DET_VERSION)

    if config.CLS_VERSION: 
        classification_model = pw_classification.SpeciesNetTFInference(version=config.CLS_VERSION, 
                                                                       run_mode='multi_thread')
        classification_model.id_to_label[len(classification_model.id_to_label)] = "unknown"
    
    ## Run detection, classification, visualisation, and sorting of videos
    outs = []
    video_files = find_videos(config.SOURCE_DIR, recursive=True)
    for video_file in tqdm(video_files):     

        ## Initiate supervision objects
        source_video_info = sv.VideoInfo.from_video_path(video_path=video_file)
        tracker_det = sv.ByteTrack(frame_rate=source_video_info.fps)
        smoother_det = sv.DetectionsSmoother()
        bbox_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=.5)
        if config.CLS_VERSION: 
            smoother_cls = sv.DetectionsSmoother()
        else:
            classification_callback = None

        ## Process a single video
        video_class = process_video(source_video_file = video_file, 
                                    source_video_dir = config.SOURCE_DIR, 
                                    target_dir = config.TARGET_DIR, 
                                    detection_callback = detection_callback, 
                                    classification_callback = classification_callback, 
                                    vis_media = config.VIS_MEDIA, 
                                    codec = config.CODEC)
        
        ## Save out the results
        outs.append({'path': video_file, 'pred': video_class})

    ## Output the results as csv
    outs_df = pd.DataFrame(outs)
    outs_df.to_csv(os.path.join(config.TARGET_DIR, "results.csv"), index = False)
