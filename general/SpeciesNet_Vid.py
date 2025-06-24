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
from itertools import chain
from typing import Callable
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from sc_utils.smoother import ClassificationSmoother
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


def classification_callback(frame_path: str, results_det = None) -> np.ndarray:
    """
    Callback function to process each video frame with SpeciesNet classifier. 
    """

    clf_conf_thres = 0.8
    clf_labels = []
    for i in range(len(results_det['detections'].xyxy)):
        r = {
            'img_id': frame_path,
            'normalized_coords': [results_det['detections'].xyxy[i]]
            }
    results_clf = classification_model.single_image_classification(frame_path, det_results=r)[0]
    clf_labels.append("{} {:.2f}".format(
        results_clf["prediction"] if results_clf["confidence"] > clf_conf_thres else "Unknown",
        results_clf["confidence"]
    ))
    

    # spp_dets = []
    # for xyxy, tracker_id in zip(results_det["detections"].xyxy, results_det["detections"].tracker_id):

    #     cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
    #     spp_results = classification_model.single_image_classification(img=cropped_image, 
    #                                                                    img_id=results_det['img_id'])
        
    #     spp_det = sv.Detections(
    #         xyxy = np.array([xyxy]),
    #         confidence = np.array([spp_results["confidence"]]), 
    #         class_id = np.array([spp_results["class_id"]]),
    #         tracker_id = np.array([tracker_id]), 
    #         data = {'all_confs': np.array([[conf[1] for conf in spp_results["all_confidences"]]]), 
    #                 'all_class_id': np.array([[conf[0] for conf in spp_results["all_confidences"]]])}
    #     )
        
    #     spp_det = smoother_cls.update_with_detections(spp_det)
    #     spp_dets.append(spp_det)

    # results_clf = {"img_id": results_det["img_id"]}
    # results_clf["detections"] = sv.Detections.merge(spp_dets)

    return results_clf


def get_video_class(results, model):
    """
    Get the overall class name of the video across all frames. Class names will be based on the 
    model provided (either the detection or classification model). 
    """
    confs_all = []
    class_id_all = []
    for result in results: 
        if result["detections"].xyxy.size != 0:
            confs_all.append(result["detections"].confidence)
            class_id_all.append(result["detections"].class_id)
    
    if not class_id_all:
        video_class = 'empty'
    else:
        confs_all_chain = list(chain.from_iterable(confs_all))
        class_id_all_chain = np.array(list(chain.from_iterable(class_id_all)))
        max_conf_idx = list(np.where(np.array(confs_all_chain) == max(confs_all_chain))[0])
        max_conf_class_ids = class_id_all_chain[max_conf_idx].tolist()
        max_conf_class_id = max(max_conf_class_ids, key = max_conf_class_ids.count)
        video_class = model.CLASS_NAMES[max_conf_class_id]

    return video_class


def vis_video(results, video_class, model, source_video_file, source_video_dir, target_dir, codec):
    """
    Visualise videos with the annotated bounding boxes from MegaDetector (or species classifier, 
    if available). 
    """
    ## Get annotated frames with labels 
    annotated_frames = []
    for result, frame in zip(results, sv.get_video_frames_generator(source_path=source_video_file)):
        
        labels = []
        for class_id, conf in zip(result["detections"].class_id, result["detections"].confidence):
            class_name = model.CLASS_NAMES[class_id]
            labels.append("{} {:.2f}".format(class_name, conf))

        annotated_frame = bbox_annotator.annotate(scene=frame, detections=result["detections"])
        annotated_frame = label_annotator.annotate(annotated_frame, 
                                                   detections=result["detections"],
                                                   labels=labels)
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
    
    results_dets = []
    for index, frame in enumerate(
        sv.get_video_frames_generator(source_path=source_video_file)
    ):
        ## Run MegaDetector
        frame_id = Path(source_video_file).stem + "_frame" + str(index).zfill(3)
        results_det = detection_callback(frame, frame_id = frame_id)
        
        ## Run SpeciesNet Classifier
        with tempfile.TemporaryDirectory(prefix = "frame_folder") as tmpdir: 
            frame_path = os.path.join(tmpdir, frame_id + '.jpg')
            cv2.imwrite(frame_path, frame)    
        
            results_clf = classification_callback(frame_path, results_det)
        
        
        results_dets.append(results_det)

    ## Run species classifier only if classification_callback is provided and video is detected to 
    ## be animal
    video_class = get_video_class(results_dets, detection_model)

    if classification_callback is None or video_class != "animal": 
        results = results_dets
        model = detection_model
    elif video_class == "animal": 
        results_clfs = []
        for results_det, frame in zip(
            results_dets, sv.get_video_frames_generator(source_path=source_video_file)
        ):
            results_clf = classification_callback(frame, results_det)
            results_clfs.append(results_clf)

        video_class = get_video_class(results_clfs, classification_model)
        results = results_clfs
        model = classification_model

    ## Save out videos with annotated bounding boxes
    if vis_media: 
        vis_video(results, video_class, model, source_video_file, source_video_dir, target_dir, codec)

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
        classification_model = pw_classification.SpeciesNetTFInferenceMD6(version=config.CLS_VERSION, 
                                                                          run_mode='multi_thread')
    
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
            smoother_cls = ClassificationSmoother()
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
