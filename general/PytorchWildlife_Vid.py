import os 
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import supervision as sv
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils


## Arguments
SOURCE_VIDEO_DIR = 'C:\\TempDataForSpeed\\example_test_set'
TARGET_VIDEO_DIR = 'results\\example_test_set'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Prep models and settings
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
trans_det = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                               stride=detection_model.STRIDE)

# classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, pretrained=True)
# trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=.5)

## Process videos
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results_det = detection_model.single_image_detection(trans_det(frame), frame.shape, index)

    ## Labels from MegaDetector
    labels = []
    if results_det["detections"].xyxy.size != 0:
        labels.append("{} {:.2f}".format(results_det["detections"].class_id[0], 
                                        results_det["detections"].confidence[0]))
    
    # ## Labels from Classifier
    # labels = []
    # for xyxy in results_det["detections"].xyxy:
    #     cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
    #     results_clf = classification_model.single_image_classification(trans_clf(Image.fromarray(cropped_image)))
    #     print(results_clf)
    #     print(results_clf["prediction"])
    #     labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))
    
    annotated_frame = box_annotator.annotate(scene=frame, detections=results_det["detections"], labels=labels)
    return annotated_frame

video_files = glob.glob(os.path.join(SOURCE_VIDEO_DIR, '**/*.AVI'), recursive=True)

for video in tqdm(video_files):

    vid_path_parts=Path(video).parts
    last_input_dir=Path(SOURCE_VIDEO_DIR).parts[-1]
    relative_dir=Path(*vid_path_parts[vid_path_parts.index(last_input_dir)+1:-1])
    full_output_dir = os.path.join(TARGET_VIDEO_DIR, relative_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    video_name=Path(video).parts[-1]
    target_video_path=os.path.join(full_output_dir,video_name)

    pw_utils.process_video(source_path=video, target_path=target_video_path, 
                           callback=callback, target_fps=100)
