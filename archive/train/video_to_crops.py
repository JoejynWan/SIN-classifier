import os
import cv2
import yaml
import torch
from tqdm import tqdm
from munch import Munch
from pathlib import Path
import supervision as sv
from sc_utils.check_corrupt import check_corrupt_dir, find_videos
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans


def save_crop_image(frame, entry, output_dir, overwrite=False):
    """
    Save cropped images based on the detection bounding boxes.

    Args:
        results (list):
            Detection results containing image ID and detections.
        output_dir (str):
            Directory to save the cropped images.
        overwrite (bool):
            Whether overwriting existing image folders. Default to False.
    """   

    for xyxy, _, _, cat, _ in entry["detections"]:

        cropped_img = sv.crop_image(image=frame, xyxy=xyxy)
        with sv.ImageSink(target_dir_path=output_dir, overwrite=overwrite) as sink:
            sink.save_image(
                image=cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                image_name="{}_{}".format(
                    int(cat), entry["img_id"].rsplit(os.sep, 1)[1]
                ),
            )


def main(
    config:str='./train/config.yaml'
):
    """
    Main function for converting video to animal crops to prep for training. 

    Args:
        config (str): Path to the configuration file
    """

    # Load and set configurations from the YAML file
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))

    # Check for corrupted videos in the input_dir
    check_corrupt_dir(conf.input_dir, conf.output_dir, vid_duration_threshold = 0) 

    # Detect for animals with MegaDetector and crop out detections
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running detection/classification on {}".format(DEVICE))
    
    detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
    trans_det = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                   stride=detection_model.STRIDE)

    for video_path in tqdm(find_videos(conf.input_dir, recursive=True)):

        for frame_number, frame in enumerate(sv.get_video_frames_generator(source_path=video_path)):
            
            img_id_parts=Path(video_path).parts
            last_input_dir=Path(conf.input_dir).parts[-1]
            relative_dir=Path(*img_id_parts[img_id_parts.index(last_input_dir)+1:])
            full_output_dir = os.path.join(conf.output_dir, relative_dir)
            os.makedirs(full_output_dir, exist_ok=True)

            frame_filename='frame{:05d}.jpg'.format(frame_number)
            output_frame_path=os.path.join(full_output_dir, frame_filename)

            results_det = detection_model.single_image_detection(
                trans_det(frame), frame.shape, output_frame_path)
            
            save_crop_image(frame, results_det, full_output_dir, overwrite=False)


if __name__ == '__main__':
    main()
