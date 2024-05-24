import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data
from PytorchWildlife import utils as pw_utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

## Run detection in batch
tgt_folder_path = 'C:\\TempDataForSpeed\\TrainVal_20221012_Trial'

dataset = pw_data.DetectionImageFolder(
    tgt_folder_path,
    transform=pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                 stride=detection_model.STRIDE)
)
loader = DataLoader(dataset, batch_size=32, shuffle=False,
                    pin_memory=True, num_workers=0, drop_last=False)
results = detection_model.batch_image_detection(loader)

pw_utils.save_detection_images(results, 'results\\TrainVal_20221012_Trial\\Images')
pw_utils.save_crop_images(results, 'results\\TrainVal_20221012_Trial\\Crops')
pw_utils.save_detection_json(results, 'results\\TrainVal_20221012_Trial\\batch_output.json',
                             categories=detection_model.CLASS_NAMES)
