import yaml
import torch
from munch import Munch
from torch.utils.data import DataLoader
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data
from PytorchWildlife import utils as pw_utils


if __name__ == '__main__':
    
    ## Set the general arguments
    CONFIG_PATH = './general/config.yaml'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load and set configurations from the YAML file
    with open(CONFIG_PATH) as f:
        config = Munch(yaml.load(f, Loader=yaml.FullLoader))

    ## Load the detection model
    detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

    ## Run detection in batch
    dataset = pw_data.DetectionImageFolder(
        config.SOURCE_DIR,
        transform=pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                    stride=detection_model.STRIDE)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        pin_memory=True, num_workers=0, drop_last=False)
    results = detection_model.batch_image_detection(loader)

    pw_utils.save_detection_images(results, 
                                   output_dir='results\\TrainVal_20221012_Trial\\Images', 
                                   input_dir=config.SOURCE_DIR)
    pw_utils.save_crop_images(results, 
                              output_dir='results\\TrainVal_20221012_Trial\\Crops', 
                              input_dir=config.SOURCE_DIR)
    pw_utils.save_detection_json(results, 'results\\TrainVal_20221012_Trial\\batch_output.json',
                                 categories=detection_model.CLASS_NAMES)
