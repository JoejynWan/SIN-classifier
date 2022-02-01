import sys
sys.path.append('D:/CNN_Animal_ID/MegaDetector/CameraTraps')
from detection.run_tf_detector_batch import load_and_run_detector_batch, write_results_to_file

import os
import time
import argparse
import humanfriendly
import tensorflow as tf 
import pandas as pd


def load_image_paths(image_dir, image_names_datasheet):
    xy_datasheet = pd.read_csv(image_names_datasheet)

    files = xy_datasheet['MediaFilename'].tolist()
    img_paths = []
    for file in files:
        file = file + ".jpg"
        file = os.path.join(image_dir, file)
        img_paths.append(file)
    
    return img_paths


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to draw bounding boxes of animals on videos using a trained MegaDetector model')
    parser.add_argument(
        '--megadetector_model', default = default_megadetector_model, 
        help = 'Path to .pb MegaDetector model file.'
    )
    parser.add_argument(
        '--image_dir', default = default_image_dir,
        help = 'Path to directory containing images.'
    )
    parser.add_argument(
        '--image_names_datasheet', default = default_image_names_datasheet,
        help = 'Path to .csv datasheet containing the image file names (in column "MediaFilename").'
    )
    parser.add_argument(
        '--output_file', default = default_output_file,
        help = 'Path to output JSON results file, should end with a .json extension'
    )
    return parser


def main(): 
    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    ## Defining TF configurations for the device
    if len(tf.config.list_physical_devices('GPU')) != 0:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True) #allocates only as much GPU memory as needed for the runtime allocations

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        print("NOTE: No GPU is available")
    
    # tf.debugging.set_log_device_placement(True) #Run this to check which processes are run on which device (CPU or GPU)


    ## Loading image paths 
    img_paths = load_image_paths(args.image_dir, args.image_names_datasheet)

    ## Predict using some images
    start_time = time.time()
    predictions = load_and_run_detector_batch(args.megadetector_model, img_paths[0:100])
    elapsed = time.time() - start_time
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    write_results_to_file(predictions, args.output_file)

if __name__ == '__main__':
    ## Defining parameters
    default_megadetector_model = "MegaDetectorModel_v4.1/md_v4.1.0.pb"
    default_image_dir = "C:/temp_for_SSD_speed/big4_datasets/testing_sets/jb_test_resized"
    default_image_names_datasheet = "data/splits/big4_20210810_test_jb_sheet_resized.csv"
    default_output_file = "results/test_inference_images.json"
    
    main()
