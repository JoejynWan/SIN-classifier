import argparse
from vis_detections import load_detector_output

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    images, detector_label_map = load_detector_output(args.full_det_frames_json)

    

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to convert detections saved in a json file to csv dataframe.')
    parser.add_argument('--full_det_frames_json', type=str,
                        default = default_full_det_frames_json, 
                        help = 'Path of json file with all detections for each frame.'
    )

if __name__ == '__main__':
    default_full_det_frames_json = 'results/example_test_set/example_test_set_full_det_frames.json'

    main()