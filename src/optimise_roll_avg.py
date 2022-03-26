import os
import argparse
import pandas as pd
from tqdm import tqdm

# Functions imported from this project
import config
from shared_utils import VideoOptions, load_detector_output
from shared_utils import check_output_dir, default_path_from_none
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


def summarise_cat(csv_file_path):
    """
    Summarises dataset into two columns: 'FullVideoPath' and 'Category', 
        where each row is a unique FullVideoPath. If a FullVideoPath has multiple 
        Categories (more than one detection or animal), they are summarised 
        using the following rules:
    1) Where there are multiples of the same category, only one is kept
    2) Where there is an animal (1) detection, and human (2) and/or vehicle (3) detection, 
        the video is considered to be an animal category (1)
    3) Where there is a human (2) and vehicle (3), the video is considered to be a human category (2)
    """

    full_df = pd.read_csv(csv_file_path)

    video_cat = full_df[['FullVideoPath', 'Category']]
    summ_cat = video_cat.sort_values(
        by = ['FullVideoPath', 'Category']
        ).drop_duplicates(
        subset = ['FullVideoPath'], keep = 'first', ignore_index = True)

    return summ_cat


def replace_pos_neg_cat(cat_data, positive_cat, negative_cat):

    cat_data = cat_data.replace(dict.fromkeys(positive_cat, 1))
    cat_data = cat_data.replace(dict.fromkeys(negative_cat, 0))

    return cat_data


def replace_TP_TN_FP_FN(cat_data):

    cat_data = cat_data.replace('00', 'TN') #true negatives
    cat_data = cat_data.replace('01', 'FP') #true negatives
    cat_data = cat_data.replace('10', 'FN') #true negatives
    cat_data = cat_data.replace('11', 'TP') #true negatives

    return(cat_data)


def confusion_mat(true_cat, predicted_cat):

    true_cat = true_cat.rename(columns = {'Category': 'CategoryTrue'})
    predicted_cat = predicted_cat.rename(columns = {'Category': 'CategoryPredicted'})

    both_cat = pd.merge(true_cat, predicted_cat, on = ['FullVideoPath'])
    
    positive_categories = [1] #animal is positive class
    nagative_categories = [0, 2, 3] #empty, human, and vehicle is negative classes
    both_cat = replace_pos_neg_cat(both_cat, positive_categories, nagative_categories)

    both_cat['CategoryTrue'] = both_cat['CategoryTrue'].apply(str)
    both_cat['CategoryPredicted'] = both_cat['CategoryPredicted'].apply(str)
    both_cat['CategoryBoth'] = both_cat['CategoryTrue'] + both_cat['CategoryPredicted']
    both_cat = replace_TP_TN_FP_FN(both_cat)

    confusion_matrix = both_cat.groupby(['CategoryBoth']).size().reset_index(name='counts')

    return confusion_matrix


def acc_metrics(confusion_matrix):

    FN = int(confusion_matrix['counts'][confusion_matrix['CategoryBoth'] == 'FN'])
    FP = int(confusion_matrix['counts'][confusion_matrix['CategoryBoth'] == 'FP'])
    TN = int(confusion_matrix['counts'][confusion_matrix['CategoryBoth'] == 'TN'])
    TP = int(confusion_matrix['counts'][confusion_matrix['CategoryBoth'] == 'TP'])

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    acc_dict = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Recall': recall, 
        'Precision': precision,
        'F1Score': f1_score
    }

    return acc_dict


def true_vs_pred(options):
    summ_manual = summarise_cat(options.manual_id_csv)
    summ_md = summarise_cat(options.roll_avg_video_csv)
    
    confusion_matrix = confusion_mat(summ_manual, summ_md)
    
    acc_dict = acc_metrics(confusion_matrix)

    return acc_dict


def optimise_roll_avg(options):
    
    md_images, detector_label_map, Fs = load_detector_output(options.full_det_frames_json)
    
    all_combi_acc = pd.DataFrame()
    num_combi = len(options.rolling_avg_size_range) * len(options.iou_threshold_range) * len(options.conf_threshold_buf_range)
    
    with tqdm(total = num_combi) as pbar:
        for rolling_avg_size in options.rolling_avg_size_range:
            options.rolling_avg_size = rolling_avg_size
            
            for iou_threshold in options.iou_threshold_range:
                options.iou_threshold = iou_threshold
                
                for conf_threshold_buf in options.conf_threshold_buf_range:
                    options.conf_threshold_buf = conf_threshold_buf
                    
                    # results of rolling avg with each combination of args will be saved in 
                    # options.roll_avg_video_csv automatically 
                    rolling_avg(options, md_images, Fs, mute = True) 

                    acc_dict = true_vs_pred(options)
                    
                    roll_avg_results_dict = {
                        'ManualIDCsv': options.manual_id_csv,
                        'RollAvgCsv': options.roll_avg_video_csv,
                        'RollAvgSize': options.rolling_avg_size,
                        'IOUThreshold': options.iou_threshold,
                        'ConfThresholdBuffer': options.conf_threshold_buf
                    }
                    
                    roll_avg_results_dict.update(acc_dict)
                    combi_pd = pd.DataFrame(roll_avg_results_dict, index = [0])
                    all_combi_acc = pd.concat([all_combi_acc, combi_pd])

                    pbar.update(1)
    
    all_combi_acc = all_combi_acc.sort_values(by = ['F1Score'], ascending = False)
    
    options.optimise_roll_avg_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.optimise_roll_avg_csv, '_optimise_roll_avg.csv'
    )
    all_combi_acc.to_csv(options.optimise_roll_avg_csv, index = False)


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    ## Create output folder if not present
    check_output_dir(options)
    os.makedirs(options.output_dir, exist_ok=True)

    ## Run optimiser
    optimise_roll_avg(options) #TODO [not impt] parallise for faster speed


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to optimise rolling prediction averaging results.')
    parser.add_argument('--input_dir', type=str,
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the videos and datasheets needed.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--manual_id_csv', type=str,
                        default = config.MANUAL_ID_CSV, 
                        help = 'Path to csv file containing the results of manual identification detections.'
    )
    parser.add_argument('--optimise_roll_avg_csv', type=str,
                        default = config.OPTIMISE_ROLL_AVG_CSV, 
                        help = 'Path to csv file for which the optimiser results will be saved.'
    )
    parser.add_argument('--full_det_frames_json', type=str,
                        default = config.FULL_DET_FRAMES_JSON, 
                        help = 'Path to json file containing the frame-level results of all MegaDetector detections.'
    )
    parser.add_argument('--rolling_avg_size_range', type=int,
                        default = config.ROLLING_AVG_SIZE_RANGE, 
                        help = 'Range of rolling average sizes to test.'
    )
    parser.add_argument('--iou_threshold_range', type=float,
                        default = config.IOU_THRESHOLD_RANGE, 
                        help = 'Range of IOU threshold values to test.'
    )
    parser.add_argument('--conf_threshold_buf_range', type=str,
                        default = config.CONF_THRESHOLD_BUF_RANGE, 
                        help = 'Range of confidence threshold buffer values to test.'
    )
    return parser
    

if __name__ == '__main__':

    main()