import os
import pandas as pd
import argparse

# Functions imported from this project
import config
from shared_utils import VideoOptions


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


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    # Create output folder if not present
    os.makedirs(options.output_dir, exist_ok=True)

    summ_manual = summarise_cat(options.manual_id_csv)
    summ_md = summarise_cat(options.roll_avg_video_csv)
    
    confusion_matrix = confusion_mat(summ_manual, summ_md)
    print(confusion_matrix)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to optimise rolling prediction averaging results.')
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--manual_id_csv', type=str,
                        default = config.MANUAL_ID_CSV, 
                        help = 'Path to csv file containing the results of manual identification detections.'
    )
    parser.add_argument('--roll_avg_video_csv', type=str,
                        default = config.ROLL_AVG_VIDEO_CSV, 
                        help = 'Path to csv file containing the results of MegaDetector detections after rolling averaging.'
    )
    return parser
    

if __name__ == '__main__':

    main()