import os
import argparse
from matplotlib.pyplot import close
from numpy import dtype
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing as mp

# Functions imported from this project
import config
from shared_utils import VideoOptions, load_detector_output
from shared_utils import check_output_dir, default_path_from_none
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


def summarise_cat(full_df):
    """
    Summarises dataset into two columns: 'UniqueFileName' and 'Category', 
        where each row is a unique UniqueFileName. If a UniqueFileName has multiple 
        Categories (more than one detection or animal), they are summarised 
        using the following rules:
    1) Where there are multiples of the same category, only one is kept
    2) Where there is an animal (1) detection, and human (2) and/or vehicle (3) detection, 
        the video is considered to be an animal category (1)
    3) Where there is a human (2) and vehicle (3), the video is considered to be a human category (2)
    """

    video_cat = full_df[['UniqueFileName', 'Category']]
    summ_cat = video_cat.sort_values(
        by = ['UniqueFileName', 'Category']
        ).drop_duplicates(
        subset = ['UniqueFileName'], keep = 'first', ignore_index = True)

    return summ_cat


def replace_pos_neg_cat(cat_data, positive_cat, negative_cat):

    cat_data = cat_data.replace(dict.fromkeys(positive_cat, '1'))
    cat_data = cat_data.replace(dict.fromkeys(negative_cat, '0'))

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

    both_cat = pd.merge(true_cat, predicted_cat, on = ['UniqueFileName'])
    
    positive_categories = ['1'] #animal is positive class
    nagative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    both_cat['CategoryTrue'] = both_cat['CategoryTrue'].apply(str)
    both_cat['CategoryPredicted'] = both_cat['CategoryPredicted'].apply(str)
    both_cat = replace_pos_neg_cat(both_cat, positive_categories, nagative_categories)

    both_cat['CategoryBoth'] = both_cat['CategoryTrue'] + both_cat['CategoryPredicted']
    both_cat = replace_TP_TN_FP_FN(both_cat)

    confusion_matrix = both_cat.groupby(['CategoryBoth']).size().reset_index(name='counts')

    acc_mets = ['FN', 'FP', 'TN', 'TP']
    for acc_met in acc_mets: 
        if not (acc_met in confusion_matrix['CategoryBoth'].tolist()):
            fill_0 = pd.DataFrame({'CategoryBoth': acc_met, 'counts': 0}, index = [0])

            confusion_matrix = pd.concat([confusion_matrix, fill_0])

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


def condense_md(megadetector_df):

    manual_df_subset = megadetector_df[['UniqueFileName', 'Category', 'DetectedObj']]

    df_subset = manual_df_subset.copy()
    df_subset['AllSpeciesMD'] = "cat_" + df_subset['Category'].map(str) + "_" + df_subset['DetectedObj'].map(str)

    df_subset = df_subset.groupby(['UniqueFileName'])['AllSpeciesMD'].agg(lambda col: ','.join(col))

    cat_summ = summarise_cat(megadetector_df)
    cat_summ = cat_summ.rename(columns = {'Category': 'CategoryMD'})

    condense_df = pd.merge(df_subset, cat_summ, on = ['UniqueFileName'])

    return condense_df


def condense_manual(manual_df):

    manual_df_subset = manual_df[[
        'UniqueFileName', 'Station', 'SamplingDate', 'DateTime', 'Remarks', 
        'ScientificName', 'Quantity']]
    
    df_subset = manual_df_subset.copy()

    df_subset['Quantity'] = df_subset['Quantity'].astype(str)
    df_subset['SpeciesQty'] = df_subset[['ScientificName', 'Quantity']].agg(' '.join, axis = 1)

    df_subset['AllSpeciesManual'] = df_subset.groupby(
        ['UniqueFileName']
        )['SpeciesQty'].transform(lambda x: ','.join(x))

    df_subset = df_subset.drop(['ScientificName', 'Quantity', 'SpeciesQty'], axis = 1)

    cat_summ = summarise_cat(manual_df)
    cat_summ = cat_summ.rename(columns = {'Category': 'CategoryManual'})
    
    condense_df = pd.merge(df_subset, cat_summ, on = ['UniqueFileName'])
    
    return condense_df


def true_vs_pred(options):

    ## Load true and predicted results
    manual_df = pd.read_csv(options.manual_id_csv, dtype=str)
    megadetector_df = pd.read_csv(options.roll_avg_video_csv, dtype=str)
    
    ## Calculate recall, precision, and F1 score
    summ_manual = summarise_cat(manual_df)
    summ_md = summarise_cat(megadetector_df)
    
    confusion_matrix = confusion_mat(summ_manual, summ_md)
    
    acc_dict = acc_metrics(confusion_matrix)
    
    roll_avg_args_dict = {
        'ManualIDCsv': options.manual_id_csv,
        'RollAvgCsv': options.roll_avg_video_csv,
        'RollAvgSize': options.rolling_avg_size,
        'IOUThreshold': options.iou_threshold,
        'RenderingConfThreshold': options.rendering_confidence_threshold, 
        'ConfThresholdBuffer': options.conf_threshold_buf
    }
    
    roll_avg_args_dict.update(acc_dict)
    
    acc_pd = pd.DataFrame(roll_avg_args_dict, index = [0])

    ## Export comparision file for one video per row
    video_summ_manual = condense_manual(manual_df)
    video_summ_md = condense_md(megadetector_df)

    video_summ_pd = pd.merge(video_summ_manual, video_summ_md, on = ['UniqueFileName'])

    positive_categories = ['1'] #animal is positive class
    nagative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    video_summ_pd = replace_pos_neg_cat(video_summ_pd, positive_categories, nagative_categories)

    video_summ_pd['CategoryManual'] = video_summ_pd['CategoryManual'].apply(str)
    video_summ_pd['CategoryMD'] = video_summ_pd['CategoryMD'].apply(str)
    video_summ_pd['CategoryBoth'] = video_summ_pd['CategoryManual'] + video_summ_pd['CategoryMD']
    video_summ_pd = replace_TP_TN_FP_FN(video_summ_pd)

    video_summ_pd['ManualCheckRemarks'] = ""

    return acc_pd, video_summ_pd


def roll_avg_combi(options, images, Fs, arg_combi):
    options.rolling_avg_size = arg_combi[0]
    options.iou_threshold = arg_combi[1]
    options.conf_threshold_buf = arg_combi[2]

    ## Run rolling prediction averaging using the unique combination of arguments
    ## roll_avg_video_csv will be exported to be used later in true_vs_pred()
    rolling_avg(options, images, Fs, mute = True) 

    ## Compare the manual ID with the rolling prediction averaging results just conducted
    acc_pd, video_summ_pd = true_vs_pred(options)

    ## Export out manual_vs_md.csv for each unique combination of rolling average arguments
    options.manual_vs_md_csv = None
    options.manual_vs_md_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.manual_vs_md_csv, 
        "_manual_vs_md_size_{}_iou_{}_buf_{}.csv".format(
            options.rolling_avg_size, 
            options.iou_threshold,
            options.conf_threshold_buf
        )
    )
    video_summ_pd.to_csv(options.manual_vs_md_csv, index = False)

    return acc_pd


def optimise_roll_avg(options, parallel = False):
    
    print("Loading MegaDetector detections from {}...".format(options.full_det_frames_json))
    md_images, detector_label_map, Fs = load_detector_output(options.full_det_frames_json)
    
    num_combi = len(options.rolling_avg_size_range) * len(options.iou_threshold_range) * len(options.conf_threshold_buf_range)
    
    arg_combis = list(product(
        options.rolling_avg_size_range, 
        options.iou_threshold_range,
        options.conf_threshold_buf_range))

    print("Running rolling prediction averaging for {} unique combination of arguments.".format(len(arg_combis)))
    
    if parallel: #problems with memory. Should parallel the rolling_avg instead
        all_combi_acc = []
        def callback_func(result):
            
            all_combi_acc.append(result)
            pbar.update()

        pool = mp.Pool(mp.cpu_count())

        pbar = tqdm(total = len(arg_combis))
        for arg_combi in arg_combis:
            pool.apply_async(
                roll_avg_combi, args = (options, md_images, Fs, arg_combi), 
                callback = callback_func)

        pool.close()
        pool.join()

        ## Concatenate into one dataframe
        all_combi_acc_pd = pd.concat(all_combi_acc)

    else:
        all_combi_acc_pd = pd.DataFrame()
        for arg_combi in tqdm(arg_combis):

            acc_pd = roll_avg_combi(
                options, md_images, Fs, arg_combi)

            all_combi_acc_pd = pd.concat([all_combi_acc_pd, acc_pd])
    
    all_combi_acc_pd = all_combi_acc_pd.sort_values(by = ['F1Score'], ascending = False)
    
    options.roll_avg_acc_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.roll_avg_acc_csv, '_roll_avg_acc.csv'
    )
    all_combi_acc_pd.to_csv(options.roll_avg_acc_csv, index = False)


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
    parser.add_argument('--check_accuracy', type=bool,
                        default = config.CHECK_ACCURACY, 
                        help="Whether to run accuracy test against manual ID"
    )
    return parser
    

if __name__ == '__main__':

    main()