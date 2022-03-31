import os
import argparse
import pandas as pd
from tqdm import tqdm
from itertools import product

# Functions imported from this project
import config
from shared_utils import VideoOptions, load_detector_output
from shared_utils import check_output_dir, default_path_from_none
from rolling_avg import rolling_avg

# Functions imported from Microsoft/CameraTraps github repository
from ct_utils import args_to_object


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


def confusion_mat(video_summ_pd):

    confusion_matrix = video_summ_pd.groupby(['AccClass']).size().reset_index(name='counts')

    acc_mets = ['FN', 'FP', 'TN', 'TP']
    for acc_met in acc_mets: 
        if not (acc_met in confusion_matrix['AccClass'].tolist()):
            fill_0 = pd.DataFrame({'AccClass': acc_met, 'counts': 0}, index = [0])

            confusion_matrix = pd.concat([confusion_matrix, fill_0])

    return confusion_matrix


def acc_metrics(confusion_matrix):

    FN = int(confusion_matrix['counts'][confusion_matrix['AccClass'] == 'FN'])
    FP = int(confusion_matrix['counts'][confusion_matrix['AccClass'] == 'FP'])
    TN = int(confusion_matrix['counts'][confusion_matrix['AccClass'] == 'TN'])
    TP = int(confusion_matrix['counts'][confusion_matrix['AccClass'] == 'TP'])

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    acc_dict = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Recall': round(recall*100, 1), 
        'Precision': round(precision*100, 1),
        'F1Score': round(f1_score*100, 1)
    }

    return acc_dict


def quantity_acc(video_summ_pd):

    qty = video_summ_pd.copy()
    qty = qty[['Total_Qty_Diff', 'UniqueFileName']]
    
    qty['QtyAcc'] = ""
    qty.loc[qty['Total_Qty_Diff'] > 0, 'QtyAcc'] = "Overestimation"
    qty.loc[qty['Total_Qty_Diff'] == 0, 'QtyAcc'] = "CorrectQuantity"
    qty.loc[qty['Total_Qty_Diff'] < 0, 'QtyAcc'] = "Underestimation"
    
    qty_acc = qty.groupby(['QtyAcc']).size().reset_index(name='counts')
    
    correct = qty_acc['counts'][qty_acc['QtyAcc'] == 'CorrectQuantity']
    over = qty_acc['counts'][qty_acc['QtyAcc'] == 'Overestimation']
    under = qty_acc['counts'][qty_acc['QtyAcc'] == 'Underestimation']
    
    if over.empty: 
        over = 0

    if under.empty: 
        under = 0

    qty_dict = {
        'Num_CorrectQty': correct,
        'Num_Overestimated': over,
        'Num_Underestimated': under,
        'Perc_CorrectQty': round(correct/(correct+over+under)*100, 1)
    }

    return qty_dict


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


def condense_md(megadetector_df):
    
    megadetector_df_copy = megadetector_df.copy()
    megadetector_df_copy['Category'] = megadetector_df_copy['Category'].astype(str)
    positive_categories = ['1'] #animal is positive class
    nagative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    megadetector_df_copy = replace_pos_neg_cat(megadetector_df_copy, positive_categories, nagative_categories)

    ## Summarising quantity per class
    md_qty_subset = megadetector_df_copy[['UniqueFileName', 'Category']]
    md_cat_qty = md_qty_subset.copy()
    
    df_summ_qty = md_cat_qty.groupby(['UniqueFileName', 'Category']).size().reset_index(name = "Quantity")
    
    df_summ_qty = df_summ_qty.pivot_table(index = 'UniqueFileName', columns = 'Category', values = 'Quantity')
    df_summ_qty = df_summ_qty.rename(columns = {
        '0': 'MD_Human_Qty', '1': 'MD_Animal_Qty'
    }).fillna(0)

    ## Creating AllSpeciesMD column
    md_as_subset = megadetector_df_copy[['UniqueFileName', 'Category', 'DetectedObj']]

    md_as = md_as_subset.copy()
    md_as['MD_AllSpecies'] = "cat_" + md_as['Category'].map(str) + "_" + md_as['DetectedObj'].map(str)

    md_as = md_as.groupby(['UniqueFileName'])['MD_AllSpecies'].agg(lambda col: ','.join(col))

    ## Create unique category for each video 
    cat_summ = summarise_cat(megadetector_df_copy)
    cat_summ = cat_summ.rename(columns = {'Category': 'MD_Cat'})

    ## Merge the dataframes into one
    condense_df = pd.merge(md_as, cat_summ, on = ['UniqueFileName'])
    condense_df = pd.merge(condense_df, df_summ_qty, on = ['UniqueFileName'])
    
    return condense_df


def condense_manual(manual_df):

    manual_df_copy = manual_df.copy()
    manual_df_copy['Quantity'] = manual_df_copy['Quantity'].astype(int)
    manual_df_copy['Category'] = manual_df_copy['Category'].astype(str)
    positive_categories = ['1'] #animal is positive class
    nagative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    manual_df_copy = replace_pos_neg_cat(manual_df_copy, positive_categories, nagative_categories)

    ## Summarising quantity per class
    man_qty_subset = manual_df_copy[['UniqueFileName', 'Category', 'Quantity']]
    man_cat_qty = man_qty_subset.copy()

    df_summ_qty = man_cat_qty.groupby(['UniqueFileName', 'Category']).sum()
    df_summ_qty = df_summ_qty.pivot_table(index = 'UniqueFileName', columns = 'Category', values = 'Quantity')
    df_summ_qty = df_summ_qty.rename(columns = {
        '0': 'Manual_Human_Qty', '1': 'Manual_Animal_Qty'
    }).fillna(0)

    ## Creating AllSpeciesManual column
    df_subset = manual_df_copy[[
        'UniqueFileName', 'FullVideoPath', 'Station', 'SamplingDate', 'DateTime', 'Manual_Remarks', 
        'ScientificName', 'Quantity']]
    df_all_species = df_subset.copy()

    df_all_species['Quantity'] = df_all_species['Quantity'].astype(str)
    df_all_species['SpeciesQty'] = df_all_species[['ScientificName', 'Quantity']].agg(' '.join, axis = 1)

    df_all_species['Manual_AllSpecies'] = df_all_species.groupby(
        ['UniqueFileName']
        )['SpeciesQty'].transform(lambda x: ','.join(x))

    df_all_species = df_all_species.drop(['ScientificName', 'Quantity', 'SpeciesQty'], axis = 1)

    ## Create unique category for each video 
    cat_summ = summarise_cat(manual_df_copy)
    cat_summ = cat_summ.rename(columns = {'Category': 'Manual_Cat'})
    
    ## Merge the dataframes into one
    condense_df = pd.merge(df_all_species, cat_summ, on = ['UniqueFileName'])
    condense_df = pd.merge(condense_df, df_summ_qty, on = ['UniqueFileName'])
    
    return condense_df


def true_vs_pred(options):

    ## Load true and predicted results
    manual_df = pd.read_csv(options.manual_id_csv, dtype=str)
    megadetector_df = pd.read_csv(options.roll_avg_video_csv, dtype=str)

    ## Create comparision file for one video per row (manual_vs_md.csv)
    video_summ_manual = condense_manual(manual_df)
    video_summ_md = condense_md(megadetector_df)

    video_summ_pd = pd.merge(video_summ_manual, video_summ_md, on = ['UniqueFileName'])

    # Replace positive and negative categories to accuracy class
    video_summ_pd['Manual_Cat'] = video_summ_pd['Manual_Cat'].apply(str)
    video_summ_pd['MD_Cat'] = video_summ_pd['MD_Cat'].apply(str)
    video_summ_pd['AccClass'] = video_summ_pd['Manual_Cat'] + video_summ_pd['MD_Cat']
    video_summ_pd = replace_TP_TN_FP_FN(video_summ_pd)

    # Columns for difference in quantity number
    video_summ_pd['Diff_Human_Qty'] = video_summ_pd['MD_Human_Qty'] - video_summ_pd['Manual_Human_Qty']
    video_summ_pd['Diff_Animal_Qty'] = video_summ_pd['MD_Animal_Qty'] - video_summ_pd['Manual_Animal_Qty']
    video_summ_pd['Total_Qty_Diff'] = video_summ_pd['Diff_Human_Qty'] + video_summ_pd['Diff_Animal_Qty']

    # Columns for second check of false negatives
    video_summ_pd['SmallAnimal'] = ""
    video_summ_pd['FastMoving'] = ""
    video_summ_pd['EdgeOfFrame'] = ""
    video_summ_pd['Unidentified'] = ""
    video_summ_pd['WrongManualID'] = ""
    video_summ_pd['SecondCheckRemarks'] = ""

    video_summ_pd = video_summ_pd.sort_values(by = ['Station', 'AccClass'])

    ## Calculate accuracy metrics for all videos
    # Recall, precision and F1 score 
    confusion_matrix = confusion_mat(video_summ_pd)
    acc_dict = acc_metrics(confusion_matrix)

    # Percetage of correct quantities
    qty_dict = quantity_acc(video_summ_pd)

    # Merge dictionaries and convert to pandas df
    roll_avg_args_dict = {
        'ManualIDCsv': options.manual_id_csv,
        'RollAvgCsv': options.roll_avg_video_csv,
        'RollAvgSize': options.rolling_avg_size,
        'IOUThreshold': options.iou_threshold,
        'RenderingConfThreshold': options.rendering_confidence_threshold, 
        'ConfThresholdBuffer': options.conf_threshold_buf
    }
    
    roll_avg_args_dict.update(qty_dict)
    roll_avg_args_dict.update(acc_dict)
    
    acc_pd = pd.DataFrame(roll_avg_args_dict, index = [0])

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


def optimise_roll_avg(options):
    
    print("Loading MegaDetector detections from {}...".format(options.full_det_frames_json))
    md_images, detector_label_map, Fs = load_detector_output(options.full_det_frames_json)
    
    arg_combis = list(product(
        options.rolling_avg_size_range, 
        options.iou_threshold_range,
        options.conf_threshold_buf_range))

    print("Running rolling prediction averaging for {} unique combination of arguments.".format(len(arg_combis)))
    
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
    optimise_roll_avg(options) 


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