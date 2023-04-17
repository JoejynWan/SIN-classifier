import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

# Functions imported from this project
import general.config as config
from general.shared_utils import VideoOptions
from detect.detect_utils import summarise_cat, check_output_dir, default_path_from_none
from detect.rolling_avg import rolling_avg

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

    if TP + FN == 0: 
        recall = 0
    else: 
        recall = TP / (TP + FN)
    
    if TP + FP == 0: 
        precision = 0
    else:
        precision = TP / (TP + FP)

    if precision + recall == 0:
        f1_score = 0
    else:
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
    qty = qty[qty['AccClass'] == 'TP']
    qty = qty[['Total_Qty_Diff', 'UniqueFileName']]
    
    qty['QtyAcc'] = ""
    qty.loc[qty['Total_Qty_Diff'] > 0, 'QtyAcc'] = "Overestimation"
    qty.loc[qty['Total_Qty_Diff'] == 0, 'QtyAcc'] = "CorrectQuantity"
    qty.loc[qty['Total_Qty_Diff'] < 0, 'QtyAcc'] = "Underestimation"
    
    qty_acc = qty.groupby(['QtyAcc']).size().reset_index(name='counts')

    over = qty_acc['counts'][qty_acc['QtyAcc'] == 'Overestimation']
    under = qty_acc['counts'][qty_acc['QtyAcc'] == 'Underestimation']
    correct = qty_acc['counts'][qty_acc['QtyAcc'] == 'CorrectQuantity']

    if over.empty: 
        over = 0
    else:
        over = int(over)

    if under.empty: 
        under = 0
    else:
        under = int(under)

    if correct.empty: 
        correct = 0
    else:
        correct = int(correct)
    
    if correct+over+under == 0:
        prec_correct = 0
    else:
        prec_correct = float(round(correct/(correct+over+under)*100, 1))
    
    qty_dict = {
        'Num_CorrectQty': correct,
        'Num_Overestimated': over,
        'Num_Underestimated': under,
        'Perc_CorrectQty': prec_correct
    }
    
    return qty_dict


def condense_md(megadetector_df):
    
    megadetector_df_copy = megadetector_df.copy()
    megadetector_df_copy['Category'] = megadetector_df_copy['Category'].astype(str)

    ## Creating AllSpeciesMD column
    md_as_subset = megadetector_df_copy[['UniqueFileName', 'Category', 'DetectedObj']]
    md_as = md_as_subset.copy()

    md_as['MD_AllSpecies'] = "cat_" + md_as['Category'].map(str) + "_" + md_as['DetectedObj'].map(str)
    md_as.loc[md_as['MD_AllSpecies'] == "cat_0_nan", 'MD_AllSpecies'] = None
    md_as = md_as.groupby(['UniqueFileName'])['MD_AllSpecies'].agg(lambda col: ','.join(filter(None, col)))

    ## Create unique category for each video 
    cat_summ = summarise_cat(megadetector_df_copy)
    positive_categories = ['1'] #animal is positive class
    negative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    cat_summ = replace_pos_neg_cat(cat_summ, positive_categories, negative_categories)
    cat_summ = cat_summ.rename(columns = {'Category': 'MD_Cat'})

    ## Summarising quantity per class
    #human and vehicle considered human
    megadetector_df_NoVeh = megadetector_df_copy.replace(dict.fromkeys(['2', '3'], '2')) 
    md_qty_subset = megadetector_df_NoVeh[['UniqueFileName', 'Category']]
    md_cat_qty = md_qty_subset.copy()
    
    df_summ_qty = md_cat_qty.groupby(['UniqueFileName', 'Category']).size().reset_index(name = "Quantity")
    
    df_summ_qty = df_summ_qty.pivot_table(index = 'UniqueFileName', columns = 'Category', values = 'Quantity')
    if '0' in df_summ_qty.columns: 
        df_summ_qty = df_summ_qty.drop('0', axis = 1) #no need the number of empty
    if not '2' in df_summ_qty.columns: 
        df_summ_qty['2'] = 0
    df_summ_qty = df_summ_qty.rename(columns = {
        '2': 'MD_Human_Qty', '1': 'MD_Animal_Qty'
    }).fillna(0)

    ## Merge the dataframes into one
    condense_df = pd.merge(md_as, cat_summ, on = ['UniqueFileName'])
    condense_df = pd.merge(condense_df, df_summ_qty, on = ['UniqueFileName'])
    
    return condense_df


def condense_manual(manual_df):

    manual_df_copy = manual_df.copy()
    manual_df_copy['Quantity'] = manual_df_copy['Quantity'].astype(int)
    manual_df_copy['Category'] = manual_df_copy['Category'].astype(str)

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
    positive_categories = ['1'] #animal is positive class
    negative_categories = ['0', '2', '3'] #empty, human, and vehicle is negative classes
    cat_summ = replace_pos_neg_cat(cat_summ, positive_categories, negative_categories)
    cat_summ = cat_summ.rename(columns = {'Category': 'Manual_Cat'})

    ## Summarising quantity per class
    manual_df_posneg = replace_pos_neg_cat(manual_df_copy, positive_categories, negative_categories)

    man_qty_subset = manual_df_posneg[['UniqueFileName', 'Category', 'Quantity']]
    man_cat_qty = man_qty_subset.copy()

    df_summ_qty = man_cat_qty.groupby(['UniqueFileName', 'Category']).sum()
    df_summ_qty = df_summ_qty.pivot_table(
        index = 'UniqueFileName', columns = 'Category', values = 'Quantity')

    if not '0' in df_summ_qty.columns:
        df_summ_qty['0'] = np.nan #If there are no videos of the negative category
    if not '1' in df_summ_qty.columns:
        df_summ_qty['1'] = np.nan #If there are no videos of the positive category

    df_summ_qty = df_summ_qty.rename(columns = {
        '0': 'Manual_Human_Qty', '1': 'Manual_Animal_Qty'
    }).fillna(0)

    ## Merge the dataframes into one
    condense_df = pd.merge(df_all_species, cat_summ, on = ['UniqueFileName'])
    condense_df = pd.merge(condense_df, df_summ_qty, on = ['UniqueFileName'])
    
    return condense_df


def gen_manual_vs_md(manual_df, megadetector_df):
    
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
    video_summ_pd['FastMoving'] = ""
    video_summ_pd['TooCloseToLens'] = ""
    video_summ_pd['EdgeOfFrame'] = ""
    video_summ_pd['UnidentifiableSpecies'] = ""
    video_summ_pd['PartlyCoveredByVegetation'] = ""
    video_summ_pd['BWVideo'] = ""
    video_summ_pd['EyeShine'] = ""
    video_summ_pd['TargetInBackground'] = ""
    video_summ_pd['BriefAppearance'] = ""
    video_summ_pd['PartialAnimal'] = ""
    video_summ_pd['WrongManualID'] = ""
    video_summ_pd['SecondCheckRemarks'] = ""

    video_summ_pd = video_summ_pd.sort_values(by = ['AccClass', 'Station'])

    return video_summ_pd


def gen_roll_avg_acc_row(options, video_summ_pd):

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

    return acc_pd


def true_vs_pred(options):

    print("\nComparing MegaDetector detections to manual ID results now...")

    ## Load true and predicted results
    manual_df = pd.read_csv(options.manual_id_csv, dtype=str)
    megadetector_df = pd.read_csv(options.roll_avg_video_csv, dtype=str)
    
    ## Create comparision file for one video per row (manual_vs_md.csv)
    video_summ_pd = gen_manual_vs_md(manual_df, megadetector_df)

    ## Calculate accuracy metrics for all videos
    acc_pd = gen_roll_avg_acc_row(options, video_summ_pd)

    return acc_pd, video_summ_pd


def roll_avg_combi(options, arg_combi):
    options.rolling_avg_size = arg_combi[0]
    options.iou_threshold = arg_combi[1]
    options.conf_threshold_buf = arg_combi[2]
    options.rendering_confidence_threshold = arg_combi[3]

    ## Run rolling prediction averaging using the unique combination of arguments
    ## roll_avg_video_csv will be exported to be used later in true_vs_pred()
    rolling_avg(options, mute = True) 

    ## Compare the manual ID with the rolling prediction averaging results just conducted
    acc_pd, video_summ_pd = true_vs_pred(options)

    ## Export out manual_vs_md.csv for each unique combination of rolling average arguments
    options.manual_vs_md_csv = None #reset the name of manual_vs_md.csv
    options.manual_vs_md_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.manual_vs_md_csv, 
        "_manual_vs_md_size_{}_iou_{}_conf_{}_confbuf_{}.csv".format(
            options.rolling_avg_size, 
            options.iou_threshold,
            options.rendering_confidence_threshold, 
            options.conf_threshold_buf
        )
    )
    video_summ_pd.to_csv(options.manual_vs_md_csv, index = False)

    return acc_pd


def optimise_roll_avg(options):
    
    arg_combis = list(product(
        options.rolling_avg_size_range, 
        options.iou_threshold_range,
        options.conf_threshold_buf_range, 
        options.rendering_confidence_threshold_range)) 
    
    print("Running rolling prediction averaging for {} unique combination of arguments.".format(len(arg_combis)))
    
    all_combi_acc_pd = pd.DataFrame()
    for arg_combi in tqdm(arg_combis):

        acc_pd = roll_avg_combi(options, arg_combi)

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

    ## Check that required files and directories exist are pathed
    assert os.path.isfile(options.full_det_frames_json), '{} is missing.'.format(options.full_det_frames_json)
    assert os.path.isfile(options.manual_id_csv), '{} is missing.'.format(options.manual_id_csv)

    # Create output folder if not present
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
    parser.add_argument('--conf_threshold_buf_range', type=float,
                        default = config.CONF_THRESHOLD_BUF_RANGE, 
                        help = 'Range of confidence threshold buffer values to test.'
    )
    parser.add_argument('--rendering_confidence_threshold_range', type=float,
                        default = config.RENDERING_CONFIDENCE_THRESHOLD_RANGE, 
                        help = 'Range of rendering confidence threshold values to test.'
    )
    parser.add_argument('--check_accuracy', type=bool,
                        default = config.CHECK_ACCURACY, 
                        help="Whether to run accuracy test against manual ID"
    )
    return parser
    

if __name__ == '__main__':

    ## Four Inputs required: 
    # input_dir, output_dir, full_det_frames_json, manual_id_csv
    # check_accuracy = True
    main()
