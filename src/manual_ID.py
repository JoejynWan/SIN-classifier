import os
import argparse
import pandas as pd
from datetime import datetime

# Functions imported from this project
import config
from shared_utils import default_path_from_none, unique, VideoOptions

# Functions imported from Microsoft/CameraTraps github repository
from detection.video_utils import find_videos
from ct_utils import args_to_object


def manual_ID_results(options):

    input_dir_full_paths = find_videos(options.input_dir, recursive=True)
    vids_rel_paths = [os.path.relpath(s,options.input_dir) for s in input_dir_full_paths]
    vids_rel_paths = [s.replace('\\','/') for s in vids_rel_paths]

    true_vids = pd.DataFrame()
    for vids_rel_path in vids_rel_paths:

        station_sampledate, species_abundance, vid_name = vids_rel_path.split("/")
        station, sampledate =  station_sampledate.split("_")
        genus, _, species_quantity = species_abundance.partition(" ")
        species, _, quantity_remarks = species_quantity.partition(" ")
        quantity, _, remarks = quantity_remarks.partition(" ")
        if quantity == "":
            quantity = 1
        file = os.path.join(station_sampledate, vid_name).replace('\\','/')

        vid_full_path = os.path.join(options.input_dir, vids_rel_path)
        creation_datetime = datetime.fromtimestamp(os.path.getctime(vid_full_path))

        true_vid_dict = {
            'FullVideoPath': vids_rel_path, 
            'UniqueFileName': file, 
            'FileName': vid_name, 
            'Station': station,
            'SamplingDate': sampledate,
            'Date': str(creation_datetime.date()),
            'Time': str(creation_datetime.time()),
            'DateTime': str(creation_datetime),
            'FolderSpeciesName': str(genus + " " + species),
            'Quantity': quantity,
            'Manual_Remarks': remarks
        }
        
        true_vid_pd = pd.DataFrame(true_vid_dict, index = [0])
        true_vids = true_vids.append(true_vid_pd)

    ## Merge dataframes based on the FolderSpeciesName key from the species_database
    species_database = pd.read_csv(options.species_database_file)
    true_vids_output = pd.merge(true_vids, species_database, on = 'FolderSpeciesName', how = 'left')
    
    ## Check if there are missing species in the species_database
    true_vids_output.reset_index()
    
    missing_species = []
    for idx, row in true_vids_output.iterrows():
        if pd.isna(row['Category']):
            missing_species.append(row['FolderSpeciesName'])
    
    missing_species = unique(missing_species)
    if missing_species:
        raise ValueError(
            "There are missing species in species_database.csv: {}. Please add them to the species database.".format(
                missing_species))

    ## Rearrange the columns
    true_vids_cols = true_vids.columns.to_list()
    species_database_cols = species_database.columns.to_list()
    col_order = true_vids_cols[0:-3] + species_database_cols[1:] + true_vids_cols[-2:]
    true_vids_output = true_vids_output.fillna('NA')
    true_vids_output = true_vids_output[col_order]

    ## Write output file
    options.manual_id_csv = default_path_from_none(
        options.output_dir, options.input_dir, 
        options.manual_id_csv, "_manual_ID.csv"
    )
    true_vids_output.to_csv(options.manual_id_csv, index = False)
    print('Output file saved at {}'.format(options.manual_id_csv))


def main():

    ## Process Command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    options = VideoOptions()
    args_to_object(args, options)

    os.makedirs(options.output_dir, exist_ok = True)

    ## Produce and export .csv file containing manual identification results
    manual_ID_results(options)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Module to produce a csv depicting the true detections from manual identification based on folder names.')
    parser.add_argument('--input_dir', type=str, 
                        default = config.INPUT_DIR, 
                        help = 'Path to folder containing the video(s) to be processed.'
    )
    parser.add_argument('--output_dir', type=str,
                        default = config.OUTPUT_DIR, 
                        help = 'Path to folder where videos will be saved.'
    )
    parser.add_argument('--species_database_file', type=str,
                        default = config.SPECIES_DATABASE_FILE, 
                        help = 'Path to the species_database.csv which describes details of species.'
    )
    parser.add_argument('--manual_id_csv', type=str,
                        default = config.MANUAL_ID_CSV, 
                        help = 'Path to csv file that contains the results of manual identification.'
    )


if __name__ == '__main__':

    main()