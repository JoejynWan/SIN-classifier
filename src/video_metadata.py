import os
import pandas as pd
from datetime import datetime

# Functions imported from this project
from shared_utils import make_output_path, unique

# Functions imported from Microsoft/CameraTraps github repository
from detection.video_utils import find_videos


def manual_ID_csv(input_dir, species_database_file, csv_output_file):

    input_dir_full_paths = find_videos(input_dir, recursive=True)
    vids_rel_paths = [os.path.relpath(s,input_dir) for s in input_dir_full_paths]
    vids_rel_paths = [s.replace('\\','/') for s in vids_rel_paths]

    true_vids = pd.DataFrame()
    for vids_rel_path in vids_rel_paths:

        station_sampledate, species_abundance, vid_name = vids_rel_path.split("/")
        station, sampledate =  station_sampledate.split("_")
        genus, _, species_quantity = species_abundance.partition(" ")
        species, _, quantity_remarks = species_quantity.partition(" ")
        quantity, _, remarks = quantity_remarks.partition(" ")
        if quantity == "":
            quantity = 0
        file = os.path.join(station_sampledate, vid_name).replace('\\','/')

        vid_full_path = os.path.join(input_dir, vids_rel_path)
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
            'Remarks': remarks
        }
        
        true_vid_pd = pd.DataFrame(true_vid_dict, index = [0])
        true_vids = true_vids.append(true_vid_pd)

    ## Merge dataframes based on the FolderSpeciesName key from the species_database
    species_database = pd.read_csv(species_database_file)
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

    true_vids_output.to_csv(csv_output_file, index = False)


def main():

    os.makedirs(output_dir, exist_ok = True)

    manual_ID_csv(input_dir, species_database_file, csv_output_file)

if __name__ == '__main__':
    input_dir = 'data/20211119'
    output_dir = 'results/20211119_test'
    species_database_file = 'data/species_database.csv'
    csv_output_file = make_output_path(output_dir, input_dir, "_manual_ID.csv")

    main()