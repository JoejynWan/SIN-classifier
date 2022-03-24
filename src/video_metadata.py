import os
import pandas as pd
from datetime import datetime

from shared_utils import make_output_path

from detection.video_utils import find_videos


def manual_ID_csv(input_dir, csv_file):

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
        file = os.path.join(station_sampledate, vid_name).replace('\\','/')

        vid_full_path = os.path.join(input_dir, vids_rel_path)
        creation_datetime = datetime.fromtimestamp(os.path.getctime(vid_full_path))

        true_vid_dict = {
            'UniqueFileName': file, 
            'FileName': vid_name, 
            'Station': station,
            'SamplingDate': sampledate,
            'Date': str(creation_datetime.date()),
            'Time': str(creation_datetime.time()),
            'DateTime': str(creation_datetime),
            'Genus': str(genus),
            'Species': str(species),
            'ScientificName': str(genus + " " + species),
            'Quantity': quantity,
            'Remarks': remarks
        }
        
        true_vid_pd = pd.DataFrame(true_vid_dict, index = [0])
        true_vids = true_vids.append(true_vid_pd)

    true_vids.to_csv(csv_file, index = False)

def main():

    os.makedirs(output_dir, exist_ok = True)

    manual_ID_csv(input_dir, csv_file)

if __name__ == '__main__':
    input_dir = 'data/20211119'
    output_dir = 'results/20211119'
    csv_file = make_output_path(output_dir, input_dir, "_manual_ID.csv")

    main()