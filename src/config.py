import multiprocessing


## Paths of inputs and outputs and must be filled in
INPUT_DIR = 'C:/TempDataForSpeed/example_test_set'
OUTPUT_DIR = 'results/example_test_set'


## Paths to required datasets
MODEL_FILE = 'models/md_v5a.0.0.pt'
SPECIES_DATABASE_FILE = 'data/species_database.csv'


## Settings for MegaDetector detection
RECURSIVE = True 
N_CORES = multiprocessing.cpu_count() - 1 # No. of available cores minus 1 to not over tax the system

FULL_DET_FRAMES_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_full_det_frames.json'
FULL_DET_VIDEO_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_full_det_videos.json'

RESUME_FROM_CHECKPOINT = None
CHECKPOINT_PATH = None #Defaults to '[OUTPUT_DIR]/checkpoint_[datetime].json'
CHECKPOINT_FREQUENCY = 10000 #Set to -1 to not run any checkpointing

FRAME_SAMPLE = None
DEBUG_MAX_FRAMES = -1
REUSE_RESULTS_IF_AVAILABLE = False
JSON_CONFIDENCE_THRESHOLD = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold


## Settings for rendering of bounding boxes over videos
RENDER_OUTPUT_VIDEO = True
DELETE_OUTPUT_FRAMES = True
FRAME_FOLDER = None #Defaults to temp folder
RENDERING_CONFIDENCE_THRESHOLD = 0.2 #0.8 for MDv4, 0.2 for MDv5


## Settings for rolling prediction averaging
ROLLING_AVG_SIZE = 32
IOU_THRESHOLD = 0.2
CONF_THRESHOLD_BUF = 0.7
NTH_HIGHEST_CONFIDENCE = 1

ROLL_AVG_FRAMES_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_frames.json'
ROLL_AVG_VIDEO_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_videos.json'
ROLL_AVG_VIDEO_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_videos.csv'


## Settings for comparing MegaDetector results with manual identification results
CHECK_ACCURACY = True
MANUAL_ID_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_manual_ID.csv'
MANUAL_VS_MD_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_manual_vs_md.csv'
SPECIES_LIST_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_species_list.csv'


## Settings for optimising rolling prediction averaging 
ROLLING_AVG_SIZE_RANGE = [32]
IOU_THRESHOLD_RANGE = [0.2]
CONF_THRESHOLD_BUF_RANGE = [0.7]
ROLL_AVG_ACC_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_optimise_roll_avg.csv'
