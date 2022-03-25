import multiprocessing


## Paths of inputs and outputs and must be filled in
INPUT_DIR = 'data/20211119_test'
OUTPUT_DIR = 'results/20211119_test'


## Paths to required datasets
MODEL_FILE = 'models/md_v4.1.0.pb'
SPECIES_DATABASE_FILE = 'data/species_database.csv'


## Settings for MegaDetector detection
RECURSIVE = True 
N_CORES = multiprocessing.cpu_count() - 1 # No. of available cores minus 1 to not over tax the system

FULL_DET_FRAMES_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_full_det_frames.json'
FULL_DET_VIDEO_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_full_det_videos.json'

FRAME_SAMPLE = None
DEBUG_MAX_FRAMES = -1
REUSE_RESULTS_IF_AVAILABLE = False
JSON_CONFIDENCE_THRESHOLD = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold


## Settings for rendering of bounding boxes over videos
RENDER_OUTPUT_VIDEO = True
DELETE_OUTPUT_FRAMES = True
FRAME_FOLDER = None #Defaults to temp folder
RENDERING_CONFIDENCE_THRESHOLD = 0.8


## Settings for rolling prediction averaging
ROLLING_AVG_SIZE = 32
IOU_THRESHOLD = 0.5
CONF_THRESHOLD_BUF = 0.7
NTH_HIGHEST_CONFIDENCE = 1

ROLL_AVG_FRAMES_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_frames.json'
ROLL_AVG_VIDEO_JSON = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_videos.json'
ROLL_AVG_VIDEO_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_roll_avg_videos.csv'

## Settings for comparing MegaDetector results with manual identification results
CHECK_ACCURACY = True
MANUAL_ID_CSV = None #Defaults to '[OUTPUT_DIR]/basename(INPUT_DIR)_manual_ID.csv'
