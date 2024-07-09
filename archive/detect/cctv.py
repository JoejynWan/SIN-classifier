import numpy as np
import cv2 as cv 
from PIL import Image
from tqdm import tqdm

from detection.run_detector import is_gpu_available, load_detector
from md_visualization import visualization_utils as vis_utils

## Variables
model_file = 'models/md_v5a.0.0.pt'
video_path = 'data/RopeBridge_CCTV/IMG_0177.AVI'
target_object = 'animal' #'animal', 'person', or 'vehicle'
duration_buffer = 1 #in seconds
output_path = 'data/RopeBridge_CCTV/out_vid1.AVI'

## Label mapping for MegaDetector
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'  # available in megadetector v4+
}

## Load MegaDetector Model
print('GPU available: {}'.format(is_gpu_available(model_file)))
detector = load_detector(model_file)

## Detect animals frame-by-frame to not exceed RAM capacity
cap = cv.VideoCapture(video_path)

fps = cap.get(cv.CAP_PROP_FPS)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))    
size = (frame_width, frame_height) 

frames_buffer = duration_buffer*fps
target_key = [k for k, v in DEFAULT_DETECTOR_LABEL_MAP.items() if v == target_object][0]

current_frame = 0
target_frames = []
pbar = tqdm(total = total_frames)
while cap.isOpened():

    ret, frame = cap.read()

    if ret == True: 

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        im = Image.fromarray(rgb) #convert from np.array to PIL Image

        result = detector.generate_detections_one_image(im, 'im_file',
                                                        detection_threshold=0.2)
        detections = result['detections']
        categories = [d['category'] for d in detections]
        
        if any(cat == target_key for cat in categories):

            last_target_frame = current_frame
        
        if (current_frame <= last_target_frame+frames_buffer):

            vis_utils.render_detection_bounding_boxes(
                detections, im, label_map = DEFAULT_DETECTOR_LABEL_MAP
            )
            im_array = np.asarray(im)
            
            target_frame = {
                'target_frame_num': current_frame, 
                'result': result,
                'labeled_frame': im_array
            }
            target_frames.append(target_frame)

        current_frame = current_frame + 1
        pbar.update(1)

    else: # no frame read, so end of vid
        break
    
cap.release()
pbar.close()

output_vid = cv.VideoWriter(output_path, 
                            cv.VideoWriter_fourcc(*'MJPG'), 
                            fps, size) 

for target_frame in target_frames: 

    labeled_frame = target_frame['labeled_frame']
    output_vid.write(labeled_frame)

output_vid.release()
