class VideoOptions:

    input_dir = ''
    output_dir = None

    model_file = ''
    species_database_file = None

    recursive = True 
    n_cores = 1
    
    full_det_frames_json = None
    full_det_video_json = None
    
    resume_from_checkpoint = None
    checkpoint_path = None
    checkpoint_frequency = -1 #default of -1 to not run any checkpointing

    frame_sample = None
    debug_max_frames = -1
    reuse_results_if_available = False
    json_confidence_threshold = 0.0 # Outdated: will be overridden by rolling prediction averaging. Description: don't include boxes in the .json file with confidence below this threshold

    render_output_video = False
    delete_output_frames = True
    frame_folder = None
    rendering_confidence_threshold = 0.2

    rolling_avg_size = 32
    iou_threshold = 0.5
    conf_threshold_buf = 0.7
    nth_highest_confidence = 1

    roll_avg_frames_json = None
    roll_avg_video_json = None
    roll_avg_video_csv = None
    
    check_accuracy = False
    manual_id_csv = None
    manual_vs_md_csv = None
    species_list_csv = None

    rolling_avg_size_range = None
    iou_threshold_range = None
    conf_threshold_buf_range = None
    rendering_confidence_threshold_range = None
    roll_avg_acc_csv = None

    cropped_images_dir = None


def unique(x):
    x = list(sorted(set(x)))
    return x
    