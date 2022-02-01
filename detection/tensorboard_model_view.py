import sys
sys.path.append('D:/CNN_Animal_ID/MegaDetector/CameraTraps')
from detection.run_tf_detector import TFDetector

import datetime
import tensorflow as tf
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard


## View the saved model on Tensorboard
saved_model_dir = "D:/CNN_Animal_ID/CamphoraClassifier/MegaDetectorModel_v4.1/md_v4.1.0_saved_model/saved_model"
logdir = "D:/CNN_Animal_ID/CamphoraClassifier/tensorboard_logs/model_summary_logs/log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import_to_tensorboard(saved_model_dir, logdir, "serve")

if False: #Currently does not work becuase loaded frozen model is a AutoTrackable object
    ## Viewing the frozen model
    frozen_model_path = "D:/CNN_Animal_ID/CamphoraClassifier/MegaDetectorModel_v4.1/md_v4.1.0.pb"
    md_detector = TFDetector(frozen_model_path)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)