import datetime
import tensorflow.compat.v1 as tf
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard

# Functions imported from Microsoft/CameraTraps github repository
from detection.run_tf_detector import TFDetector
from visualization import visualization_utils as viz_utils

## View the saved model on Tensorboard
# saved_model_dir = "D:/CNN_Animal_ID/CamphoraClassifier/MegaDetectorModel_v4.1/md_v4.1.0_saved_model/saved_model"
# logdir = "D:/CNN_Animal_ID/CamphoraClassifier/tensorboard_logs/model_summary_logs/log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# import_to_tensorboard(saved_model_dir, logdir, "serve")


## Load TFDetector class




def load_detection_graph(model_path):
    print('TFDetector: Loading graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('TFDetector: Detection graph loaded.')

    return detection_graph

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
frozen_model_path = "D:/CNN_Animal_ID/CamphoraClassifier/MegaDetectorModel_v4.1/md_v4.1.0.pb"
image_path = "data/00b82552-0372-4f71-9862-c396fa2d636b.jpg"

detector = TFDetector(frozen_model_path)

image = viz_utils.load_image(image_path)

(box_tensor_out, score_tensor_out, class_tensor_out) = detector._generate_detections_one_image(image)
trainable_params = detector.trainable_variables()
print(trainable_params)

# detection_graph = load_detection_graph(frozen_model_path)
# tf_session = tf.Session(graph=detection_graph)

# image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
# score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
# class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

# tf_session.run([box_tensor, score_tensor, class_tensor])

# print(image_tensor)
# print(box_tensor)
# print(score_tensor)
# print(class_tensor)






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