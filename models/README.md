# Models
This folder is for the storage of trained models that will be used for the detection of animals within camera trap videos. 

## MegaDetector Model
The first part of the workflow involves the drawing of bounding boxes around objects of interest (i.e., animals, humans, and vehicles). The drawing of these bounding boxes is done automatically using a pre-trained model called [MegaDetector](https://github.com/microsoft/CameraTraps) . 

This repository is currently compatible with version 4.1.0 of MegaDetector. [Download](https://github.com/microsoft/CameraTraps/releases/tag/v4.1) the trained model and place the md_v4.1.0.pb file in this folder. The model will then be automatically be used for the drawing of bounding boxes. 
