# SIN-classifier
A project to classify animals of Singapore captured in camera trap images and videos. 

## Workflow 
1) Detect and draw bounding boxes around animals in camera trap videos using MegaDetector by Microsoft. 
2) Crop the bounding boxes and create a classifier to classify the detected animals to species level. 

# Initial Setup
## Virtual environment and packages
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has been used to manage the Python package dependencies of this project. Do [install](https://docs.anaconda.com/anaconda/install/index.html) the correct version of Anaconda for your OS. 

The required Python packages and its versions are listed in the environment.yml file found in the root directory of this repository. To use this file to set up your virtural environment, issue the following command in your terminal: 
```
conda env create --file environment.yml
```

To enter the conda virtual environment for use, run:
```
conda activate camphora_classifier
```
You should now see (camphora_classifier) prepended to the command line prompt. Invoking python will now be using the interpreter and packages available in this virtual environment.

To exit the virtual env, issue:
```
conda deactivate
```

## Cloning of other git repos for use
Some scripts (e.g., for the drawing of bounding boxes) require the use of functions from other git repositories. These git repositories have to be cloned, and its path appended to `PYTHONPATH`. On Windows, append a path to `PYTHONPATH` for the current shell session by executing the following on Windows:
```
set PYTHONPATH="%PYTHONPATH%;c:\path\to\the\git\repo"
```