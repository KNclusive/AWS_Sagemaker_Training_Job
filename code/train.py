#!/usr/bin/env python3
# Above line is a shebang or hash-bang or hash-pound which basically tells the interpretor what to use to run the below code.

import os
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

# READING TRAINING_FILE_NAME ENV VARIABLE which contains the configuration file
configuration_file_name = os.getenv("CONFIGURATION_FILE_NAME", "training_configuration.yaml")

#STANDARD SAGEMAKER PATH inside the container
prefix = "/opt/ml/input"

#TRAINING DATA PATH
configuration_file_path = os.path.join(prefix, "data/train/configuration_folder", configuration_file_name)

# # HYPERPARAMETERS PATH
# hyperparameters_path = os.path.join(
#     prefix, "config", "hyperparameters.json"
# )

# try:
#     hyperparameters = json.loads(hyperparameters_path)
# except:
#     hyperparameters = {}

logging.info('Model will start training now')

# Load a COCO-pretrained YOLO11n model
model = YOLO("/opt/ml/code/yolo11m.pt")

# Train the model with custom configuration
model.train(cfg=configuration_file_path)

logging.info('Model training finished')