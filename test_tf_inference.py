import os
import sys

import numpy
import cv2
import tensorflow as tf

import tensorflow_util
# GLOBAL VARIABLES
PROJECT_DIR = os.getcwd()

IMAGE_DIR = os.path.join(PROJECT_DIR, "jpeg_images")
MODEL_PATH = os.path.join(PROJECT_DIR, "model/frozen_inference_graph.pb")
LABEL_MAP = os.path.join(PROJECT_DIR, "model/mscoco_label_map.pbtxt")
ANNOTATION_DIR = os.path.join(PROJECT_DIR, "annotation")

detection_graph = tensorflow_util.get_detection_graph(MODEL_PATH)

print("Detection Graph:", type(detection_graph))
print  (detection_graph)