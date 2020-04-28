import os
import sys
import time

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

import tensorflow_util
# GLOBAL VARIABLES
PROJECT_DIR = os.getcwd()

IMAGE_DIR = os.path.join(PROJECT_DIR, "jpeg_images")
MODEL_PATH = os.path.join(PROJECT_DIR, "model/frozen_inference_graph.pb")
LABEL_MAP = os.path.join(PROJECT_DIR, "model/security_label_map.pbtxt")
ANNOTATION_DIR = os.path.join(PROJECT_DIR, "annotation")

# get a frozen graph
detection_graph = tensorflow_util.get_detection_graph(MODEL_PATH)
print("Detection Graph:", type(detection_graph))

# get images list
dir_list = os.listdir(IMAGE_DIR)
image_list = list()
for f in dir_list:
    full_path = os.path.join(IMAGE_DIR, f)
    if os.path.isfile(full_path) and os.path.splitext(f)[1].lower() == '.jpg':
        image_list.append(full_path)

# limitations with the way we are displaying
image_list = image_list[5:1000]
print ("Image Count:", len(image_list))
print ("Sample:",image_list[0])

# run all of the images through

# i = tensorflow_util.send_image_to_frozen_graph(image_list, detection_graph)

sess, tensor_dict, image_tensor = tensorflow_util.get_tf_session(detection_graph)
print ("session:", sess)
print ("tensor_dict:", tensor_dict)
print ("image_tensor:", image_tensor)

for i,image_path in enumerate(image_list):
    print (i, image_path)
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = tensorflow_util.load_image_into_numpy_array(image_path)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    start = time.perf_counter()
    # -- run model
    output_tuple = tensorflow_util.send_image_to_tf_sess(image_np_expanded, sess, tensor_dict, image_tensor)
    # print ("Output tuple: \n", type(output_tuple), output_tuple)

    # get data for relavant detections
    num_detections = len(output_tuple[0])
    detection_scores = output_tuple[0]
    detection_classes = output_tuple[1]
    detection_boxes = output_tuple[2]

    print ("Number of Detections:", num_detections)
    print ("scores:", detection_scores)
    print ("classes:", detection_classes)
    print ("bboxes:", detection_boxes)


    finish = time.perf_counter()
    print (f'Finished in {round(finish - start, 2)} seconds(s)')
    if i > 50:
        break


