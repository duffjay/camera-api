import os
import sys
import numpy as np

from math import atan2, degrees


# add the tensorflow models project to the Python path
# github tensorflow/models
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)

from object_detection.utils.np_box_ops import area



# HELPER FUNCTIONS

def get_centers(bbox_array):
    bbox_centers = []
    for bbox in bbox_array:
        center_0 = (bbox[2] - bbox[0]) / 2 + bbox[0]
        center_1 = (bbox[3] - bbox[1]) / 2 + bbox[1]
        bbox_centers.append([center_0, center_1])
    return np.asarray(bbox_centers)

# spatial utils
# points = numpy array == [y, x]
def radius(target, actual):
    return np.linalg.norm(target - actual)
def angle(target, actual):
    '''
    12:00 == -90
     3:00 == 180
     6:00 ==  90
     9:00 ==   0 

    '''
    rad = atan2((target[0] - actual[0]), (target[1] - actual[1]))
    deg = degrees(rad)
    return rad, deg





# C L A S S
#
# RegionDetection (parent)
# -- ModelInference (child)


class RegionDetection():
    def __init__(self, image_time, camera_id, region_id, new_objects, dup_objects, model_inference):
        self.image_time = image_time  # time * 10
        self.camera_id = camera_id
        self.region_id = region_id
        self.new_objects = new_objects
        self.dup_objects = dup_objects
        self.model_inference = model_inference
    def __str__(self):
        region_string = f'RegionDection - camera: {self.camera_id}  region: {self.region_id}  new: {self.new_objects}  dup: {self.dup_objects}\n--{self.model_inference}'
        return region_string


class ModelInference():
    def __init__(self, class_array, prob_array, bbox_array):
        self.class_array = class_array
        self.prob_array = prob_array
        self.bbox_array = bbox_array
        self.bbox_center_array = get_centers(self.bbox_array)
        self.bbox_area_array = area(self.bbox_array)
    def __str__(self):
        inference_string = f'ModelInference \n\
        -- classes: {self.class_array}\n\
        -- probs:   {self.prob_array}\n\
        -- bbox:    {self.bbox_array}\n\
        -- centers: {self.bbox_center_array}\n\
        -- areas:   {self.bbox_area_array}  '
        return inference_string
