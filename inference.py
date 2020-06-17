import os
import sys
import numpy as np
import logging

from math import atan2, degrees


# add the tensorflow models project to the Python path
# github tensorflow/models
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)

from object_detection.utils.np_box_ops import area

log = logging.getLogger(__name__)

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
    12:00 ==  90
     3:00 == 180
     6:00 ==  -90
     9:00 ==   0 

    '''
    rad = atan2((target[0] - actual[0]), (target[1] - actual[1]))
    deg = degrees(rad)
    return rad, deg


def is_area_valid(region_id, area_limits, area):
    '''
    is the area reasonable?
    - could be bad inference
    - could be door in motion
    '''
    valid = False
    # area of gmark detection within min max for given region
    if area >= area_limits[0] and area <= area_limits[1]:
        valid = True
    return valid

def is_on_target(target_center, detection_center, distance_limit, min_angle, max_angle):
    log.info(f'GarageStatus.is_on_target -- target:    ({target_center[0]:02.2f}, {target_center[1]:02.2f})')
    log.info(f'GarageStatus.is_on_target -- detection: ({detection_center[0]:02.2f}, {detection_center[1]:02.2f})'
        f' dist limit: {distance_limit}  {min_angle}  {max_angle} ')
    # defaults
    within_radius = False
    on_travel_path = False
    # now check radius
    # - within_dist == within reasonable distance of the target
    radius_from_target = radius(target_center, detection_center)
    if radius_from_target < distance_limit:
        within_radius = True
    # now check angle
    # - primarily relevant at farther radius
    # - discard radians, keep  the angle in degrees
    _, angle_from_target = angle(target_center, detection_center)
    if angle_from_target >= min_angle and angle_from_target <= max_angle:
        on_travel_path = True

    log.info(f'GarageStatus.is_on_target -- RESULT: ' 
        f' rad: {radius_from_target:02.2f}  ang: {angle_from_target:02.2f}'
        f' within_radius: {within_radius}  on_travel_path: {on_travel_path}')
    return within_radius, on_travel_path

def get_save_detection_path(rule_num, det):
    '''
    rule 1 = save only if there are new objects - all cameras
    rule 2 = only priority camera + region, only new objects
    rule 3 = all images from a specific list of camera/regions
    '''

    # default values
    image_name = None  # default path, None == don't save it
    annotation_name = None
    save = False
    priority = False

    # camera 6 = side yard included
    camera_priority_dict = {
        0 : "True", 
        1 : "True",
        2 : "False",
        3 : "False",
        4 : "True",
        5 : "True",
        6 : "True"}
    # no problem adding regions that don't exist
    region_priority_dict = {
        0 : {0: "True", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        1 : {0: "False", 1 : "False", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        2 : {0: "True", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        3 : {0: "True", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        4 : {0: "False", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        5 : {0: "True", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
        6 : {0: "True", 1 : "True", 2 : "True", 3 : "True", 4 : "True", 5 : "True", 6 : "True", 7 : "True"},
    }

    priority = camera_priority_dict[det.camera_id]
    if priority == 'True':
        priority = region_priority_dict[det.camera_id][det.region_id]
    
    # figure out the rules
    # - just use save = True/False

    # this is saving all regions wtih new objects
    if rule_num == 1 and det.new_objects > 0:
        save = True

    # this is saving high priority images with new objects
    if rule_num == 2 and priority == 'True' and det.new_objects > 0:
        save = True
    
    # this is like saving the stream
    if rule_num == 3:
        if det.camera_id == 5:
            if det.region_id = 1:
                save = True

    if save == True:
    # base name == sssssss-camera_id-region_id-c/g
        color_code = 'c'            # g == grayscale, c == color
        if det.is_color == 0:
            color_code = 'g'                                
        base_name = '{}-{}-{}-{}'.format(det.image_time, det.camera_id, det.region_id, color_code)   
        image_name = os.path.join(settings.image_path,  base_name + '.jpg')
        annotation_name = os.path.join(settings.annotation_path,  base_name + '.xml')
        log.debug(f"image_consumer/get_save_detection_path -- saving: {image_name} {image.shape} {annotation_name}")
    return image_name, annotation_name

# C L A S S
#
# RegionDetection (parent)
# -- ModelInference (child)


class RegionDetection():
    def __init__(self, image_time, camera_id, region_id, is_color, new_objects, dup_objects, model_inference):
        self.image_time = image_time  # time * 10
        self.camera_id = camera_id
        self.region_id = region_id
        self.is_color = is_color
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
