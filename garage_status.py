import logging
import numpy as np
import time

# import gen_util

from inference import RegionDetection
from inference import ModelInference
from inference import radius, angle
from inference import is_area_valid, is_on_target

# If new camera
# - you'll have to adjust centers
# - maybe?  area

log = logging.getLogger(__name__)

# important assumptions
unknown = -1
cam_garage_outdoor = 0  # ~ 1 sec camera
cam_garage_indoor = 2 


cam_reg_garage_indoor_full = 0
cam_reg_garage_indoor_left_door = 1
cam_reg_garage_indoor_right_door = 2
cam_reg_garage_indoor_window = 3

gmark_class_id = 35
# attributes are embedded lists
# region_id = list index
#  i.e.   region_id = 3 == list[3]
#  i took some short cuts where only regions 0, 1 are relevent - i.e.  no 2,3,4

# gmark -- only relevent in regions 0 & 1
# areas - [(reg0_min, reg0_max), (reg1_min, reg1_max)]
# ** these are good for door status and car status gmarks
gmark_areas = [[0.006, 0.012], [0.027, 0.035]]
gmark_dist_limits = [0.15, 0.2]

# region specific
# regions = (y, x) - normalized
# get the centers from the log - look for:
#   GarageStatus.update - (cam) 2  (reg) 0,1 = centers: [[y,x]]
gmark_door = 0
gmark_door_centers = np.asarray([[0.18, 0.23], [0.12, 0.52]])
gmark_car = 1
gmark_car_centers = np.asarray([[0.505, 0.178], [0.842, 0.416]])

garage_door_closed = 0
garage_door_moving = 5
garage_door_open = 10

garage_car_absent = 0
garage_car_moving = 5
garage_car_present = 10

history_depth = 120     # 1/2 second increments, depth of stack

# common helpers

def update_history(history, image_time, status_code, comment=''):
    '''
    update the correct "slot" of the array with the status
    '''
    
    # remember - timestamp == time * 10 == 1/10s of seconds
    hist_timestamp = history[0]                                 # history array timestamp
    det_timestamp = image_time                                  # timestamp when image was grabbed
    index = int(((hist_timestamp - det_timestamp) + 1) / 10)    # add  1 because position 0 == timestamp
    
    log.info(f'GarageStatus.update_history: {history.tolist()[:10]}  image_time:{image_time}  status:{status_code}  index: {index} {comment}')
    if index < (history_depth) and index > 0:
        history[index] = status_code
    return history



    
def is_gmark_door(region_id, detection_center):
    '''
    you can verify:
    - this is a garage door mark
    - if the door is moving

    BUT - you cannot verify that the door is open
        - could be, you just didn't detect the gmark
    '''
    # default
    gmark_id = unknown
    door_status = unknown
    # is this a door status mark?  -- most likely
    # - get the distance from true center
    # - get the angle from true center
    within_radius, on_travel_path = is_on_target(gmark_door_centers[region_id], detection_center, 
        gmark_dist_limits[region_id], 80, 100)

    # on target
    if within_radius == True:
        gmark_id = gmark_door
        door_status = garage_door_closed
    # not on target
    # - but is it moving within path?
    # WEAKNESS - not validing the radius in this condition, but geometrically, it's a safe bet
    #            given the door rolls up to the top
    else:
        if on_travel_path == True:
            gmark_id = gmark_door
            door_status = garage_door_moving

    return gmark_id, door_status

def is_gmark_car(region_id, detection_center):
    '''
    you can verify:
    - if this is a car gmark
      THUS - car is NOT present
    - if door is moving

    BUT - you cannot verify that a car is present
        - could be, you just didn't detect the gmark
    '''
    # default
    gmark_id = unknown
    car_status = unknown
    # is this a car status mark? 
    # - get the distance from true center
    # - get the angle from true center
    within_radius, on_travel_path = is_on_target(gmark_car_centers[region_id], detection_center, 
        gmark_dist_limits[region_id], 80, 100)

    # on target
    if within_radius == True:
        gmark_id = gmark_car
        car_status = garage_car_absent
    # not on target
    # - but is it moving within path?
    else:
        if on_travel_path == True:
            gmark_id = 0
            car_status = garage_car_moving

    return gmark_id, car_status


def interpret_gmark_position(image_time, region_id, detection_center):
    '''
    GIVEN - you know this is the garage interior camera
          - area of object (bbox) was already validated
          - but could be region 0 (full) or region 1 (left side of door)

    ONLY 1 detection -- not the full collection of detections

    INTERPRET a gmark (garage marker)
    - can be in two regions (full, left door)
      so the parameters are subscripted corresponding to the region_id
    - could be the door marker or the car marker

    detection sent here if:
    - valid area/size
    - valid region

    determine
    - which mark
    - what status
    '''
    # defaults == unknown
    gmark_id = unknown
    door_status = unknown
    car_status = unknown

    # make the detections - exclusive
    gmark_id, door_status = is_gmark_door(region_id, detection_center)
    if gmark_id == unknown:
        gmark_id, car_status = is_gmark_car(region_id, detection_center)

    log.info(f'GarageStatus.interpret_gmark_position -- {region_id}' 
            f' gmark# {gmark_id}  door: {door_status}   car: {car_status}')
        
    return gmark_id, door_status, car_status


def get_gmark_status(garage_status, region_id, det):
    '''
    given full detection (all detected objects)

    car_status == car MARK status   (not the actual car detection)
    '''

    # default == unknown
    gmark_id = unknown
    door_status = unknown
    car_status = unknown

    # get indexes = gmark
    log.debug(f'GarageStatus.get_gmark_status.np.where --- {det.model_inference.class_array.tolist()}')
    gmark_array = np.where(det.model_inference.class_array == gmark_class_id )[0]       # results for single axis == 0
                                                                                        # returns array like [2,3]
    log.debug(f'GarageStatus.get_gmark_status.gmark_array {gmark_array.tolist()}')
    # iterate through the indexes -> pointing to the detections == gmark
    for idx in gmark_array:
        center = det.model_inference.bbox_center_array[idx]
        area = det.model_inference.bbox_area_array[idx]
        area_valid = is_area_valid(region_id, gmark_areas[region_id], area)
        # only if area is value
        if area_valid == True:
            gmark_id, door_status, car_status = interpret_gmark_position(det.image_time, region_id, center)

        log.info(f'GarageStatus.update gmark: idx = {idx} -- {det.image_time} {det.camera_id} {det.region_id} area valid: {area_valid}'
            f' gmark_id: {gmark_id} door_status: {door_status}  car_status: {car_status}')
        # update history
        # - update based on region
        # - only update if you have a known status
        #   - e.g.  an unknown (e.g. car) while evaluating a door mark  would possibly overwrite a known door mark status

        #    -- full image region 0 --
        if det.region_id == cam_reg_garage_indoor_full:
            if door_status >= 0:
                update_history(garage_status.door_r0_history, det.image_time, door_status, 'r0:door')
            if car_status >= 0:
                update_history(garage_status.car_mark_r0_history, det.image_time, car_status, 'r0:car_mark')

        #    -- left door region 1 --
        else:
            if door_status >= 0:
                update_history(garage_status.door_r1_history, det.image_time, door_status, 'r1:door')
            if car_status >= 0:
                update_history(garage_status.car_mark_r1_history, det.image_time, car_status, 'r1:car_mark')

            
    return door_status, car_status

class GarageStatus:
    '''
    history == 121
       [0] == timestamp
       [1:history_depth] == stack

       door_r0 == gmark on region 0 (full res)
       door_r1 == gmark on region 1 (left side of door close-up)
    '''
    def __init__(self, door_status, car_present, person_present, light_status):
        self.door_status = door_status
        self.car_present = car_present
        self.person_present = person_present
        self.light_status = light_status
        # history
        # -- gmark 0 == door, but there are two regional views of it, so two history stacks
        self.door_r0_history = np.full((history_depth), -1, dtype=int)
        self.door_r0_history[0] = int(time.time() * 10)
        self.door_r1_history = np.full((history_depth), -1, dtype=int)
        self.door_r1_history[0] = int(time.time() * 10)
        # -- gmark 1 == car, name it car mark because you have car also
        self.car_mark_r0_history = np.full((history_depth), -1, dtype=int)
        self.car_mark_r0_history[0] = int(time.time() * 10)
        self.car_mark_r1_history = np.full((history_depth), -1, dtype=int)
        self.car_mark_r1_history[0] = int(time.time() * 10)
        # -- person
        # -- car present - in the garage

        return
    def __str__(self):
        garage_status_string = f".garage_status -- {self.door_status}"
        return garage_status_string

    def update_from_detection(self, det):
        '''
        Update GarageStatus from a Detection (== det)
        '''
        
        log.info(f'GarageStatus.update_from_detection -- {det.camera_id} {det.region_id} = classes: {det.model_inference.class_array.tolist()}')
        log.info(f'GarageStatus.update_from_detection -- {det.camera_id} {det.region_id} = centers: {det.model_inference.bbox_center_array.tolist()}')
        log.info(f'GarageStatus.update_from_detection -- {det.camera_id} {det.region_id} = areas:   {det.model_inference.bbox_area_array.tolist()}')

        # regions w/ gmarks
        #  - region 0 == full res
        #  - region 1 == left door
        # for gmarks, strict position validation
        if det.region_id == cam_reg_garage_indoor_left_door or \
            det.region_id == cam_reg_garage_indoor_full:
            log.info(f'GarageStatus.update_from_detection -- check gmarks')
            self.door_status, self.car_present = get_gmark_status(self, det.region_id, det)

            log.info(f'GarageStatus door (r0) history: {self.door_r0_history.tolist()[:20]}')
            log.info(f'GarageStatus door (r1) history: {self.door_r1_history.tolist()[:20]}')
            
            log.info(f'GarageStatus car mark (r0) history: {self.car_mark_r0_history.tolist()[:20]}')
            log.info(f'GarageStatus car mark (r1) history: {self.car_mark_r1_history.tolist()[:20]}')
        return

        # regions w/ (actual car) - region 0 - full only


        # regions w/ people