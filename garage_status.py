import logging
import numpy as np
import time


from inference import RegionDetection
from inference import ModelInference
from inference import radius, angle



log = logging.getLogger(__name__)

# important assumptions

cam_garage_outdoor = 0  # ~ 1 sec camera
cam_garage_indoor = 2 


cam_reg_full = 0
cam_reg_garage_indoor_left_door = 1
cam_reg_garage_indoor_right_door = 2
cam_reg_garage_indoor_window = 3

gmark_class_id = 35
# attributes are embedded lists
# region_id = list index
#  i.e.   region_id = 3 == list[3]
#  i took some short cuts where only regions 0, 1 are relevent - i.e.  no 2,3,4

# gmark -- only relevent in regions 0 & 1
# areas - [(region 0), region 1)]
#          (region 0) = (min, max)
gmark_areas = [[0.02, 0.03], [0.027, 0.035]]
gmark_dist_limits = [0.15, 0.2]
# regions = (y, x) - normalized
gmark_door = 0
gmark_door_centers = np.asarray([[0.7, 0.25], [0.12, 0.52]])
gmark_car = 1
gmark_car_centers = np.asarray([[0.5, 0.3], [0.7, 0.25]])

garage_door_unk = -1
garage_door_closed = 0
garage_door_moving = 5
garage_door_open = 10

garage_car_unk = -1
garage_car_absent = 0
garage_car_moving = 5
garage_car_present = 10

history_depth = 120     # 1/2 second increments, depth of stack

# common helpers

def update_history(history, image_time, status_code):
    '''
    update the correct "slot" of the array with the status
    '''
    # remember - timestamp == time * 10 == 1/10s of seconds
    hist_timestamp = history[0]                                 # history array timestamp
    det_timestamp = image_time                                  # timestamp when image was grabbed
    index = int(((hist_timestamp - det_timestamp) + 1) / 10)    # add  1 because position 0 == timestamp
    
    log.info(f'GarageStatus.update history  history: {history.tolist()[:10]}  image_time:{image_time}  status:{status_code}  index: {index}')
    if index < (history_depth) and index > 0:
        history[index] = status_code
    return history

def is_gmark_area_valid(region_id, area):
    '''
    is the area reasonable?
    - could be bad inference
    - could be door in motion
    '''
    valid = False
    # area of gmark detection within min max for given region
    if area >= gmark_areas[region_id][0] and area <= gmark_areas[region_id][1]:
        valid = True
    return valid

def interpret_gmark(image_time, region_id, detection_center):
    '''
    INTERPRET a gmark (garage marker)
    - can be in two regions (full, left door)
      so the parameters are subscripted corresponding to the region_id

    detection sent here if:
    - valid area/size
    - valid region

    determine
    - which mark
    - what status
    '''
    # defaults == unknown
    gmark_id = -1  
    door_status = garage_door_unk
    car_status = garage_car_unk

    # is this a door closed mark?  -- most likely
    # - get the distance from true center
    # - get the angle from true center
    radius_from_gmark_door = radius(gmark_door_centers[region_id], detection_center)
    _, angle_from_gmark_door = angle(gmark_door_centers[region_id], detection_center)
    # is the detection within target of the garage door gmark?  (radius and angle)
    if radius_from_gmark_door < gmark_dist_limits[region_id]:
        within_dist = True
    # within the travel path
    if angle_from_gmark_door >= -100. and angle_from_gmark_door <= -80:
        travel_path = True


    if within_dist == True:
        gmark_id = 0
        door_status = garage_door_closed
    else:
        if travel_path == True:
            gmark_id = 0
            door_status = garage_door_moving

    log.info(f'GarageStatus.update door gmark -- {region_id}' 
        f' rad: {radius_from_gmark_door:02.2f}  ang: {angle_from_gmark_door:02.2f}'
        f' gmark# {gmark_id}  door: {door_status}   car: {car_status}')

    # if still unknown (not a door)
    # check for car gmark
    # -- untested
    if gmark_id == -1:
        radius_from_gmark_car = radius(gmark_car_centers[region_id], detection_center)
        _, angle_from_gmark_car = angle(gmark_car_centers[region_id], detection_center)
        if radius_from_gmark_car < gmark_dist_limits[region_id]:
            gmark_id = 1
            car_status = 'pres'
        elif angle_from_gmark_car >= -100. and angle_from_gmark_door <= -80:
            gmark_id = 1
            car_status = 'run'
        log.debug(f'GarageStatus.update car gmark -- {region_id}' 
            f' rad: {radius_from_gmark_car:02.2f}  ang: {angle_from_gmark_car:02.2f}'
            f' gmark# {gmark_id}  door: {door_status}   car: {car_status}')
        
    return gmark_id, door_status, car_status
    
    



def get_gmark_status(region_id, det):
    door_status = garage_door_unk
    car_present = 'unk'
    if det.region_id == cam_reg_garage_indoor_left_door:
        # get indexes = gmark
        gmark_array = np.where(det.model_inference.class_array)[0]      # results for single axis == 0
                                                                        # returns array like [2,3]
        for idx in gmark_array:
            center = det.model_inference.bbox_center_array[idx]
            area = det.model_inference.bbox_area_array[idx]
            area_valid = is_gmark_area_valid(region_id, area)
            gmark_id, door_status, car_status = interpret_gmark(det.image_time, region_id, center)
            # valid area
            # door axis
            # car axis
            log.info(f'GarageStatus.update gmark -- {det.image_time} {det.camera_id} {det.region_id} area valid: {area_valid}')
    return door_status, car_present

class GarageStatus:
    '''
    history == 121
       [0] == timestamp
       [1:121] == stack
    '''
    def __init__(self, door_status, car_present, person_present, light_status):
        self.door_status = door_status
        self.car_present = car_present
        self.person_present = person_present
        self.light_status = light_status
        # history
        self.door_status_history = np.full((history_depth), -1, dtype=int)
        self.door_status_history[0] = int(time.time() * 10)
        return
    def __str__(self):
        garage_status_string = f".garage_status -- {self.door_status}"
        return garage_status_string

    def update_from_detection(self, det):
        '''
        Update GarageStatus from a Detection (== det)
        '''
        
        log.info(f'GarageStatus.update -- {det.camera_id} {det.region_id} = classes: {det.model_inference.class_array}')
        log.info(f'GarageStatus.update -- {det.camera_id} {det.region_id} = centers: {det.model_inference.bbox_center_array}')
        log.info(f'GarageStatus.update -- {det.camera_id} {det.region_id} = areas:   {det.model_inference.bbox_area_array}')
        if det.region_id == cam_reg_garage_indoor_left_door:
            self.door_status, self.car_present = get_gmark_status(det.region_id, det)
            update_history(self.door_status_history, det.image_time, self.door_status)
            log.info(f'GarageStatus door history: {self.door_status_history.tolist()[:20]}')
        return