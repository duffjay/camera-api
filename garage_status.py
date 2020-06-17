import logging
import numpy as np
import time

# my modules
import settings

from inference import RegionDetection
from inference import ModelInference
from inference import radius, angle
from inference import is_area_valid, is_on_target

import status
from person import get_person_status

# If new camera
# - you'll have to adjust centers
# - maybe?  area

log = logging.getLogger(__name__)

# important assumptions
unknown = 0

# TODO -- this needs to be moved
history_depth = 120     # 1/2 second increments, depth of stack

cam_garage_outdoor = 0  # ~ 1 sec camera
cam_reg_garage_outdoor_full = 0
cam_reg_garage_outdoor_driveway = 1
cam_reg_garage_outdoor_backdoor = 2
cam_reg_garage_outdoor_parking_pad = 3
cam_reg_garage_outdoor_close = 4
cam_reg_garage_outdoor_backyard = 5

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
gmark_areas = [[0.006, 0.02], [0.027, 0.045]]               # area gets bigger while traveling up
gmark_dist_limits = [0.10, 0.2]

# region specific
# regions = (y, x) - normalized
# get the centers from the log - look for:
#   GarageStatus.update - (cam) 2  (reg) 0,1 = centers: [[y,x]]
gmark_door = 0
gmark_door_centers = np.asarray([[0.18, 0.23], [0.12, 0.52]])
gmark_car = 1
gmark_car_centers = np.asarray([[0.505, 0.178], [0.842, 0.416]])

# gmarks - must be in correct location
#        - 0 == not detected in position
#        - 1 == detected in position
garage_door_closed = 1
garage_door_open = 0

garage_car_absent = 0
garage_car_present = 1

# car present IN the garage
# only 1 region, but use same data structure
#    index in the list == region_id
car_class_id = 3                        # get this from the label protobuf in /model
car_inside_areas = [[0.1, 0.41]]        # min/max;  area gets smaller FAST as you pull out
                                        # 2' = 0.38,  4' = 0.26,  6' = 0.17  8' = 0.16  10' = 0.11
car_inside_centers = [[0.69, 0.317]]
car_inside_dist_limits = [[0.05]]       # angles are hard coded below
car_inside_absent = 0   # not used - we can be sure, maybe just not detected
car_inside_present = 1


    
def is_gmark_door(region_id, detection_center):
    '''
    you can verify:
    - this is a garage door mark == door is closed
    - but the logic is, gmark_door detected in position 


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
        gmark_dist_limits[region_id], 75, 100)

    # on target
    if within_radius == True:
        gmark_id = gmark_door
        door_status = garage_door_closed

    return gmark_id, door_status

def is_gmark_car(region_id, detection_center):
    '''
    you can verify:
    - if this is a car gmark - in position

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
        gmark_dist_limits[region_id], 75, 100)

    # on target
    if within_radius == True:
        gmark_id = gmark_car
        car_status = garage_car_absent

    return gmark_id, car_status


def get_gmark_status(home_status, region_id, det):
    '''
    This is ONLY called if camera = 2 (garage indoor)
     AND only called for regions 0 & 1

     called per region (a detection can have many objects)

    given full detection (all detected objects)
    car_status == car MARK status   (not the actual car detection)
    '''

    # default == unknown
    gmark_id = unknown
    door_status = unknown
    car_status = unknown

    # get indexes = gmark
    log.debug(f'GarageStatus.get_gmark_status - all detected classes = {det.model_inference.class_array.tolist()}')
    gmark_array = np.where(det.model_inference.class_array == gmark_class_id )[0]       # results for single axis == 0
                                                                                        # returns array like [2,3]
    log.debug(f'GarageStatus.get_gmark_status.gmark_array {gmark_array.tolist()}')
    # iterate through the indexes -> pointing to the detections == gmark
    for idx in gmark_array:
        center = det.model_inference.bbox_center_array[idx]
        area = det.model_inference.bbox_area_array[idx]
        area_valid = is_area_valid(region_id, gmark_areas[region_id], area)
        # only if area is value -- not worth checking otherwise
        # - with valid area, where it is determines door & car status
        if area_valid == True:
            gmark_id, door_status = is_gmark_door(region_id, center)
            if gmark_id == unknown:
                gmark_id, car_status = is_gmark_car(region_id, center)

            log.info(f'GarageStatus.interpret_gmark_position -- {region_id}' 
                    f' gmark# {gmark_id}  door: {door_status}   car: {car_status}')

        log.info(f'GarageStatus.update gmark: idx = {idx} -- {det.image_time} {det.camera_id} {det.region_id} area valid: {area_valid}'
            f' gmark_id: {gmark_id} door_status: {door_status}  car_status: {car_status}')
        # update history
        # - update based on region
        # - only update if you have a known status
        #   - e.g.  an unknown (e.g. car) while evaluating a door mark  would possibly overwrite a known door mark status

        #  this 'if' is superfulous, it's not necessary - you already know this is camera 2 only and it you already filtered down
        #                             to gmarks only;  it's here for safety later
        if (det.camera_id == cam_garage_indoor) and (det.region_id in (cam_reg_garage_indoor_full, cam_reg_garage_indoor_left_door)):
            # if this detected object (in the camera/region) was a gmark_door
            if door_status >= 0:
                history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "gmark_door")
                home_status.update_history(det.image_time, history_row_num, door_status, comment=f'gmark_door-{det.camera_id}:{det.region_id}')
            # if this detected object (in the camera/region) was a gmark_car
            if car_status >= 0:
                history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "gmark_car")
                home_status.update_history(det.image_time, history_row_num, car_status, comment=f'gmark_car-{det.camera_id}:{det.region_id}')
            
    return door_status, car_status

# - - - - - - - - - - - - - car present in garage - - - - - - - - - - - - - - 
# camera 2 (garage indoor) car history
# - technically, this camera can see cars OUTSIDE of the garage when the door is open
# - so, car must be in position
def get_car_inside_status(home_status, region_id, det):
    '''
    don't confuse this car_status with the car_gmark status
    - car gmark - if you can see it, tells you definitely that a car is NOT present
                - if you don't see it, PROBABLY a car is blocking it but you don't know for sure
    - car status - this means you detect a car in the parked position
                 - or pulling out

    Only checking the full resultion (region 0) region

    you could have a car in view (through an open door)

    '''
    car_status = unknown

    # get indexes = gmark
    log.debug(f'GarageStatus.get_car_insidestatus - all detected classes = {det.model_inference.class_array.tolist()}')
    car_array = np.where(det.model_inference.class_array == car_class_id )[0]           # results for single axis == 0
                                                                                        # returns array like [2,3]
    
    # should only have 1 - you could possibly have a shadow/duplicate detection
    for idx in car_array:
        car_status = unknown                                    # reset between iteration
        within_radius = unknown

        center = det.model_inference.bbox_center_array[idx]
        area = det.model_inference.bbox_area_array[idx]
        area_valid = is_area_valid(region_id, car_inside_areas[region_id], area)
        # only if area is value
        if area_valid == True:
            # is this a car in the parked position? 
            # - get the distance from true center
            # - get the angle from true center
            within_radius, on_travel_path = is_on_target(car_inside_centers[region_id], center, 
                car_inside_dist_limits[region_id], 50, 82)

            # on target
            if within_radius == True:
                car_status = car_inside_present


        log.info(f'GarageStatus.update car_inside: idx = {idx} -- {det.image_time} {det.camera_id} {det.region_id} center: {center}  area: {area}')
        log.info(f'GarageStatus.update car_inside: idx = {idx} -- {det.image_time} area_valid: {area_valid}  within radius: {within_radius}'
            f'  on path: {on_travel_path}  car inside status: {car_status}')

        if car_status >= 0:
                history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "car")
                home_status.update_history(det.image_time, history_row_num, car_status, comment=f'car-{det.camera_id}:{det.region_id}')

    return car_status



def update_garage_status(home_status, det):
    '''
    Update GarageStatus from a Detection (== det)
    '''
    
    log.info(f'garage_status/update_garage_status -- {det.camera_id} {det.region_id} = classes: {det.model_inference.class_array.tolist()}')
    log.info(f'garage_status/update_garage_status -- {det.camera_id} {det.region_id} = centers: {det.model_inference.bbox_center_array.tolist()}')
    log.info(f'garage_status/update_garage_status -- {det.camera_id} {det.region_id} = areas:   {det.model_inference.bbox_area_array.tolist()}')

    # camera = indoor 
    if det.camera_id == cam_garage_indoor:
        # regions w/ gmarks
        #  - region 0 == full res
        #  - region 1 == left door
        # for gmarks, strict position validation
        if det.region_id == cam_reg_garage_indoor_left_door or \
            det.region_id == cam_reg_garage_indoor_full:
            log.info(f'garage_status/update_garage_status -- check gmarks')
            get_gmark_status(home_status, det.region_id, det)
    
        # regions w/ (actual car) - region 0 - full only
        if det.region_id == cam_reg_garage_indoor_full:
            log.info(f'garage_status/update_garage_status -- check car present indoor -- (full) region 0 only')
            get_car_inside_status(home_status, det.region_id, det)

    # regions w/ people
    # all cameras, all regions
    get_person_status(home_status, det)

    return        
