import logging
import numpy as np


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

def interpret_gmark(region_id, detection_center):
    '''
    detection sent here if:
    - valid area/size
    - valid region

    determine
    - which mark
    - what status
    '''
    # defaults == unknown
    gmark_id = -1  
    door_status = 'unk'
    car_status = 'unk'
    # is this a door closed mark?  -- most likely
    radius_from_gmark_door = radius(gmark_door_centers[region_id], detection_center)
    _, angle_from_gmark_door = angle(gmark_door_centers[region_id], detection_center)
    if radius_from_gmark_door < gmark_dist_limits[region_id]:
        gmark_id = 0
        door_status = 'cls'
    elif angle_from_gmark_door >= -100. and angle_from_gmark_door <= -80:
        gmark_id = 0
        door_status = 'run'
    log.debug(f'GarageStatus.update door gmark -- {region_id}' 
        f' rad: {radius_from_gmark_door:02.2f}  ang: {angle_from_gmark_door:02.2f}'
        f' gmark# {gmark_id}  door: {door_status}   car: {car_status}')

    # if still unknown (not a door)
    # check for car gmark
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
    door_status = 'unk'
    car_present = 'unk'
    if det.region_id == cam_reg_garage_indoor_left_door:
        # get indexes = gmark
        gmark_array = np.where(det.model_inference.class_array)[0]      # results for single axis == 0
                                                                        # returns array like [2,3]
        for idx in gmark_array:
            center = det.model_inference.bbox_center_array[idx]
            area = det.model_inference.bbox_area_array[idx]
            area_valid = is_gmark_area_valid(region_id, area)
            gmark_id = interpret_gmark(region_id, center)
            # valid area
            # door axis
            # car axis
            log.debug(f'GarageStatus.update gmark -- {det.camera_id} {det.region_id} area valid: {area_valid}')
    return door_status, car_present

class GarageStatus:
    def __init__(self, door_status, car_present, person_present, light_status):
        self.door_status = door_status
        self.car_present = car_present
        self.person_present = person_present
        self.light_status = light_status
        return
    def __str__(self):
        garage_status_string = f".garage_status -- {self.door_status}"
        return garage_status_string

    def update_from_detection(self, det):
        '''
        Update GarageStatus from a Detection (== det)
        '''
        
        log.debug(f'GarageStatus.update -- {det.camera_id} {det.region_id} = classes: {det.model_inference.class_array}')
        log.debug(f'GarageStatus.update -- {det.camera_id} {det.region_id} = centers: {det.model_inference.bbox_center_array}')
        log.debug(f'GarageStatus.update -- {det.camera_id} {det.region_id} = areas:   {det.model_inference.bbox_area_array}')
        if det.region_id == cam_reg_garage_indoor_left_door:
            self.door_status, self.car_present = get_gmark_status(det.region_id, det)

        return