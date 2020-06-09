#
# common functions related to detected 'car'
# in a detection
#
#  !!! DOES NOT INCLUDE:
#      - car parked in the garage, for that (see garage_status)
#        we check location
#
# this is kinda brute force, if there is a car, that's it
# multiple cars == car present
# - don't care about qty
# - don't care about moving
# - don't care about location

import logging
import settings
import numpy as np

# gets moved to 
import status

log = logging.getLogger(__name__)
unknown = 0
car_class_id = 3
car_present = 1

def get_car_status(home_status, det):
    '''
    given full detection: (all detected objects)
                          1 camera : 1 region

    '''
    car_status = 0

    log.info(f'get_car_status - cam: {det.camera_id}-{det.region_id}')
    # # default == unknown
    car_status = unknown

    # get indexes = person
    log.info(f'get_car_status  all classes detected = {det.model_inference.class_array.tolist()}')
    car_array = np.where(det.model_inference.class_array == car_class_id )[0]       # results for single axis == 0
                                                                                          # returns array like [2,3]
    log.info(f'get_car_status  car_array {car_array.tolist()}')
    # NOT processing multiple car objects
    # - we just care that there was at least one
    if car_array.shape[0] > 0:
        # All areas for a person will be valid - no reason to validate
        # - no reason to check center, area
        # - there is no logic for car location - so no reason to interpret the size, location, travel path (compared to gmark)
        # area_valid = is_area_valid(region_id, gmark_areas[region_id], area)
        car_status = car_present
        # update_history
        history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "car")
        home_status.update_history(det.image_time, history_row_num, car_status, comment=f'car-{det.camera_id}:{det.region_id}')

    return car_status, car_array.shape[0]