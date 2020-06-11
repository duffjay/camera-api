#
# common functions related to detected 2-whlrs (bikes & motorcycles)
# in a detection
#
# this is kinda brute force, if there is a bicycle OR motorcycle, that's it
# multiple two wheelers == 2whlr present -- that's all
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
bicycle_class_id = 2
motorcycle_class_id = 4
two_whlr_present = 1

def get_2whlr_status(home_status, det):
    '''
    given full detection: (all detected objects)
                          1 camera : 1 region

    '''
    two_whlr_status = 0

    log.info(f'get_2whlr_status - cam: {det.camera_id}-{det.region_id}')
    # # default == unknown
    two_whlr_status = unknown

    # get indexes = person
    log.info(f'get_2whlr_status  all classes detected = {det.model_inference.class_array.tolist()}')
    two_whlr_array = np.where(np.logical_or(det.model_inference.class_array == bicycle_class_id,
        det.model_inference.class_array == motorcycle_class_id))[0]                          # results for single axis == 0
                                                                                            # returns array like [2,3]
    log.info(f'get_2whlr_status  2whlr_array {two_whlr_array.tolist()}')
    # NOT processing multiple 2 whlr objects
    # - we just care that there was at least one
    if two_whlr_array.shape[0] > 0:
        # All areas for a bicycle/motorcycle will be valid - no reason to validate
        # - no reason to check center, area
        # - there is no logic for object location - so no reason to interpret the size, location, travel path (compared to gmark)
        two_whlr_status = two_whlr_present
        # update_history
        history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "2whlr")
        home_status.update_history(det.image_time, history_row_num, two_whlr_status, comment=f'2whlr-{det.camera_id}:{det.region_id}')

    return two_whlr_status, two_whlr_array.shape[0]