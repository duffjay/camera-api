#
# common functions related to detected 'person'
# in a detection
#
import logging
import settings
import numpy as np
import traceback

# gets moved to 
import status

log = logging.getLogger(__name__)
unknown = 0
person_class_id = 1
person_present = 1

def get_person_status(home_status, det):
    '''
    given full detection: (all detected objects)
                          1 camera : 1 region

    validate, update history stack

    This is a many to one -- there can be many persons - doesn't matter, 
    we're only grabbing the fact there was A person
    '''
    person_status = 0

    log.info(f'person.get_person_status - cam: {det.camera_id}-{det.region_id}')

    if det.model_inference is not None:
        # # default == unknown
        person_status = unknown

        # get indexes = person
        log.info(f'get_person_status  all classes detected = {det.model_inference.class_array.tolist()}')
        person_array = np.where(det.model_inference.class_array == person_class_id )[0]       # results for single axis == 0
                                                                                            # returns array like [2,3]
        log.info(f'person.get_person_status  person_array {person_array.tolist()}')
        # NOT processing multiple person objects
        # - we just care that there was at least one
        try:
            if person_array.shape[0] > 0:
                # All areas for a person will be valid - no reason to validate
                # - no reason to chack center, area
                # - there is no logic for person location - so no reason to interpret the size, location, travel path (compared to gmark)
                # area_valid = is_area_valid(region_id, gmark_areas[region_id], area)
                person_status = person_present
                # update_history
                history_row_num = status.get_history_np_row(settings.configured_history_map, det.camera_id, det.region_id, "person")
                home_status.update_history(det.image_time, history_row_num, person_status, comment=f'person-{det.camera_id}:{det.region_id}')
        except Exception as e:
            with settings.safe_print:
                log.error(f'person.get_person_status:  ERROR {e}')
                traceback.print_exc()
                log.info(f' person.get_person -- check configuration of history map')            

    return person_status