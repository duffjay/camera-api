import time
import logging
import traceback
import numpy as np

import status
import settings

log = logging.getLogger(__name__)


def process_front_lights(home_status):

    log.info(f'PROCESS_FRONT_LIGHTS started')
    
    while True:
        new_objects = 0             # default to 0
        try:
            # Consumer tasks
            # - once/frame
            #    - ALL regions of the frame
            color_code_idx_start = status.status_meta_index["color_code_start"]
            color_code_idx_end = color_code_idx_start + 10
            color_array = home_status.history[0, color_code_idx_start:color_code_idx_end]
            unique, counts = np.unique(color_array, return_counts=True)         # get counts of 0's and 1's
            color_counts = dict(zip(unique, counts))                            # make a dict
            log.info(f'process_front_lights - color: {color_array} night vision: {color_counts[0]}')

            
        except Exception as e:
            with settings.safe_print:
                log.error(f'PROCESS_FRONT_LIGHTS: !!! ERROR - Exception ')
                log.error(f'  --  process front lights: exception: {e}')
                log.error(traceback.format_exc())
        
        time.sleep(10)
        
    
    return