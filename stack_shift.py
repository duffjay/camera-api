
import time
import threading
import queue
import logging
import numpy as np
import traceback

# my stuff
import settings

log = logging.getLogger(__name__)

# preallocate empty array and assign slice by chrisaycock
def shift(stack):
    '''
    proven to be fastest way to shift (faster than np.roll)
    stack == original array
    don't forget timestamp in stack[0]
    '''
    # num == how much to shift
    timestamp = time.time() * 10
    base_time = stack[0]
    shift_increment = int((timestamp - base_time) / 10)
    log.info(f'time_shift -- shift - time: {timestamp}  base: {base_time}  increment:  {shift_increment}')

    fill_value = -1

    # shift  
    #   - NOT handling negative shift (shift to left)
    #     not applicable and not testing it 
    result = np.empty_like(stack)       # same as stack - but empty
    if shift_increment > 0:
        result[0] = timestamp                           # write time in slot 0
        result[1:shift_increment] = fill_value          # fill the new part, start at pos 1
        # paste positon = [0] + increment
        start_pos = shift_increment + 1                 # start shift (not forgetting timestamp in pos 0) 
        result[start_pos:] = stack[1:-shift_increment]  # copy original in
    else:
        result[:] = stack
        result[0] = timestamp
    return result


def stack_shift(home_status):
    while True:
        timestamp = time.time() * 10
        log.info(f'time_shift -- BEFORE garage door history: {home_status.garage_status.door_status_history.tolist()[:10]} ')



        # with the lock
        with settings.safe_status_update:
            home_status.garage_status.door_status_history = shift(home_status.garage_status.door_status_history)

        log.info(f'time_shift --  AFTER garage door history: {home_status.garage_status.door_status_history.tolist()[:10]} ')
        elapsed = (time.time() * 10) - timestamp
        if elapsed < 1.0:
            sleep_time = 1.0 - elapsed
        else:
            sleep_time = 0.0

        log.info(f'time_shift -- sleep:  {time.time()}  {timestamp}  elapsed: {elapsed} sleep: {sleep_time}')
        time.sleep(sleep_time)

        # stop?
        if settings.run_state == False:
            log.info(f'******* stack_shift thread shutdown *******')
            break