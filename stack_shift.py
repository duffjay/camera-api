
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
    stack structure:
        row 0 : column 0 == timestamp
        row 1 : 0 : history_depth = the status stack
    '''
    # meta row == data NOT to shift
    meta_row = stack[0:1, :]
    history_rows = stack[1:]

    # num == how much to shift
    timestamp = int(time.time() * 10)
    base_time = meta_row[0, 0]
    shift_increment = int((timestamp - base_time) / 10)

    fill_value = -1

    # shift  
    #   - NOT handling negative shift (shift to left)
    #     not applicable and not testing it 
    result = np.empty_like(history_rows)                # same as history rows - but empty
    if shift_increment > 0:
        log.info(f'time_shift -- shift - time: {timestamp}  base: {base_time}  increment:  {shift_increment}')
        # fill all rows (:) but only the first columns (up to shift increment) with the fill value
        result[:, 0:shift_increment] = fill_value          # fill the new part, start at pos 1
        # paste positon = [0] + increment 
        result[:, shift_increment:] = history_rows[:, 0:-shift_increment]  # copy original in
    else:
        log.info(f'time_shift -- shift - time: {timestamp}  base: {base_time}  increment:  {shift_increment} NO SHIFT')
        result[:] = history_rows

    # recombine meta + history
    meta_row[0,0] = time.time() * 10                      # update time in the meta row
    shifted_stack = np.concatenate((meta_row, result), axis=0)

    return shifted_stack

def stack_shift(home_status):
    while True:
        start = time.time()

        # with the lock
        with settings.safe_status_update:
            home_status.history = shift(home_status.history)
            home_status.log_history(list(range(75)))

        elapsed = time.time() - start
        if elapsed < 1.0:
            sleep_time = 1.0 - elapsed
        else:
            sleep_time = 0.0

        log.info(f'time_shift -- sleep:  {time.time()}  {start}  elapsed: {elapsed} sleep: {sleep_time}')
        time.sleep(sleep_time)

        # stop?
        if settings.run_state == False:
            log.debug(f'******* stack_shift thread shutdown *******')
            break