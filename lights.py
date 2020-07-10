import time
import logging
import traceback
import numpy as np

import status
import hue_util

import settings

log = logging.getLogger(__name__)


# this gets called with each shift (1 time/second)

def determine_is_day(home_status):
    '''
    determine is_day == True
    based on number of cameras that are not color
    - remember 1st couple of cycle, all cameras are 0 until data comes in
    '''
    is_day = False

    # get range of camera color/grayscale (color == 1)
    color_code_idx_start = status.status_meta_index["color_code_start"]
    color_code_idx_end = color_code_idx_start + 10

    color_array = home_status.history[0, color_code_idx_start:color_code_idx_end]
    unique, counts = np.unique(color_array, return_counts=True)         # get counts of 0's and 1's
    color_counts = dict(zip(unique, counts))                            # make a dict

    if color_counts[0] < 6:
        is_day = True

    log.info(f'lights.determine_is_day - color: {color_array} night vision: {color_counts[0]} is day: {is_day}')

    return is_day

def get_group_status(group_id):
    '''
     0 == off
     + == brightness
    '''
    group = settings.light_groups[group_id]
    if group.data['action']['on'] == True:
        group_status = group.data['action']['bri']
    else:
        group_status = 0
    log.info(f'lights.get_group_status {group_id} = {group_status}')
    return group_status


def update_group_status():
    # update the data from bridge
    settings.light_groups = hue_util.get_groups(hue_util.get_bridge(settings.hue_bridge_ip, settings.hue_bridge_username))

    # hardcoded - it's short and simple
    settings.light_group_status["front_porch"] = get_group_status(settings.front_porch_group_id)
    settings.light_group_status["bar"] = get_group_status(settings.bar_group_id)
    settings.light_group_status["sunroom"] = get_group_status(settings.sunroom_group_id)

    return 

def process_front_lights(home_status):
    '''
    if NOT is_day
        if a person is present (home_status)
        turn the lights up
    '''
    log.info(f'PROCESS_FRONT_LIGHTS started')
    

        
    
    return

def update_lights(home_status):
    is_day = determine_is_day(home_status)
    update_group_status()
    log.info(f'lights.update_lights - start: {settings.light_group_status}')
    # update front lights
    front_light_status = process_front_lights(home_status)

    log.info(f'lights.update_lights -   end: {settings.light_group_status}')
    return