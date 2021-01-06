import os
import sys
import time
import cv2
import threading
import logging
import queue
import numpy as np

import camera_util

import settings

log = logging.getLogger(__name__)

def compute_sleep_time(frame_elapsed_time, qsize, camera_id):
    # value 0 - 10
    #   0 == don't sleep, go as fast as you can
    #  10 == sleep if the queue is deep
    # 
    # example:
    #   camera_sleep_factor = 7
    #   imageQueue depth = 20
    #   unversal sleep factor = 0.01
    #   elapsed time on frame = 1.5
    # sleep_time = (20 * 0.01 * 10) - (1.5) = 0.5       
    sleep_factor = camera_util.get_camera_sleep_factor(settings.config, camera_id)
    sleep_time = (qsize * settings.universal_sleep_factor * sleep_factor) - frame_elapsed_time
    # can't be a negative value
    if sleep_time < 0.0:
        sleep_time = 0.0
    
    log.debug(f'compute_sleep_time - camera_id: {camera_id}  frame_elapsed: {frame_elapsed_time}  qsize: {qsize}  sleep_time: {sleep_time}')
    return sleep_time

def determine_push(push_of_n, push_history):
    '''
    determine if the image should pushed (processed) or skipped 

    return  true = push, updated history
    '''
    push = False        # default
    # expecting history length of 10
    if True in push_history[-push_of_n:]:
        push = False
    else:
        push = True
    
    # update the history
    push_history.append(push)
    push_history = push_history[1:11]

    return push, push_history

def image_producer(camera_id, camera_config, camera_snapshot_times):
    '''
    There is one (1) producer per camera
    '''
    # create a history - initially all False, to set up the ability to keep only a small portion
    push_history = [False, False, False, False, False, False, False, False, False, False]
    push_of_n = int(camera_config["push_of_n"])
    if settings.imageQueue.qsize() > 0:
        push_of_n = int(settings.imageQueue.qsize()/5) + 1


    stream = False
    if camera_config["stream"] == 1:
        stream = True

    # if Honeywell, open the video stream
    # if camera_config['mfr'] == 'Honeywell':
    video_stream = camera_util.open_video_stream(camera_id, camera_config, stream)

    while True:
        start_time = time.perf_counter()

        # Honeywell
        # if camera_config['mfr'] == 'Honeywell':
        frame = camera_util.get_camera_full(camera_id, video_stream)

        should_push, push_history = determine_push(push_of_n, push_history)
        with settings.safe_print:
            log.debug (f"  IMAGE-PRODUCER:>>{camera_id} {should_push} {push_history}")

        if should_push:
            camera_name, np_images, is_color = camera_util.get_camera_regions_from_full(frame, camera_id, camera_config, stream)

            # Reolink Snapshot
            # if camera_config['mfr'] == 'Reolink':
            #     camera_name, np_images, is_color = camera_util.get_camera_regions(camera_id, camera_config, stream)

            snapshot_elapsed =  time.perf_counter() - camera_snapshot_times[camera_id]      # elapsed time between snapshots
            camera_snapshot_times[camera_id] = time.perf_counter()                          # update the time == start time for the next snapshot
            # pushes to the stack if there was a frame captured
            if np_images is not None:
                image_time = int(time.time() * 10)  # multiply time x 10 to pick up 10ths
                settings.imageQueue.put((camera_id, camera_name, image_time, np_images, is_color))
                with settings.safe_print:
                    log_msg = "  IMAGE-PRODUCER:>>{} np_images: {}  {:02.2f} secs  stream: {}".format(camera_name, np_images.shape, snapshot_elapsed, stream)
                    log.info(log_msg)
            else:
                with settings.safe_print:
                    log.info(f"  IMAGE-PRODUCER:--{camera_name } np_images: None")


        # stop?        
        if settings.run_state == False:
            break
    return