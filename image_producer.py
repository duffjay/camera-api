import os
import sys
import time
import cv2
import threading
import logging
import queue

import imutils
import cv2

import numpy as np

# -- my stuff ---
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


def determine_push(push_of_n, push_history, stream):
    '''
    determine if the image should pushed (processed) or skipped 
    push_of_n == 1 means every other value (1 means there must be a False)
    push_of_n == 0 means every value
    return  true = push, updated history
    '''

    if stream:
        push = True
    else:
        push = False        # default
        # expecting history length of 10
        if push_of_n == 0:
            push = True
        else:
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

    speed - the put time (in the log) must be < than the snapshot period or you'll get H264 decode errors.
    e.g.    put time = 0.35 < (2 fps = 0.5 sec period) 
            don't turn the cameras above 2 frames/sec
    '''
    # create a history - initially all False, to set up the ability to keep only a small portion
    push_history = [False, False, False, False, False, False, False, False, False, False]
    push_of_n = int(camera_config["push_of_n"])
    print (f'---- {camera_id} {camera_config} {push_of_n}')

    # setup consecutive image comparison
    prev_frame = np.zeros((settings.config["model_input_height"], settings.config["model_input_width"], 3))
    image_difference_threshold = float(camera_config["image_difference_threshold"])
    score = 0.0   # first time through, you need a value

    stream = False
    if camera_config["stream"] == 1:
        stream = True

    video_stream = camera_util.open_video_stream(camera_id, camera_config, stream)
    frame_none_count = 0

    # use the operational attribute to turn off a camera
    first_loop = True
    while camera_config["operational"] == 1:
        start_time = time.perf_counter()
        image_time = int(time.time() * 10)  # multiply time x 10 to pick up 10ths

        # get 'should push' based on previous images processed
        # - it's better to drop then here from the video stream
        if settings.imageQueue.qsize() > 0:
            adj_push_of_n = push_of_n + int(settings.imageQueue.qsize()/5) + 1
            log.info(f'  IMAGE-PRODUCER:>>{camera_id} push_of_n throttled:  {push_of_n} -> {adj_push_of_n}')
        else:
            adj_push_of_n = push_of_n

        should_push, push_history = determine_push(adj_push_of_n, push_history, stream)

        # get frame and compare
        frame = camera_util.get_camera_full(camera_id, video_stream)
        if frame is not None and should_push:
            frame_none_count = 0
            # process & resize the frame => np_images
            camera_name, np_images, is_color = camera_util.get_camera_regions_from_full(frame, camera_id, camera_config, stream)
            # append prev frame gray + np_images = old frame, new frame + regions
            #  - expand_dims to go from (h,w,c) => (1,h,w,c) so you can append
            if first_loop:
                prev_frame = np_images[0]
                first_loop = False

            np_images = np.append(np.expand_dims(prev_frame, axis=0), np_images,  axis = 0)

            snapshot_elapsed =  time.perf_counter() - camera_snapshot_times[camera_id]      # elapsed time between snapshots
            camera_snapshot_times[camera_id] = time.perf_counter()                          # update the time == start time for the next snapshot
            # pushes to the stack if there was a frame captured
            with settings.safe_print:
                log.info (f"  IMAGE-PRODUCER:>>{camera_id:02}v {image_time} -- stream: {stream} push_of_n: {push_of_n}:{adj_push_of_n} seq_should:{should_push} {push_history}")

            # push the images to the queue
            # - can't be None
            # - must be different -or- stream
            if np_images is not None:
                settings.imageQueue.put((camera_id, camera_name, image_time, np_images, is_color))

                # old = [0], new = [1]
                prev_frame = np_images[1]

                total_put_time = time.perf_counter() -start_time
                with settings.safe_print:
                    log.info(f"  IMAGE-PRODUCER:>>{camera_id:02}^ put time: {total_put_time:02.2f} np_images: {np_images.shape}  {snapshot_elapsed:02.2f} secs  stream: {stream}")
            else:
                with settings.safe_print:
                    log.info(f"  IMAGE-PRODUCER:>>{camera_id:02}^ --not pushed-- np_images: {type(np_images)} difference score: {score}")
        else:
            log.info(f'  IMAGE-PRODUCER:>>{camera_id:02}^ video_stream: {type(video_stream)} frame:  {type(frame)}  should push: {should_push} ')
            if frame is None:
                time.sleep(3)
                frame_none_count = frame_none_count + 1
                if frame_none_count > 5:
                    # try resetting the video stream
                    video_stream = None
                    log.info(f'  IMAGE-PRODUCER:>>{camera_id:02} frame_none_count: {frame_none_count} resetting vide_stream')
            else:
                # frame was good, but should_push == False
                frame_none_count = 0

            if video_stream is None:
                video_stream = camera_util.open_video_stream(camera_id, camera_config, stream)
                if video_stream is not None:
                    log.info(f"  IMAGE-PRODUCER:>>{camera_id:02} video stream re-acquired")

        # stop?        
        if settings.run_state == False:
            break
    return