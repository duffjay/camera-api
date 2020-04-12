import os
import sys
import time
import cv2
import threading
import queue
import numpy as np

import camera_util

import settings


def image_producer(camera_id, camera_config, camera_snapshot_times):
    
    while True:
        camera_name, np_images = camera_util.get_camera_regions(camera_config)
        snapshot_elapsed =  time.perf_counter() - camera_snapshot_times[camera_id]      # elapsed time between snapshots
        camera_snapshot_times[camera_id] = time.perf_counter()                          # update the time == start time for the next snapshot
        # pushes to the stack if there was a frame captured
        if np_images is not None:
            image_time = int(time.time() * 10)  # multiply time x 10 to pick up 10ths
            settings.imageQueue.put((camera_id, camera_name, image_time, np_images))
            with settings.safe_print:
                print ("  IMAGE-PRODUCER:>>{} np_images: {}  {:02.2f} secs".format(camera_name, np_images.shape, snapshot_elapsed))
        else:
            with settings.safe_print:
                print ("  IMAGE-PRODUCER:--{} np_images: None".format(camera_name))
        # stop?        
        if settings.run_state == False:
            break
    return