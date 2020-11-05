import os
import sys
import time
import re
import cv2
import threading
import logging
from logging.handlers import TimedRotatingFileHandler


import numpy as np
import traceback

# add the tensorflow models project to the Python path
# github tensorflow/models
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)

# modules - part of the tensorflow models repo
from object_detection.utils.np_box_ops import iou

# models - part of this project
import gen_util
import label_map_util
import tensorflow_util
import camera_util
import display
import annotation
import rekognition_util

import image_producer
import image_consumer
import face_consumer
import stack_shift
import lights

import settings


def main():
    # args
    config_filename = sys.argv[1]   # 0 based

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # file handler
    logname = "security_app.log"
    handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
    handler.suffix = "%Y%m%d"
    handler.extMatch = re.compile(r"^\d{8}$")
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # init the variables in security settings
    # - init only in main()
    # https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    logger.debug(f'main - settings.init() - - setting global variables')
    settings.init(config_filename)

    # configure the model
    model_config = settings.config["model"]
    sess, tensor_dict, image_tensor, model_input_dim, label_map, label_dict = tensorflow_util.configure_tensorflow_model(model_config)

    # camera config
    # - includes getting the data structures to track detections
    camera_config_list, camera_count, camera_snapshot_times, bbox_stack_lists, bbox_push_lists = camera_util.config_camara_data(settings.config)
    logger.info("Camera Count: %d", camera_count)
    logger.info("Facial Detection Enabled: %s", settings.facial_detection_enabled)

    #   I M A G E    C O N S U M E R S
    #   == face producers
    # 
    # !!! Cannot display multi-threaded - must have just 1 display !!!
    #     Thus, consumer count must = 1
    #     watch the queue size as it runs
    consumer_count = 48
    for i in range(consumer_count):
        logger.debug(f'Starting Consumer: {i}')
        thread = threading.Thread(target=image_consumer.image_consumer, 
            args=(i, 
                sess, tensor_dict, image_tensor, bbox_stack_lists, bbox_push_lists, model_input_dim, label_dict))
        thread.daemon = True
        thread.start()

    #   I M A G E    P R O D U C E R S
    #    
    for camera_id, camera_config in enumerate(camera_config_list):
        thread = threading.Thread(target=image_producer.image_producer, 
            args=(camera_id, camera_config, camera_snapshot_times))
        thread.start()

    #   F A C E    C O N S U M E R S
    #   == run Rekognition
    # 
    if settings.facial_detection_enabled == True:
        consumer_count = 1
        for i in range(consumer_count):
            thread = threading.Thread(target=face_consumer.face_consumer, 
                args=(i,))
            thread.daemon = True
            thread.start()

    #   S T A C K    S H I F T
    #   - shift the history stack as function of time
    #   settings.home_status == the singleton, overall status object
    thread = threading.Thread(target=stack_shift.stack_shift, args=(settings.home_status,))
    thread.daemon = True
    thread.start()

    #   P R O C E S S    L I G H T S
    #   - front lights (front porch & sunroom)
    #   - back lights (back porch)
    thread = threading.Thread(target=lights.update_lights, args=(settings.home_status,))
    thread.daemon = True
    thread.start()

    logger.info("main() exit")



if __name__ == '__main__':
    main()
