import os
import sys
import time
import threading
import queue
import numpy as np

import gen_util
import aws_util

# for tracking status
import status

cwd = os.getcwd()

# Globals
#    to be used across files

def init(config_filename):

    #
    # T O P    L E V E L   
    #    app config 
    # 
    global config
    config = gen_util.read_app_config(config_filename)

    global run_inferences, save_inference, annotation_dir, snapshot_dir
    run_inferences = config["run_inferences"]
    save_inference = config["save_inference"]
    annotation_dir = config["annotation_dir"]
    snapshot_dir = config["snapshot_dir"]
    status_dir = config["status_dir"]

    # global image_path
    global image_path, annotation_path, status_path
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))
    status_path = os.path.abspath(os.path.join(cwd, status_dir))

    # Queues
    global imageQueue, faceQueue
    imageQueue = queue.Queue()
    faceQueue = queue.Queue()

    global run_state
    run_state = True

    global safe_print, safe_imshow
    safe_print = threading.Lock()
    safe_imshow = threading.Lock()

    global safe_stack_update
    safe_stack_update = threading.Lock()

    # AWS
    global aws_session, aws_profile
    aws_profile = config["aws_profile"]
    aws_session = aws_util.get_session()

    global aws_s3_public_image
    aws_s3_public_image = config["aws_s3_public_image"]

    # faces
    global facial_detection_enabled
    global last_recognized_face_id
    global last_recognized_face_time

    if config["facial_detection_enabled"] == "True":
        facial_detection_enabled = True 
    last_recognized_face_id = 0
    last_recognized_face_time = 0.0

    # new object IoU Threshold
    global iou_threshold
    iou_threshold = 0.8

    # universal sleep factor
    # - base multiplier to make the cameras sleep
    #   increase this if the imageQueue gets too big
    global universal_sleep_factor
    universal_sleep_factor = 0.01

    # image auto correct
    # matrix
    # - rows = camera
    # - columns = regions
    # 8 regions max
    # 0.0 == don't autocorrect
    global color_image_auto_correct_clips_array
    color_clip_list = config["color_image_auto_correct_clips"]
    color_image_auto_correct_clips_array = np.asarray(color_clip_list)

    global gray_image_auto_correct_clips_array
    gray_clip_list = config["gray_image_auto_correct_clips"]
    gray_image_auto_correct_clips_array = np.asarray(gray_clip_list)

   
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # S T A T U S    T R A C K I N G
    #

    global safe_status_update
    safe_status_update = threading.Lock()
    
    global configured_history_map, history_row_count, row_num_dict
    configured_history_map, history_row_count, row_num_dict = status.configure_history_map(status.status_history_dict)

    global home_status
    home_status = status.Status(time.time())

    # - - - - - - - - 
    # H U E  
    #  light control

    global hue_bridge_ip, hue_bridge_username
    hue_bridge_ip = config['hue_bridge_ip']
    hue_bridge_username = config['hue_bridge_username']