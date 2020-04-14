import os
import sys
import threading
import queue

import gen_util
import aws_util

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

    # global image_path
    global image_path, annotation_path
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))

    # Queues
    global imageQueue, faceQueue
    imageQueue = queue.Queue()
    faceQueue = queue.Queue()


    print ("security_settings:  Setting Globals")
    global run_state
    run_state = True

    global safe_print
    safe_print = threading.Lock()

    # AWS
    global aws_session, aws_profile
    aws_profile = config["aws_profile"]
    aws_session = aws_util.get_session()

    # faces
    global facial_detection_enabled
    global last_recognized_face_id
    global last_recognized_face_time

    if config["facial_detection_enabled"] == "True":
        facial_detection_enabled = True 
    last_recognized_face_id = 0
    last_recognized_face_time = 0.0
