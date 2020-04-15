import os
import sys
import time
import cv2
import threading

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

import settings


def main():
    # args
    config_filename = sys.argv[1]   # 0 based

    # init the variables in security settings
    # - init only in main()
    # https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    settings.init(config_filename)

    # configure the model
    model_config = settings.config["model"]
    sess, tensor_dict, image_tensor, model_input_dim, label_map, label_dict = tensorflow_util.configure_tensorflow_model(model_config)

    # camera config
    # - includes getting the data structures to track detections
    camera_config_list, camera_count, camera_snapshot_times, bbox_stack_lists, bbox_push_lists = camera_util.config_camara_data(settings.config)
    print ("Camera Count:", camera_count)
    print ("Facial Detection Enabled", settings.facial_detection_enabled)

    #   I M A G E    C O N S U M E R S
    #   == face producers
    # 
    # !!! Cannot display multi-threaded - must have just 1 display !!!
    #     Thus, consumer count must = 1
    #     watch the queue size as it runs
    consumer_count = 4  
    for i in range(consumer_count):
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


    # time.sleep(240)
    # print ("main() sleep timed out")
    # run_state = False

    cv2.destroyAllWindows()
    print ("main() exit")



        # camera_name, np_images = camera_util.get_camera_regions(camera_config)
        # if np_images is not None:
        #     print ("np_images:", np_images.shape)
        #     for i, image in enumerate(np_images):
        #         print ("image {}  shape {}".format(i, image.shape))

        #         window_name = "{}-{}".format(camera_name, i)
        #         cv2.imshow(window_name,image)
        #     cv2.waitKey(0)
        # else:
        #     print ("nothing returned")

if __name__ == '__main__':
    main()
