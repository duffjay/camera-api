import os
import sys
import time
import cv2
import threading
import queue
import numpy as np


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

import settings

#
#  convert the facial detection list of lists 
#  to: list of tuple-2
def convert_facial_lists(regions_lists):
    regions_tuples = []
    for regions_list in regions_lists:
        regions_tuples.append((regions_list[0], regions_list[1]))
    return regions_tuples

# return  True -- if the camera-region is in the facial detection list
#                 -- i.e.  regions to check faces
#              -- AND - there was a person detected
def check_faces(camera_id, region_id, region_list, class_array):
    # 1st - is this region in the regions to check list?
    match = False
    submit_to_queue = False
    for region_tuple in region_list:
        if region_tuple == (camera_id, region_id):
            match = True
            break
    # 2nd - was there a person detected
    if match == True:
        for clss in class_array:
            if clss == 1:
                submit_to_queue = True
                return submit_to_queue
    # return == True if it it in the regions to check & there was a person detected
    return submit_to_queue

# Pull images off the imageQueue
# - produce faceQueue (check the faces in the images)
def image_consumer(consumer_id, 
        sess, tensor_dict, image_tensor, bbox_stack_lists, bbox_push_lists, model_input_dim, label_dict):

        # configuration
    facial_detection_regions = convert_facial_lists(settings.config["facial_detection_regions"]) # list of lists converted to list of tuple-2  (camera_id, regions_id)

    while True:
        try:
            # Consumer tasks
            # - once/frame
            #    - ALL regions of the frame
            data = settings.imageQueue.get(block=False)
            camera_id, camera_name, image_time, np_images = data
            pushed_to_face_queue = False
            start = time.perf_counter()
            
            # loop through the regions in the frame
            for region_id, np_image in enumerate(np_images):
                orig_image = np_image.copy()     # np_image will become inference image - it's NOT immutable
                np_image_expanded = np.expand_dims(np_image, axis=0)
                # -- run model
                prob_array, class_array, bbox_array = tensorflow_util.send_image_to_tf_sess(np_image_expanded, sess, tensor_dict, image_tensor)
                # get data for relavant detections
                num_detections = prob_array.shape[0]
              
                # check for new detections
                # - per camera - per region
                iou_threshold = 0.9    # IOU > 0.90
                if num_detections > 0:
                    # need bbox_array as a numpy
                    new_objects, dup_objects = tensorflow_util.identify_new_detections(
                        iou_threshold, camera_id, region_id, bbox_array, bbox_stack_lists[camera_id], bbox_push_lists[camera_id])
                
                # display
                window_name = '{}-{}'.format(camera_name, region_id)
                if num_detections > 0:
                    image = np_image    
                    # TODO - get rid of the threshold
                    probability_threshold = 0.7
                    inference_image, orig_image_dim, detected_objects = display.inference_to_image( 
                        image,
                        bbox_array, class_array, prob_array, 
                        model_input_dim, label_dict, probability_threshold)
                    cv2.imshow(window_name, inference_image)
                else:
                    cv2.imshow(window_name, np_image)

                # Facial Detection
                if settings.facial_detection_enabled == True:
                    submit_face_queue = check_faces(camera_id, region_id, facial_detection_regions, class_array)  # is this region in the regions to check?
                    if submit_face_queue == True:
                        settings.faceQueue.put((camera_id, region_id, image_time, np_image))
                        pushed_to_face_queue = True

                # S A V E
                # - if there were detections
                #     - and they was at lea# cascade classifier
                saved = False
                if num_detections > 0:
                    if settings.save_inference and new_objects > 0:
                        base_name = '{}-{}-{}'.format(image_time, camera_id, region_id)   
                        image_name = os.path.join(settings.image_path,  base_name + '.jpg')
                        annotation_name = os.path.join(settings.annotation_path,  base_name + '.xml')
                        # print ("saving:", image_name, image.shape, annotation_name)
                        # original image - h: 480  w: 640
                        saved = True
                        cv2.imwrite(image_name, orig_image)
                        # this function generates & saves the XML annotation
                        annotation_xml = annotation.inference_to_xml(settings.image_path, image_name,orig_image_dim, detected_objects, settings.annotation_path )
                
                with settings.safe_print:
                    print ('  IMAGE-CONSUMER:<< image queue size: {}, camera name: {} region: {} image_timestamp {}  inference time:{:02.2f} sec  detections: {}'.format(
                        settings.imageQueue.qsize(), camera_name, region_id, image_time, (time.perf_counter() - start), num_detections))
                    for detection_id in range(num_detections):
                        print ('      detection {}  classes detected{} new: {}  repeated: {}'.format(detection_id, class_array, new_objects, dup_objects))
                        print ('           Scores: {}   Classes: {}'.format(prob_array, class_array))
                        for bbox in bbox_array:
                            print ('           bbox:', bbox)
                    if pushed_to_face_queue == True:
                        print ('      pushed to faceQueue')
                    if saved == True:
                        print ("      Saved: stale objects: {}  new objects: {}   image_name: {}".format( dup_objects, new_objects, image_name))
                    else:
                        print ("      No new objects detected --- not saved")


            # stop all on a 'q' in a display window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                settings.run_state = False
                print (" *********** SHUTDOWN REQUESTED ***********")
                break

        except queue.Empty:
            pass

        except Exception as e:
            with settings.safe_print:
                print ('  IMAGE-CONSUMER:!!! ERROR - Exception  Consumer ID: {}'.format(consumer_id))

    # stop?
    if settings.run_state == False:
        print (" ******* image consummer {} shutdown *******".format(consumer_id))
        
    time.sleep(0.1)
    return
