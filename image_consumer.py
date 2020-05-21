import os
import sys
import time
import cv2
import threading
import queue
import logging
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

import inference
import status

import settings

log = logging.getLogger(__name__)

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

# TODO
# change this to RegionDetection object
def is_save_inference(rule_num, camera_id, region_id, num_detections, new_objects):
    save_inference = False
    # rule 1
    if rule_num == 1 and num_detections > 0:
        if settings.save_inference and new_objects > 0:
            save_inference = True
    
    # rule 2
    # garage door - left door region -or- full image
    # save all
    if rule_num == 2 and camera_id == 2:
        if region_id == 1 or region_id == 0:
            save_inference = True

    log.debug(f'image_consumer/is_save_inference -- rule# {rule_num}  {camera_id}  {region_id}  {num_detections}  {new_objects} ==> {save_inference}')
    return save_inference


# Pull images off the imageQueue
# - produce faceQueue (check the faces in the images)
def image_consumer(consumer_id, 
        sess, tensor_dict, image_tensor, bbox_stack_lists, bbox_push_lists, model_input_dim, label_dict):

    log.info(f'IMAGE-CONSUMER started #{consumer_id}')
        # configuration
    facial_detection_regions = convert_facial_lists(settings.config["facial_detection_regions"]) # list of lists converted to list of tuple-2  (camera_id, regions_id)

    while True:
        new_objects = 0             # default to 0
        try:
            # Consumer tasks
            # - once/frame
            #    - ALL regions of the frame
            data = settings.imageQueue.get(block=False)
            camera_id, camera_name, image_time, np_images, is_color = data
            pushed_to_face_queue = False
            start = time.perf_counter()
            # loop through the regions in the frame
            for region_id, np_image in enumerate(np_images):
                orig_image = np_image.copy()     # np_image will become inference image - it's NOT immutable
                np_image_expanded = np.expand_dims(np_image, axis=0)
                # -- run model
                inf = tensorflow_util.send_image_to_tf_sess(np_image_expanded, sess, tensor_dict, image_tensor)
                # get data for relavant detections
                num_detections = inf.prob_array.shape[0]
              
                # check for new detections
                # - per camera - per region 
                if num_detections > 0:
                    # need bbox_array as a numpy
                    log.debug(f'image_consumer#{consumer_id} - cam#{camera_id}  reg#{region_id}  bbox_stack_list len: {len(bbox_stack_lists)}')
                    log.debug(f'-- bbox_stack_lists[{camera_id}] => bbox_stack_list')
                    log.debug(f'-- bbox_stack_list: {bbox_stack_lists[camera_id]}')
                    with settings.safe_stack_update:
                        new_objects, dup_objects = tensorflow_util.identify_new_detections(image_time,
                            settings.iou_threshold, camera_id, region_id, inf.bbox_array, bbox_stack_lists[camera_id], bbox_push_lists[camera_id])
                    # create a detection & update home_status
                    det = inference.RegionDetection(image_time, camera_id, region_id, new_objects, dup_objects, inf)
                    with settings.safe_status_update:
                        settings.home_status.update_from_detection(det)
                
                # display - 
                # NOTE - Displaying w/ cv2 in multi-threads is a problem!!   1 consumer only if you want to enable this
                # window_name = '{}-{}'.format(camera_name, region_id)
                if num_detections > 0:
                    image = np_image    
                    # TODO - get rid of the threshold
                    probability_threshold = 0.7
                    inference_image, orig_image_dim, detected_objects = display.inference_to_image( 
                        image,
                        inf, 
                        model_input_dim, label_dict, probability_threshold)
                    
                # Facial Detection
                if settings.facial_detection_enabled == True:
                    submit_face_queue = check_faces(camera_id, region_id, facial_detection_regions, inf.class_array)  # is this region in the regions to check?
                    if submit_face_queue == True:
                        settings.faceQueue.put((camera_id, region_id, image_time, np_image))
                        pushed_to_face_queue = True

                # S A V E
                # - set up a couple of rules for saving
                  # default
                rule_num = 1
                save_inference = is_save_inference(rule_num, camera_id, region_id, num_detections, new_objects)

                saved = False
                if save_inference == True:
                    # base name == sssssss-camera_id-region_id-c/g
                    color_code = 'c'            # g == grayscale, c == color
                    if is_color == 0:
                        color_code = 'g'                                
                    base_name = '{}-{}-{}-{}'.format(image_time, camera_id, region_id, color_code)   
                    image_name = os.path.join(settings.image_path,  base_name + '.jpg')
                    annotation_name = os.path.join(settings.annotation_path,  base_name + '.xml')
                    log.debug(f"image_consumer -- saving: {image_name} {image.shape} {annotation_name}")
                    # original image - h: 480  w: 640
                    saved = True
                    cv2.imwrite(image_name, orig_image)
                    # this function generates & saves the XML annotation
                    annotation_xml = annotation.inference_to_xml(settings.image_path, image_name,orig_image_dim, detected_objects, settings.annotation_path )
            
                with settings.safe_print:
                    log.info(
                        f'  IMAGE-CONSUMER:<< {consumer_id} qsize: {settings.imageQueue.qsize()}'
                        f'  cam: {camera_name} reg: {region_id} timestamp {image_time}'
                        f'  inftime:{(time.perf_counter() - start):02.2f} sec  dets: {num_detections} new: {new_objects}  saved: {saved}'
                    )
                    if num_detections > 0:
                        log.debug(f'image_consumer - detection: {det}')
                    if pushed_to_face_queue == True:
                        log.info('      pushed to faceQueue')
                    if saved == True:
                        log.debug(f"      Saved: stale objects - image_name: {image_name}")
                    else:
                        log.debug("      No new objects detected --- not saved")




        except queue.Empty:
            pass

        except Exception as e:
            with settings.safe_print:
                log.error(f'  IMAGE-CONSUMER:!!! ERROR - Exception  Consumer ID: {consumer_id}')
                log.error(f'  --  image consumer exception: {e}')
                log.error(traceback.format_exc())
        

    # stop?
    if settings.run_state == False:
        log.info(f'******* image consummer {consumer_id} shutdown *******')
        
    
    return
