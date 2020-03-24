import os
import sys
import time
import cv2
import numpy as np
import imutils

import camera_util
import gen_util
import tensorflow_util
import label_map_util 
import display

import annotation

# add some paths that will pull in other software
# -- don't add to the path over and over
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)

from object_detection.utils.np_box_ops import iou



global PROBABILITY_THRESHOLD
PROBABILITY_THRESHOLD = 0.6    # only display objects with a 0.6+ probability


global bbox_iou_threshold
bbox_iou_threshold = 0.9
# global bbox_stack_list     # this will be a list of varied shape numpy arrays - to keep rolling bbox history
# global bbox_push_list      # list / camera of objects pushed to the stack with each cycle
global camera_name            # one camera per running program, this is the camera name

#
#  Each running instance of this program correpsonds to ONE camera
#
#  Setup for Reolink cameras only
#  - doesn't use cv2.VideoCapture
#  - uses simple http post

#  TODO
#  
#  parameters
#  - input a camera number 
#  - config file


def calc_iou_with_previous(region_id, bbox_stack_list, bbox_push_list, bbox_array):
    '''
    use IOU algorithm to compare current with previous
    stack is a all bboxes from previous inferences (where something was detected)
      size = DEDUP_DEPTH

    SIMPLE - 1 object detected
    consider:
     b1 = np.array([[0.10, 0.20, 0.30, 0.40], [0.12, 0.22, 0.32, 0.42], [0.08, 0.18, 0.28, 0.38], [0.4, 0.6, 0.4, 0.6], [0.10, 0.2, 0.3, 0.4]])
     b2 = np.array([[0.10, 0.20, 0.30, 0.40]])

    # iou = [[1], [0.68], [0.68], [0.], [1]]   average = 0.67
    match_rates = iou(b1,b2).reshape(-1,)
    matches = np.argwhere(match_rates > 0.8).size

    COMPLEX - 2 objects detected (history = 2 object, 1 object, 2 objects)
    b1 = np.array([[0.1, 0.11, 0.2, 0.22], [0.3, 0.33, 0.4, 0.44], [0.1, 0.11, 0.2, 0.22], [0.1, 0.11, 0.2, 0.22], [0.3, 0.33, 0.4, 0.44]])
    b2 = [[0.1, 0.11, 0.2, 0.22], [0.3, 0.33, 0.4, 0.44]]
    match_rates = iou(b1,b2)

    np.count_nonzero(match_rates[:,0] > 0.8)
    np.count_nonzero(match_rates[:,1] > 0.8)

    returns = IOU
    '''

    # get the bbox_stack
    bbox_stack = bbox_stack_list[region_id]
    bbox = bbox_array.reshape(-1,4)

    # print ("\n\n ---- BEFORE ----")
    # print ("Stack Before:", bbox_stack)
    # print ("bbox:", bbox)
    # print ("push list BEFORE:", bbox_push_list[region_id])

    match_rates = iou(bbox_stack, bbox)
    det_obj_count = bbox.shape[0]

    # for the number of detected objects (1 object = 1 bb0x)
    # how many matches?
    match_counts = []
    for i in range(det_obj_count):
        object_match_count = np.count_nonzero(match_rates[:,i] > bbox_iou_threshold)
        match_counts.append(object_match_count)

    # push match count to the bbox push list (so you know how many to pop off)
    # push list is nexted list - 1 list per region
    bbox_push_list[region_id].append(det_obj_count)         # push the object count into the list

    objects_to_remove = bbox_push_list[region_id][0]    # how many rows to remove from bbox_stack
                                                            # zero based index so subtract 1
    bbox_push_list[region_id].pop(0)                        # pop the first

    # print ("----- AFTER -----")
    # print ("-- bbox_stack ", bbox_stack.shape)
    # print ("-- bbox       ", bbox.shape)
    bbox_stack_list[region_id] = np.append(bbox_stack, bbox, 0)
    # print ("-- box_stack_list-appended", region_id, bbox_stack_list[region_id].shape)
    # print ("Stack BEFORE delete:", bbox_stack_list[region_id])

    # print ("Slice:", objects_to_remove)
    bbox_stack_list[region_id] = np.delete(bbox_stack_list[region_id], slice(0, objects_to_remove), 0)

    # print ("Match Counts:", match_counts)
    # print ("Stack After:", bbox_stack_list[region_id])
    # print ("Push List:", bbox_push_list[region_id])

    # print (" -- match counts:", match_counts)
    return match_counts


def run_inference(image, base_name, region, region_idx, bbox_stack_list, bbox_push_list, save_inference):
    '''
    run the inference with the given image
      image = full size image 
      base_name = time to make the file unique
          you'll append region_id to keep it unique
      region = dimension data to pull out, region = ((ymin, ymax), (xmin, xmax))
             last region is the full image = all 0 values
      region_idx
    save using the {base_name}_{region_idx}
    '''
    
    if region:
        (ymin, ymax) = region[0]
        (xmin, xmax) = region[1]
        # TODO
        #   after getting rid of orig/fullsize
        #   you can simplify this - you'll always have region
        if ymax > 0:
            image = image[ymin:ymax, xmin:xmax].copy()

    base_name = "{}_{}".format(base_name, region_idx)
    # This is destructive
    # when you do the display inference
    orig_image = image.copy()

    # pre-process the frame -> a compatible numpy array for the model
    preprocessed_image = tensorflow_util.preprocess_image(image, interpreter, model_image_dim, model_input_dim)
    # run the model
    bbox_array, class_id_array, prob_array = tensorflow_util.send_image_to_model(preprocessed_image, interpreter, PROBABILITY_THRESHOLD)

    # check detected objects against the stack
    new_objects = 0
    dup_objects = 0
    if prob_array is not None:
        match_counts = calc_iou_with_previous(region_idx, bbox_stack_list, bbox_push_list, bbox_array)
        for match_count in match_counts:
            if match_count >= 3:
                dup_objects = dup_objects + 1
            else:
                new_objects = new_objects + 1

    inference_image, orig_image_dim, detected_objects = display.inference_to_image( 
            image,
            bbox_array, class_id_array, prob_array, 
            model_input_dim, label_dict, PROBABILITY_THRESHOLD)

    # if the detected objects were repetitive - don't save the image
    #  get IOU


    # testing the format
    # convert detected_objexts to XML
    # detected_objects = list [ (class_id, class_name, probability, xmin, ymin, xmax, ymax)]
    if len(detected_objects) > 0:
        print ("       Objects:", base_name, detected_objects)
        if save_inference and new_objects > 0:
            image_name = os.path.join(image_path,  base_name + '.jpg')
            annotation_name = os.path.join(annotation_path,  base_name + '.xml')
            # print ("saving:", image_name, image.shape, annotation_name)
            # original image - h: 480  w: 640
            print ("  Saved: match count: {}  new objects: {}   image_name: {}".format( match_counts, new_objects, image_name))
            cv2.imwrite(image_name, orig_image)
            # this function generates & saves the XML annotation
            annotation_xml = annotation.inference_to_xml(image_path, image_name,orig_image_dim, detected_objects, annotation_path )
        elif save_inference and new_objects == 0:
            print ("  No new objects detected --- not saved")

    return inference_image, detected_objects, bbox_array




def main():
    # args
    camera_number = int(sys.argv[1])   # 0 based

    # get the app config - including passwords
    config = gen_util.read_app_config('app_config.json')

    # set some flags based on the config
    run_inferences = config["run_inferences"]
    save_inference = config["save_inference"]
    annotation_dir = config["annotation_dir"]
    snapshot_dir = config["snapshot_dir"]


    # set up tflite model
    global label_dict 
    label_dict = label_map_util.get_label_map_dict(config['label_map'], 'id')

    global interpreter
    interpreter = tensorflow_util.get_tflite_interpreter('model/output_tflite_graph.tflite')

    global model_image_dim, model_input_dim, output_details
    model_image_dim, model_input_dim, output_details = tensorflow_util.get_tflite_attributes(interpreter)

    # define your paths here - just once (not in the loop)
    global image_path, annotation_path
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))

    

    # Set up Camera 
    # TODO - should be a list
    #   - but it's just one camera now

    # for name, capture, flip in camera_list:
    camera_config = camera_util.get_camera_config(config, camera_number)
    camera_name = camera_config['name']
    url = camera_util.get_reolink_url('http', camera_config['ip'])      # pass the url base - not just the ip
    print ("Camera Config:", camera_config)

    # based on the config, config all camera regions
    # - includes building the bbox stacks
    regions, bbox_stack_list, bbox_push_list = camera_util.config_camera_regions(camera_config)

    snapshot_count = 0
    while True:

        start_time = time.time()
        base_name = "{}_{}".format(str(int(start_time)), camera_number)
        # frame returned as a numpy array ready for cv2
        # not resized
        angle = camera_config['rotation_angle']
        frame = camera_util.get_reolink_snapshot(url, camera_config['username'], camera_config['password'])

        if frame is not None:
            frame = imutils.rotate(frame, angle)                 # rotate frame
            orig_image_dim = (frame.shape[0], frame.shape[1])    #  dim = (height, width),
            orig_image = frame.copy()                            # preserve the original - full resolution
            # corner is top left
        
            print ('\n-- {} snap captured: {}'.format(snapshot_count, frame.shape), '{0:.2f} seconds'.format(time.time() - start_time))

            # True == run it through the model
            if run_inferences:
                inference_start_time = time.time()
                # loop through 0:n sub-regions of the frame
                # last one is the full resolution
                for i, region in enumerate(regions):
                    crop_start_time = time.time()
                    
                    inference_image, detected_objects, bbox_array = run_inference(orig_image, base_name, region, i, bbox_stack_list, bbox_push_list, True)
                    print ('     crop {}'.format(i), ' inference: {0:.2f} seconds'.format(time.time() - crop_start_time))
                    # enlarged_inference = cv2.resize(inference_image, (1440, 1440), interpolation = cv2.INTER_AREA)
                    window_name = "{} crop {}".format(camera_name, i)
                    cv2.imshow(window_name, inference_image)   # show the inferance

                print ('   TOTAL inference: {0:.2f} seconds'.format(time.time() - inference_start_time))

            else:
                cv2.imshow(camera_name, frame)
            snapshot_count = snapshot_count + 1
        else:
            print ("-- no frame returned -- ")
            

        # time.sleep(3)

        # Use key 'q' to close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
