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

# globals

safeprint = threading.Lock()
imageQueue = queue.Queue()
# inferenceQueue = queue.Queue()
run_state = True



def image_producer(camera_id, camera_config, camera_snapshot_times, imageQueue):
    while True:
        camera_name, np_images = camera_util.get_camera_regions(camera_config)
        snapshot_elapsed =  time.perf_counter() - camera_snapshot_times[camera_id]      # elapsed time between snapshots
        camera_snapshot_times[camera_id] = time.perf_counter()                          # update the time == start time for the next snapshot
        # pushes to the stack if there was a frame captured
        if np_images is not None:
            image_time = int(time.time())
            imageQueue.put((camera_id, camera_name, image_time, np_images))
            with safeprint:
                print ("  IMAGE-PRODUCER:>>{} np_images: {}  {:02.2f} secs".format(camera_name, np_images.shape, snapshot_elapsed))
        else:
            with safeprint:
                print ("  IMAGE-PRODUCER:--{} np_images: None".format(camera_name))
        # stop?        
        global run_state                        # use a global to shut down the thread
        if run_state == False:
            break
    return

# Pull images off the imageQueue
# - produce inferences
def image_consumer(consumer_id, imageQueue, sess, tensor_dict, image_tensor, bbox_stack_lists, bbox_push_lists, model_input_dim, label_dict):
    save_inference = True
    while True:
        try:
            # Consumer tasks
            # - once/frame
            #    - ALL regions of the frame
            data = imageQueue.get(block=False)
            camera_id, camera_name, image_time, np_images = data
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
                
                with safeprint:
                    print ('  IMAGE-CONSUMER:<<Camera Name: {} region: {} image_timestamp {}  inference time:{:02.2f} sec  detections: {}'.format(
                        camera_name, region_id, image_time, (time.perf_counter() - start), num_detections))
                    for detection_id in range(num_detections):
                        print ('      detection {}  classes detected{} new: {}  repeated: {}'.format(detection_id, class_array, new_objects, dup_objects))
                        print ('           Scores: {}   Classes: {}    BBoxes: {}'.format(prob_array, class_array, bbox_array))
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

                # S A V E
                # - if there were detections
                #     - and they was at least 1 new one
                # save the annotation also
                if num_detections > 0:
                    if save_inference and new_objects > 0:
                        base_name = '{}-{}-{}'.format(image_time, camera_id, region_id)
                        global image_path
                        image_name = os.path.join(image_path,  base_name + '.jpg')
                        global annotation_path
                        annotation_name = os.path.join(annotation_path,  base_name + '.xml')
                        # print ("saving:", image_name, image.shape, annotation_name)
                        # original image - h: 480  w: 640
                        print ("  Saved: stale objects: {}  new objects: {}   image_name: {}".format( dup_objects, new_objects, image_name))
                        cv2.imwrite(image_name, orig_image)
                        # this function generates & saves the XML annotation
                        annotation_xml = annotation.inference_to_xml(image_path, image_name,orig_image_dim, detected_objects, annotation_path )
                    elif save_inference and new_objects == 0:
                        print ("  No new objects detected --- not saved")

            # stop all on a 'q' in a display window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                global run_state
                run_state = False
                break

        except queue.Empty:
            pass

        # stop?
        # ?? why no global run_state here ??
        if run_state == False:
            break
    time.sleep(0.1)
    return

def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    #
    # T O P    L E V E L   
    #    app config 
    # 
    config = gen_util.read_app_config(config_filename)
    run_inferences = config["run_inferences"]
    save_inference = config["save_inference"]
    annotation_dir = config["annotation_dir"]
    snapshot_dir = config["snapshot_dir"]

    global image_path
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    global annotation_path
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))

    # configure the model
    model_config = config["model"]
    sess, tensor_dict, image_tensor, model_input_dim, label_map, label_dict = tensorflow_util.configure_tensorflow_model(model_config)

    # camera config
    # - includes getting the data structures to track detections
    camera_config_list, camera_count, camera_snapshot_times, bbox_stack_lists, bbox_push_lists = camera_util.config_camara_data(config)
    print ("Camera Count:", camera_count)

    global run_state
    run_state = True
    #   I M A G E    C O N S U M E R S
    #   == inference producers
    # 
    consumer_count = 1
    for i in range(1):
        thread = threading.Thread(target=image_consumer, 
            args=(i, imageQueue, sess, tensor_dict, image_tensor, bbox_stack_lists, bbox_push_lists, model_input_dim, label_dict))
        thread.daemon = True
        thread.start()

    #   I M A G E    P R O D U C E R S
    #    
    for camera_id, camera_config in enumerate(camera_config_list):
        thread = threading.Thread(target=image_producer, 
            args=(camera_id, camera_config, camera_snapshot_times, imageQueue))
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
