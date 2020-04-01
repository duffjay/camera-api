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

# globals

safeprint = threading.Lock()
imageQueue = queue.Queue()
inferenceQueue = queue.Queue()
run_state = True


def image_producer(camera_id, camera_config, camera_snapshot_times, imageQueue):
    while True:
        camera_name, np_images = camera_util.get_camera_regions(camera_config)
        snapshot_elapsed =  time.perf_counter() - camera_snapshot_times[camera_id]      # elapsed time between snapshots
        camera_snapshot_times[camera_id] = time.perf_counter()                          # update the time == start time for the next snapshot
        if np_images is not None:
            image_time = time.time()
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
def image_consumer(consumer_id, imageQueue, sess, tensor_dict, image_tensor):
    while True:
        try:
            data = imageQueue.get(block=False)
            camera_id, camera_name, image_time, np_images = data
            start = time.perf_counter()
            for i, np_image in enumerate(np_images):
                np_image_expanded = np.expand_dims(np_image, axis=0)
                # -- run model
                output_dict = tensorflow_util.send_image_to_tf_sess(np_image_expanded, sess, tensor_dict, image_tensor)
                # get data for relavant detections
                num_detections = output_dict['num_detections']
                detection_scores = output_dict['detection_scores'][0:num_detections]
                detection_classes = output_dict['detection_classes'][0:num_detections]
                detection_boxes = output_dict['detection_boxes'][0:num_detections]
                with safeprint:
                    print ('  IMAGE-CONSUMER:<<Camera Name: {}  {:02.2f}'.format(camera_name, (time.perf_counter() - start)))
                    for i in range(num_detections):
                        print ('      region {}  classes detected{}'.format(i, detection_classes))
        except queue.Empty:
            pass

        # stop?
        # ?? why no global run_state here ??
        if run_state == False:
            break
    time.sleep(0.5)
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

    
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))

    # configure the model
    model_config = config["model"]
    sess, tensor_dict, image_tensor, model_input_dim, label_map, label_dict = tensorflow_util.configure_tensorflow_model(model_config)

    # camera config
    camera_config_list, camera_count, camera_snapshot_times = camera_util.config_camara_data(config)
    print ("Camera Count:", camera_count)

    global run_state
    run_state = True
    #   I M A G E    C O N S U M E R S
    #   == inference producers
    # 
    for i in range(len(camera_config_list)):
        thread = threading.Thread(target=image_consumer, args=(i, imageQueue, sess, tensor_dict, image_tensor))
        thread.daemon = True
        thread.start()

    #   I M A G E    P R O D U C E R S
    #    
    for camera_id, camera_config in enumerate(camera_config_list):
        thread = threading.Thread(target=image_producer, args=(camera_id, camera_config, camera_snapshot_times, imageQueue))
        thread.start()

    time.sleep(15)
    print ("main() sleep timed out")
    run_state = False
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
