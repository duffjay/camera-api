import os
import sys
import time
import cv2

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
    framework, tflite_interpreter, tensorflow_session, model_input_dim, output_details, label_map, label_dict = tensorflow_util.configure_model(model_config)

    # camera config
    camera_config_list = config['camera']
    camera_count = len(camera_config_list)
    print ("Camera Count:", camera_count)

    for camera_config in camera_config_list:
        camera_name, np_images = camera_util.get_camera_regions(camera_config)
        if np_images is not None:
            print ("np_images:", np_images.shape)
            for i, image in enumerate(np_images):
                print ("image {}  shape {}".format(i, image.shape))

                window_name = "{}-{}".format(camera_name, i)
                cv2.imshow(window_name,image)
            cv2.waitKey(0)
        else:
            print ("nothing returned")

if __name__ == '__main__':
    main()
