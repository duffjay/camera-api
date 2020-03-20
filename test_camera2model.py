import os
import sys
import time
import cv2
import numpy as np

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


PROBABILITY_THRESHOLD = 0.6    # only display objects with a 0.6+ probability

#
#  Set up for Amcrest Cameras
#

def main():
    # get the app config - including passwords
    config = gen_util.read_app_config('app_config.json')

    # set some flags based on the config
    run_inference = config["run_inference"]
    save_inference = config["save_inference"]
    annotation_dir = config["annotation_dir"]
    snapshot_dir = config["snapshot_dir"]


    # set up camerass
    camera_list = camera_util.configure_cameras(config)

    # set up tflite model
    label_dict = label_map_util.get_label_map_dict(config['label_map'], 'id')
    interpreter = tensorflow_util.get_tflite_interpreter('model/output_tflite_graph.tflite')
    model_image_dim, model_input_dim, output_details = tensorflow_util.get_tflite_attributes(interpreter)

    # define your paths here - just once (not in the loop)
    image_path = os.path.abspath(os.path.join(cwd, snapshot_dir))
    annotation_path = os.path.abspath(os.path.join(cwd, annotation_dir))

    run_with_camera_number = 0   # 0 based

    snapshot_count = 0
    while True:

        # for name, capture, flip in camera_list:
        name, capture, flip = camera_list[run_with_camera_number]  # running with 1 camera only
        start_time = time.time()
        print(name, snapshot_count)

        ret, frame = capture.read()                          #  frame.shape (height, width, depth)

        if frame is not None:
            orig_image_dim = (frame.shape[0], frame.shape[1])    #  dim = (height, width),
            orig_image = frame.copy()
            snapshot_count = snapshot_count + 1

            print ('captured:', frame.shape, time.time() - start_time)

            if flip == "vert":
                frame = cv2.flip(frame, 0)

                

                

            # True == run it through the model
            if run_inference:
                # pre-process the frame -> a compatible numpy array for the model
                preprocessed_image = tensorflow_util.preprocess_image(frame, interpreter, model_image_dim, model_input_dim)
                bbox_array, class_id_array, prob_array = tensorflow_util.send_image_to_model(preprocessed_image, interpreter)
                print ('inference:', frame.shape, time.time() - start_time)

                inference_image, orig_image_dim, detected_objects = display.inference_to_image( 
                        frame,
                        bbox_array, class_id_array, prob_array, 
                        model_input_dim, label_dict, PROBABILITY_THRESHOLD)

                
                # testing the format
                # convert detected_objexts to XML
                # detected_objects = list [ (class_id, class_name, probability, xmin, ymin, xmax, ymax)]
                if len(detected_objects) > 0:
                    print (detected_objects)
                    if save_inference:
                        image_base_name = str(int(start_time))
                        image_name = os.path.join(image_path,  image_base_name + '.jpg')
                        annotation_name = os.path.join(annotation_path,  image_base_name + '.xml')
                        print ("saving:", image_name, frame.shape, annotation_name)
                        # original image - h: 480  w: 640
                        cv2.imwrite(image_name, orig_image)
                        # this function generates & saves the XML annotation
                        annotation_xml = annotation.inference_to_xml(name, image_name,orig_image_dim, detected_objects, annotation_dir )


                # enlarged_inference = cv2.resize(inference_image, (1440, 1440), interpolation = cv2.INTER_AREA)
                cv2.imshow(name, inference_image)   # show the inferance
                # cv2.imshow(name, orig_image)     # show the raw image from the camera
            else:
                cv2.imshow(name, frame)
        else:
            print ("-- no frame returned -- ")
            

        # time.sleep(3)

        # Use key 'q' to close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
