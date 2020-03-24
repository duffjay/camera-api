import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

# add some paths that will pull in other software
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)


from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
#import tflite_runtime.interpreter as tflite

def make_objects(boxes, class_ids_rev):
    '''
    boxes = list of bounding boxes
    class_ids_rev = 
    '''
    objects = ''
    for box in boxes:
        if str(box["class_id"]) not in class_ids_rev:
            continue
        if box['xmin'] == box['xmax'] or box['ymin'] == box['ymax']:
            print('Box size zero removed')
            continue

        class_name = class_ids_rev[str(box["class_id"])]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        objects += """<object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>""".format(class_name, xmin, ymin, xmax, ymax)
    return objects

def make_xml_string(folder, filename, path, width, height, objects, verified):
    base = """<annotation{}>
    <folder>{}</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {}
    </annotation>""".format(verified, folder, filename, path, width, height, objects)
    return base

def get_tflite_interpreter(model_path):
    print ('TF Lite Model loading...')
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()  # failure to do this will give you a core dump
    return interpreter

def get_tflite_attributes(interpreter):
    # using the interpreter - get some of the model attributes
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
        
    model_input_shape = input_details[0]['shape']   # (batch, h, w, channels)
    model_image_dim = (model_input_shape[1], model_input_shape[2])    # model - image dimension
    model_input_dim = (1, model_input_shape[1], model_input_shape[2], 3) # model - batch of images dimensions
    print ("Model Input Dimension: {}".format(model_input_dim))
    return model_image_dim, model_input_dim, output_details

def send_image_to_model(preprocessed_image, interpreter, threshold):
    '''
    send the image into the model
    return only inferences with probability > threshold
    '''
    # input (image) is (1,300,300,3) - shaped like a batch of size 1
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)    # model input is a batch of images
    interpreter.invoke()        # this invokes the model, creating output data

    # model has created an inference from a batch of data
    # - the model creates output like a batch of size = 1
    # - size must be 1, so simplify the shape by taking first row only
    #   [0] at the end effectivtly means bbox (1,10,4) becomes (10,4)
    bbox_data = interpreter.get_tensor(output_details[0]['index'])[0]
    class_data = interpreter.get_tensor(output_details[1]['index'])[0]
    prob_data = interpreter.get_tensor(output_details[2]['index'])[0]

    # print ("DEBUG prob_data:", prob_data)
    prob_data = np.nan_to_num(prob_data)   # replace nan with 0
    bbox_data = np.nan_to_num(bbox_data)
    reduction_index = np.argwhere((prob_data > threshold) & (prob_data <= 1.0))
    if reduction_index.size > 0:
        return bbox_data[reduction_index], class_data[reduction_index], prob_data[reduction_index]
    else:
        return None, None, None

def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image, interpreter, model_image_dim, model_input_dim):
    resized_image = cv2.resize(image, model_image_dim, interpolation = cv2.INTER_AREA)  # resized to 300x300xRGB
    reshaped_image = np.reshape(resized_image, model_input_dim)                         # reshape for model (1,300,300,3)
    return reshaped_image

def detected_objects_to_annotation(detected_objects):
    '''
    input:   detected objects = list(tuplie)
                                (class_id, class_name, probability, xmin, ymin, xmax, ymax)

    '''

    return