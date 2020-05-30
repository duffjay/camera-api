import re
import logging
from logging.handlers import TimedRotatingFileHandler

# tensorflow logger
# -- trying (unsuccessfully) to direct the default tensorflow log to file handler
# -- 
logTF = logging.getLogger('tensorflow')
logTF.setLevel(logging.INFO)
logname = 'tensorflow.log'
handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
handler.suffix = "%Y%m%d"
handler.extMatch = re.compile(r"^\d{8}$")
# formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logTF.addHandler(handler)

import os
import sys
import time
import cv2
from PIL import Image
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
from object_detection.utils.np_box_ops import iou

#import tflite_runtime.interpreter as tflite

import label_map_util
import inference
import status

# main logger
log = logging.getLogger(__name__)

# H E L P E R    F U N C T I O N S

# load an image and resturn a numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# C O N F I G U R A T I O N
def configure_tensorflow_model(model_config):
    log.info(f'tensorflow model config: {model_config}')
    framework = model_config['model_framework']
    model_path = os.path.join('model', model_config['model_path'])
    
    # get a frozen graph
    detection_graph = get_detection_graph(model_path)
    sess, tensor_dict, image_tensor = get_tf_session(detection_graph)

    label_map = model_config['label_map']
    label_dict = label_map_util.get_label_map_dict(label_map, 'id')

    # Model Input Dimensions
    # - tflite will give it to you, but not tensorflow frozen graph
    #   so I put it in the config - this is overwriting whatever tflite reported - beware
    model_input_dim = model_config['model_input_dim']
    model_image_dim = (model_config['model_input_dim'][1], model_config['model_input_dim'][2])
    # print ("Model Framework: {}   model input dim: {}   image dim: {}".format(framework, model_input_dim, model_image_dim))
    # print ("      Label Map: {}".format(label_map))
    log.info(f'Model Framework: {framework}   model input dim: {model_input_dim}   image dim: {model_image_dim}')
    return sess, tensor_dict, image_tensor, model_input_dim, label_map, label_dict


# NOT TESTE$T
# TODO - test reolink2model.py
#      - using this function
def configure_tflite_model(model_config):
    print (model_config)
    framework = model_config['model_framework']
    model_path = os.path.join('model', model_config['model_path'])
    #
    # S S D   M O D E L   F R A M E W O R K
    # TF Lite
    if framework == 'tflite':
        interpreter = tensorflow_util.get_tflite_interpreter(model_path)
        model_image_dim, model_input_dim, output_details = get_tflite_attributes(interpreter)

    label_map = model_config['label_map']
    label_dict = label_map_util.get_label_map_dict(label_map, 'id')

    # Model Input Dimensions
    # - tflite will give it to you, but not tensorflow frozen graph
    #   so I put it in the config - this is overwriting whatever tflite reported - beware
    model_input_dim = model_config['model_input_dim']
    model_image_dim = (model_config['model_input_dim'][1], model_config['model_input_dim'][2])
    print ("Model Framework: {}   model input dim: {}   image dim: {}".format(framework, model_input_dim, model_image_dim))
    print ("      Label Map: {}".format(label_map))
    return framework, interpreter, model_input_dim, output_details, label_map, label_dict


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
            log.info('Box size zero removed')
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

def get_detection_graph(model_path):
    '''
    open frozen TF graph
    return the object
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

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

def get_tf_session(graph):

    with graph.as_default():
        sess = tf.Session()
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        # print ("ALL model operations:", type(ops), len(ops), ops)
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        log.debug(f'tensor names: {type(all_tensor_names)} : {len(all_tensor_names)}')
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_value = tf.get_default_graph().get_tensor_by_name(tensor_name)
                log.debug(f'tensor name - {tensor_name} : {tensor_value}')
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            log.debug("*** detection mask in the tensor dict ***")
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return sess, tensor_dict, image_tensor

# the model yields all detections over 0 probabiliity
# keep only the ones over threshold
# return ModelInferenceObject
def convert_output_to_inference_object(output_dict, threshold):
    scores = output_dict['detection_scores']
    good_detection_count = len([i for i in scores if i >= threshold])
    # convert ONLY GOOD inferences to numpy
    prob_array = np.array(output_dict['detection_scores'][:good_detection_count])
    class_array = np.array(output_dict['detection_classes'][:good_detection_count])
    bbox_array = np.array(output_dict['detection_boxes'][:good_detection_count])
    inf = inference.ModelInference(class_array, prob_array, bbox_array)
    return inf


def send_image_to_tf_sess(image_np_expanded, sess, tensor_dict, image_tensor):
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image_np_expanded})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    
    # Convert good detections to ModelInference object
    threshold = 0.7
    inf = convert_output_to_inference_object(output_dict, threshold)
    return inf

# frozen graph
# -- development only -- this should be discarded at some point
def send_imagelist_to_frozen_graph(image_list, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      # print ("ALL model operations:", type(ops), len(ops), ops)
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      log.debug(f'tensor names: {type(all_tensor_names)} : {len(all_tensor_names)}')
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_value = tf.get_default_graph().get_tensor_by_name(tensor_name)
          log.debug(f'tensor name - {tensor_name} : {tensor_value}')
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      if 'detection_masks' in tensor_dict:
        log.debug("*** detection mask in the tensor dict ***")
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      log.debug(f'image_tensor: {image_tensor}')
        
      # Run inference
      log.info(" --- run the model ---")


      for i,image_path in enumerate(image_list):
        log.info(f'index: {i} {image_path}')
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image_path)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        start = time.perf_counter()
        # -- run model
        # output_dict = tensorflow_util.send_image_to_frozen_graph(image_np_expanded, detection_graph)
        output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image_np_expanded})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        # print ("Detection:", output_dict)
        finish = time.perf_counter()
        log.info(f'Finished in {round(finish - start, 2)} seconds(s)')


  return i

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - -
#    B O U N D I N G    B O X    U T I L I T I E S
# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def calc_iou_with_previous(image_time, bbox_iou_threshold, camera_id, region_id, bbox_stack_list, bbox_push_list, bbox_array):
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
    bbox_stack = bbox_stack_list[region_id]  # should be [4, depth]
    bbox = bbox_array.reshape(-1,4)

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} ---- BEFORE ----')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Stack Before: {bbox_stack.shape}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} stack: {bbox_stack.tolist()}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} bbox: {bbox}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} push list BEFORE:  {bbox_push_list[region_id]}')

    match_rates = iou(bbox_stack, bbox)
    det_obj_count = bbox.shape[0]

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} match_rates: {match_rates.tolist()}')
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

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} ----- AFTER -----')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} -- bbox_stack {bbox_stack.shape}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} -- bbox       {bbox.shape}')
    bbox_stack_list[region_id] = np.append(bbox_stack, bbox, 0)
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} -- box_stack_list-appended: {region_id} {bbox_stack_list[region_id].shape}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Stack BEFORE delete: {bbox_stack_list[region_id].tolist()}')

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Slice: {objects_to_remove}')
    bbox_stack_list[region_id] = np.delete(bbox_stack_list[region_id], slice(0, objects_to_remove), 0)

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Match Counts: {match_counts}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Stack After:  {bbox_stack_list[region_id].tolist()}')
    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} Push List:    {bbox_push_list[region_id]}')

    log.debug(f'id new -- cam#: {camera_id} reg# {region_id} {image_time} -- match counts: {match_counts}')
    return match_counts


# identify new & old (repetitive) detections
# this is a per camera-region function
# - this is ONLY called if there were predictions
def identify_new_detections(image_time, bbox_iou_threshold, camera_id, region_id, bbox_array, bbox_stack_list, bbox_push_list):
    # check detected objects against the stack
    new_objects = 0
    dup_objects = 0

    match_counts = calc_iou_with_previous(image_time, bbox_iou_threshold, camera_id, region_id, bbox_stack_list, bbox_push_list, bbox_array)
    for match_count in match_counts:
        if match_count >= 3:
            dup_objects = dup_objects + 1
        else:
            new_objects = new_objects + 1
    return new_objects, dup_objects