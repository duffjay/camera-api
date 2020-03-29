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
#import tflite_runtime.interpreter as tflite

# Helper Function
# load an image and resturn a numpy array

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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
        print ("tensor names:", type(all_tensor_names), len(all_tensor_names))
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_value = tf.get_default_graph().get_tensor_by_name(tensor_name)
                print (tensor_name, tensor_value)
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            print ("*** detection mask in the tensor dict ***")
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
    return output_dict

# frozen graph
# -- development only -- this should be discarded at some point
def send_imagelist_to_frozen_graph(image_list, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      # print ("ALL model operations:", type(ops), len(ops), ops)
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      print ("tensor names:", type(all_tensor_names), len(all_tensor_names))
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_value = tf.get_default_graph().get_tensor_by_name(tensor_name)
          print (tensor_name, tensor_value)
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      if 'detection_masks' in tensor_dict:
        print ("*** detection mask in the tensor dict ***")
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
      print ("image_tensor:", image_tensor)
        
      # Run inference
      print (" --- run the model ---")


      for i,image_path in enumerate(image_list):
        print (i, image_path)
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
        print (f'Finished in {round(finish - start, 2)} seconds(s)')


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