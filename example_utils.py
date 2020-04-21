import json
import os
import sys
import time

from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup

from PIL import Image
import IPython.display as display

import numpy as np
import tensorflow as tf
import contextlib2

# this is a different version (MY VERSION) of utils - because of inconsistencies in the pbtxt
from label_map_util import get_label_map_dict

from object_detection.utils import ops as utils_ops
from object_detection.utils.visualization_utils import STANDARD_COLORS
from object_detection.utils.visualization_utils import draw_bounding_box_on_image
from object_detection.dataset_tools import tf_record_creation_util


# in this project, moved to the general code from ssd-dag/code/cfa_utils/
from gen_imagesets import gen_imageset_list

# taken from: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# use this as the standard feature definition for MobileNet SSD Example
# this comes from the models/mobilenet code - it's not CFA specific
# but it's specific to the model - must be consistent through the pipeline

feature_obj_detect = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/class/text': tf.io.VarLenFeature(tf.string),
        'image/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/area': tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
        'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        'image/object/group_of': tf.io.VarLenFeature(tf.int64),
        'image/object/weight': tf.io.VarLenFeature(tf.float32)
    }

# Default values for the annotations
# - seems you must supply a value for every attr in the  dict
#   or you'll get a MergeFrom error
IMG_FORMAT = b'jpg'
IMG_SHA256 = b''
IMG_CLASS_NAMES = [b'cfa_prod']
IMG_CLASS_IDS = [1]

OBJ_AREA = 1.0
OBJ_IS_CROWD = 0
OBJ_DIFFICULT = 0  # the DIFFICULT in XML is per image - not per object
OBJ_GROUP_OF = 1
OBJ_WEIGHT = 1


NUM_SHARDS = 12

# return a list of imagesets in given directory
def get_imagesets(imageset_dir):
    file_list = []
    for f in listdir(imageset_dir):
        file_part = os.path.splitext(f)
        if (isfile(join(imageset_dir, f)) and (file_part[1] == ".txt") ):
            file_list.append(f)
    return file_list

def incr_class_count (class_dict, class_id):
    try:
        class_dict[class_id] = class_dict[class_id] + 1
    except:
        class_dict.update({ class_id : 1})

    return class_dict

def invert_dict(orig_dict):
    inverted_dict = dict([[v,k] for k,v in orig_dict.items()])
    return inverted_dict

# from VOC compliant images/annotations/sets
# convert to tfrecords
def voc_to_tfrecord_file(image_dir,
                    annotation_dir,
                    label_map_file,
                    tfrecord_dir,
                    training_split_tuple,
                    include_classes = "all",
                    exclude_truncated=False,
                    exclude_difficult=False):
    # this uses only TensorFlow libraries
    # - no P Ferrari classes

    label_map = get_label_map_dict(label_map_file, 'value')
    # label_map_dict = invert_dict(origin_label_map_dict)    # we need the id, not the name as the key

    train_list, val_list, test_list = gen_imageset_list(annotation_dir, training_split_tuple)

    print (label_map)

    # iterate through each image_id (file name w/o extension) in the image list
    # this list will give you the variables needed to iterate through train/val/test
    imageset_list = [(train_list, 'train'), (val_list, 'val'), (test_list, 'test')]
    j = 0
    # image_list == train, val, test
    for (image_list, imageset_name) in imageset_list:

        # you can create/open the tfrecord writer
        output_path = os.path.join(tfrecord_dir, imageset_name, imageset_name + ".record")
        # tf_writer = tf.python_io.TFRecordWriter(output_path)
        print (" -- images", len(image_list), " writing to:", output_path)
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_path, NUM_SHARDS)

            image_count = 0   # simple image cuonter
            class_dict = {}   # dict to keep class count

            # loop through each image in the image list

            for image_id in image_list:
                if image_id.startswith('.'):
                    continue
                # get annotation information
                annotation_path = os.path.join(annotation_dir, image_id + '.xml')
                with open(annotation_path) as f:
                        soup = BeautifulSoup(f, 'xml')

                        folder = soup.folder.text
                        filename = soup.filename.text
                        # size = soup.size.text
                        sizeWidth = float(soup.size.width.text)     # you need everything as floats
                        sizeHeight = float(soup.size.height.text)
                        sizeDepth = float(soup.size.depth.text)

                        boxes = [] # We'll store all boxes for this image here
                        objects = soup.find_all('object') # Get a list of all objects in this image

                        # Parse the data for each object
                        for obj in objects:
                            class_name = obj.find('name').text
                            try:
                                class_id = label_map[class_name]
                                class_dict = incr_class_count(class_dict, class_id)
                            except:
                                print ("!!! label map error:", image_id, class_name, " skipped")
                                continue
                            # Check if this class is supposed to be included in the dataset
                            if (not include_classes == 'all') and (not class_id in include_classes): continue
                            pose = obj.pose.text
                            truncated = int(obj.truncated.text)
                            if exclude_truncated and (truncated == 1): continue
                            difficult = int(obj.difficult.text)
                            if exclude_difficult and (difficult == 1): continue
                            # print (image_id, image_count, "xmin:", obj.bndbox.xmin.text)
                            xmin = int(obj.bndbox.xmin.text.split('.')[0])  # encountered a few decimals - that will throw an error
                            ymin = int(obj.bndbox.ymin.text.split('.')[0])
                            xmax = int(obj.bndbox.xmax.text.split('.')[0])
                            ymax = int(obj.bndbox.ymax.text.split('.')[0])
                            item_dict = {'class_name': class_name,
                                        'class_id': class_id,
                                        'pose': pose,
                                        'truncated': truncated,
                                        'difficult': difficult,
                                        'xmin': xmin,
                                        'ymin': ymin,
                                        'xmax': xmax,
                                        'ymax': ymax}
                            boxes.append(item_dict)
        
                # get the encoded image
                img_path = os.path.join(image_dir, image_id + ".jpg")
                with tf.io.gfile.GFile(img_path, 'rb') as fid:
                    encoded_jpg = fid.read()

                # now you have everything necessary to create a tf.example
                # tf.Example proto 
                # print ("    ", filename)
                # print ("     ", class_name, class_id)
                # print ("     ", sizeHeight, sizeWidth, sizeDepth)
                # print ("     ", len(boxes))

                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                class_names = []
                class_ids = []
                obj_areas = []
                obj_is_crowds = []
                obj_difficults = []
                obj_group_ofs = []
                obj_weights = []

                # loop through each bbox to make a list of each
                for box in boxes:
                    # print ("       ", box)
                    # create lists of bbox dimensions
                    xmins.append(box['xmin'] / sizeWidth)
                    xmaxs.append(box['xmax'] / sizeWidth)

                    ymins.append(box['ymin'] / sizeHeight)
                    ymaxs.append(box['ymax'] / sizeHeight)

                    class_names.append(str.encode(box['class_name']))
                    class_ids.append(int(box['class_id']))

                    obj_areas.append(OBJ_AREA)
                    obj_is_crowds.append(OBJ_IS_CROWD)
                    obj_difficults.append(OBJ_DIFFICULT)
                    obj_group_ofs.append(OBJ_GROUP_OF)
                    obj_weights.append(OBJ_WEIGHT)

                # use the commonly defined feature dictionary
                feature = feature_obj_detect.copy()
                # thus you have a common structure for writing & reading
                # these image features

                # per image attributes
                feature['image/encoded'] = bytes_feature(encoded_jpg)
                feature['image/format'] = bytes_feature(IMG_FORMAT)
                feature['image/filename'] = bytes_feature(str.encode(filename))
                feature['image/key/sha256'] = bytes_feature(IMG_SHA256)
                feature['image/source_id'] = bytes_feature(str.encode(image_id))

                feature['image/height'] = int64_feature(int(sizeHeight))
                feature['image/width'] = int64_feature(int(sizeWidth))

                feature['image/class/text'] = bytes_list_feature(IMG_CLASS_NAMES)
                feature['image/class/label'] = int64_list_feature(IMG_CLASS_IDS)


                # per image/object attributes
                feature['image/object/bbox/xmin'] = float_list_feature(xmins)
                feature['image/object/bbox/xmax'] = float_list_feature(xmaxs)
                feature['image/object/bbox/ymin'] = float_list_feature(ymins)
                feature['image/object/bbox/ymax'] = float_list_feature(ymaxs)
                feature['image/object/class/text'] = bytes_list_feature(class_names)
                feature['image/object/class/label'] = int64_list_feature(class_ids)

                # these are all taken from default values
                feature['image/object/area'] = float_list_feature(obj_areas)
                feature['image/object/is_crowd'] = int64_list_feature(obj_is_crowds)
                feature['image/object/difficult'] = int64_list_feature(obj_difficults)
                feature['image/object/group_of'] = int64_list_feature(obj_group_ofs)
                feature['image/object/weight'] = float_list_feature(obj_weights)


                features = tf.train.Features(feature=feature)

                tf_example = tf.train.Example(features=features)
                # write to the tfrecords writer
                tf_writer.write(tf_example.SerializeToString())
                image_count = image_count + 1

        # end of loop
        # TODO - shard on larger sets
        #        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        # tf_writer.close()                   # close the writer
        print ('     image count:', image_count, "  class_count:", class_dict)
    return 1

def parse_function(example_proto):
    # Parse the input using the standard dictionary
    feature = tf.io.parse_single_example(example_proto, feature_obj_detect)
    return feature

def get_class_names(label_map):
    """
    get the class names - as a dict of byte arrays
    input:  full absolute path o the label map protobuf
    output: dict of byte arrays {class id:  b'name'}
    """
    label_map = get_label_map_dict(label_map)
    label_map_reverse = {}
    for (k,v) in label_map.items():
        label_map_reverse[v] = k
    for (k,v) in label_map_reverse.items():
        label_map_reverse[k] = str.encode(v)
    return label_map_reverse

def get_dataset_length(ds):
    """
    get dataset size
      brute force iteration to get record count
    """
    num_elements = 0
    for element in ds:
        num_elements += 1
    return num_elements

def display_detection(image_tensor, class_names, threshold, prediction):
    """
    utilility to display image & prediction
    input:  image - EagerTensor
            class names - dict {class id: b'name'}
            threshold - 0.0 - 1.0  (keep if score > threshold)
            prediction - model response for this instance
    """
    # post processed prediction
    # - without the threshold applied
    detect_scores = np.asarray(prediction['detection_scores'], dtype=np.float32)
    detect_classes = np.asarray(prediction['detection_classes'], dtype=np.int8)
    detect_boxes = np.asarray(prediction['detection_boxes'], dtype=np.float32)

    # keep == w/ threshold applied
    keep_idx = np.argwhere(detect_scores > threshold)
    class_ids = detect_classes[keep_idx]
    (obj_count, discard) = class_ids.shape
    # get the class names you need to keep
    obj_class_ids = class_ids.reshape(-1)
    obj_class_names = []
    for i in obj_class_ids:
        obj_class_names.append(class_names[i])
    obj_class_names = np.asarray(obj_class_names)
    # get the bounding boxes you need to keep
    bboxes = detect_boxes[keep_idx]

    # just keep data from here on
    image_decoded_tensor = tf.image.decode_jpeg(image_tensor)
    image = image_decoded_tensor.numpy()
    pil_image = Image.fromarray(image)
    bb = bboxes.reshape(-1,4)
    ymins = bb[:,0]
    xmins = bb[:,1]
    ymaxs = bb[:,2]
    xmaxs = bb[:,3]

    print ("detected:", obj_count, obj_class_names)

    for idx in range(obj_count):
        draw_bounding_box_on_image(pil_image,ymins[idx],xmins[idx], ymaxs[idx], xmaxs[idx],
                color=STANDARD_COLORS[obj_class_ids[idx]], 
                thickness=4, display_str_list=[obj_class_names[idx]],
                use_normalized_coordinates=True)
    display.display(pil_image)
