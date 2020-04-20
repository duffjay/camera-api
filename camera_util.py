import io
import random
import string
import urllib
import numpy as np
import time

from PIL import Image
import cv2
import imutils

# DEPRECATED
def get_camera_config(config, camera_num):
    '''
    given the config JSON and a camera number, 
    return the config 
    '''
    camera_config = config['camera'][camera_num]
    return camera_config

# is the image color?
#  return g == grayscale
#         c == color
# append on the file names
def is_color_gc(image):
    height, width, channels = image.shape
    # grab a center section 40x40
    size = 40
    start_y = int((height / 2) - (size / 2))
    start_x = int((width / 2) - (size / 2))
    end_y = start_y + size
    end_x = start_x + size
    # arbitrarily calling the planes r, g, b  (not really sure if it's rgb or bgr)
    #             but it doesn't matter
    r = image[start_y:end_y, start_x:end_x, 0]          # gotta be integers
    b = image[start_y:end_y, start_x:end_x, 1]
    g = image[start_y:end_y, start_x:end_x, 2]
    # take difference - not exhaustive but good enough
    diff_rg = np.sum(np.absolute(r - b))                # get absolute values - then sum the differences
    diff_gb = np.sum(np.absolute(g - b))
    # use a threshold
    if (diff_rg + diff_gb) > 0.10:
        result = 'c'
    else:
        result = 'g'

    return result 

# sets up memory structures for ALL  cameras
def config_camara_data(config):
    camera_config_list = config['camera']            # get list of camera configs
    camera_count = len(camera_config_list)                  # count of cameras in this config
    # a list of snapshot start times - used for measuring elapsed times
    camera_snapshot_times = []
    for i in range(camera_count):
        camera_snapshot_times.append(time.perf_counter())

    # generate memory structures for the identify_new_detections
    # Each Camaera has:
    #   bbox_stack_list - list of numpy arrays - 1 array per region -- 1 row per detected objects
    #   bbox_push_list  - list of lists,       - 1 list per region  -- how many objects were pushed to the stack
    # With multiple cameras..
    #  this becomes:
    #   bbox_stack_lists - one list per  camera
    #   bbox_push_lists  - one list per camera
    bbox_stack_lists = []
    bbox_push_lists = []
    for camera_id in range(camera_count):
        camera_config = camera_config_list[camera_id]
        print ("Camera_Config:", camera_config)
        regions_config = camera_config['regions']
        dedup_depth = camera_config['dedup_depth']
        bbox_stack_list = []
        bbox_push_list = []
        for i, region in enumerate(regions_config):
            bbox_stack_list.append(np.zeros((dedup_depth,4)))
            # need a list of object count pushed to the stack
            # we are pushing 1 bbox == object for dedup_DEPTH
            # so that is dedup_depth x 1
            region_push_list = []
            for i in range(dedup_depth):
                region_push_list.append(1)
            bbox_push_list.append(region_push_list)
        bbox_stack_lists.append(bbox_stack_list)
        bbox_push_lists.append(bbox_push_list)

    return camera_config_list, camera_count, camera_snapshot_times, bbox_stack_lists, bbox_push_lists

def get_reolink_url(scheme, ip):
    '''
    construct the camera base URL
    '''
    url = f"{scheme}://{ip}/cgi-bin/api.cgi"
    return url

def get_reolink_snapshot(url, username, password):
    '''
    return a snapshot from the camera
    - full resolutioin
    - numpy array
    '''

    # Create the URL
    randomstr = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    snap = url + "?cmd=Snap&channel=0&rs=" \
            + randomstr \
            + "&user=" + username \
            + "&password=" + password

    try:
        # debug
        # print ("Camera Request HTTP Request: ", snap)
        reader = urllib.request.urlopen(snap, timeout=10)
        img_bytes = bytearray(reader.read())
        img_array = Image.open(io.BytesIO(img_bytes))
        img_numpy = numpy.array(img_array)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        # print (" get_reolink_snapshot: ** captured ** ", img_numpy.shape)
        return img_bgr
    except:
        print ("get_snap:", url)
        return None


def append_crop_region(regions, crop_corner, crop_size):
    xmin = crop_corner[1]
    xmax = xmin + crop_size[1]
    ymin = crop_corner[0]
    ymax = ymin + crop_size[0]

    regions.append(((ymin, ymax), (xmin, xmax)))

    return regions

# Deprecate - reolink2tflite only
def config_camera_regions(camera_config):
    # 
    # - - Camera Regions - - - 
    #     should be in JSON as [xmin, ymin, height, width]

    dedup_depth = camera_config['dedup_depth']
    regions_config = camera_config['regions']


    regions = []
    bbox_stack_list = []
    bbox_push_list = []
    for i, region in enumerate(regions_config):
        append_crop_region(regions, (region[0], region[1]), (region[2], region[3]))
        print ("Crop {} Dimensions: corner: {}:{}  size: {}:{}".format(i, region[0], region[1], region[2], region[3]))
        bbox_stack_list.append(np.zeros((dedup_depth,4)))
        # need a list of object count pushed to the stack
        # we are pushing 1 bbox == object for dedup_DEPTH
        # so that is dedup_depth x 1
        region_push_list = []
        for i in range(dedup_depth):
            region_push_list.append(1)
        bbox_push_list.append(region_push_list)

    return regions, bbox_stack_list, bbox_push_list

def extract_regions(config, image):
    regions_config = config['regions']

    for i, region in enumerate(regions_config):
        ymin = region[0]
        xmin = region[1]
        ymax = ymin + region[2]
        xmax = xmin + region[3]
        # extract (crop it out)
        regional_image = image[ymin:ymax, xmin:xmax]
        # resize to 480x640 -- but it's (width, height) !!
        width = 640
        height = 480
        dim = (width, height)
        # resize image
        resized_image = cv2.resize(regional_image, dim, interpolation = cv2.INTER_AREA)
        # reshape to (1,480,640,3)
        np_image = resized_image.reshape((1,height,width,3))

        if i == 0:
            np_images = np_image
        else:
            np_images = np.append(np_images, np_image, 0)

    return np_images
# 
#
# Given camera config
#   get frame
#   extract regions
#   return numpy array [regions, 480, 640, 3]
def get_camera_regions(config):
    start = time.perf_counter()
    
    name = config['name']
    url = get_reolink_url('http', config['ip'])
    username = config['username']
    password = config['password']
    rotation_angle = int(config['rotation_angle'])
    # get the full image
    # print ("get_camera_regions: url = ", url)
    full_image = get_reolink_snapshot(url, username, password)
    
    # if the image exists..
    if full_image is not None:
        # rotate the image
        rot_full_image = imutils.rotate(full_image, rotation_angle)
        # print ("Camera: {} {} -- Frame Captured".format(name, start))
        np_images = extract_regions(config, rot_full_image)
    else:
        print ("Camera: {} {} -- snapshot timeout".format(name, start))
        np_images = None
    
    return name, np_images