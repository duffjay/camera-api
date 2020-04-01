import io
import cv2
import numpy
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

# DEPRECATED
def DELETE_get_camera(ip, port, username, password, mfr):
    '''
    get the camera object
    '''

    if mfr == 'Amcrest':
        # construct camera URL
        URL = "http://{}:{}@{}/cgi-bin/mjpg/video.cgi?channel=0&subtype=1".format(username, password, ip)
    elif mfr == 'Reolink':
        # URL = "http://{}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user={}&password={}".format(ip, username, password)
        URL = "http://{}/cgi-bin/api.cgi?cmd=Snap&channel=0&user={}&password={}".format(ip, username, password)
    else:
        print ("** bad mfr value: ", mfr)

    # http://192.168.1.122/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user=admin&password=sT1nkeye
    # rtsp://admin:sT1nk&ye@192.168.1.122:554//h264Preview_01_main

    camera = cv2.VideoCapture(URL)     # returns a VideoCapture object
    camera.set(cv2.CAP_PROP_FPS, 20)   # set the capture rate - not sure this did anyting
    # return the VideoCapture object
    return camera

def config_camara_data(camera_config):
    camera_config_list = camera_config['camera']
    camera_count = len(camera_config_list)
    camera_snapshot_times = []
    for i in range(camera_count):
        camera_snapshot_times.append(time.perf_counter())
    return camera_config_list, camera_count, camera_snapshot_times

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