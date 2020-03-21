import io
import cv2
import numpy
import random
import string
import urllib
import numpy as np

from PIL import Image

def get_camera_config(config, camera_num):
    '''
    given the config JSON and a camera number, 
    return the config 
    '''
    camera_config = config['camera'][camera_num]
    return camera_config

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
        reader = urllib.request.urlopen(snap)
        img_bytes = bytearray(reader.read())
        img_array = Image.open(io.BytesIO(img_bytes))
        img_numpy = numpy.array(img_array)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
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