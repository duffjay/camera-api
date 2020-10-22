import numpy as np
import cv2
import imutils

import io
import os


import random
import string
import urllib


import time
import logging

import cv2
from PIL import Image

import settings


log = logging.getLogger(__name__)


def automatic_brightness_and_contrast(image, camera_id, region_id, is_color):
    if is_color == 1:
        clip_hist_percent = settings.color_image_auto_correct_clips_array[camera_id, region_id]
    else:
        clip_hist_percent = settings.gray_image_auto_correct_clips_array[camera_id, region_id]

    
    if clip_hist_percent > 0.0:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate grayscale histogram
            hist = cv2.calcHist([gray],[0],None,[256],[0,256])
            hist_size = len(hist)
            # Calculate cumulative distribution from the histogram
            accumulator = []
            accumulator.append(float(hist[0]))
            for index in range(1, hist_size):
                accumulator.append(accumulator[index -1] + float(hist[index]))

            # Locate points to clip
            maximum = accumulator[-1]
            clip_hist_percent *= (maximum/100.0)
            clip_hist_percent /= 2.0

            # Locate left cut
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1

            # Locate right cut
            maximum_gray = hist_size -1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1
            gray_range = maximum_gray - minimum_gray

            # Calculate alpha and beta values
            if (maximum_gray - minimum_gray) == 0:
                log.warning('camera-util/automatic_brihtness_and_contrast warning')
                log.warning('-- divide by zero (max-min) = 0')
                auto_result = image
                alpha = 0
                beta = 0
            else:
                alpha = 255 / (maximum_gray - minimum_gray)
                beta = -minimum_gray * alpha
                '''
                # Calculate new histogram with desired range and show histogram 
                new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
                plt.plot(hist)
                plt.plot(new_hist)
                plt.xlim([0,256])
                plt.show()
                '''
                auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            log.debug(
                f'camera-util/auto_bright_cont - is_color: {is_color}  clip hist % {clip_hist_percent:02.2f}'\
                f'  hist size: {hist_size}  max/max: {maximum_gray}/{minimum_gray} {gray_range}  a/b: {alpha:02.2f}/{beta:02.2f}')

        except Exception as e:
            log.error("camera-util/automatic_brihtness_and_contrast error")
            log.error(f"-- eneral xception: {e}")
            auto_result = image
            alpha = 0
            beta = 0
    else:
        auto_result = image
        alpha = 0
        beta = 0

    return (auto_result, alpha, beta)

# is the image color?
#  return 0 == grayscale
#         1 == color
# append on the file names
def get_color(image):
    height, width, channels = image.shape
    # grab a center section 40x40
    size = 40
    start_y = int((height / 2) - (size / 2))
    start_x = int((width / 2) - (size / 2))
    end_y = start_y + size
    end_x = start_x + size
    # arbitrarily calling the planes r, g, b  (not really sure if it's rgb or bgr)
    #             but it doesn't matter
    r = image[:, :, 0]          # gotta be integers
    b = image[:, :, 1]
    g = image[:, :, 2]
    # take difference - not exhaustive but good enough
    diff_rg = np.sum(np.absolute(r - b))                # get absolute values - then sum the differences
    diff_gb = np.sum(np.absolute(g - b))
    # use a threshold
    if (diff_rg + diff_gb) > 0.10:
        result = 1
    else:
        result = 0

    return result 


def get_camera_sleep_factor(config, camera_id):
    '''
    look up config and get the sleep_factor for this camera
    '''
    camera_config_list = config['camera']
    camera_config = camera_config_list[camera_id]
    sleep_factor = int(camera_config['sleep_factor'])
    log.debug(f'get sleep factor - camera_id: {camera_id}  sleep_factor: {sleep_factor}')
    return sleep_factor

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
        log.info(f'Camera_Config: {camera_config}')
        regions_config = camera_config['regions']
        dedup_depth = camera_config['dedup_depth']
        bbox_stack_list = []
        bbox_push_list = []
        for region_id, region in enumerate(regions_config):
            bbox_stack_list.append(np.zeros((dedup_depth,4)))
            # need a list of object count pushed to the stack
            # we are pushing 1 bbox == object for dedup_DEPTH
            # so that is dedup_depth x 1
            region_push_list = []
            for i in range(dedup_depth):
                region_push_list.append(1)
            bbox_push_list.append(region_push_list)
            # debug log
            log.debug(f'config_camera_data - cam#{camera_id} reg#{region_id}')
            log.debug(f'-- bbox_stack_list: {bbox_stack_list}')
            log.debug(f'-- bbox_push_list:  {bbox_push_list}')
        bbox_stack_lists.append(bbox_stack_list)
        bbox_push_lists.append(bbox_push_list)

    return camera_config_list, camera_count, camera_snapshot_times, bbox_stack_lists, bbox_push_lists



def get_reolink_url(scheme, ip):
    '''
    construct the camera base URL
    '''
    url = f"{scheme}://{ip}/cgi-bin/api.cgi"
    return url

def get_honeywell_rtsp(config):
    '''
    construct Honeywell rtsp
    '''
    usr = config['username']
    pw = config['password']
    ip = config['ip']
    url = f"rtsp://{usr}:{pw}@{ip}:554/live.sdp"
    return url

# https://support.reolink.com/hc/en-us/articles/360007010473-How-to-Live-View-Reolink-Cameras-via-VLC-Media-Player
# rtsp://admin:password@ip_address:554//h264Preview_01_main
def get_reolink_rtsp(config):
    '''
    construct Reolink rtsp
    '''
    usr = config['username']
    pw = config['password']
    ip = config['ip']
    url = f"rtsp://{usr}:{pw}@{ip}:554//h264Preview_01_main"
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
        img_numpy = np.array(img_array)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        # print (" get_reolink_snapshot: ** captured ** ", img_numpy.shape)
        return img_bgr
    except Exception as e:
        log.error(f'get_snap failed: {snap}')
        log.error(f'ERROR:  {e}')
        return None


def append_crop_region(regions, crop_corner, crop_size):
    xmin = crop_corner[1]
    xmax = xmin + crop_size[1]
    ymin = crop_corner[0]
    ymax = ymin + crop_size[0]

    regions.append(((ymin, ymax), (xmin, xmax)))

    return regions


def extract_regions(config, camera_id, image, is_color):
    regions_config = config['regions']

    for region_id, region in enumerate(regions_config):
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

        #  auto correct (but not the raw full image - its already correct)
        #  !!! Auto Correct is effectively disabled with > 20 (no camera will have 20 regions)
        #      there is some problem with the Honeywell image - axis 7 ???
        if region_id > 20:
            clip_hist_percent = 10
            resized_image, alpha, beta = automatic_brightness_and_contrast(resized_image, camera_id, region_id, is_color)
        # reshape to (1,480,640,3)
        np_image = resized_image.reshape((1,height,width,3))
        # build the array of images (as a numpy array)
        if region_id == 0:
            np_images = np_image
        else:
            np_images = np.append(np_images, np_image, 0)

    return np_images

# Verson 2 - for stream


def get_camera_regions_from_full(full_image, camera_id, config, stream):
    '''
    get the regions from a full image
      frame == full_image
    '''
    start = time.perf_counter()
    # config stuff
    name = config['name']
    rotation_angle = int(config['rotation_angle'])
    
    # if the image exists..
    if full_image is not None:
        is_color = get_color(full_image)
        # rotate the image
        rot_full_image = imutils.rotate(full_image, rotation_angle)
        # if streaming is on for this camera, store the full image
        if stream:
            cam_dir = f'cam{camera_id}'
            filename = f'{int(time.time())}.jpg'
            full_path = os.path.join("stream", cam_dir,  filename)
            cv2.imwrite(full_path, full_image)
        # print ("Camera: {} {} -- Frame Captured".format(name, start))
        np_images = extract_regions(config, camera_id, rot_full_image, is_color)
    else:
        log.warning(f'Camera: {name} {start} -- snapshot timeout')
        np_images = None
        is_color = 0
    return name, np_images, is_color

def get_camera_full(camera_id, video_stream):
    try:
        ret, frame = video_stream.read()
    except Exception as e:
        print (f'-- get_camera_full: {camera_id} ERROR {e}')
        traceback.print_exc()

    return frame



# 
#
# Given camera config
#   get frame
#   extract regions
#   return numpy array [regions, 480, 640, 3]
def get_camera_regions(camera_id, config, stream):
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
        is_color = get_color(full_image)
        # rotate the image
        rot_full_image = imutils.rotate(full_image, rotation_angle)
        # if streaming is on for this camera, store the full image
        if stream:
            cam_dir = f'cam{camera_id}'
            filename = f'{int(time.time())}.jpg'
            full_path = os.path.join("stream", cam_dir,  filename)
            cv2.imwrite(full_path, full_image)
        # print ("Camera: {} {} -- Frame Captured".format(name, start))
        np_images = extract_regions(config, camera_id, rot_full_image, is_color)
    else:
        log.warning(f'Camera: {name} {start} -- snapshot timeout')
        np_images = None
        is_color = 0
    
    return name, np_images, is_color

# moving to video stream functionality for Honeywell
#



def open_video_stream(camera_id, config, stream):
    # Honeywell
    if config['mfr'] == "Honeywell":
        url = get_honeywell_rtsp(config)            # url == the rtsp url
        video_stream = cv2.VideoCapture(url)
    # Reolink
    if config['mfr'] == "Reolink":
        url = get_reolink_rtsp(config)            # url == the rtsp url
        video_stream = cv2.VideoCapture(url)

    # print (f'Camera Stream: {url}')
    return video_stream

#
# Optical Zoom
#
def start_zoom(camera_id, config, direction, speed):
    '''
    begin the zoom operation
    '''
    if direction == 'in':
        operation = 'ZoomInc'
    elif direction == 'out':
        operation = 'ZoomDec'
    

    data = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": operation, "speed": speed}}]

    return  data

def stop_zoom(camera_id, config):
    data = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}]
    return data