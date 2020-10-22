import os
import io
import sys

import numpy
import cv2
from PIL import Image

import time
import base64
import urllib.request

import gen_util
import camera_util



import settings


# works
url1 = 'http://10.0.0.22/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=KL94LT46X6&user=admin&password=honey_E4362'
url2 = 'http://10.0.0.22/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=KL94LT46X6&user=admin&password=honey_E4362'
url3 = 'http://10.0.0.22/img/video.mjpeg'

# from William Opperman
rtsp1 = "rtsp://USER:PASSWORD@IPADDRESS:554/live.sdp"
rtsp2 = "rtsp://USER:PASSWORD@IPADDRESS:554/live2.sdp"
rtsp = "rtsp://admin:honey_E4362@10.0.0.22:554/live.sdp"

# google: honeywell HC30 cameras
# browser (chrome - not Firefox)  10.0.0.22
# this is an unsafe connection
# also verify with ping

# the manual is in  Downloads/30Series ..pdf
# Google Drive Security


# vcap = cv.VideoCapture("rtsp://192.168.1.2:8080/out.h264")





# vcap = cv2.VideoCapture(rtsp)
# i = 0
# while(1):
#     ret, frame = vcap.read()
#     cv2.imshow('VIDEO', frame)
#     cv2.waitKey(1)
#     i = i + 1
#     print (type(frame), i, time.time())



def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    config = gen_util.read_app_config(config_filename)

    # set the global values
    settings.init(config_filename)

    # camera config - list all found in the configuratio0n
    camera_config_list = config['camera']
    camera_count = len(camera_config_list)
    for i, camera_config in enumerate(camera_config_list):
        print (i, camera_config, '\n')
    # choose a camera
    camera_id = int(input("Enter Camera ID> "))  

    # review image regions
    camera_config = camera_config_list[camera_id]
    regions_config = camera_config['regions']
    for i, region_config in enumerate(regions_config):
        print ("top left corner (y, x): ({}, {})   region size (width, height): ({}, {})".format(
            region_config[0], region_config[1], region_config[2], region_config[3]))

    
    print ("\n - - If you don't like these values, edit app_*.json and rerun this - - ")
    resize_dimensions = [(640,480), (1200,720), (1920,1440)]
    resize_input = int(input ("Resize Factor (width, height), \n{}\nEnter 0,1,2 > ".format(resize_dimensions)))
    dim = resize_dimensions[resize_input]


    # must be a honeywell 
    print ( f'camera mfr = {camera_config["mfr"]}')
    assert camera_config['mfr'] == "Honeywell"

    # open the video stream
    stream = False
    video_stream = camera_util.open_video_stream(camera_id, camera_config, stream)

    for i in range(2500):
        start_time = time.perf_counter()  

        # Honeywell video stream
        frame = camera_util.get_camera_full(camera_id, video_stream)
        camera_name, np_images, is_color = camera_util.get_camera_regions_from_full(frame, camera_id, camera_config, stream)

        #camera_name, np_images, is_color = camera_util.get_camera_regions(camera_id, camera_config, False)
        print (" {:04d}  main -- camera: {}  secs: {:02.2f}".format(
            i, camera_name, (time.perf_counter() - start_time)))
        
        if np_images is not None:
            for i, np_image in enumerate(np_images):
                # img_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                region_name = 'region_{}'.format(i)
                
                resized_img = cv2.resize(np_image, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow(region_name, resized_img)
        else:
            # troubleshooting tips
            print ("- - NO Image returned - - ")
            print (" - access via Reolink app on phone")
            print (" - check IP of camera")
            print (" - edit: camera_util.get_reolink_snapshot print out the HTTP string")
            print ("         and test in a browser")

        # time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()