
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

# add the ReolinkAPI project
# 
cwd = os.getcwd()
reolink_api = os.path.abspath(os.path.join(cwd, '..', 'ReolinkCameraAPI'))
sys.path.append(reolink_api)

from Camera import Camera

import settings

# works
url = 'http://192.168.1.122/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=KL94LT46X6&user=admin&password=sT1nkeye'

data = {
    'username' : 'admin',
    'password' : 'sT1nkeye'
}

error_count = 0
success_count = 0


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


    # if camera has PTZ functionality
    # - only camera_id = 5
    if camera_id == 5:
        # login to specified camera
        ip = camera_config['ip']
        username = camera_config['username']
        password = camera_config['password']
        camera_5 = Camera(ip, username, password)
        print ("camera #5 logged in & instantiated")
        # get a zoom factor
        zoom_instruction = int(input("Enter seconds to zoom (+X to zoom in, -X to zoom out, 0 do do thing)"))
        if zoom_instruction > 0:
            camera_5.start_zooming_in()
            time.sleep(zoom_instruction)
            camera_5.stop_zooming()
        if zoom_instruction < 0:
            camera_5.start_zooming_out()
            time.sleep(-zoom_instruction)
            camera_5.stop_zooming()

    for i in range(250):
        start_time = time.perf_counter()  
        camera_name, np_images, is_color = camera_util.get_camera_regions(camera_id, camera_config, False)
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



