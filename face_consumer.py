import os
import sys
import time
import json

import cv2
import threading
import queue
import numpy as np
import traceback

import rekognition_util

import settings

# Pull images off the faceQueue
# - check with OpenCV if there is a face
# - submit to AWS Rekognition
# If known face detected, then write status
def face_consumer(consumer_id):

    # cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        try:
            # Consumer tasks
            # - once/frame
            #    - ALL regions of the frame
            data = settings.faceQueue.get(block=False)
            camera_id, region_id, image_time, np_image = data
            start = time.perf_counter()  
            face_count = 0
            print ("------------------------------------------------------- pulled", camera_id, region_id, image_time, np_image.shape)
            # check if there is a face w/ OpenCV
            match_id = 0
            similarity = 0.0
            try:
                # cv2.imshow('face candidate', np_image)
                # # stop all on a 'q' in a display window
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     global run_state
                #     run_state = False
                #     break
                # process the image
                image_name = "faces/{}-{}-{}.jpg".format(image_time, camera_id, region_id)
                cv2.imwrite(image_name, np_image)
                image_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
                # print (image_gray)
                faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)
                face_count = len(faces)
                # send to Rekognition
                if face_count > 0:
                    match_id, similarity = rekognition_util.search_faces_by_image("family", np_image)
                # write a action JSON
                detection_json_str = { "image_name" : image_name, "face_count" : face_count, "match_id" : match_id, "similarity" : similarity}
                detection_json = json.dumps(detection_json_str)
                json_name = "faces/{}-{}-{}.json".format(image_time, camera_id, region_id)
                with open(json_name, "w") as json_file:
                    json_file.write(detection_json)
                    json_file.close()

            except Exception as e:
                print ("--------------------------------------------------- opencv ERROR:", e)
                
            elapsed = time.perf_counter() - start
            with settings.safe_print:
                print ('   FACE-CONSUMER:--Camera ID: {} region ID: {} image_timestamp {}  inference time:{:02.2f} sec  faces: {} {} {:.2f}'.format(camera_id, region_id, image_time, elapsed, face_count, match_id, similarity))

            


        except queue.Empty:
            with settings.safe_print:
                print ('   FACE-CONSUMER:  nothing on queue')
            pass
        
        except Exception as e:
            with settings.safe_print:
                print ('   FACE-CONSUMER:  ERROR {}'.format(e))
                traceback.print_exc()


        time.sleep(0.5)
        # stop?
        if settings.run_state == False:
            break
        
    return 