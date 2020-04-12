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


def get_validated_face_count(faces):
    face_sizes = []
    sufficient_size = False
    face_count = 0
    time_eligible = False
    # valid only if you haven't seen a face in 20 minutes
    time_since_last_match = (time.perf_counter() - settings.last_recognized_face_time)
    print (" time since:", time_since_last_match)
    if time_since_last_match > 1200.:
        time_eligible = True
    for (x, y, w, h) in faces:
        size = int(w * h)
        if size > 5000:
            sufficient_size = True
        face_sizes.append(size)
        face_count = face_count + 1 
    # return i == int, faces_sizes = strings (so you can serialize easily)
    
    return face_count, face_sizes, sufficient_size, time_eligible

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
                face_count, face_sizes, sufficient_size, time_eligible = get_validated_face_count(faces)
                print ("************** face count:", face_count)
                # send to Rekognition
                if time_eligible == True and face_count > 0:
                    match_id, similarity = rekognition_util.search_faces_by_image("family", np_image)
                    if similarity > 0.8:
                        settings.last_recognized_face_id = match_id
                        settings.last_recognized_face_time = time.perf_counter()
                # write a action JSON
                last_timestamp = "{:.2f}".format(settings.last_recognized_face_time)
                detection_json_str = { "image_name" : image_name, 
                    "face_count" : face_count, 
                    "face_sizes" : face_sizes, 
                    "sufficient_size" : sufficient_size, 
                    "last_recognition_time" : last_timestamp, 
                    "time_eligible" : time_eligible,
                    "match_id" : match_id, 
                    "similarity" : similarity}
                detection_json = json.dumps(detection_json_str)
                json_name = "faces/{}-{}-{}.json".format(image_time, camera_id, region_id)
                with open(json_name, "w") as json_file:
                    json_file.write(detection_json)
                    json_file.close()
                elapsed = time.perf_counter() - start
                with settings.safe_print:
                    print ('   FACE-CONSUMER:--Camera ID: {} region ID: {} image_timestamp {}  inference time:{:02.2f} sec  time eligible: {} faces: {} {} {:.2f}'.format(
                        camera_id, region_id, image_time, elapsed, time_eligible, face_count, match_id, similarity))

            except Exception as e:
                print ("--------------------------------------------------- processing faces - ERROR:", e)
                           
        except queue.Empty:
            pass
        
        except Exception as e:
            with settings.safe_print:
                print ('   FACE-CONSUMER:  ERROR {}'.format(e))
                traceback.print_exc()

        time.sleep(0.1)
        # stop?
        if settings.run_state == False:
            break
        
    return 