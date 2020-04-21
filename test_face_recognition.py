import os
import sys
import cv2
import time
import string

import boto3

import rekognition_util

# cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# client = session.client('rekognition')
collection_id = 'family'
aws_profile = "jmduff"

# aws session
def get_sessiion(profile):
    return boto3.session.Session(profile_name = 'jmduff')


#  input:  full image file path
# output:  bytes
def get_image_bytes_from_file(image_path):
    with open(image_pagh, 'rb') as image_file:
        return image_file.read()
    

# get image list from a directory
def get_image_file_list(path):
    files_list = []
    files = os.listdir(path)

    for file in files:
        if file.endswith('.jpg'):
            files_list.append(file)
    return files_list

image_dir = os.path.join("/home/jay/Downloads", "faces")
image_list = get_image_file_list(image_dir)

session = rekognition_util.get_session(aws_profile)

for i,image_filename in enumerate(image_list):
    start = time.perf_counter()

    # get & display the image
    image_path = os.path.join(image_dir, image_filename)
    print (i, image_path)
    image = cv2.imread(image_path)
    cv2.imshow('raw', image)

    # process the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print (image_gray)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)
    face_count = len(faces)
    print ("  Face Count: {}".format(face_count))

    # Draw the rectangle around each face
    if face_count > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        cv2.imshow('faces', image)

        # send to Rekognition
        match_id = rekognition_util.search_faces_by_image(session, aws_profile, "family", image)
        print ("Search Result:", match_id)
    else:
        cv2.destroyWindow("faces")

    cv2.waitKey(0)


    finish = time.perf_counter()
    print (f'Finished in {round(finish - start, 2)} seconds(s)')
    # input ('next')

cv2.destroyAllWindows()