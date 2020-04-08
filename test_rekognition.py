import boto3

import s3_util
import rekognition_util

# global values
aws_profile = 'jmduff'
collection_id = 'family'
bucket='jmduff.rekognition'
folder='family/'

def update_session(aws_profile):
    return boto3.session.Session(profile_name = aws_profile)

def main():

    session = update_session(aws_profile)

    photo_list = s3_util.get_object_list_from_s3(session, bucket, folder)
    print ('{} photos in the list'.format(len(photo_list)))
    face_count = rekognition_util.add_faces_to_collection(session, bucket, photo_list, collection_id)


if __name__ == "__main__":
    main()