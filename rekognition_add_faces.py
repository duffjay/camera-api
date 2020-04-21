import sys
import boto3

import s3_util
import rekognition_util

import settings

# global values
aws_profile = 'jmduff'
collection_id = 'family'
bucket='jmduff.rekognition'

# - - pick a folder - - 
#folder='family/'
folder='mailman_1/'


def update_session(aws_profile):
    return boto3.session.Session(profile_name = aws_profile)

def main():
    # args
    config_filename = sys.argv[1]   # 0 based

    # init the variables in security settings
    # - init only in main()
    # https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    settings.init(config_filename)

    # not finsished - use settings.aws_session
    photo_list = s3_util.get_object_list_from_s3(bucket, folder)
    print ('{} photos in the list'.format(len(photo_list)))
    face_count = rekognition_util.add_faces_to_collection(bucket, photo_list, collection_id)
    print ("FINISHED - face count:", face_count)

if __name__ == "__main__":
    main()