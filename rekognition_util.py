import cv2
import boto3
from botocore.exceptions import ClientError

# https://docs.aws.amazon.com/rekognition/latest/dg/rekognition-dg.pdf

# aws session
def get_sessiion(profile):
    return boto3.session.Session(profile_name = profile)

def create_collection(session, collection_id):

    client=session.client('rekognition')

    #Create a collection
    print('Creating collection:' + collection_id)
    response=client.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')
    return response['CollectionArn']

def add_faces_to_collection(session, bucket, photo_list, collection_id):
    client = session.client("rekognition")

    # for photo in photo_list:
    #     image = {'S3Object':{'Bucket':bucket, 'Name:':photo}}
    #             {'S3Object':{'Bucket':bucket,'Name':photo}}
    #     print ("photo to add:", image)
    image = {"S3Object": {"Bucket" : 'jmduff.rekognition', "Name" : 'family/20160918_135541.jpg'}}
    photo_path = 'family/20160918_135541.jpg'
    photo = '20160918_135541.jpg'
    print ("Image JSON:", image)
    face_count = 0

    # while - to handle the errors w/ retry
    retry_count = 0
    while True:
        try:
            response = client.index_faces(CollectionId = collection_id,
                Image=image,
                ExternalImageId=photo,
                MaxFaces=3,
                QualityFilter="AUTO",
                DetectionAttributes=['ALL']
                )
            print ("Add Faces To Collection - Reults:", photo)
            print ("  Faces indexed:")
            for faceRecord in response['FaceRecords']:
                print ('    Face ID: ' + faceRecord['Face']['FaceId'])
                print ('    Location: {}'.format(faceRecord['Face']['BoundingBox']))

            print ("  Faces NOT indexed:")
            for unindexFace in response['UnindexedFaces']:
                print ('   Location {}'.fromat(unindexedFace['Face']['FaceId']))
                print ('   Reaosons:')
                for reason in unindexedFace['Reasons']:
                    print ('     ' + reason)
            face_count = len(response['FaceRecords'])
            break    # finished happily
        except ClientError as e:
            if e.response['Error']['Code'] == 'ExpiredTokenException':
                print(" - - - UPDATE SESSION - - - ")

            elif e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(" - - - creating collection - - - ")
                create_collection(session, collection_id)

            else:
                print ("unhandled AWS ClientError:", e)
                break
        else:
            print("Unexpected error")
            break
        retry_count += 1
        if retry_count > 3:
            print ("!!! Retry Count exceeded !!!")
            break

    
    return face_count

def search_faces_by_image(session, profile, collection_id, np_image):
    region = "us-east-1"

    # image_bytes = np_image.tobytes()
    retval, image_bytes = cv2.imencode('.jpg', np_image)
    # image_bytes = open ("/home/jay/Downloads/faces/1586378250-0-3.jpg", "rb")
    print ("image_bytes type:", type(image_bytes))
    image = {"Bytes" : bytearray(image_bytes)}

    retry_count = 0
    # while True:
    #     try:
    rekognition = session.client("rekognition", region)
    response = rekognition.search_faces_by_image(
        CollectionId=collection_id,
        FaceMatchThreshold=70.0,
        Image=image,
        MaxFaces=1,
        QualityFilter='AUTO'
    )
    #     break
    # except ClientError as e:
    #     print ("unhandled AWS ClientError:", e)
    #     break
    # else:
    #     print("Unexpected error")
    #     break
    
    # retry_count += 1
    # if retry_count > 3:
    #     print ("!!! Retry Count exceeded !!!")
    #     break

    return response