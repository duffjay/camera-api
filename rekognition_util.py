import cv2
import boto3
from botocore.exceptions import ClientError

import aws_util
import settings

# https://docs.aws.amazon.com/rekognition/latest/dg/rekognition-dg.pdf



def create_collection(session, collection_id):

    client=session.client('rekognition')

    #Create a collection
    print('Creating collection:' + collection_id)
    response=client.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')
    return response['CollectionArn']

def add_faces_to_collection(bucket, photo_list, collection_id):
    client = settings.aws_session.client("rekognition")

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
                settings.aws_session = aws_util.get_session()

            elif e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(" - - - creating collection - - - ")
                create_collection(session, collection_id)

            else:
                print ("unhandled AWS ClientError:", e)
                break
        except Exception as e:
            print ("General Exception:", e)
        # else:
        #     print("Unexpected error")
        #     break
        retry_count += 1
        if retry_count > 3:
            print ("!!! Retry Count exceeded !!!")
            break

    
    return face_count

def search_faces_by_image(collection_id, np_image):
    region = "us-east-1"

    retval, image_bytes = cv2.imencode('.jpg', np_image)
    image = {"Bytes" : bytearray(image_bytes)}      # without this, the type = ndarray and won't pass validation test on aws

    retry_count = 0
    face_id = 0
    similarity = 0.0
    while True:
        try:
            rekognition = settings.aws_session.client("rekognition", region)
            response = rekognition.search_faces_by_image(
                CollectionId=collection_id,
                FaceMatchThreshold=70.0,
                Image=image,
                MaxFaces=1,
                QualityFilter='AUTO'
            )
            # process the JSON response
            # searched = response['SearchedFaceBoundingBox']
            # print ("searched: ", searched)
            face_matches = response['FaceMatches']
            for match in face_matches:
                similarity = float(match['Similarity'])
                face = match['Face']
                face_id = face['FaceId']
                # print ("Matched Face: {}  {:.2f}".format(face_id, similarity))
            break
        except ClientError as e:
            if e.response['Error']['Code'] == 'ExpiredTokenException':
                print(" - - - UPDATE SESSION - - - ")
                settings.aws_session = aws_util.get_session()
                # no break - you can retry

            elif e.response['Error']['Code'] == 'InvalidParameterException':
                print(" - - - no faces? - - - ")
                break       # no reason to try again

            else:
                print ("unhandled AWS ClientError:", e)
                break       # no reason to try again
        
        except Exception as e:
            print ("General Exception:", e)
            break

        # else:
        #     print("Unexpected error")
        #     break       # no reason to try again
        
        retry_count += 1
        if retry_count > 3:
            print ("!!! Retry Count exceeded !!!")
            break

    return (face_id, similarity)



     # {'SearchedFaceBoundingBox': {'Width': 0.08219089359045029, 'Height': 0.1460799276828766, 'Left': 0.12093808501958847, 'Top': 0.32774946093559265}, 'SearchedFaceConfidence': 99.99976348876953, 'FaceMatches': [{'Similarity': 96.39073181152344, 'Face': {'FaceId': 'de74929a-deb4-492e-b39b-e5708bc481c8', 'BoundingBox': {'Width': 0.0609435997903347, 'Height': 0.14026999473571777, 'Left': 0.19517099857330322, 'Top': 0.1603659987449646}, 'ImageId': '6ea067a3-66ec-333f-ad05-531aeec3acac', 'ExternalImageId': '20160918_135541.jpg', 'Confidence': 100.0}}], 'FaceModelVersion': '4.0', 'ResponseMetadata': {'RequestId': 'd5369d25-301b-4d9f-b8a9-339c148cb13f', 'HTTPStatusCode': 200, 'HTTPHeaders': {'content-type': 'application/x-amz-json-1.1', 'date': 'Thu, 09 Apr 2020 17:31:53 GMT', 'x-amzn-requestid': 'd5369d25-301b-4d9f-b8a9-339c148cb13f', 'content-length': '544', 'connection': 'keep-alive'}, 'RetryAttempts': 0}}
