import logging
import cv2
import boto3
from botocore.exceptions import ClientError

import aws_util
import settings

# https://docs.aws.amazon.com/rekognition/latest/dg/rekognition-dg.pdf

log = logging.getLogger(__name__)


def create_collection(collection_id):

    client=settings.aws_session.client('rekognition')

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
    for photo_path in photo_list:
        image = {"S3Object": {"Bucket" : 'jmduff.rekognition', "Name" : photo_path}}
        photo = photo_path[(photo_path.rfind('/') + 1):]
        print ("\nImage JSON:", image)
        print (photo)
        face_count = 0

        # while - to handle the errors w/ retry
        retry_count = 0
        while True:
            try:
                response = client.index_faces(CollectionId = collection_id,
                    Image=image,
                    ExternalImageId=photo,
                    MaxFaces=24,
                    QualityFilter="AUTO",
                    DetectionAttributes=['ALL']
                    )
                print ("Add Faces To Collection - Reults:", photo)
                print ("  Faces indexed:")
                for faceRecord in response['FaceRecords']:
                    print ('    Face ID: ' + faceRecord['Face']['FaceId'])
                    print ('    Location: {}'.format(faceRecord['Face']['BoundingBox']))

                print ("  Faces NOT indexed:")
                for unindexedFace in response['UnindexedFaces']:
                    # print (' unindexedFace', unindexedFace)
                    # print ('   Location {}'.format(unindexedFace['Face']['FaceId']))
                    print ('   Reaosons:')
                    for reason in unindexedFace['Reasons']:
                        print ('     ' + reason)
                face_count = len(response['FaceRecords'])
                print ("Added {} face count: {}".format(photo_path, face_count))
                break    # finished happily
            except ClientError as e:
                if e.response['Error']['Code'] == 'ExpiredTokenException':
                    print(" - - - UPDATE SESSION - - - ")
                    settings.aws_session = aws_util.get_session()

                elif e.response['Error']['Code'] == 'ResourceNotFoundException':
                    print(" - - - creating collection - - - ")
                    create_collection(collection_id)

                else:
                    print ("unhandled AWS ClientError:", e)
                    break
            except Exception as e:
                print ("General Exception:", e)
            else:
                print("Unexpected error")
                break
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
                log.warning(f'rekognition: UPDATE SESSION')
                settings.aws_session = aws_util.get_session()
                # no break - you can retry

            elif e.response['Error']['Code'] == 'InvalidParameterException':
                log.warning(f'rekognition: no faces?')
                break       # no reason to try again

            else:
                log.error(f'unhandled AWS ClientError: {e}')
                break       # no reason to try again
        
        except Exception as e:
            log.error(f'rekognition - General Exception: {e}')
            break

        # else:
        #     print("Unexpected error")
        #     break       # no reason to try again
        
        retry_count += 1
        if retry_count > 3:
            log.error(f'rekognition: !!! Retry Count exceeded !!!')
            break

    return (face_id, similarity)
