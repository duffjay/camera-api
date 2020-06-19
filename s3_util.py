import string
import logging
import boto3
from botocore.exceptions import ClientError

import aws_util
import settings

log = logging.getLogger(__name__)


def get_object_list_from_s3(bucket, folder):
    s3 = settings.aws_session.resource('s3')
    s3_bucket = s3.Bucket(bucket)

    object_list = []
    for object in s3_bucket.objects.all():
        # now filter by the folder key
        if object.key.startswith(folder):
            # omit the folder itself
            if object.key != folder:
                object_list.append(object.key)
                # print (object.key)

    return object_list

def upload_file(local_file_name, bucket, s3_key):

    s3 = settings.aws_session.resource('s3')
    
    # while - to handle the errors w/ retry
    url = None
    retry_count = 0
    while True:
        try:
            response = s3.meta.client.upload_file(local_file_name, bucket, s3_key)
            url = f'https://s3.amazonaws.com/{bucket}/{s3_key}'
            break    # finished happily
        except ClientError as e:
            if e.response['Error']['Code'] == 'ExpiredTokenException':
                log.info(f's3_util/upload_file - expired token, updating ... ')
                settings.aws_session = aws_util.get_session()

            else:
                log.error (f's3_util/upload_file - unhandled AWS ClientError:{e}')
                break
        except Exception as e:
            log.error ("General Exception:", e)
        else:
            log.error(f's3_util/upload_file - Unexpected error')
            break
        retry_count += 1
        if retry_count > 3:
            log.error (f's3_util/upload_file - !!! Retry Count exceeded !!!')
            break

    return url