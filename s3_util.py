import string
import boto3


def get_object_list_from_s3(session, bucket, folder):
    s3 = session.resource('s3')
    s3_bucket = s3.Bucket(bucket)

    object_list = []
    for object in s3_bucket.objects.all():
        # now filter by the folder key
        if object.key.startswith(folder):
            # omit the folder itself
            if object.key != folder:
                object_list.append(object.key)
                print (object.key)

    return object_list