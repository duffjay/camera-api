import sys
import base64



import aws_util
import s3_util
import settings



def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    settings.init(config_filename)

    # try not sending it an active session
    #ses = settings.aws_session.client('s3')
  
    bucket = settings.aws_s3_public_image
    s3_key = '20200619/15920694963-3-3.jpg'
    file_name = 'faces/15920694963-3-3.jpg'

    print (f'BEFORE present in {bucket} / {s3_key}')
    object_list = s3_util.get_object_list_from_s3(bucket, s3_key)
    for obj in object_list:
        print (obj)


    print (f'PUTTING file {file_name} -> {bucket} {s3_key}')
    response = s3_util.upload_file(file_name, bucket, s3_key)
    print (f'AFTER - response: {response}')



if __name__ == "__main__":
    main()