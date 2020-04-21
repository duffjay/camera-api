import boto3

import settings

# aws session
def get_session():
    return boto3.session.Session(profile_name = settings.aws_profile)