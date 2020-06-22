import logging
import cv2
import boto3
from botocore.exceptions import ClientError

import aws_util
import settings

log = logging.getLogger(__name__)

