import sys

import boto3

import aws_util
import notification_util
import settings


def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    settings.init(config_filename)

    sns = settings.aws_session.client('sns')
  

    # Send message to SQS queue
    response = sns.publish(
        TopicArn='arn:aws:sns:us-east-1:333408648190:home_security_event',

        Message='test #13 event:  backdoor \nhttps://photos.app.goo.gl/p8yPhr5m6v7H4Fr96',
        Subject='test_notification',
        MessageStructure='string'
    )

    print(f"SNS Publish Response: {response['MessageId']}")



if __name__ == "__main__":
    main()