import sys
import base64

import boto3
from botocore.exceptions import ClientError

import aws_util
import notification_util
import settings

def image_to_data_url(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        img = f.read()
    return prefix + base64.b64encode(img).decode('utf-8')

def main():
    # args
    config_filename = sys.argv[1]   # 0 based
    settings.init(config_filename)

    ses = settings.aws_session.client('ses')
  
    encoded_image = image_to_data_url('faces/15920694963-3-3.jpg')


    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "Jay Duff <jay.duff@cfacorp.com>"

    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    # RECIPIENT = "duffjay@gmail.com"
    RECIPIENT = "jay.duff@cfacorp.com"

    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the 
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    CONFIGURATION_SET = "myconfig"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = "home security 18:37"

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = ("Amazon SES Test (Python)\r\n"
                "This email was sent with Amazon SES using the "
                "AWS SDK for Python (Boto)."
                )
                
    # f'<img src="{encoded_image}" alt="Red dot" />'
    # The HTML body of the email.
    BODY_HTML = (f'<html>'
        f'<head></head>'
        f'  <body>'
        f'    <h1>Amazon SES Test (SDK for Python)</h1>'
        f'    <p>This email was sent with</p>'
        f'    <img src="{encoded_image}" alt="Red dot" />'
        f'  </body>'
        f'</html>')

    print ("HTML: \n", BODY_HTML)
          

    # The character encoding for the email.
    CHARSET = "UTF-8"


    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = ses.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])



if __name__ == "__main__":
    main()